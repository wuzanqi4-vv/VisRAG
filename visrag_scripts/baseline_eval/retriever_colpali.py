"""ColPali v1.3 multi-vector retrieval wrapper.

Supports late interaction (ColBERT / MaxSim) across document page visual embeddings
and query token embeddings.
"""

import os
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
from PIL import Image


class ColPaliRetriever:
    """Wrapper for ColPali v1.3 multi-vector retrieval model."""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self._load_model()

    def _load_model(self):
        """Loads ColPali model either via sentence-transformers or colpali-engine / transformers."""
        try:
            from sentence_transformers import MultiVectorEncoder
            print(f"[ColPali] Loading with sentence_transformers from {self.model_path}...")
            self.model = MultiVectorEncoder(self.model_path)
            self.backend = "sentence_transformers"
        except Exception as e:
            try:
                from colpali_engine.models import ColPali, ColPaliProcessor
                print(f"[ColPali] Loading with colpali_engine from {self.model_path}...")
                self.processor = ColPaliProcessor.from_pretrained(self.model_path)
                self.model = ColPali.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    device_map=self.device,
                )
                self.model.eval()
                self.backend = "colpali_engine"
            except Exception as e_engine:
                # Fallback to direct PaliGemma / Transformers
                from transformers import AutoProcessor, AutoModel
                print(f"[ColPali] Loading with transformers fallback from {self.model_path}...")
                self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
                self.model = AutoModel.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True,
                ).to(self.device)
                self.model.eval()
                self.backend = "transformers"

    def encode_query(self, query: str) -> torch.Tensor:
        """Encodes query into multi-vector token embeddings: shape (seq_len, dim)."""
        if self.backend == "sentence_transformers":
            emb = self.model.encode_query([query])[0]
            return torch.from_numpy(emb).to(self.device)
        elif self.backend == "colpali_engine":
            inputs = self.processor.process_queries([query]).to(self.device)
            with torch.no_grad():
                emb = self.model(**inputs)[0]
            return emb
        else:
            inputs = self.processor(text=[query], return_tensors="pt").to(self.device)
            with torch.no_grad():
                emb = self.model(**inputs)[0]
            return emb

    def encode_document_pages(
        self,
        image_paths: List[str],
        cache_file: Path = None,
    ) -> List[torch.Tensor]:
        """Encodes page images into multi-vector embeddings per page."""
        if cache_file and cache_file.exists():
            data = np.load(cache_file, allow_pickle=True)
            return [torch.from_numpy(arr).to(self.device) for arr in data]

        images = [Image.open(p).convert("RGB") for p in image_paths]
        embeddings = []

        if self.backend == "sentence_transformers":
            encoded = self.model.encode_document(images)
            embeddings = [torch.from_numpy(x).to(self.device) for x in encoded]
        elif self.backend == "colpali_engine":
            for img in images:
                inputs = self.processor.process_images([img]).to(self.device)
                with torch.no_grad():
                    emb = self.model(**inputs)[0]
                embeddings.append(emb)
        else:
            for img in images:
                inputs = self.processor(images=[img], return_tensors="pt").to(self.device)
                with torch.no_grad():
                    emb = self.model(**inputs)[0]
                embeddings.append(emb)

        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np_data = [emb.cpu().float().numpy() for emb in embeddings]
            np.save(cache_file, np.array(np_data, dtype=object), allow_pickle=True)

        return embeddings

    def score_maxsim(self, query_emb: torch.Tensor, doc_emb: torch.Tensor) -> float:
        """Computes ColBERT MaxSim score between query and document tokens.

        MaxSim = Sum_{q in Query} Max_{d in Doc} (q . d)
        """
        # query_emb: (q_len, dim), doc_emb: (d_len, dim)
        similarity_matrix = torch.matmul(query_emb, doc_emb.T)  # (q_len, d_len)
        max_sims = torch.max(similarity_matrix, dim=-1).values  # (q_len,)
        score = torch.sum(max_sims).item()
        return float(score)

    def retrieve(
        self,
        query: str,
        image_paths: List[str],
        top_k: int = 5,
        cache_file: Path = None,
    ) -> List[Tuple[int, float, str]]:
        """Retrieves top-k pages using MaxSim late interaction scoring."""
        total_pages = len(image_paths)
        actual_k = min(top_k, total_pages)
        if actual_k <= 0:
            return []

        doc_embs = self.encode_document_pages(image_paths, cache_file=cache_file)
        query_emb = self.encode_query(query)

        scored_pages = []
        for idx, doc_emb in enumerate(doc_embs):
            score = self.score_maxsim(query_emb, doc_emb)
            page_num = idx + 1
            scored_pages.append((page_num, score, image_paths[idx]))

        scored_pages.sort(key=lambda x: x[1], reverse=True)
        return scored_pages[:actual_k]
