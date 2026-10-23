"""VisRAG-Retriever baseline wrapper.

Provides document-level encoding and dense retrieval using VisRAG-Ret
(MiniCPM-V based vision-text dual-encoder).
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from io_utils import atomic_path


def weighted_mean_pooling(
    hidden: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Matches weighted pooling in the VisRAG reference implementation."""
    attention_mask_ = attention_mask * attention_mask.cumsum(dim=1)
    s = torch.sum(hidden * attention_mask_.unsqueeze(-1).float(), dim=1)
    d = attention_mask_.sum(dim=1, keepdim=True).float()
    return s / d


class VisRAGRetriever:
    """Wrapper for VisRAG-Ret dense dual-encoder model."""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"[VisRAG] Loading {model_path} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=True)
        self.model = AutoModel.from_pretrained(
            model_path,
            attn_implementation="sdpa",
            torch_dtype=(torch.bfloat16 if torch.cuda.is_available()
                         else torch.float32),
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model.eval()
        self.model.to(self.device)
        print("[VisRAG] VisRAG-Ret loaded successfully!")

    @torch.no_grad()
    def encode(self, text_or_image_list: List) -> np.ndarray:
        """Encodes queries or page images into normalized embeddings."""
        if isinstance(text_or_image_list[0], str):
            inputs = {
                "text": text_or_image_list,
                "image": [None] * len(text_or_image_list),
                "tokenizer": self.tokenizer,
            }
        else:
            inputs = {
                "text": [""] * len(text_or_image_list),
                "image": text_or_image_list,
                "tokenizer": self.tokenizer,
            }
        outputs = self.model(**inputs)
        attention_mask = outputs.attention_mask
        hidden = outputs.last_hidden_state

        reps = weighted_mean_pooling(hidden, attention_mask)
        embeddings = F.normalize(reps, p=2, dim=1).detach().cpu().numpy()
        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """Encodes a single query with retrieval prompt instruction."""
        prompt_query = (
            "Represent this query for retrieving relevant documents: " + query)
        return self.encode([prompt_query])  # shape: (1, dim)

    def encode_document_pages(
        self,
        image_paths: List[str],
        cache_file: Path = None,
    ) -> np.ndarray:
        """Encodes a list of page images for a single document."""
        if cache_file and cache_file.exists():
            try:
                cached = np.load(cache_file, allow_pickle=False)
                if (cached.ndim != 2 or len(cached) != len(image_paths)
                        or not np.isfinite(cached).all()):
                    raise ValueError("Invalid cached page vectors")
                return cached.astype(np.float32)
            except (OSError, ValueError, EOFError):
                pass

        reps_list = []
        batch_size = 4
        for i in range(0, len(image_paths), batch_size):
            batch = []
            try:
                for path in image_paths[i:i + batch_size]:
                    with Image.open(path) as source:
                        batch.append(source.convert("RGB"))
                reps_list.append(self.encode(batch).astype(np.float32))
            finally:
                for image in batch:
                    image.close()

        doc_reps = np.concatenate(reps_list, axis=0)

        if cache_file:
            with atomic_path(cache_file) as temporary:
                with temporary.open("wb") as output:
                    np.save(output, doc_reps, allow_pickle=False)

        return doc_reps

    def retrieve(
        self,
        query: str,
        image_paths: List[str],
        top_k: int = 5,
        cache_file: Path = None,
    ) -> List[Tuple[int, float, str]]:
        """Performs document-level dense retrieval for a query."""
        total_pages = len(image_paths)
        actual_k = min(top_k, total_pages)
        if actual_k <= 0:
            raise ValueError("Positive top_k and nonempty pages required.")

        doc_reps = self.encode_document_pages(
            image_paths, cache_file=cache_file)
        query_rep = self.encode_query(query)

        doc_reps_t = torch.from_numpy(doc_reps).to(self.device)
        query_rep_t = torch.from_numpy(query_rep).to(self.device)

        scores = torch.matmul(query_rep_t, doc_reps_t.T).squeeze(0)
        values = scores.cpu().tolist()  # Synchronizes completed GPU work.
        indices = sorted(range(total_pages), key=lambda i: (-values[i], i))
        return [(i + 1, float(values[i]), image_paths[i])
                for i in indices[:actual_k]]
