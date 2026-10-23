"""VisRAG-Retriever baseline wrapper.

Provides document-level encoding and dense retrieval using VisRAG-Ret
(MiniCPM-V based vision-text dual-encoder).
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer


class VisRAGRetriever:
    """Wrapper for VisRAG-Ret dense dual-encoder model."""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"[VisRAG] Loading VisRAG-Ret from {model_path} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        self.model.to(self.device)
        self.model.eval()

    def encode(self, inputs: List[Dict]) -> np.ndarray:
        """Encodes queries or images using VisRAG's internal encode method."""
        with torch.no_grad():
            outputs = self.model(data=inputs, tokenizer=self.tokenizer)
            reps = outputs.cpu().float().numpy()
        return reps

    def encode_query(self, query: str) -> np.ndarray:
        """Encodes a single query with retrieval prompt instruction."""
        prompt_query = "Represent this query for retrieving relevant document: " + query
        inputs = [{"text": prompt_query}]
        return self.encode(inputs)  # shape: (1, dim)

    def encode_document_pages(
        self,
        image_paths: List[str],
        cache_file: Path = None,
    ) -> np.ndarray:
        """Encodes a list of page images for a single document.

        Args:
            image_paths: List of file paths to rendered page images.
            cache_file: Optional path to save/load precomputed representations.

        Returns:
            np.ndarray of shape (num_pages, dim)
        """
        if cache_file and cache_file.exists():
            return np.load(cache_file)

        inputs = []
        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            inputs.append({"image": img})

        reps_list = []
        # Process in batches to avoid GPU OOM
        batch_size = 4
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i : i + batch_size]
            batch_reps = self.encode(batch)
            reps_list.append(batch_reps)

        doc_reps = np.concatenate(reps_list, axis=0)

        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, doc_reps)

        return doc_reps

    def retrieve(
        self,
        query: str,
        image_paths: List[str],
        top_k: int = 5,
        cache_file: Path = None,
    ) -> List[Tuple[int, float, str]]:
        """Performs document-level dense retrieval for a query.

        Args:
            query: The user query string.
            image_paths: List of page image paths for the document.
            top_k: Number of pages to retrieve.
            cache_file: Cache path for document representations.

        Returns:
            List of (page_num_1_indexed, score, image_path) sorted by score desc.
        """
        total_pages = len(image_paths)
        actual_k = min(top_k, total_pages)
        if actual_k <= 0:
            return []

        doc_reps = self.encode_document_pages(image_paths, cache_file=cache_file)
        query_rep = self.encode_query(query)

        doc_reps_t = torch.from_numpy(doc_reps).to(self.device)
        query_rep_t = torch.from_numpy(query_rep).to(self.device)

        # Cosine similarity / inner product
        scores = torch.matmul(query_rep_t, doc_reps_t.T).squeeze(0)
        top_scores, top_indices = torch.topk(scores, k=actual_k)

        results = []
        for score, idx in zip(top_scores.cpu().tolist(), top_indices.cpu().tolist()):
            page_num = idx + 1
            results.append((page_num, float(score), image_paths[idx]))

        return results
