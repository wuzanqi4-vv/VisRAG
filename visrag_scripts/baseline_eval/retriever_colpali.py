"""Transformers-native ColPali HF retrieval with numeric page caches."""
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from io_utils import atomic_path


class ColPaliRetriever:
    """Uses the retrieval head shipped with colpali-v1.3-hf."""

    def __init__(self, model_path, device="cuda"):
        try:
            from transformers import ColPaliForRetrieval, ColPaliProcessor
        except ImportError as error:
            raise RuntimeError(
                "Use a separate ColPali environment with Transformers "
                "supporting ColPaliForRetrieval."
            ) from error
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = ColPaliForRetrieval.from_pretrained(
            model_path, local_files_only=True,
            torch_dtype=(torch.bfloat16 if self.device.startswith("cuda")
                         else torch.float32),
        ).to(self.device).eval()
        self.processor = ColPaliProcessor.from_pretrained(
            model_path, local_files_only=True, use_fast=False)
        # The selected checkpoint was trained with Query:, not Question:.
        self.processor.query_prefix = "Query: "

    @torch.no_grad()
    def encode_query(self, query):
        """Returns query vectors without the batch dimension."""
        inputs = self.processor(text=[query], return_tensors="pt")
        return self.model(**inputs.to(self.device)).embeddings[0].float()

    @torch.no_grad()
    def encode_document_pages(self, image_paths, cache_file=None):
        """Caches float32 arrays individually, without object/pickle data."""
        if cache_file and Path(cache_file).exists():
            try:
                with np.load(cache_file, allow_pickle=False) as archive:
                    vectors = [archive[f"page_{i}"].copy()
                               for i in range(len(image_paths))]
                if any(v.ndim != 2 or v.dtype != np.float32
                       or v.shape[1] != 128 or not np.isfinite(v).all()
                       for v in vectors):
                    raise ValueError("Invalid cached vectors")
                return [torch.from_numpy(v) for v in vectors]
            except (OSError, ValueError, KeyError, EOFError):
                pass
        vectors = []
        for path in image_paths:
            with Image.open(path) as source:
                image = source.convert("RGB")
            try:
                inputs = self.processor(images=[image], return_tensors="pt")
                vectors.append(self.model(
                    **inputs.to(self.device)).embeddings[0].float().cpu())
            finally:
                image.close()
        if cache_file:
            with atomic_path(cache_file) as temporary:
                with temporary.open("wb") as output:
                    np.savez(output, **{
                        f"page_{i}": v.numpy() for i, v in enumerate(vectors)})
        return vectors

    def score_maxsim(self, query_emb, doc_emb):
        """Scores one page without padded batches."""
        similarities = query_emb @ doc_emb.to(
            device=query_emb.device, dtype=query_emb.dtype).T
        return float(similarities.max(dim=-1).values.sum().item())

    def retrieve(self, query, image_paths, top_k=5, cache_file=None):
        """Returns stable (physical page, score, image path) tuples."""
        if top_k <= 0 or not image_paths:
            raise ValueError("Positive top_k and nonempty pages required.")
        pages = self.encode_document_pages(image_paths, cache_file)
        query_vectors = self.encode_query(query)
        ranked = [(i + 1, self.score_maxsim(query_vectors, v), path)
                  for i, (v, path) in enumerate(zip(pages, image_paths))]
        return sorted(ranked, key=lambda row: (-row[1], row[0]))[:top_k]
