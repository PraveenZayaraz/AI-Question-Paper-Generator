"""Text embedders. Every embedder returns L2-normalised float32 vectors,
so a dot product between two vectors is their cosine similarity."""
from __future__ import annotations

import hashlib
import re
from typing import List, Sequence

import numpy as np

_WORD = re.compile(r"[a-z0-9]+")


class SentenceTransformerEmbedder:
    """Dense semantic embeddings (default model: all-MiniLM-L6-v2, 384 dimensions)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.name = f"st:{model_name}"
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(
            list(texts), batch_size=64, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype=np.float32)


class HashingEmbedder:
    """Dependency-free fallback: hashed bag of words and word pairs.

    It only captures word overlap, not meaning, so retrieval quality is lower
    than with sentence-transformers. It exists so tests and CI run in seconds
    without downloading a model.
    """

    def __init__(self, dim: int = 2048):
        self.name = f"hashing:{dim}"
        self.dim = dim

    def _features(self, text: str) -> List[str]:
        words = _WORD.findall(text.lower())
        return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature in self._features(text):
                digest = hashlib.md5(feature.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] & 1 else -1.0
                out[row, bucket] += sign
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


def make_embedder(kind: str, model_name: str = "all-MiniLM-L6-v2"):
    if kind == "hashing":
        return HashingEmbedder()
    if kind == "st":
        return SentenceTransformerEmbedder(model_name)
    raise ValueError(f"Unknown embedder '{kind}'. Use 'st' or 'hashing'.")
