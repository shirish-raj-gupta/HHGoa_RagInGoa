"""
Exact (brute-force) cosine search.

Why this exists: the chunking ablation compares CHUNKING STRATEGIES. Running
it over an approximate index means every measured difference is chunking
signal plus HNSW noise, and HNSW noise on this corpus turned out to be larger
than the effect being measured (an arm scored R@5 0.316 purely because the
graph was badly built). Exact search removes that confound entirely, so a
difference in the ablation table is a difference in chunking.

It is also simply faster at ablation scale: 2,707 queries x 53k chunks is one
144M-dot-product matmul, a couple of seconds in numpy, versus building an HNSW
graph per arm.

The production path still uses HNSW - 14.3M vectors cannot be scanned inside
200ms. This class is the ground truth that HNSW is TUNED AGAINST
(bench/sweep_hnsw.py), not a replacement for it.
"""
from __future__ import annotations

import numpy as np

from .dense import Hit


class ExactPartition:
    """Ground-truth cosine search over L2-normalized vectors."""

    def __init__(self, lang: str, dim: int = 384):
        self.lang, self.dim = lang, dim
        self.V: np.ndarray | None = None
        self.chunk_ids: list[str] = []
        self.passage_ids: list[str] = []

    def add(self, vectors: np.ndarray, chunk_ids: list[str],
            passage_ids: list[str]) -> None:
        V = np.ascontiguousarray(vectors, dtype=np.float32)
        # normalize defensively; cosine == dot product only if we do
        V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
        self.V = V if self.V is None else np.vstack([self.V, V])
        self.chunk_ids.extend(chunk_ids)
        self.passage_ids.extend(passage_ids)

    def search(self, qvec: np.ndarray, k: int = 10,
               expansion_search: int | None = None) -> list[Hit]:
        """`expansion_search` accepted and ignored - exact search has no knobs."""
        if self.V is None or not self.chunk_ids:
            return []
        q = np.asarray(qvec, dtype=np.float32).reshape(-1)
        q /= (np.linalg.norm(q) + 1e-9)
        sims = self.V @ q
        k = min(k, sims.shape[0])
        # argpartition is O(n); a full sort of 53k per query is wasted work
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [Hit(chunk_id=self.chunk_ids[int(i)],
                    passage_id=self.passage_ids[int(i)],
                    score=float(sims[i]), lang=self.lang, rank=r)
                for r, i in enumerate(idx)]

    def search_batch(self, Q: np.ndarray, k: int = 10) -> np.ndarray:
        """Top-k indices for many queries at once. One matmul, no Python loop."""
        Qn = np.ascontiguousarray(Q, dtype=np.float32)
        Qn /= (np.linalg.norm(Qn, axis=1, keepdims=True) + 1e-9)
        sims = Qn @ self.V.T
        k = min(k, sims.shape[1])
        idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        rows = np.arange(sims.shape[0])[:, None]
        return idx[rows, np.argsort(-sims[rows, idx], axis=1)]

    def self_retrieval_rate(self, n: int = 200, seed: int = 0) -> float:
        """Exact search is exact; this is 1.0 by construction, kept for parity."""
        return 1.0

    @property
    def size_bytes(self) -> int:
        return 0 if self.V is None else self.V.nbytes
