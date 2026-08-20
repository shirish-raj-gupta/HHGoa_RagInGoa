"""
Reciprocal Rank Fusion + MMR diversification.

RRF is used instead of score normalization because dense cosine and BM25 live
on incomparable scales; RRF only needs rank order, so it cannot be broken by a
score distribution shift in either arm.

MMR then trades a little relevance for diversity, which matters here because
the corpus carries ~3.4% near-duplicate passages - without it the top-k can be
five phrasings of the same passage, which looks like strong retrieval and
gives the generator nothing extra to ground on.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class FusedHit:
    passage_id: str
    score: float
    rank: int = 0
    dense_rank: int | None = None
    sparse_rank: int | None = None
    dense_score: float | None = None
    sparse_score: float | None = None
    sources: list[str] = field(default_factory=list)


def rrf(dense: list, sparse: list, k: int = 60, top_k: int = 20) -> list[FusedHit]:
    """
    Reciprocal Rank Fusion: score = sum over systems of 1/(k + rank).

    k=60 is the value from the original Cormack et al. paper and is a
    deliberate default, not a tuned one - it is flat enough that tuning it on
    our own eval set would be overfitting for no measurable gain.
    """
    acc: dict[str, FusedHit] = {}

    for r, h in enumerate(dense):
        f = acc.setdefault(h.passage_id, FusedHit(h.passage_id, 0.0))
        f.score += 1.0 / (k + r + 1)
        f.dense_rank, f.dense_score = r, getattr(h, "score", None)
        f.sources.append("dense")

    for r, h in enumerate(sparse):
        f = acc.setdefault(h.passage_id, FusedHit(h.passage_id, 0.0))
        f.score += 1.0 / (k + r + 1)
        f.sparse_rank, f.sparse_score = r, getattr(h, "score", None)
        f.sources.append("sparse")

    out = sorted(acc.values(), key=lambda f: -f.score)[:top_k]
    for i, f in enumerate(out):
        f.rank = i
    return out


def mmr(hits: list[FusedHit], vectors: dict[str, np.ndarray],
        lam: float = 0.7, k: int = 5) -> list[FusedHit]:
    """
    Maximal Marginal Relevance.

    lam=0.7 leans towards relevance; at lam=1.0 this is a no-op passthrough.
    Passages without a vector are treated as maximally diverse rather than
    dropped - losing a hit to a missing vector would be a silent recall bug.
    """
    if not hits or lam >= 1.0:
        return hits[:k]

    selected: list[FusedHit] = []
    pool = list(hits)
    best = max(hits, key=lambda h: h.score)
    denom = best.score or 1.0

    while pool and len(selected) < k:
        scored = []
        for h in pool:
            rel = h.score / denom
            v = vectors.get(h.passage_id)
            if v is None or not selected:
                div = 0.0
            else:
                sims = [float(v @ vectors[s.passage_id])
                        for s in selected if s.passage_id in vectors]
                div = max(sims) if sims else 0.0
            scored.append((lam * rel - (1 - lam) * div, h))
        scored.sort(key=lambda t: -t[0])
        pick = scored[0][1]
        selected.append(pick)
        pool.remove(pick)

    for i, h in enumerate(selected):
        h.rank = i
    return selected
