"""
In-process dense index (usearch HNSW).

No network hop, ever - a hosted vector DB on the critical path would make the
<200ms CORE_RAG_LOOP claim impossible (ADR 0001 section 4).

Language partitioning lives here. The corpus is parallel across 14 languages,
so a Hindi query needs the Hindi partition plus English for cross-lingual
fallback: ~1.9M vectors searched instead of 14.3M. That is what makes the
full-validation corpus viable at all.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from usearch.index import Index, MetricKind, ScalarKind


@dataclass(slots=True)
class Hit:
    chunk_id: str
    passage_id: str
    score: float
    lang: str
    rank: int


class DensePartition:
    """One HNSW index over one language's chunks."""

    def __init__(self, lang: str, dim: int = 384, *,
                 connectivity: int = 16, expansion_add: int = 128,
                 expansion_search: int = 64, quantize: bool = True):
        self.lang, self.dim = lang, dim
        self.expansion_search = expansion_search
        self.index = Index(
            ndim=dim,
            metric=MetricKind.Cos,
            # int8 keeps the full-14 corpus at ~384 B/vector; measured cosine
            # agreement with fp32 is 0.990 mean / 0.987 min across scripts
            dtype=ScalarKind.I8 if quantize else ScalarKind.F32,
            connectivity=connectivity,
            expansion_add=expansion_add,
            expansion_search=expansion_search,
        )
        self.chunk_ids: list[str] = []
        self.passage_ids: list[str] = []

    def add(self, vectors: np.ndarray, chunk_ids: list[str],
            passage_ids: list[str]) -> None:
        start = len(self.chunk_ids)
        keys = np.arange(start, start + len(chunk_ids), dtype=np.uint64)
        self.index.add(keys, vectors.astype(np.float32), log=False)
        self.chunk_ids.extend(chunk_ids)
        self.passage_ids.extend(passage_ids)

    def search(self, qvec: np.ndarray, k: int = 10,
               expansion_search: int | None = None) -> list[Hit]:
        if len(self.chunk_ids) == 0:
            return []
        if expansion_search is not None:
            # deadline-aware degradation: the budget can drop ef_search
            self.index.expansion_search = expansion_search
        m = self.index.search(qvec.astype(np.float32).reshape(1, -1),
                              min(k, len(self.chunk_ids)), log=False)
        keys = np.atleast_1d(m.keys.flatten())
        dists = np.atleast_1d(m.distances.flatten())
        return [Hit(chunk_id=self.chunk_ids[int(kk)],
                    passage_id=self.passage_ids[int(kk)],
                    score=float(1.0 - d), lang=self.lang, rank=r)
                for r, (kk, d) in enumerate(zip(keys, dists))]

    @property
    def size_bytes(self) -> int:
        return self.index.memory_usage

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.index.save(str(path))
        path.with_suffix(".meta.json").write_text(json.dumps({
            "lang": self.lang, "dim": self.dim,
            "chunk_ids": self.chunk_ids, "passage_ids": self.passage_ids,
        }), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, *, view: bool = True) -> "DensePartition":
        meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        p = cls(meta["lang"], meta["dim"])
        # view() memory-maps instead of reading into RAM, so resident memory
        # tracks the hot set rather than the full partition
        p.index.view(str(path)) if view else p.index.load(str(path))
        p.chunk_ids, p.passage_ids = meta["chunk_ids"], meta["passage_ids"]
        return p


class DenseIndex:
    """Language-partitioned dense index with cross-lingual fallback routing."""

    def __init__(self, dim: int = 384, **kw):
        self.dim, self._kw = dim, kw
        self.partitions: dict[str, DensePartition] = {}

    def partition(self, lang: str) -> DensePartition:
        if lang not in self.partitions:
            self.partitions[lang] = DensePartition(lang, self.dim, **self._kw)
        return self.partitions[lang]

    def add(self, lang: str, vectors, chunk_ids, passage_ids) -> None:
        self.partition(lang).add(vectors, chunk_ids, passage_ids)

    def route(self, lang: str, fallback: str = "eng_Latn") -> list[str]:
        """Which partitions to search for a query in `lang`."""
        out = [lang] if lang in self.partitions else []
        if fallback in self.partitions and fallback != lang:
            out.append(fallback)
        return out or list(self.partitions)

    def search(self, qvec: np.ndarray, lang: str, k: int = 10, *,
               fallback: str = "eng_Latn",
               expansion_search: int | None = None) -> list[Hit]:
        hits: list[Hit] = []
        for lg in self.route(lang, fallback):
            hits.extend(self.partitions[lg].search(qvec, k, expansion_search))
        hits.sort(key=lambda h: -h.score)
        for r, h in enumerate(hits):
            h.rank = r
        return hits[:k]

    @property
    def size_bytes(self) -> int:
        return sum(p.size_bytes for p in self.partitions.values())

    @property
    def n_chunks(self) -> int:
        return sum(len(p.chunk_ids) for p in self.partitions.values())
