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

    # MEASURED, do not "optimize" back to I8. usearch's ScalarKind.I8 expects
    # values in int8 range, not L2-normalized floats in [-1,1], and it does not
    # scale them for you. Feeding it unit vectors destroys the distance function
    # and therefore the HNSW graph. Self-retrieval (search a vector against an
    # index that CONTAINS it, expect rank 0) on 20k random unit vectors:
    #
    #     dtype   ef_search=64   ef_search=256
    #     F32          99.0%          100.0%
    #     F16         100.0%           99.5%
    #     I8            9.0%            1.0%     <- broken
    #
    # Pre-scaling by 127 does not help (9.5%); casting to int8 is worse (6.5%).
    # The failure is not uniform, which is what makes it dangerous: on real
    # embeddings it produced a plausible ablation table with one inexplicable
    # row (fixed_128_o0 R@5=0.474 against 0.864 for its neighbours), because
    # graph quality became a lottery per build rather than consistently bad.
    #
    # F16 costs 2 B/dim instead of 1 but is numerically sound. This is NOT the
    # same int8 as the ONNX model quantization, which IS validated and kept
    # (0.990 mean cosine agreement with fp32 across five scripts).
    DTYPE = ScalarKind.F16

    def __init__(self, lang: str, dim: int = 384, *,
                 connectivity: int = 16, expansion_add: int = 128,
                 expansion_search: int = 64, dtype: ScalarKind | None = None):
        self.lang, self.dim = lang, dim
        self.expansion_search = expansion_search
        self.index = Index(
            ndim=dim,
            metric=MetricKind.Cos,
            dtype=dtype or self.DTYPE,
            connectivity=connectivity,
            expansion_add=expansion_add,
            expansion_search=expansion_search,
        )
        self.chunk_ids: list[str] = []
        self.passage_ids: list[str] = []
        # passage_id -> index key, so MMR can read vectors instead of re-embedding
        self._key_of_passage: dict[str, int] = {}

    def add(self, vectors: np.ndarray, chunk_ids: list[str],
            passage_ids: list[str]) -> None:
        start = len(self.chunk_ids)
        keys = np.arange(start, start + len(chunk_ids), dtype=np.uint64)
        self.index.add(keys, vectors.astype(np.float32), log=False)
        self.chunk_ids.extend(chunk_ids)
        self.passage_ids.extend(passage_ids)
        for off, pid in enumerate(passage_ids):
            self._key_of_passage.setdefault(pid, start + off)

    def get_vectors(self, passage_ids: list[str]) -> dict[str, np.ndarray]:
        """
        Read stored vectors back out of the index, by passage.

        MMR needs candidate vectors. The orchestrator used to get them by
        re-embedding the passage TEXT, one forward pass per candidate, on the
        critical path - which cost 260-420ms of a 200ms budget and made the
        fuse stage dominate the whole loop. The vectors are already in the
        index; reading them is a memory lookup.

        Vectors come back at index precision (F16), which is ample for a
        diversity comparison.
        """
        keys, out = [], {}
        for pid in passage_ids:
            k = self._key_of_passage.get(pid)
            if k is not None:
                keys.append((pid, k))
        if not keys:
            return out
        got = self.index.get(np.asarray([k for _, k in keys], dtype=np.uint64))
        if got is None:
            return out
        arr = np.atleast_2d(np.asarray(got, dtype=np.float32))
        for (pid, _), row in zip(keys, arr):
            out[pid] = row
        return out

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

    def self_retrieval_rate(self, n: int = 200, seed: int = 0) -> float:
        """
        Sanity guard: a vector searched against an index that CONTAINS it must
        come back at rank 0. Anything below ~0.95 means the distance function
        is broken (wrong dtype, unnormalized vectors, dimension mismatch)
        rather than merely imprecise.

        This exists because a silently broken index still returns plausible
        top-k results and quietly ruins every retrieval metric downstream.
        """
        if not self.chunk_ids:
            return 1.0
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self.chunk_ids), min(n, len(self.chunk_ids)),
                         replace=False)
        vecs = self.index.get(np.asarray(idx, dtype=np.uint64))
        if vecs is None:
            return 1.0
        vecs = np.atleast_2d(np.asarray(vecs, dtype=np.float32))
        hit = 0
        for row, key in zip(vecs, idx):
            m = self.index.search(row.reshape(1, -1), 1, log=False)
            keys = np.atleast_1d(m.keys.flatten())
            if len(keys) and int(keys[0]) == int(key):
                hit += 1
        return hit / len(idx)

    @property
    def size_bytes(self) -> int:
        """
        Actual stored bytes. usearch's `memory_usage` reports an allocation
        figure that did not track vector count in testing (51.2 MB for both
        49,611 and 53,274 vectors), so it is not trustworthy for the ablation's
        "Index MB" column - compute it instead.
        """
        bytes_per_scalar = {ScalarKind.F32: 4, ScalarKind.F16: 2,
                            ScalarKind.I8: 1}.get(self.DTYPE, 4)
        vectors = len(self.chunk_ids) * self.dim * bytes_per_scalar
        graph = len(self.chunk_ids) * self.index.connectivity * 2 * 4
        return vectors + graph

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
        # rebuild the passage->key map; without it MMR silently gets no vectors
        # and falls back to plain RRF order
        for off, pid in enumerate(p.passage_ids):
            p._key_of_passage.setdefault(pid, off)
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
