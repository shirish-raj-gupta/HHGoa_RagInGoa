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

import concurrent.futures
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

    # Chosen by measurement (bench/sweep_hnsw.py), against EXACT search as
    # ground truth. On 49,611 real e5 vectors:
    #
    #   dtype  M   ea   es   recall@10 vs exact   self-retr   p50      size
    #   f16    16  256  128  0.9925               0.995       1.80ms   44.5MB
    #   f32    32  256   64  0.9882               0.995       0.42ms   88.9MB
    #   i8     16  256  512  0.9412               1.000       3.58ms   22.3MB
    #
    # F16 gives the best recall per byte, and at 15 partitions memory is the
    # binding constraint (12.8GB vs 21.9GB for f32 across the full corpus).
    #
    # A WARNING ABOUT THE SANITY CHECK BELOW, because it misled this project
    # once. Self-retrieval on RANDOM unit vectors reports I8 at 0.085 and looks
    # like a smoking gun. It is an artifact: random vectors in 384 dimensions
    # are isotropic and mutually near-equidistant, so int8 quantization erases
    # the only distinctions there are. Real embeddings are anisotropic - e5
    # concentrates in a narrow cone, so a vector's true self-match sits far
    # above its neighbours and survives quantization intact. Measured on real
    # e5 vectors at identical component scale (sd 0.0510 in both cases):
    #
    #   random unit vectors   F32 0.980   F16 0.995   I8 0.085
    #   real e5 vectors       F32 0.990   F16 0.990   I8 0.990
    #
    # So I8 is NOT broken - it simply costs ~5 points of recall against exact.
    # The real lesson is about the test, not the dtype: a synthetic sanity
    # check can produce a confident false positive, and this one did.
    DTYPE = ScalarKind.F16

    def __init__(self, lang: str, dim: int = 384, *,
                 connectivity: int = 16, expansion_add: int = 128,
                 expansion_search: int = 64, dtype: ScalarKind | None = None):
        self.lang, self.dim = lang, dim
        self.expansion_search = expansion_search
        # the ACTUAL dtype, not the class default - size_bytes read
        # self.DTYPE and so under-reported f32 by 2x, which made the
        # HNSW sweep pick f32 believing it was the smaller option
        self.dtype = dtype or self.DTYPE
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
                            ScalarKind.I8: 1}.get(self.dtype, 4)
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
        routes = self.route(lang, fallback)
        if not routes:
            return []
        if len(routes) == 1:
            return self.partitions[routes[0]].search(qvec, k, expansion_search)

        hits: list[Hit] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(routes)) as ex:
            futures = [ex.submit(self.partitions[lg].search, qvec, k, expansion_search)
                       for lg in routes if lg in self.partitions]
            for f in concurrent.futures.as_completed(futures):
                hits.extend(f.result())
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
