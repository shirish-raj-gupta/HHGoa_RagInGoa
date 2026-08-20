"""
HNSW recall-vs-latency sweep, and the operating point we actually chose.

The brief asks for this curve. It turned out to be load-bearing rather than
decorative: the default parameters (M=16, ef_add=128, ef_search=64) silently
built a graph with 0.84 self-retrieval on real e5 embeddings at 50k vectors,
which dragged one ablation arm to R@5 0.316. Real embeddings cluster far more
tightly than the random vectors a naive sanity check uses, so HNSW has a much
harder time navigating them.

Ground truth is EXACT cosine search (src/index/exact.py), so "recall" here is
"what fraction of the exact top-k does the approximate index return", which is
the only definition that separates index error from retrieval quality.

    python -m bench.sweep_hnsw --slice data/slice --lang eng_Latn [--gpu]

Writes bench/hnsw_sweep.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from usearch.index import ScalarKind

from src.index.dense import DensePartition
from src.index.exact import ExactPartition
from src.index.embedder import OnnxEmbedder

ONNX_DIR = Path("artifacts/e5-small-onnx")
DTYPES = {"f32": ScalarKind.F32, "f16": ScalarKind.F16, "i8": ScalarKind.I8}


def recall_at_k(approx: list[list[str]], exact: list[list[str]], k: int) -> float:
    """Fraction of the exact top-k that the approximate index also returned."""
    tot = 0.0
    for a, e in zip(approx, exact):
        if not e:
            continue
        tot += len(set(a[:k]) & set(e[:k])) / min(k, len(e))
    return tot / max(1, len(exact))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", type=Path, default=Path("data/slice"))
    ap.add_argument("--lang", default="eng_Latn")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--queries", type=int, default=500)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", type=Path, default=Path("bench/hnsw_sweep.json"))
    a = ap.parse_args()

    d = a.slice / a.lang
    corpus = pd.read_parquet(d / "corpus.parquet")
    qs = pd.read_parquet(d / "queries.parquet")
    qs = qs[qs.answerable].head(a.queries).reset_index(drop=True)

    model = (ONNX_DIR / "fp32" / "model.onnx") if a.gpu else (ONNX_DIR / "model_int8.onnx")
    embed = OnnxEmbedder(model, ONNX_DIR, threads=a.threads, use_gpu=a.gpu)
    print(f"embedding {len(corpus):,} passages on {embed.provider}", flush=True)
    V = embed.encode_passages(corpus.text.tolist(), batch=16 if a.gpu else 64)
    QV = embed.encode_queries(qs["query"].tolist(), batch=16 if a.gpu else 64)

    # ground truth
    ex = ExactPartition(a.lang, dim=V.shape[1])
    ex.add(V, corpus.passage_id.tolist(), corpus.passage_id.tolist())
    t0 = time.perf_counter_ns()
    exact_top = [[h.passage_id for h in ex.search(QV[i], k=a.k)] for i in range(len(qs))]
    exact_ms = (time.perf_counter_ns() - t0) / 1e6 / len(qs)
    print(f"exact baseline: {exact_ms:.3f} ms/query over {len(corpus):,} vectors")

    rows = []
    grid = [(dt, M, ea, es)
            for dt in ("f16", "f32", "i8")
            for M in (16, 32, 48)
            for ea in (128, 256)
            for es in (64, 128, 256, 512)]
    for dt, M, ea, es in grid:
        t0 = time.perf_counter_ns()
        part = DensePartition(a.lang, dim=V.shape[1], connectivity=M,
                              expansion_add=ea, expansion_search=es,
                              dtype=DTYPES[dt])
        part.add(V, corpus.passage_id.tolist(), corpus.passage_id.tolist())
        build_s = (time.perf_counter_ns() - t0) / 1e9

        lat = []
        approx = []
        for i in range(len(qs)):
            t = time.perf_counter_ns()
            hits = part.search(QV[i], k=a.k)
            lat.append((time.perf_counter_ns() - t) / 1e6)
            approx.append([h.passage_id for h in hits])

        row = {
            "dtype": dt, "M": M, "expansion_add": ea, "expansion_search": es,
            "recall_at_10_vs_exact": round(recall_at_k(approx, exact_top, a.k), 4),
            "recall_at_1_vs_exact": round(recall_at_k(approx, exact_top, 1), 4),
            "self_retrieval": round(part.self_retrieval_rate(200), 4),
            "p50_ms": round(float(np.percentile(lat, 50)), 3),
            "p95_ms": round(float(np.percentile(lat, 95)), 3),
            "p100_ms": round(float(np.max(lat)), 3),
            "build_s": round(build_s, 1),
            "index_mb": round(part.size_bytes / 1e6, 1),
        }
        rows.append(row)
        print(f"  {dt} M={M:<3} ea={ea:<4} es={es:<4} "
              f"R@10={row['recall_at_10_vs_exact']:.4f} "
              f"self={row['self_retrieval']:.3f} "
              f"p50={row['p50_ms']:.3f}ms p95={row['p95_ms']:.3f}ms "
              f"{row['index_mb']:.0f}MB build={row['build_s']:.0f}s", flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({
            "lang": a.lang, "corpus": len(corpus), "queries": len(qs), "k": a.k,
            "exact_ms_per_query": round(exact_ms, 3),
            "provider": embed.provider, "rows": rows}, indent=1), encoding="utf-8")

    # Operating point: cheapest config that keeps recall-vs-exact >= 0.98 AND
    # self-retrieval >= 0.98. Recall is chosen over latency because the whole
    # budget only has ~200ms and retrieval is single-digit ms either way -
    # there is no reason to trade recall for time we are not short of.
    ok = [r for r in rows
          if r["recall_at_10_vs_exact"] >= 0.98 and r["self_retrieval"] >= 0.98]
    chosen = min(ok, key=lambda r: (r["p95_ms"], r["index_mb"])) if ok else \
        max(rows, key=lambda r: r["recall_at_10_vs_exact"])
    print(f"\nCHOSEN: {chosen}")
    data = json.loads(a.out.read_text(encoding="utf-8"))
    data["chosen"] = chosen
    a.out.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
