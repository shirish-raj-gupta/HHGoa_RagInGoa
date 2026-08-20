"""
Gate B - the chunking ablation.

Produces docs/chunking-ablation.md and bench/chunking_results.json.

Method
------
Corpus  : every candidate passage of N sampled queries (real distractors).
Queries : only the labelled subset (>=1 is_selected passage) - ~54%.
Gold    : passages.is_selected, the dataset's own relevance labels. These are
          REAL labels, so Recall@5 / MRR@10 / nDCG@10 are real metrics, not a
          silver standard (docs/discovery/report.md section 4).
Scoring : chunks resolve to their passage_id and the best rank per passage
          wins, so a strategy is never rewarded for flooding the top-k with
          many chunks of the same passage.

Dense retrieval only. Chunking is a property of what gets embedded, so mixing
in BM25 here would confound the comparison; hybrid fusion is measured at Gate C.

Retrieval is EXACT (brute-force cosine), not ANN. An approximate index adds
graph-quality noise that, on this corpus, was larger than the chunking effect
being measured - one arm scored R@5 0.316 purely because its HNSW graph built
badly. Exact search makes a difference in this table a difference in chunking.
HNSW parameters are tuned and published separately (bench/sweep_hnsw.py).

    python -m bench.ablate_chunking --slice data/slice --langs eng_Latn
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.chunking.base import Passage
from src.chunking.strategies import build_registry
from src.index.exact import ExactPartition
from src.index.embedder import MODEL_ID, OnnxEmbedder, E5Tokenizer

ONNX_DIR = Path("artifacts/e5-small-onnx")


# ------------------------------------------------------------------ metrics
def dcg(rels: list[int]) -> float:
    return sum(r / np.log2(i + 2) for i, r in enumerate(rels))


def score_run(ranked_pids: list[str], gold: set[str], k_recall=5, k=10) -> dict:
    top = ranked_pids[:k]
    rels = [1 if p in gold else 0 for p in top]
    rr = next((1.0 / (i + 1) for i, r in enumerate(rels) if r), 0.0)
    ideal = sorted(rels, reverse=True)
    return {
        "recall_at_5": float(any(p in gold for p in ranked_pids[:k_recall])),
        "mrr_at_10": rr,
        "ndcg_at_10": dcg(rels) / dcg(ideal) if any(ideal) else 0.0,
    }


def dedupe_by_passage(hits) -> list[str]:
    """Best rank per passage - a strategy cannot win by flooding top-k."""
    seen, out = set(), []
    for h in hits:
        if h.passage_id not in seen:
            seen.add(h.passage_id)
            out.append(h.passage_id)
    return out


# ------------------------------------------------------------------ one arm
def run_arm(name, chunker, corpus_df, queries_df, embed, lang,
            k=10, latency_sample=200, seed=0, vec_cache: dict | None = None,
            lat_embed=None) -> dict:
    t0 = time.perf_counter_ns()
    passages = [Passage(r.passage_id, r.doc_id, r.text, r.lang, r.script)
                for r in corpus_df.itertuples()]
    chunks = chunker.chunk_many(passages)
    t_chunk = (time.perf_counter_ns() - t0) / 1e6

    n_degen = sum(c.degenerate for c in chunks)
    is_late = any(c.extra.get("late_chunk") for c in chunks)

    # Fingerprint the emitted chunk text. Arms that produce byte-identical
    # chunks are the SAME retrieval system and must score identically - so we
    # reuse the vectors and record which arm this one collapsed onto. This
    # turns "fixed_512 is a no-op on this corpus" from a claim into a proof,
    # and it is why the table shows exact ties rather than near-ties.
    fp = None
    if not is_late:
        import hashlib
        h = hashlib.blake2b(digest_size=16)
        for c in chunks:
            h.update(c.text.encode("utf-8"))
            h.update(b"\x00")
        fp = h.hexdigest()

    identical_to = None
    cached = vec_cache.get(fp) if (vec_cache is not None and fp) else None

    # ---- embed the chunks
    t0 = time.perf_counter_ns()
    if cached is not None:
        V, identical_to = cached["V"], cached["arm"]
    elif is_late:
        # late chunking: one full-passage forward pass, then mean-pool spans
        by_pid: dict[str, list] = {}
        for c in chunks:
            by_pid.setdefault(c.passage_id, []).append(c)
        text_of = dict(zip(corpus_df.passage_id, corpus_df.text))
        vecs, order = [], []
        for pid, cs in by_pid.items():
            spans = [(c.extra.get("tok_lo", 0), c.extra.get("tok_hi", 512)) for c in cs]
            vecs.append(embed.encode_late(text_of[pid], spans))
            order.extend(cs)
        V = np.vstack(vecs)
        chunks = order
    else:
        V = embed.encode_passages([c.text for c in chunks], batch=64)
    t_embed_corpus = (time.perf_counter_ns() - t0) / 1e6
    if vec_cache is not None and fp and cached is None:
        vec_cache[fp] = {"V": V, "arm": name}

    # ---- index
    # EXACT search on purpose: this ablation measures chunking, and an
    # approximate index injects graph-quality noise larger than the effect
    # being measured. HNSW is tuned separately in bench/sweep_hnsw.py.
    part = ExactPartition(lang, dim=V.shape[1])
    t0 = time.perf_counter_ns()
    part.add(V, [c.chunk_id for c in chunks], [c.passage_id for c in chunks])
    t_index = (time.perf_counter_ns() - t0) / 1e6

    # Guard: a vector must retrieve itself at rank 0. Below 0.95 the distance
    # function is broken, not merely imprecise - this is what caught usearch's
    # ScalarKind.I8 silently ruining one arm's numbers (see src/index/dense.py).
    self_retr = part.self_retrieval_rate(200)

    # ---- retrieve
    qs = queries_df[queries_df.answerable].reset_index(drop=True)
    QV = embed.encode_queries(qs["query"].tolist(), batch=64)

    agg = {"recall_at_5": [], "mrr_at_10": [], "ndcg_at_10": []}
    per_type: dict[str, list] = {}
    t_ret = []
    for i in range(len(qs)):
        t = time.perf_counter_ns()
        hits = part.search(QV[i], k=k * 3)
        t_ret.append((time.perf_counter_ns() - t) / 1e6)
        m = score_run(dedupe_by_passage(hits), set(qs.gold.iloc[i]))
        for kk, v in m.items():
            agg[kk].append(v)
        per_type.setdefault(qs.query_type.iloc[i], []).append(m["recall_at_5"])

    # ---- single-query embed latency, measured separately from the batch path
    rng = np.random.default_rng(seed)
    sample = qs["query"].iloc[rng.choice(len(qs), min(latency_sample, len(qs)),
                                         replace=False)].tolist()
    # Always measured on the CPU int8 session, even when the corpus was
    # embedded on GPU: the Space has no GPU, so a CUDA number here would be a
    # latency the deployed system can never achieve.
    lat = lat_embed or embed
    t_emb_single = []
    for q in sample:
        t = time.perf_counter_ns()
        lat.encode_queries([q])
        t_emb_single.append((time.perf_counter_ns() - t) / 1e6)

    return {
        "strategy": name, "lang": lang,
        "chunks": len(chunks), "passages": len(passages),
        "chunk_fingerprint": fp,
        "identical_to": identical_to,
        "chunks_per_passage": round(len(chunks) / max(1, len(passages)), 3),
        "degenerate_chunks": n_degen,
        "degenerate_pct": round(100 * n_degen / max(1, len(chunks)), 2),
        "index_mb": round(part.size_bytes / 1e6, 1),
        "self_retrieval": round(self_retr, 4),
        "index_ok": bool(self_retr >= 0.95),
        "recall_at_5": round(float(np.mean(agg["recall_at_5"])), 4),
        "mrr_at_10": round(float(np.mean(agg["mrr_at_10"])), 4),
        "ndcg_at_10": round(float(np.mean(agg["ndcg_at_10"])), 4),
        "embed_p50_ms": round(float(np.percentile(t_emb_single, 50)), 2),
        "embed_p95_ms": round(float(np.percentile(t_emb_single, 95)), 2),
        "retrieve_p50_ms": round(float(np.percentile(t_ret, 50)), 3),
        "retrieve_p95_ms": round(float(np.percentile(t_ret, 95)), 3),
        "build_chunk_ms": round(t_chunk, 1),
        "build_embed_ms": round(t_embed_corpus, 1),
        "build_index_ms": round(t_index, 1),
        "eval_queries": len(qs),
        "corpus_embed_provider": getattr(embed, "provider", "?"),
        "latency_provider": getattr(lat, "provider", "?"),
        "recall_by_query_type": {k: round(float(np.mean(v)), 4)
                                 for k, v in sorted(per_type.items())},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", type=Path, default=Path("data/slice"))
    ap.add_argument("--langs", default="eng_Latn")
    ap.add_argument("--arms", default="", help="comma list; empty = all")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("bench/chunking_results.json"))
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--gpu", action="store_true",
                    help="embed the corpus on CUDA using the fp32 export "
                         "(int8 has no CUDA kernels and crashes the EP)")
    a = ap.parse_args()

    tok = E5Tokenizer(ONNX_DIR)
    # CPU int8 session: production path, and the only one used for latency
    lat_embed = OnnxEmbedder(ONNX_DIR / "model_int8.onnx", ONNX_DIR,
                             threads=a.threads)
    if a.gpu:
        embed = OnnxEmbedder(ONNX_DIR / "fp32" / "model.onnx", ONNX_DIR,
                             threads=a.threads, use_gpu=True)
        print(f"corpus embedding on {embed.provider}; "
              f"latency on {lat_embed.provider}")
    else:
        embed = lat_embed

    def embed_fn(texts):                       # for SemanticBreakpointChunker
        return embed.encode_passages(list(texts), batch=64)

    registry = build_registry(tok, embed_fn=embed_fn)
    names = [n.strip() for n in a.arms.split(",") if n.strip()] or list(registry)

    results, existing = [], []
    vec_cache: dict = {}
    if a.out.exists():
        existing = json.loads(a.out.read_text(encoding="utf-8")).get("results", [])

    for lang in a.langs.split(","):
        d = a.slice / lang
        vec_cache.clear()   # vectors are per-language
        corpus = pd.read_parquet(d / "corpus.parquet")
        queries = pd.read_parquet(d / "queries.parquet")
        if a.max_queries:
            queries = queries.head(a.max_queries)
        print(f"\n=== {lang}: {len(corpus):,} passages, "
              f"{int(queries.answerable.sum()):,} labelled queries ===", flush=True)
        for n in names:
            if n not in registry:
                print(f"  !! unknown arm {n}"); continue
            t0 = time.time()
            r = run_arm(n, registry[n], corpus, queries, embed, lang,
                        vec_cache=vec_cache, lat_embed=lat_embed)
            results.append(r)
            print(f"  {n:22s} chunks={r['chunks']:>7,} "
                  f"R@5={r['recall_at_5']:.4f} MRR@10={r['mrr_at_10']:.4f} "
                  f"nDCG@10={r['ndcg_at_10']:.4f} "
                  f"idx={r['index_mb']:>5.1f}MB emb_p50={r['embed_p50_ms']:.2f} "
                  f"ret_p50={r['retrieve_p50_ms']:.3f} degen={r['degenerate_pct']:.0f}% "
                  f"{'==' + r['identical_to'] if r['identical_to'] else ''} "
                  f"{'' if r['index_ok'] else '!!INDEX_BROKEN sr=%.2f!!' % r['self_retrieval']} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(json.dumps({
                "model": MODEL_ID, "threads": a.threads,
                "results": existing + results}, indent=1), encoding="utf-8")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
