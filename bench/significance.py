"""
Is the chunking ablation measuring anything, or is it measuring noise?

The English arms span Recall@5 0.8678-0.8814 and the Hindi arms span
0.6675-0.6793. Before claiming an ordering - especially before claiming that
sentence-packing "beats" the control on Hindi by 0.0004 - those gaps have to
be tested. A 0.0004 difference on 2,707 queries is one query.

Method: **paired bootstrap over queries**. Paired is the right test because
every arm is evaluated on the SAME queries, and most queries behave
identically across arms (only the queries whose gold passage got split can
change). An unpaired proportion test would badly overstate the variance and
call everything a tie.

    n_boot resamples of the query set, with replacement
    for each resample, recompute Recall@5 for both arms and take the delta
    report the 95% interval of the delta and P(arm > baseline)

An interval straddling zero means the two arms are indistinguishable on this
data, which is a result worth publishing, not a failure to find one.

    python -m bench.significance --slice data/slice --lang eng_Latn --gpu
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.chunking.base import Passage
from src.chunking.strategies import build_registry
from src.index.embedder import OnnxEmbedder, E5Tokenizer
from src.index.exact import ExactPartition

ONNX_DIR = Path("artifacts/e5-small-onnx")
DEFAULT_ARMS = ["passage_atomic", "sentence_pack_128", "late_chunk_96",
                "fixed_128_o0", "semantic_p85"]


def hits_per_query(corpus: pd.DataFrame, queries: pd.DataFrame, chunker,
                   embed, lang: str, k: int = 5, gpu: bool = False) -> np.ndarray:
    """Boolean vector: did any gold passage land in the top-k for this query?"""
    passages = [Passage(r.passage_id, r.doc_id, r.text, r.lang, r.script)
                for r in corpus.itertuples()]
    chunks = [c for p in passages for c in chunker.chunk(p)]

    if any(c.extra.get("late_chunk") for c in chunks):
        by_pid: dict = {}
        for c in chunks:
            by_pid.setdefault(c.passage_id, []).append(c)
        text_of = dict(zip(corpus.passage_id, corpus.text))
        pids = list(by_pid)
        spans = [[(c.extra.get("tok_lo", 0), c.extra.get("tok_hi", 512))
                  for c in by_pid[p]] for p in pids]
        V = np.vstack(embed.encode_late_batch([text_of[p] for p in pids], spans,
                                              batch=32))
        chunks = [c for p in pids for c in by_pid[p]]
    else:
        V = embed.encode_passages([c.text for c in chunks],
                                  batch=16 if gpu else 64)

    part = ExactPartition(lang, dim=V.shape[1])
    part.add(V, [c.chunk_id for c in chunks], [c.passage_id for c in chunks])

    QV = embed.encode_queries(queries["query"].tolist(), batch=16 if gpu else 64)
    out = np.zeros(len(queries), dtype=bool)
    for i, gold in enumerate(queries["gold"]):
        g = set(gold)
        seen, rank = [], 0
        # resolve chunks to passages, best rank per passage, so an arm cannot
        # win by flooding the top-k with many chunks of one passage
        for h in part.search(QV[i], k=k * 8):
            if h.passage_id not in seen:
                seen.append(h.passage_id)
                rank += 1
                if rank >= k:
                    break
        out[i] = bool(g & set(seen[:k]))
    return out


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10000,
                     seed: int = 0) -> dict:
    """a, b are per-query boolean hit vectors for two arms on the SAME queries."""
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, size=(n_boot, n))
    da = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    disc_a = int((a & ~b).sum())          # queries only arm A gets
    disc_b = int((b & ~a).sum())          # queries only arm B gets
    return {
        "delta": round(float(a.mean() - b.mean()), 5),
        "ci95_lo": round(float(np.percentile(da, 2.5)), 5),
        "ci95_hi": round(float(np.percentile(da, 97.5)), 5),
        "p_better": round(float((da > 0).mean()), 4),
        "discordant_wins": disc_a, "discordant_losses": disc_b,
        "significant": bool(np.percentile(da, 2.5) > 0
                            or np.percentile(da, 97.5) < 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", type=Path, default=Path("data/slice"))
    ap.add_argument("--lang", default="eng_Latn")
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    ap.add_argument("--baseline", default="passage_atomic")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", type=Path, default=Path("bench/significance.json"))
    a = ap.parse_args()

    d = a.slice / a.lang
    corpus = pd.read_parquet(d / "corpus.parquet")
    queries = pd.read_parquet(d / "queries.parquet")
    queries = queries[queries.answerable].reset_index(drop=True)
    print(f"{a.lang}: {len(corpus):,} passages, {len(queries):,} labelled queries")

    model = (ONNX_DIR / "fp32" / "model.onnx") if a.gpu else (ONNX_DIR / "model_int8.onnx")
    embed = OnnxEmbedder(model, ONNX_DIR, threads=a.threads, use_gpu=a.gpu)
    tok = E5Tokenizer(ONNX_DIR)
    registry = build_registry(tok, embed_fn=lambda ts: embed.encode_passages(ts))

    arms = [s.strip() for s in a.arms.split(",") if s.strip()]
    hits: dict[str, np.ndarray] = {}
    for name in arms:
        if name not in registry:
            print(f"  skip unknown arm {name}")
            continue
        h = hits_per_query(corpus, queries, registry[name], embed, a.lang,
                           k=a.k, gpu=a.gpu)
        hits[name] = h
        print(f"  {name:20s} R@{a.k}={h.mean():.4f}", flush=True)

    base = hits.get(a.baseline)
    if base is None:
        print("baseline arm missing"); return 1

    print(f"\npaired bootstrap vs {a.baseline}  ({a.n_boot:,} resamples)\n")
    print(f"{'arm':20s} {'delta':>9s} {'95% CI':>20s} {'P(>base)':>9s}  verdict")
    results = {}
    for name, h in hits.items():
        if name == a.baseline:
            continue
        r = paired_bootstrap(h, base, a.n_boot)
        results[name] = r
        verdict = "SIGNIFICANT" if r["significant"] else "indistinguishable"
        print(f"{name:20s} {r['delta']:+9.5f} "
              f"[{r['ci95_lo']:+.5f},{r['ci95_hi']:+.5f}] "
              f"{r['p_better']:>9.3f}  {verdict}"
              f"  (+{r['discordant_wins']}/-{r['discordant_losses']} queries)")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "lang": a.lang, "baseline": a.baseline, "k": a.k,
        "n_queries": len(queries), "n_boot": a.n_boot,
        "recall": {n: round(float(h.mean()), 5) for n, h in hits.items()},
        "vs_baseline": results,
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
