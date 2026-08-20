"""
Calibrate the relevance threshold tau, with a real ROC behind it.

The brief is explicit that "a threshold with a curve behind it is engineering;
a hardcoded 0.5 is a guess". This dataset makes the curve possible: 44.92% of
queries are marked "No Answer Present." and that marker corresponds EXACTLY to
zero is_selected passages, with no exceptions in 97,941 rows (see
docs/discovery/report.md section 4).

So we have real positives and real negatives:

  positives  answerable queries   - a gold passage IS in the corpus
  negatives  unanswerable queries - real queries whose real candidate passages
                                    genuinely do not contain the answer

These negatives are far better than synthetic off-topic queries: they are
plausible, on-domain, and retrieve confident-looking passages. That is exactly
the case where an ungated RAG system hallucinates.

    python -m bench.calibrate_tau --slice data/slice --lang eng_Latn

Writes bench/tau_calibration.json and updates src/guardrails/thresholds.yaml.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.index.exact import ExactPartition
from src.index.embedder import OnnxEmbedder, E5Tokenizer
from src.index.fusion import rrf
from src.index.sparse import SparsePartition

ONNX_DIR = Path("artifacts/e5-small-onnx")


def roc(pos: np.ndarray, neg: np.ndarray, n: int = 400) -> list[dict]:
    lo = float(min(pos.min(), neg.min()))
    hi = float(max(pos.max(), neg.max()))
    out = []
    for t in np.linspace(lo, hi, n):
        tp = float((pos >= t).sum()); fn = float((pos < t).sum())
        fp = float((neg >= t).sum()); tn = float((neg < t).sum())
        tpr = tp / max(1e-9, tp + fn)          # answerable correctly answered
        fpr = fp / max(1e-9, fp + tn)          # unanswerable wrongly answered
        prec = tp / max(1e-9, tp + fp)
        out.append({
            "tau": float(t), "tpr": tpr, "fpr": fpr, "precision": prec,
            "f1": 2 * prec * tpr / max(1e-9, prec + tpr),
            # the number that matters for a refusal gate
            "false_answer_rate": fpr,
            "false_refusal_rate": fn / max(1e-9, tp + fn),
            "youden_j": tpr - fpr,
        })
    return out


def auc_of(curve: list[dict]) -> float:
    pts = sorted(((c["fpr"], c["tpr"]) for c in curve))
    x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
    return float(np.trapezoid(y, x)) if len(x) > 1 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", type=Path, default=Path("data/slice"))
    ap.add_argument("--lang", default="eng_Latn")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--max-neg", type=int, default=2000)
    ap.add_argument("--target-false-answer-rate", type=float, default=0.10,
                    help="operating point: max fraction of unanswerable "
                         "queries we are willing to answer anyway")
    ap.add_argument("--out", type=Path, default=Path("bench/tau_calibration.json"))
    ap.add_argument("--write-thresholds", action="store_true")
    ap.add_argument("--gpu", action="store_true",
                    help="embed on CUDA using the fp32 export")
    a = ap.parse_args()

    d = a.slice / a.lang
    corpus = pd.read_parquet(d / "corpus.parquet")
    queries = pd.read_parquet(d / "queries.parquet")

    model = (ONNX_DIR / "fp32" / "model.onnx") if a.gpu else (ONNX_DIR / "model_int8.onnx")
    embed = OnnxEmbedder(model, ONNX_DIR, threads=a.threads, use_gpu=a.gpu)
    print(f"[1/4] embedding {len(corpus):,} passages", flush=True)
    t0 = time.time()
    V = embed.encode_passages(corpus.text.tolist(), batch=16 if a.gpu else 64)
    print(f"      {time.time()-t0:.0f}s ({len(corpus)/(time.time()-t0):.0f} psg/s)")

    print("[2/4] building dense + sparse", flush=True)
    # Exact, not HNSW: tau is a threshold on fused scores, and RRF scores are
    # rank-derived, so a badly-built graph would shift every score and calibrate
    # the gate against index error instead of relevance. The deployed HNSW is
    # tuned to >=0.98 recall vs exact, so the scores it produces match.
    dense = ExactPartition(a.lang, dim=V.shape[1])
    dense.add(V, corpus.passage_id.tolist(), corpus.passage_id.tolist())
    sparse = SparsePartition(a.lang)
    sparse.build(corpus.text.tolist(), corpus.passage_id.tolist())

    print("[3/4] scoring answerable vs unanswerable", flush=True)
    pos_q = queries[queries.answerable]
    neg_q = queries[~queries.answerable].head(a.max_neg)
    scores: dict[str, list[float]] = {"pos": [], "neg": []}
    for tag, qs in (("pos", pos_q), ("neg", neg_q)):
        QV = embed.encode_queries(qs["query"].tolist(), batch=16 if a.gpu else 64)
        for i in range(len(qs)):
            dh = dense.search(QV[i], k=10)
            sh = sparse.search(qs["query"].iloc[i], k=10)
            fused = rrf(dh, sh, top_k=5)
            scores[tag].append(fused[0].score if fused else 0.0)

    pos = np.array(scores["pos"]); neg = np.array(scores["neg"])
    curve = roc(pos, neg)
    area = auc_of(curve)

    # Operating point: the LOWEST tau whose false-answer rate meets target.
    # Chosen this way round because refusing a good question is cheap (the
    # user rephrases) while confidently answering an unanswerable one is the
    # failure this whole gate exists to prevent.
    feasible = [c for c in curve if c["false_answer_rate"] <= a.target_false_answer_rate]
    chosen = min(feasible, key=lambda c: c["tau"]) if feasible else \
        max(curve, key=lambda c: c["youden_j"])
    best_j = max(curve, key=lambda c: c["youden_j"])
    best_f1 = max(curve, key=lambda c: c["f1"])

    print(f"[4/4] pos n={len(pos)} mean={pos.mean():.4f} | "
          f"neg n={len(neg)} mean={neg.mean():.4f}")
    print(f"      AUC = {area:.4f}")
    for label, c in (("target-FAR", chosen), ("Youden J", best_j), ("max F1", best_f1)):
        print(f"      {label:11s} tau={c['tau']:.5f} "
              f"false_answer={c['false_answer_rate']:.3f} "
              f"false_refusal={c['false_refusal_rate']:.3f} f1={c['f1']:.3f}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "lang": a.lang, "corpus": len(corpus),
        "n_positive": len(pos), "n_negative": len(neg),
        "pos_mean": float(pos.mean()), "neg_mean": float(neg.mean()),
        "pos_p10": float(np.percentile(pos, 10)),
        "neg_p90": float(np.percentile(neg, 90)),
        "auc": area, "chosen": chosen, "youden": best_j, "max_f1": best_f1,
        "target_false_answer_rate": a.target_false_answer_rate,
        "curve": curve,
        "raw_scores": {"pos": pos.tolist(), "neg": neg.tolist()},
    }, indent=1), encoding="utf-8")
    print(f"wrote {a.out}")

    if a.write_thresholds:
        p = Path("src/guardrails/thresholds.yaml")
        s = p.read_text(encoding="utf-8")
        s = s.replace("  tau: null            # SET BY bench/calibrate_tau.py"
                      " - null means uncalibrated",
                      f"  tau: {chosen['tau']:.5f}   # calibrated, AUC={area:.4f}")
        s = s.replace("  tau_source: uncalibrated",
                      f"  tau_source: bench/calibrate_tau.py lang={a.lang} "
                      f"n_pos={len(pos)} n_neg={len(neg)}")
        p.write_text(s, encoding="utf-8")
        print(f"updated {p} -> tau={chosen['tau']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
