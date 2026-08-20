"""
Build a real out-of-corpus negative set for calibrating the relevance gate.

Why this exists: the first tau calibration used *unanswerable* queries as
negatives and scored AUC 0.508 - random. Two separate mistakes, and the second
is the interesting one.

  1. It thresholded the RRF fused score. RRF is rank-derived, so the top-1
     fused score is ~2/60 for almost every query (43 distinct values across
     2,707 queries). It encodes "did both arms agree on the top hit", not "how
     good is the hit". It cannot support a threshold at all.

  2. It used the wrong negatives. MS MARCO "unanswerable" queries are
     ON-DOMAIN: their ten candidate passages were retrieved by a real search
     engine and are topically close, they just do not contain the answer. That
     is an ANSWER-SCOPE problem, and it belongs to the output-side
     groundedness rail. The input-side relevance gate answers a different
     question - "is this covered by the corpus at all?" - so its negatives
     must be queries whose answers are genuinely absent.

This script builds those negatives honestly: real MS MARCO queries that were
NOT sampled into the index, whose gold passages are verified absent from it by
content hash. Same distribution, same phrasing, genuinely not in the corpus -
which is exactly the refusal the gate is supposed to produce.

    python -m bench.make_negatives --shards data/shards --slice data/slice
"""
from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

NO_ANSWER = "no answer present."


def norm(t: str) -> str:
    t = unicodedata.normalize("NFC", t or "")
    return "".join(c for c in t if c == "\n" or c == "\t"
                   or unicodedata.category(c)[0] != "C").strip()


def chash(t: str) -> str:
    return hashlib.blake2b(norm(t).encode("utf-8"), digest_size=16).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=Path, default=Path("data/shards"))
    ap.add_argument("--slice", type=Path, default=Path("data/slice"))
    ap.add_argument("--shard-idx", type=int, default=0,
                    help="any shard works: Eng_Query/English_passages are "
                         "identical across all 14, they are parallel")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--out", type=Path, default=Path("bench/negatives.jsonl"))
    a = ap.parse_args()

    shard = a.shards / "default" / "validation" / f"{a.shard_idx:04d}.parquet"
    if not shard.exists():
        raise SystemExit(f"missing shard {shard} - run src.ingest.fetch_shards")

    corpus = pd.read_parquet(a.slice / "eng_Latn" / "corpus.parquet")
    indexed = set(corpus.passage_id)
    # passage_id is a content hash in build_slice, but hash the text too so this
    # does not silently depend on that
    indexed_hashes = {chash(t) for t in corpus.text}
    used_qids = set(pd.read_parquet(a.slice / "eng_Latn" / "queries.parquet").query_id)
    print(f"indexed corpus: {len(corpus):,} passages, {len(used_qids):,} queries used")

    pf = pq.ParquetFile(shard)
    t = pf.read(columns=["query_id", "Eng_Query", "Eng_Answer", "query_type",
                         "passages"])
    df = t.to_pandas()
    print(f"shard: {len(df):,} rows")

    held = df[~df.query_id.isin(used_qids)]
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(held))

    out, checked, rejected = [], 0, 0
    for pos in order:
        if len(out) >= a.n:
            break
        r = held.iloc[int(pos)]
        checked += 1
        p = r["passages"]
        sel = list(p["is_selected"])
        eng = list(p["English_passages"])
        gold = [eng[i] for i, s in enumerate(sel) if s]
        if not gold:
            continue                       # need a real gold to verify absence
        ans = str(r["Eng_Answer"] or "").strip().lower()
        if ans == NO_ANSWER:
            continue                       # that is the OTHER guardrail's job
        # the whole point: keep only queries whose answer is genuinely absent
        if any(chash(g) in indexed_hashes for g in gold):
            rejected += 1
            continue
        q = norm(str(r["Eng_Query"]))
        if len(q) < 8:
            continue
        out.append({
            "id": f"oob:{int(r['query_id'])}",
            "query": q,
            "lang": "eng_Latn",
            "query_type": r.get("query_type"),
            "category": "out_of_corpus",
            "n_gold_absent": len(gold),
        })

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out),
                     encoding="utf-8")
    print(f"\nscanned {checked:,} held-out queries")
    print(f"rejected {rejected:,} whose gold WAS in the corpus "
          f"({100*rejected/max(1,checked):.1f}% - passages are shared across "
          f"queries, so this check is not decorative)")
    print(f"wrote {len(out):,} verified out-of-corpus negatives -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
