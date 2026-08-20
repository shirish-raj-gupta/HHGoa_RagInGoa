"""
Build the stratified benchmark query set (Gate C needs >= 200).

Sampled from the real slices rather than invented, so every query is one a
user could actually ask of this corpus and every label (`answerable`,
`query_type`) is the dataset's own.

Strata, per the brief:

  language        English, Hindi, Tamil
  query type      DESCRIPTION / NUMERIC / ENTITY / PERSON / LOCATION
  short           < 5 tokens
  long            top decile by token count
  entity-heavy    ENTITY or PERSON typed
  unanswerable    is_selected all zero - the dataset's own negatives
  code-switched   synthesized, because the corpus has no genuinely
                  code-switched rows (each row is single-language). These are
                  MARKED `synthetic: true` so nobody reads them as sampled.

    python -m bench.make_queries --slice data/slice --n 240
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

# Hinglish frames: an English query stem with Hindi function words around it,
# which is what Indian users actually speak. Synthetic and labelled as such.
HINGLISH = [
    "{q} ke bare mein batao",
    "mujhe {q} ke baare mein jaanna hai",
    "{q} kya hota hai",
    "yaar {q} explain karo",
    "{q} ka matlab kya hai",
]


def take(df: pd.DataFrame, n: int, rng: random.Random) -> list[dict]:
    if df.empty or n <= 0:
        return []
    idx = list(df.index)
    rng.shuffle(idx)
    return [df.loc[i].to_dict() for i in idx[:n]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", type=Path, default=Path("data/slice"))
    ap.add_argument("--langs", default="eng_Latn,hin_Deva,tam_Taml")
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", type=Path, default=Path("bench/queries.jsonl"))
    a = ap.parse_args()

    rng = random.Random(a.seed)
    langs = [l.strip() for l in a.langs.split(",") if l.strip()]
    per_lang = a.n // (len(langs) + 1)          # +1 reserves room for synthetic
    out: list[dict] = []

    for lang in langs:
        f = a.slice / lang / "queries.parquet"
        if not f.exists():
            print(f"  skip {lang}: no queries.parquet")
            continue
        q = pd.read_parquet(f).copy()
        q["ntok"] = q["query"].str.split().str.len()

        ans = q[q.answerable]
        una = q[~q.answerable]
        long_cut = q["ntok"].quantile(0.9)

        buckets = {
            "short": ans[ans.ntok < 5],
            "long": ans[ans.ntok >= long_cut],
            "entity_heavy": ans[ans.query_type.isin(["ENTITY", "PERSON"])],
            "numeric": ans[ans.query_type == "NUMERIC"],
            "location": ans[ans.query_type == "LOCATION"],
            "description": ans[ans.query_type == "DESCRIPTION"],
            "unanswerable": una,
        }
        # split the per-language budget evenly, with unanswerable double-weighted
        # because the relevance gate is what they exercise
        weights = {k: (2 if k == "unanswerable" else 1) for k in buckets}
        total_w = sum(weights.values())
        for name, df in buckets.items():
            k = max(1, round(per_lang * weights[name] / total_w))
            for r in take(df, k, rng):
                out.append({
                    "id": f"{lang}:{name}:{r['query_id']}",
                    "query": r["query"],
                    "lang": lang,
                    "type": name,
                    "query_type": r.get("query_type"),
                    "answerable": bool(r["answerable"]),
                    "n_tokens": int(r["ntok"]),
                    "synthetic": False,
                })

    # code-switched, synthesized from English stems and marked
    eng = a.slice / "eng_Latn" / "queries.parquet"
    if eng.exists():
        src = pd.read_parquet(eng)
        src = src[src.answerable]
        for r in take(src, max(1, per_lang // 2), rng):
            stem = str(r["query"]).rstrip("?").strip()
            out.append({
                "id": f"codeswitch:{r['query_id']}",
                "query": rng.choice(HINGLISH).format(q=stem),
                "lang": "hin_Deva",
                "type": "code_switched",
                "query_type": r.get("query_type"),
                "answerable": True,
                "n_tokens": len(stem.split()) + 4,
                "synthetic": True,
                "derived_from": str(r["query_id"]),
            })

    rng.shuffle(out)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out),
                     encoding="utf-8")

    from collections import Counter
    print(f"wrote {a.out} with {len(out)} queries")
    print("\nby language:")
    for k, v in sorted(Counter(r["lang"] for r in out).items()):
        print(f"  {k:12s} {v:>4}")
    print("by stratum:")
    for k, v in sorted(Counter(r["type"] for r in out).items()):
        print(f"  {k:14s} {v:>4}")
    n_syn = sum(r["synthetic"] for r in out)
    n_una = sum(not r["answerable"] for r in out)
    print(f"\nsynthetic (code-switched): {n_syn}   unanswerable: {n_una}")
    if len(out) < 200:
        print(f"WARNING: {len(out)} < 200, the brief requires at least 200")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
