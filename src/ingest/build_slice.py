"""
Build a labelled retrieval slice from MSMARCO-XI validation shards.

The ingest unit is the QUERY with its full candidate set. Any other sampling
unit breaks `is_selected` as a retrieval label - if you drop candidates you
cannot tell "the gold passage was ranked low" from "the gold passage is not
in the corpus".

Emits, per language:
  corpus.parquet   passage_id, doc_id, text, lang, script, token_len
  queries.parquet  query_id, query, lang, query_type, answerable, gold[]

Cleaning applied (see docs/discovery/report.md section 6):
  - NFC normalize, strip control chars
  - drop MT-degenerate passages (token cap + repetition ratio), ~0.18%
  - exact dedup by content hash; English is deduped ONCE globally, since it is
    byte-identical across all 14 shards (93.1% redundant if ingested naively)
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

LANG_OF_SHARD = {
    0: "asm_Beng", 1: "ben_Beng", 2: "guj_Gujr", 3: "hin_Deva", 4: "kan_Knda",
    5: "mal_Mlym", 6: "mar_Deva", 7: "npi_Deva", 8: "ory_Orya", 9: "pan_Guru",
    10: "san_Deva", 11: "tam_Taml", 12: "tel_Telu", 13: "urd_Arab",
}
SCRIPT_OF_LANG = {
    "asm_Beng": "BENGALI", "ben_Beng": "BENGALI", "guj_Gujr": "GUJARATI",
    "hin_Deva": "DEVANAGARI", "kan_Knda": "KANNADA", "mal_Mlym": "MALAYALAM",
    "mar_Deva": "DEVANAGARI", "npi_Deva": "DEVANAGARI", "ory_Orya": "ORIYA",
    "pan_Guru": "GURMUKHI", "san_Deva": "DEVANAGARI", "tam_Taml": "TAMIL",
    "tel_Telu": "TELUGU", "urd_Arab": "ARABIC", "eng_Latn": "LATIN",
}
NO_ANSWER = "no answer present."


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    return "".join(c for c in text
                   if c in "\n\t" or unicodedata.category(c)[0] != "C").strip()


def repetition_ratio(text: str, k: int = 8) -> float:
    """Fraction of k-gram mass taken by repeats. Catches MT decoding loops."""
    w = text.split()
    if len(w) < k * 2:
        return 0.0
    grams = [" ".join(w[i:i + k]) for i in range(len(w) - k + 1)]
    c = collections.Counter(grams)
    return 1.0 - (len(c) / len(grams))


LETTER = re.compile(r"[^\W\d_]", re.UNICODE)


def is_degenerate(text: str, max_chars: int = 1500, min_letters: int = 8) -> bool:
    """
    Report section 6: 0.18% of Indic passages are MT repetition loops.

    Also drops passages with almost no LETTERS. A passage of "..." or
    "--------------" or "৩.৩৩৩৩৩৩৩৩..." cannot answer anything, and one was
    observed being returned at rank 1 against a live index. Rare - measured
    0.010% of Assamese passages - but disproportionately visible when it hits.

    NOTE for anyone re-measuring this: count letters with `re`, not
    `pandas.Series.str.count`. Pandas returns 0 for Indic text on this pattern,
    which made a first pass report 97% of every Indic corpus as junk. The real
    figure is 0.01%. Measuring the measurement was the fix.
    """
    if len(LETTER.findall(text)) < min_letters:
        return True
    return len(text) > max_chars and repetition_ratio(text) > 0.35


def _h(s: str) -> str:
    return hashlib.blake2b(s.encode("utf-8"), digest_size=12).hexdigest()


def build(shard_paths: dict[str, Path], out_dir: Path, n_queries: int,
          seed: int = 0, include_english: bool = True) -> dict:
    """shard_paths maps lang -> local parquet path. English is taken from any one."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stats: dict = {"languages": {}, "n_queries_requested": n_queries}

    # choose the query_ids ONCE so every language sees the identical sample -
    # the shards are row-aligned, so this keeps the ablation confound-free
    first = next(iter(shard_paths.values()))
    qid_tbl = pq.read_table(first, columns=["query_id"]).to_pandas()
    rng = np.random.default_rng(seed)
    row_idx = np.sort(rng.choice(len(qid_tbl), min(n_queries, len(qid_tbl)),
                                 replace=False))
    stats["row_indices_sampled"] = len(row_idx)

    english_done = False
    for lang, path in shard_paths.items():
        t = pq.read_table(path, columns=[
            "query_id", "query", "Eng_Query", "Answer", "Eng_Answer",
            "query_type", "passages"]).to_pandas().iloc[row_idx].reset_index(drop=True)

        targets = [(lang, "Translated_passages", "query", "Answer")]
        if include_english and not english_done:
            targets.append(("eng_Latn", "English_passages", "Eng_Query", "Eng_Answer"))
            english_done = True

        for out_lang, pcol, qcol, acol in targets:
            seen: dict[str, str] = {}          # content hash -> passage_id
            corpus, queries = [], []
            n_drop_degen = n_drop_dup = 0

            for i in range(len(t)):
                p = t.passages.iloc[i]
                texts = list(p[pcol])
                sel = list(p["is_selected"])
                qid = int(t.query_id.iloc[i])
                gold: list[str] = []

                for j, raw in enumerate(texts):
                    txt = normalize(raw)
                    if not txt:
                        continue
                    if is_degenerate(txt):
                        n_drop_degen += 1
                        continue
                    hh = _h(txt)
                    if hh in seen:
                        n_drop_dup += 1
                        pid = seen[hh]           # dedup: reuse the existing id
                    else:
                        pid = f"{out_lang}:{hh}"
                        seen[hh] = pid
                        corpus.append({
                            "passage_id": pid, "doc_id": f"q{qid}:{j}", "text": txt,
                            "lang": out_lang, "script": SCRIPT_OF_LANG[out_lang],
                            "char_len": len(txt),
                        })
                    if j < len(sel) and int(sel[j]) == 1:
                        gold.append(pid)

                ans = normalize(str(t[acol].iloc[i] or ""))
                queries.append({
                    "query_id": qid, "query": normalize(str(t[qcol].iloc[i] or "")),
                    "lang": out_lang, "query_type": t.query_type.iloc[i],
                    "answer": ans,
                    "answerable": bool(gold),
                    "gold": gold,
                })

            cdf, qdf = pd.DataFrame(corpus), pd.DataFrame(queries)
            d = out_dir / out_lang
            d.mkdir(parents=True, exist_ok=True)
            cdf.to_parquet(d / "corpus.parquet", index=False)
            qdf.to_parquet(d / "queries.parquet", index=False)

            stats["languages"][out_lang] = {
                "passages": len(cdf), "queries": len(qdf),
                "labelled_queries": int(qdf.answerable.sum()),
                "pct_labelled": round(100 * float(qdf.answerable.mean()), 2),
                "dropped_degenerate": n_drop_degen,
                "deduped_instances": n_drop_dup,
                "dedup_pct": round(100 * n_drop_dup / max(1, n_drop_dup + len(cdf)), 2),
                "char_len_p50": int(cdf.char_len.median()),
                "char_len_p99": int(cdf.char_len.quantile(0.99)),
            }
            print(f"  {out_lang:9s} {len(cdf):>7,} passages  {len(qdf):>6,} queries  "
                  f"{int(qdf.answerable.sum()):>6,} labelled  "
                  f"dedup {stats['languages'][out_lang]['dedup_pct']:>5.2f}%  "
                  f"degen -{n_drop_degen}", flush=True)

    (out_dir / "slice_stats.json").write_text(
        json.dumps(stats, indent=1), encoding="utf-8")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True,
                    help="dir holding default/validation/XXXX.parquet")
    ap.add_argument("--langs", default="hin_Deva,tam_Taml")
    ap.add_argument("--out", type=Path, default=Path("data/slice"))
    ap.add_argument("--n-queries", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    inv = {v: k for k, v in LANG_OF_SHARD.items()}
    paths = {}
    for lg in a.langs.split(","):
        p = Path(a.shards) / "default" / "validation" / f"{inv[lg]:04d}.parquet"
        if not p.exists():
            raise SystemExit(f"missing shard for {lg}: {p}")
        paths[lg] = p

    print(f"building slice: {a.n_queries} queries x {list(paths)} (+eng_Latn)")
    s = build(paths, a.out, a.n_queries, a.seed)
    print(json.dumps(s["languages"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
