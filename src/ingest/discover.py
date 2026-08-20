"""
Gate A - dataset discovery for ai4bharat/MSMARCO-XI.

Every number in docs/discovery/report.md is produced by this script.

    python src/ingest/discover.py --out docs/discovery

Design note: the HF datasets-server viewer is BROKEN for this dataset
(/first-rows -> 501, /rows -> 500 ArrowNotImplementedError "Nested data
conversions not implemented for chunked array outputs"), because the
`passages` column is a struct-of-lists. So we go straight to the parquet
conversion branch and read column chunks over HTTP range requests.

Cost control: each shard is a SINGLE row group (~98k rows / ~1.16 GB
uncompressed), so partial reads within a column are impossible. We instead
read only the cheap scalar columns (~17 MB/shard) for most passes, take
per-column byte sizes from parquet footer metadata (free), and download
exactly one full shard for passage-level statistics.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download

REPO = "ai4bharat/MSMARCO-XI"
REV = "refs/convert/parquet"
FS_BASE = f"datasets/{REPO}@refs%2Fconvert%2Fparquet/default"
EMBED_MODEL = "intfloat/multilingual-e5-small"

# The shard we pay to download in full for passage-level stats (hin_Deva).
PROBE_SPLIT, PROBE_SHARD = "validation", 3


def _open(split: str, i: int):
    return pq.ParquetFile(HfFileSystem().open(f"{FS_BASE}/{split}/{i:04d}.parquet", "rb"))


# ---------------------------------------------------------------- pass 1
def shard_map(splits: dict[str, int]) -> list[dict]:
    """Structural map of every shard. Uses footer metadata only - no data read."""
    out = []
    for split, n in splits.items():
        for i in range(n):
            md = _open(split, i).metadata
            rg = md.row_group(0)
            stats, sizes = {}, {}
            for j in range(rg.num_columns):
                c = rg.column(j)
                sizes[c.path_in_schema] = {
                    "compressed": c.total_compressed_size,
                    "uncompressed": c.total_uncompressed_size,
                }
                if c.statistics and c.path_in_schema in (
                        "target_lang", "source_lang", "query_id", "query_type"):
                    stats[c.path_in_schema] = [c.statistics.min, c.statistics.max]
            out.append({
                "split": split, "shard": i, "rows": md.num_rows,
                "row_groups": md.num_row_groups,
                # one shard == one language, so min == max on target_lang
                "target_lang": stats.get("target_lang", [None])[0],
                "source_lang": stats.get("source_lang", [None])[0],
                "query_id_range": stats.get("query_id"),
                "col_stats": stats, "col_sizes": sizes,
            })
            print(f"  {split}/{i:04d} {str(out[-1]['target_lang']):>9} rows={md.num_rows:>7}",
                  flush=True)
    return out


# ---------------------------------------------------------------- pass 2
def parallelism_check(n_val: int) -> dict:
    """Are the 14 language shards row-aligned? Reads query_id + Eng_Query only."""
    ref, res = None, {}
    for i in range(n_val):
        t = _open("validation", i).read(
            columns=["query_id", "Eng_Query", "target_lang"]).to_pandas()
        lang = t.target_lang.iloc[0]
        key = (t.query_id.tolist(), t.Eng_Query.fillna("").tolist())
        if ref is None:
            ref, res[lang] = key, {"reference": True}
        else:
            res[lang] = {"query_id_identical": key[0] == ref[0],
                         "eng_query_identical": key[1] == ref[1]}
        print(f"  {lang}: {res[lang]}", flush=True)
    return res


# ---------------------------------------------------------------- pass 3
def label_stats(shard_path: str) -> dict:
    """Relevance labels, answerability, and their correspondence."""
    df = pq.read_table(shard_path, columns=[
        "query_id", "query_type", "Eng_Answer", "passages"]).to_pandas()
    sel = [list(p["is_selected"]) for p in df.passages]
    n_pass = pd.Series([len(s) for s in sel])
    n_sel = pd.Series([int(sum(s)) for s in sel])
    ea = df.Eng_Answer.fillna("")
    no_ans = ea.str.strip().str.lower().eq("no answer present.")
    ct = pd.crosstab(no_ans, n_sel > 0)
    return {
        "rows": len(df),
        "unique_query_id": int(df.query_id.nunique()),
        "query_type_counts": df.query_type.value_counts().to_dict(),
        "passages_per_query": {k: float(v) for k, v in
                               n_pass.describe(percentiles=[.5, .9, .99]).items()},
        "selected_per_query": {int(k): int(v) for k, v in
                               n_sel.value_counts().sort_index().items()},
        "pct_with_label": float(100 * (n_sel >= 1).mean()),
        "pct_no_answer_present": float(100 * no_ans.mean()),
        # the key integrity result: is "No Answer Present." <=> zero selected?
        "crosstab_noanswer_x_hasselected": {str(k): {str(kk): int(vv) for kk, vv in v.items()}
                                            for k, v in ct.to_dict().items()},
    }


# ---------------------------------------------------------------- pass 4
def text_stats(shard_path: str, sample: int = 40_000, seed: int = 0) -> dict:
    """Char + real e5-tokenizer token length distributions, scripts, degeneration."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(EMBED_MODEL)

    t = pq.read_table(shard_path, columns=["query", "Eng_Query", "passages"]).to_pandas()
    flat_e = [x for p in t.passages for x in p["English_passages"]]
    flat_t = [x for p in t.passages for x in p["Translated_passages"]]

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(flat_e), min(sample, len(flat_e)), replace=False)
    se, st = [flat_e[i] for i in idx], [flat_t[i] for i in idx]

    def toklen(xs, bs=512):
        out = []
        for i in range(0, len(xs), bs):
            out += [len(x) for x in tok(xs[i:i + bs], add_special_tokens=False)["input_ids"]]
        return pd.Series(out)

    pcts = [.5, .9, .95, .99]
    series = {
        "passage_en_tokens": toklen(se),
        "passage_indic_tokens": toklen(st),
        "query_en_tokens": toklen(t.Eng_Query.fillna("").tolist()[:20_000]),
        "query_indic_tokens": toklen(t["query"].fillna("").tolist()[:20_000]),
        "passage_en_chars": pd.Series([len(x) for x in flat_e]),
        "passage_indic_chars": pd.Series([len(x) for x in flat_t]),
    }
    dist = {k: {kk: float(vv) for kk, vv in v.describe(percentiles=pcts).items()}
            for k, v in series.items()}

    te, tt = series["passage_en_tokens"], series["passage_indic_tokens"]

    def script_of(s: str) -> str:
        c = collections.Counter()
        for ch in s[:400]:
            if not ch.isalpha():
                continue
            try:
                c[unicodedata.name(ch).split()[0]] += 1
            except ValueError:
                pass
        return c.most_common(1)[0][0] if c else "NONE"

    return {
        "distributions": dist,
        "pct_en_over_512_tok": float(100 * (te > 512).mean()),
        "pct_indic_over_512_tok": float(100 * (tt > 512).mean()),
        "pct_en_under_128_tok": float(100 * (te < 128).mean()),
        "pct_indic_under_128_tok": float(100 * (tt < 128).mean()),
        "indic_over_en_token_ratio_median": float(tt.median() / te.median()),
        "script_en": dict(collections.Counter(script_of(x) for x in se[:4000])),
        "script_indic": dict(collections.Counter(script_of(x) for x in st[:4000])),
        # MT degeneration: repetition loops that also leak the translation prompt.
        # Counted over the FULL column, not the sample - these are ~0.2% of rows,
        # so a 40k sample reports 0 and understates the defect.
        "indic_passages_over_3000_chars": int((series["passage_indic_chars"] > 3000).sum()),
        "indic_passage_instances": len(flat_t),
        "degeneration_marker_hits": int(sum("Translated Text:" in x for x in flat_t)),
        "degeneration_marker_pct": float(100 * sum("Translated Text:" in x for x in flat_t) / len(flat_t)),
        "longest_indic_passage_chars": int(series["passage_indic_chars"].max()),
    }


# ---------------------------------------------------------------- pass 5
def dedup_stats(shard_path: str, sample: int = 30_000, seed: int = 0) -> dict:
    """Exact dup (full column) + MinHash-LSH near-dup (sampled)."""
    t = pq.read_table(shard_path, columns=["passages"]).to_pandas()
    flat_e = [x for p in t.passages for x in p["English_passages"]]
    flat_t = [x for p in t.passages for x in p["Translated_passages"]]

    def h(x):
        return hashlib.blake2b(x.encode(), digest_size=16).digest()

    he, ht = [h(x) for x in flat_e], [h(x) for x in flat_t]
    top = collections.Counter(he).most_common(3)

    rng = np.random.default_rng(seed)
    sub = [flat_e[i] for i in rng.choice(len(flat_e), min(sample, len(flat_e)), replace=False)]

    def sig(s, n=64, k=5):
        w = re.findall(r"\w+", s.lower())
        sh = {hash(tuple(w[i:i + k])) for i in range(max(1, len(w) - k + 1))}
        if not sh:
            return tuple([0] * n)
        return tuple(min((x * (2 * i + 1)) & 0xFFFFFFFF for x in sh) for i in range(n))

    sigs = [sig(x) for x in sub]
    bands = collections.defaultdict(list)
    for i, s in enumerate(sigs):
        for b in range(8):
            bands[(b, s[b * 8:(b + 1) * 8])].append(i)
    near = set()
    for v in bands.values():
        if 1 < len(v) < 50:
            for a in range(len(v)):
                for c in range(a + 1, len(v)):
                    if len(set(sigs[v[a]]) & set(sigs[v[c]])) / 64 >= 0.8:
                        near.add(v[c])
    return {
        "instances": len(flat_e),
        "exact_unique_en": len(set(he)), "exact_unique_indic": len(set(ht)),
        "exact_dup_pct_en": float(100 * (1 - len(set(he)) / len(he))),
        "exact_dup_pct_indic": float(100 * (1 - len(set(ht)) / len(ht))),
        "max_repeat_counts": [c for _, c in top],
        "near_dup_sample": len(sub),
        "near_dup_pct": float(100 * len(near) / len(sub)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("docs/discovery"))
    ap.add_argument("--cache", type=Path, default=Path(".cache/hf"))
    ap.add_argument("--skip-download", action="store_true",
                    help="structural passes only; skips the ~462 MB shard pull")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    R: dict = {"dataset": REPO, "revision": REV, "embed_model_for_tokens": EMBED_MODEL}

    print("[1/5] shard map (footer metadata only)")
    R["shards"] = shard_map({"validation": 14, "train": 13})

    print("[2/5] cross-language parallelism")
    R["parallelism"] = parallelism_check(14)

    if not a.skip_download:
        print(f"[3/5] downloading {PROBE_SPLIT}/{PROBE_SHARD:04d}.parquet (~462 MB)")
        p = hf_hub_download(repo_id=REPO, repo_type="dataset", revision=REV,
                            filename=f"default/{PROBE_SPLIT}/{PROBE_SHARD:04d}.parquet",
                            local_dir=str(a.cache))
        print("[4/5] label + answerability stats")
        R["labels"] = label_stats(p)
        print("[5/5] text + dedup stats")
        R["text"] = text_stats(p)
        R["dedup"] = dedup_stats(p)

    dest = a.out / "discovery.json"
    dest.write_text(json.dumps(R, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
