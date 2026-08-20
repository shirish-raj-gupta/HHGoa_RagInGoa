"""
Download the MSMARCO-XI validation parquet shards.

One shard per language, ~450 MB each, 14 of them (~6.3 GB). The train split is
deliberately not touched: it is 49 GB, and it is missing Telugu entirely (see
docs/discovery/report.md §2), so validation is both smaller AND more complete.

Column projection cannot help here - `passages` is 445 MB of each 462 MB file,
and passages are the thing we need - so the whole shard comes down. Downloads
are resumable and skipped if already present, because a dropped connection 5 GB
in should not restart from zero.

Files land at <dir>/default/validation/XXXX.parquet, the layout
src/ingest/build_slice.py expects.

    python -m src.ingest.fetch_shards --out data/shards
    python -m src.ingest.fetch_shards --out data/shards --langs hin_Deva,tam_Taml
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "ai4bharat/MSMARCO-XI"
REVISION = "refs/convert/parquet"

# Verified by reading each shard's parquet column statistics, not assumed:
# every shard holds exactly one target_lang.
LANG_OF_SHARD = {
    0: "asm_Beng", 1: "ben_Beng", 2: "guj_Gujr", 3: "hin_Deva", 4: "kan_Knda",
    5: "mal_Mlym", 6: "mar_Deva", 7: "npi_Deva", 8: "ory_Orya", 9: "pan_Guru",
    10: "san_Deva", 11: "tam_Taml", 12: "tel_Telu", 13: "urd_Arab",
}
SHARD_OF_LANG = {v: k for k, v in LANG_OF_SHARD.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/shards"))
    ap.add_argument("--langs", default=",".join(LANG_OF_SHARD.values()),
                    help="comma-separated FLORES codes; default is all 14")
    a = ap.parse_args()

    langs = [l.strip() for l in a.langs.split(",") if l.strip()]
    unknown = [l for l in langs if l not in SHARD_OF_LANG]
    if unknown:
        raise SystemExit(f"unknown languages: {unknown}")

    dest = a.out / "default" / "validation"
    dest.mkdir(parents=True, exist_ok=True)
    print(f"{len(langs)} shard(s) -> {dest}")

    total_bytes, t_all = 0, time.perf_counter()
    for i, lang in enumerate(langs, 1):
        idx = SHARD_OF_LANG[lang]
        name = f"{idx:04d}.parquet"
        target = dest / name
        if target.exists() and target.stat().st_size > 1_000_000:
            mb = target.stat().st_size / 1e6
            total_bytes += target.stat().st_size
            print(f"  [{i:2d}/{len(langs)}] {lang:10s} {name}  "
                  f"{mb:7.0f} MB  already present")
            continue

        t0 = time.perf_counter()
        got = hf_hub_download(
            repo_id=REPO, repo_type="dataset", revision=REVISION,
            filename=f"default/validation/{name}",
            local_dir=str(a.out),
        )
        dt = time.perf_counter() - t0
        size = Path(got).stat().st_size
        total_bytes += size
        print(f"  [{i:2d}/{len(langs)}] {lang:10s} {name}  "
              f"{size/1e6:7.0f} MB  {dt:6.0f}s  {size/1e6/max(dt,1e-9):5.1f} MB/s",
              flush=True)

    dt = time.perf_counter() - t_all
    print(f"\n{total_bytes/1e9:.1f} GB in {dt/60:.1f} min -> {dest}")
    print(f"next: python -m src.ingest.build_slice --shards {a.out} "
          f"--langs {','.join(langs)} --n-queries 97941")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
