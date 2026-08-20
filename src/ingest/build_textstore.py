"""
Build the per-language passage text stores.

Separate from build_index.py on purpose: this is pure CPU/IO and the index
build is GPU-bound, so the two run concurrently instead of serially. It is also
re-runnable without touching a four-hour embedding job.

    python -m src.ingest.build_textstore --slice data/full --out artifacts/index
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from ..index.textstore import TextStore


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", type=Path, default=Path("data/full"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/index"))
    ap.add_argument("--langs", default="")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    langs = [l.strip() for l in a.langs.split(",") if l.strip()] or \
        sorted(p.name for p in a.slice.iterdir()
               if p.is_dir() and (p / "corpus.parquet").exists())
    a.out.mkdir(parents=True, exist_ok=True)

    total = 0
    t_all = time.perf_counter()
    for i, lang in enumerate(langs, 1):
        db = a.out / f"{lang}.texts.db"
        if db.exists() and not a.force:
            print(f"[{i}/{len(langs)}] {lang}: exists, skipping")
            continue
        t0 = time.perf_counter()
        c = pd.read_parquet(a.slice / lang / "corpus.parquet",
                            columns=["passage_id", "lang", "text"])
        tmp = db.with_suffix(".tmp")
        tmp.unlink(missing_ok=True)
        st = TextStore(tmp, read_only=False)
        # chunked inserts keep peak memory flat on ~950k rows
        step = 100_000
        for s in range(0, len(c), step):
            part = c.iloc[s:s + step]
            st.add_many(zip(part.passage_id, part.lang, part.text))
        n = st.count()
        st.close()
        tmp.replace(db)                    # atomic: never leave a half-written db
        total += n
        print(f"[{i}/{len(langs)}] {lang}: {n:,} passages, "
              f"{db.stat().st_size/1e6:.0f}MB, {time.perf_counter()-t0:.0f}s",
              flush=True)
    print(f"\n{total:,} passages in {(time.perf_counter()-t_all)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
