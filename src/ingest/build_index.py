"""
Build the production index, one language partition at a time.

Memory is the binding constraint, not time. The full corpus is ~14.7M passages;
at 384 dims that is ~11GB of F16 vectors plus ~1.9GB of HNSW graph. This
machine has 15.7GB of RAM. A single flat index cannot be built here, and it
could not be served on a Space either.

Language partitioning is what makes it possible, and it is not a workaround -
it is the same partitioning retrieval already uses (a Hindi query searches
Hindi + English, ~1.9M vectors, never 14.7M). Each partition is ~980k vectors,
about 880MB, built and flushed independently:

    for each language:
        embed in slabs of --slab passages     (bounded peak memory)
        add each slab to the partition        (usearch keeps F16, not F32)
        verify self-retrieval                 (a broken index must not ship)
        save to disk, free, next

Resumable: a language whose partition already exists on disk is skipped, so a
four-hour build that dies at hour three costs one language, not three hours.

    python -m src.ingest.build_index --slice data/slice --out artifacts/index --gpu
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..index.dense import DensePartition
from ..index.embedder import OnnxEmbedder
from ..index.sparse import SparsePartition

ONNX_DIR = Path("artifacts/e5-small-onnx")


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:,.1f}{unit}"
        n /= 1024
    return f"{n:,.1f}TB"


def build_language(lang: str, corpus: pd.DataFrame, embed: OnnxEmbedder,
                   out: Path, slab: int, batch: int, gpu: bool) -> dict:
    t0 = time.perf_counter()
    part = DensePartition(lang, dim=384)
    texts = corpus.text.tolist()
    pids = corpus.passage_id.tolist()
    n = len(texts)

    done = 0
    for s in range(0, n, slab):
        chunk_texts = texts[s:s + slab]
        V = embed.encode_passages(chunk_texts, batch=batch)
        part.add(V, pids[s:s + slab], pids[s:s + slab])
        done += len(chunk_texts)
        del V
        gc.collect()
        el = time.perf_counter() - t0
        rate = done / max(el, 1e-9)
        eta = (n - done) / max(rate, 1e-9)
        print(f"    {done:>9,}/{n:,}  {rate:7.0f} psg/s  "
              f"elapsed {el/60:5.1f}m  eta {eta/60:5.1f}m", flush=True)

    # A silently broken index still returns plausible results. Never ship one.
    sr = part.self_retrieval_rate(300)
    if sr < 0.95:
        raise RuntimeError(f"{lang}: index is broken, self_retrieval={sr:.3f}")

    out.mkdir(parents=True, exist_ok=True)
    part.save(out / f"{lang}.usearch")

    sp = SparsePartition(lang)
    sp.build(texts, pids)
    sp.save(out / f"{lang}.bm25")

    dt = time.perf_counter() - t0
    info = {
        "lang": lang, "passages": n,
        "self_retrieval": round(sr, 4),
        "index_bytes": part.size_bytes,
        "build_s": round(dt, 1),
        "psg_per_s": round(n / max(dt, 1e-9), 1),
        "provider": embed.provider,
    }
    print(f"  {lang}: {n:,} passages, self_retrieval={sr:.3f}, "
          f"{human(part.size_bytes)}, {dt/60:.1f}m", flush=True)
    del part, sp, texts, pids
    gc.collect()
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", type=Path, default=Path("data/slice"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/index"))
    ap.add_argument("--langs", default="",
                    help="default: every language found under --slice")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--slab", type=int, default=200_000,
                    help="passages embedded before flushing to the index")
    ap.add_argument("--batch", type=int, default=0,
                    help="0 = 16 on GPU (measured optimum), 64 on CPU")
    ap.add_argument("--force", action="store_true",
                    help="rebuild languages that already have a partition")
    a = ap.parse_args()

    langs = [l.strip() for l in a.langs.split(",") if l.strip()] or \
        sorted(p.name for p in a.slice.iterdir()
               if p.is_dir() and (p / "corpus.parquet").exists())
    if not langs:
        raise SystemExit(f"no corpora under {a.slice}")

    model = (ONNX_DIR / "fp32" / "model.onnx") if a.gpu else (ONNX_DIR / "model_int8.onnx")
    batch = a.batch or (16 if a.gpu else 64)
    embed = OnnxEmbedder(model, ONNX_DIR, threads=a.threads, use_gpu=a.gpu)
    print(f"provider={embed.provider}  batch={batch}  slab={a.slab:,}")
    print(f"{len(langs)} language(s): {', '.join(langs)}\n")

    manifest_path = a.out / "manifest.json"
    manifest = {"languages": {}, "dim": 384, "dtype": "F16"}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    t_all = time.perf_counter()
    for i, lang in enumerate(langs, 1):
        target = a.out / f"{lang}.usearch"
        if target.exists() and not a.force:
            print(f"[{i}/{len(langs)}] {lang}: already built, skipping")
            continue
        corpus = pd.read_parquet(a.slice / lang / "corpus.parquet")
        print(f"[{i}/{len(langs)}] {lang}: {len(corpus):,} passages", flush=True)
        info = build_language(lang, corpus, embed, a.out, a.slab, batch, a.gpu)
        manifest["languages"][lang] = info
        del corpus
        gc.collect()
        a.out.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    total_p = sum(v["passages"] for v in manifest["languages"].values())
    total_b = sum(v["index_bytes"] for v in manifest["languages"].values())
    dt = time.perf_counter() - t_all
    print(f"\n{len(manifest['languages'])} partitions, {total_p:,} passages, "
          f"{human(total_b)} on disk, {dt/60:.1f} min")
    print(f"manifest -> {manifest_path}")
    worst = min((v["self_retrieval"] for v in manifest["languages"].values()),
                default=1.0)
    print(f"worst self_retrieval across partitions: {worst:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
