"""
Deploy the frontend to a Hugging Face **Static** Space.

Why static: this account cannot host compute on HF. Both alternatives were
tested against the live API, not assumed —

    Docker Space        402 Payment Required, needs PRO
    Gradio + ZeroGPU    402, "wait 30 days or request a community grant"
    Static Space        created fine

So the Space serves the UI and the backend runs elsewhere (a local container
behind a tunnel). `--api` bakes that URL into config.js at deploy time; a
visitor can override it with ?api=... without a redeploy.

    python -m scripts.deploy_static_space --space srg101/raginggoa \\
        --api https://<your>.trycloudflare.com
    python -m scripts.deploy_static_space ... --push
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get as cfg_get, load_dotenv   # noqa: E402

README = """---
title: RAG in Goa
emoji: 🌅
colorFrom: indigo
colorTo: yellow
sdk: static
app_file: index.html
pinned: false
short_description: Voice RAG over MS MARCO-XI, 14 Indic languages
---

# RAG in Goa

Voice-to-grounded-answer RAG over [`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) — English plus 14 Indic languages.
**HH Goa 2026 · Task #02** · `#RAGInGoa`

## This page is a frontend

The retrieval backend is **not hosted here**. A Static Space can serve files
but cannot run Python, and this account's tier returns `402 Payment Required`
for both Docker and ZeroGPU Spaces. The index is also 15 GB across 15
partitions, which does not belong on ephemeral Space storage.

So the API runs on a separate machine and this page points at it. If the
backend is down you will see a "backend unreachable" panel rather than a
broken app — that state is deliberate.

Point it anywhere with `?api=https://your-backend`, or run the whole thing
locally with `docker compose up` and then `?api=http://localhost:7860`.

## What was measured

| | |
|---|---|
| `CORE_RAG_LOOP_MS` p100 | **19.0 ms** (150k vectors) · **108.5 ms** (953k) vs a 200 ms budget |
| Chunking | 25 arms, 3 languages — **no strategy beats the passage-atomic control significantly** |
| Cross-lingual | R@5 **0.8814 → 0.6786 → 0.4972** (en → hi → ta) on *identical parallel content* |
| Guardrails | block **0.938**, false-refusal **0.154**, every failure published |
| Corpus | 14,265,074 passages, 15 languages |

Full reports, the scripts that produced every number, and the honest list of
bugs and corrections are in the repo.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, help="e.g. srg101/raginggoa")
    ap.add_argument("--api", default="", help="backend base URL baked into config.js")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--push", action="store_true",
                    help="actually write to the Hub (default is a dry run)")
    a = ap.parse_args()

    load_dotenv()
    token = cfg_get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set (put it in .env)")

    web = Path("web")
    for f in ("index.html", "tokens.css", "config.js"):
        if not (web / f).exists():
            raise SystemExit(f"missing web/{f}")

    staged = Path(tempfile.mkdtemp()) / "space"
    staged.mkdir(parents=True)
    shutil.copy2(web / "index.html", staged / "index.html")
    shutil.copy2(web / "tokens.css", staged / "tokens.css")
    (staged / "config.js").write_text(
        '// Written by scripts/deploy_static_space.py at deploy time.\n'
        '// Override without redeploying: append ?api=https://... to the URL.\n'
        f'window.RAG_API = {json.dumps(a.api)};\n', encoding="utf-8")
    (staged / "README.md").write_text(README, encoding="utf-8")

    print(f"space   : {a.space}  ({'private' if a.private else 'public'})")
    print(f"backend : {a.api or '(none set - page will show the offline state)'}")
    for f in sorted(staged.iterdir()):
        print(f"  {f.name:14s} {f.stat().st_size:>8,} B")

    if not a.push:
        print("\nDRY RUN - nothing uploaded. Re-run with --push.")
        return 0

    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(a.space, repo_type="space", space_sdk="static",
                    private=a.private, exist_ok=True)
    api.upload_folder(folder_path=str(staged), repo_id=a.space,
                      repo_type="space", commit_message="deploy frontend")
    url = f"https://huggingface.co/spaces/{a.space}"
    print(f"\ndeployed: {url}")
    print(f"direct  : https://{a.space.replace('/', '-').lower()}.static.hf.space")
    if not a.api:
        print("\nNo --api set. Deploy again with --api once the tunnel is up,")
        print("or visit the Space with ?api=https://... appended.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
