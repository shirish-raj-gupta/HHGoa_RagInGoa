"""
Deploy to Hugging Face: index artifacts to a dataset repo, code to a Space.

The index does not fit in a Space repo. Fifteen partitions are ~12.8GB of
vectors plus ~5.7GB of compressed passage text, and a Space is meant to hold
code, not an 18GB corpus. So the artifacts live in a **dataset** repo and the
Space pulls what it needs at boot, which also means redeploying the app does
not re-upload 18GB.

Nothing here runs implicitly. `--dry-run` (the default) prints the plan and
uploads nothing; `--push` is required to actually write to the Hub, and the
target repo ids must be passed explicitly rather than guessed from the token.

    python -m scripts.deploy_space --index-repo srg101/raginggoa-index \\
        --space srg101/raginggoa --langs eng_Latn,hin_Deva,tam_Taml
    python -m scripts.deploy_space ... --push
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get as cfg_get, load_dotenv   # noqa: E402

SPACE_FILES = [
    "Dockerfile", "pyproject.toml", "README.md",
    "src", "web", "bench", "docs", "tests",
]
# Never ship these to a public Space, whatever the glob says.
NEVER = {".env", ".env.local", "data", "artifacts", ".venv", ".venv-gpu",
         ".git", "__pycache__"}


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:,.1f}{u}"
        n /= 1024
    return f"{n:,.1f}TB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", type=Path, default=Path("artifacts/index"))
    ap.add_argument("--onnx-dir", type=Path,
                    default=Path("artifacts/e5-small-onnx"))
    ap.add_argument("--index-repo", required=True,
                    help="dataset repo for the index, e.g. srg101/raginggoa-index")
    ap.add_argument("--space", required=True,
                    help="Space repo id, e.g. srg101/raginggoa")
    ap.add_argument("--langs", default="",
                    help="languages to upload; default every partition present")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--push", action="store_true",
                    help="actually write to the Hub (default is a dry run)")
    a = ap.parse_args()

    load_dotenv()
    token = cfg_get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set (put it in .env)")

    from huggingface_hub import HfApi
    api = HfApi(token=token)
    who = api.whoami()["name"]
    print(f"authenticated as {who}\n")

    langs = [l.strip() for l in a.langs.split(",") if l.strip()]
    parts = sorted(a.index_dir.glob("*.usearch"))
    if langs:
        parts = [p for p in parts if p.stem in langs]
    if not parts:
        raise SystemExit(f"no partitions under {a.index_dir} (build them first)")

    payload: list[Path] = []
    for p in parts:
        lang = p.stem
        for suffix in (".usearch", ".meta.json", ".texts.db"):
            f = a.index_dir / f"{lang}{suffix}"
            if f.exists():
                payload.append(f)
        bm = a.index_dir / f"{lang}.bm25"
        if bm.is_dir():
            payload.extend(x for x in bm.rglob("*") if x.is_file())
    mani = a.index_dir / "manifest.json"
    if mani.exists():
        payload.append(mani)
    if a.onnx_dir.exists():
        payload.extend(x for x in a.onnx_dir.rglob("*")
                       if x.is_file() and "fp32" not in x.parts)

    total = sum(f.stat().st_size for f in payload)
    print(f"index repo  : {a.index_repo}  ({'private' if a.private else 'public'})")
    print(f"  {len(parts)} partition(s): {', '.join(p.stem for p in parts)}")
    print(f"  {len(payload)} files, {human(total)}")
    print(f"\nspace       : {a.space}")
    for f in SPACE_FILES:
        pth = Path(f)
        if not pth.exists():
            print(f"  MISSING {f}")
        else:
            n = sum(1 for _ in pth.rglob('*')) if pth.is_dir() else 1
            print(f"  {f:16s} {n} file(s)")
    print(f"\nexcluded from the Space: {', '.join(sorted(NEVER))}")

    if not a.push:
        print("\nDRY RUN - nothing uploaded. Re-run with --push to write to the Hub.")
        print("The Space will pull the index at boot via RAG_INDEX_REPO.")
        return 0

    print("\nuploading index artifacts...")
    api.create_repo(a.index_repo, repo_type="dataset", private=a.private,
                    exist_ok=True)
    api.upload_folder(
        folder_path=str(a.index_dir), repo_id=a.index_repo, repo_type="dataset",
        allow_patterns=[f"{p.stem}*" for p in parts] + ["manifest.json"],
        commit_message="index partitions")
    if a.onnx_dir.exists():
        api.upload_folder(
            folder_path=str(a.onnx_dir), repo_id=a.index_repo,
            repo_type="dataset", path_in_repo="e5-small-onnx",
            ignore_patterns=["fp32/*"], commit_message="onnx embedder")

    print("creating/updating the Space...")
    api.create_repo(a.space, repo_type="space", space_sdk="docker",
                    private=a.private, exist_ok=True)
    for f in SPACE_FILES:
        pth = Path(f)
        if not pth.exists():
            continue
        if pth.is_dir():
            api.upload_folder(folder_path=str(pth), repo_id=a.space,
                              repo_type="space", path_in_repo=f,
                              ignore_patterns=[f"**/{n}/**" for n in NEVER],
                              commit_message=f"deploy {f}")
        else:
            api.upload_file(path_or_fileobj=str(pth), path_in_repo=f,
                            repo_id=a.space, repo_type="space",
                            commit_message=f"deploy {f}")

    print(f"\ndone: https://huggingface.co/spaces/{a.space}")
    print("Set these as Space secrets (never in the repo):")
    print("  SARVAM_API_KEY, GROQ_API_KEY")
    print(f"Set this as a Space variable:  RAG_INDEX_REPO={a.index_repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
