"""
Secrets and config, loaded from the environment.

`.env` is read only if present and is gitignored. On HF Spaces there is no
.env - the platform injects Space secrets as environment variables - so the
same code path serves both without a branch.

Keys are never written to a log, never put in a URL query string, and never
returned by any endpoint. `describe()` exists so startup can prove a key is
loaded without printing it.
"""
from __future__ import annotations

import os
from pathlib import Path

_LOADED = False


def load_dotenv(path: str | Path = ".env", override: bool = False) -> int:
    """Minimal .env reader. No dependency, no surprises."""
    global _LOADED
    p = Path(path)
    if not p.exists():
        _LOADED = True
        return 0
    n = 0
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and (override or k not in os.environ):
            os.environ[k] = v
            n += 1
    _LOADED = True
    return n


def get(name: str, default: str | None = None) -> str | None:
    if not _LOADED:
        load_dotenv()
    return os.environ.get(name, default)


def require(name: str) -> str:
    v = get(name)
    if not v:
        raise RuntimeError(
            f"{name} is not set. Put it in .env (gitignored) or, on HF Spaces, "
            f"in the Space's secrets. Never in code.")
    return v


def describe(name: str) -> str:
    """Prove a key is loaded without revealing it."""
    v = get(name)
    if not v:
        return f"{name}: not set"
    return f"{name}: set ({len(v)} chars, ends ...{v[-4:]})"


def status() -> dict[str, bool]:
    return {k: bool(get(k)) for k in
            ("SARVAM_API_KEY", "GROQ_API_KEY", "HF_TOKEN")}


if __name__ == "__main__":
    load_dotenv()
    for k in ("SARVAM_API_KEY", "GROQ_API_KEY", "HF_TOKEN"):
        print(" ", describe(k))
