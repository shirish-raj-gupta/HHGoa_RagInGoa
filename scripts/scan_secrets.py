"""
Vendor-specific secret scan, run from .pre-commit-config.yaml.

gitleaks already runs generically. This adds the two key shapes this project
actually handles, plus the mistake that generic scanners miss: an API key
passed as a URL query parameter, which then leaks into proxy logs, browser
history and screenshots even though it never appears as an assignment.

Exits non-zero on a hit, which blocks the commit.

    python scripts/scan_secrets.py [files...]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PATTERNS = {
    "sarvam_key": re.compile(r"\bsk_[A-Za-z0-9_\-]{16,}\b"),
    "anthropic_key": re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b"),
    "hf_token": re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    "aws_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    # assignment to a key-ish name with a long literal
    "assigned_key": re.compile(
        r"(?i)\b(api[_-]?key|subscription[_-]?key|secret|token|password)\b"
        r"\s*[=:]\s*[\"'][A-Za-z0-9_\-]{20,}[\"']"),
    # the one generic scanners miss: a key in a URL query string
    "key_in_url": re.compile(
        r"(?i)[?&](api[_-]?key|subscription[_-]?key|token|access[_-]?token)="
        r"[A-Za-z0-9_\-]{12,}"),
}

# Placeholders and env lookups are the CORRECT pattern, not a finding.
ALLOW = re.compile(
    r"(?i)(os\.environ|getenv|\$\{?[A-Z_]+\}?|<[a-z_]+>|your[_-]?key|"
    r"xxx+|\bexample\b|\bplaceholder\b|\bdummy\b|\bfake\b|\*{4,})")

SKIP_DIRS = {".git", ".venv", ".venv-gpu", "__pycache__", "node_modules",
             "artifacts", "data", ".cache"}
SKIP_SUFFIX = {".parquet", ".onnx", ".bin", ".safetensors", ".png", ".jpg",
               ".webp", ".mp4", ".woff", ".woff2", ".usearch", ".lock"}


def scan(path: Path) -> list[tuple[int, str, str]]:
    if path.suffix.lower() in SKIP_SUFFIX:
        return []
    if any(part in SKIP_DIRS for part in path.parts):
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    # do not flag this scanner's own pattern definitions
    if path.name == "scan_secrets.py":
        return []

    out = []
    for i, line in enumerate(text.splitlines(), 1):
        if ALLOW.search(line):
            continue
        for name, pat in PATTERNS.items():
            if pat.search(line):
                out.append((i, name, line.strip()[:100]))
    return out


def main(argv: list[str]) -> int:
    targets = [Path(p) for p in argv[1:]] or \
        [p for p in Path(".").rglob("*") if p.is_file()]
    findings = [(p, *f) for p in targets if p.is_file() for f in scan(p)]
    if not findings:
        return 0
    print("BLOCKED: possible secrets in tracked files\n")
    for path, line_no, kind, snippet in findings:
        print(f"  {path}:{line_no}  [{kind}]\n      {snippet}")
    print("\nKeys belong in HF Space secrets and are read via os.environ.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
