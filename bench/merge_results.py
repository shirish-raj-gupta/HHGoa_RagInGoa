"""
Merge ablation result files into one.

Exists because the English run died after 11 of 13 arms and the remaining two
were run separately. Results are written incrementally per arm, so a crash
costs only the arm in flight - this reassembles the pieces instead of
re-running four GPU-hours of work.

Later files win on strategy-name collisions, so a re-run of a single arm
overrides the earlier value.

    python -m bench.merge_results bench/chunking_results.json \\
        bench/_semantic_rest.json --out bench/chunking_results.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    merged: dict = {}
    by_name: dict[str, dict] = {}
    order: list[str] = []
    for f in a.files:
        if not f.exists():
            print(f"  skip (missing) {f}")
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        merged.update({k: v for k, v in d.items() if k != "results"})
        for r in d.get("results", []):
            n = r["strategy"]
            if n not in by_name:
                order.append(n)
            by_name[n] = r
        print(f"  + {f}  ({len(d.get('results', []))} arms)")

    merged["results"] = [by_name[n] for n in order]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(merged, indent=1), encoding="utf-8")
    print(f"\nwrote {a.out} with {len(merged['results'])} arms:")
    for r in merged["results"]:
        print(f"  {r['strategy']:22s} R@5={r['recall_at_5']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
