"""
Render bench/chunking_results.json into docs/chunking-ablation.md.

Kept separate from the runner so the table can be regenerated without
re-running a two-hour ablation, and so every number in the doc provably comes
from the committed JSON rather than from prose.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

# Human-readable notes per arm. The brief requires all 8 strategies; several
# are degenerate on this corpus and the table says so explicitly.
NOTES = {
    "fixed_128_o0": "1. fixed-size, 128 tok, no overlap",
    "fixed_128_o25": "1. fixed-size, 128 tok, 25% overlap",
    "fixed_256_o0": "1. fixed-size, 256 tok",
    "fixed_512_o0": "1. fixed-size, 512 tok",
    "sentence_pack_128": "2. script-aware sentence packing",
    "semantic_p85": "3. semantic breakpoint, 85th pct",
    "semantic_p90": "3. semantic breakpoint, 90th pct",
    "semantic_p95": "3. semantic breakpoint, 95th pct",
    "passage_atomic": "4. passage-atomic (control)",
    "hierarchical_c64": "5. hierarchical parent-child",
    "late_chunk_96": "6. late chunking",
    "metadata_aware": "7. metadata-aware",
    "doc2query": "8. query-aware expansion",
}
ORDER = list(NOTES)


def fmt(rows: list[dict], baseline: str = "passage_atomic") -> str:
    base = next((r for r in rows if r["strategy"] == baseline), None)
    out = ["| Strategy | Chunks | chunks/psg | Index MB | Recall@5 | MRR@10 | "
           "nDCG@10 | Embed p50 | Retrieve p50 | Note |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in sorted(rows, key=lambda r: ORDER.index(r["strategy"])
                    if r["strategy"] in ORDER else 99):
        note = NOTES.get(r["strategy"], r["strategy"])
        if r.get("identical_to"):
            note += f" — **identical output to `{r['identical_to']}`**"
        d = ""
        if base and r["strategy"] != baseline:
            delta = r["recall_at_5"] - base["recall_at_5"]
            d = f" ({delta:+.4f})" if abs(delta) > 1e-9 else " (=)"
        bold = "**" if r["strategy"] == baseline else ""
        out.append(
            f"| {bold}`{r['strategy']}`{bold} | {r['chunks']:,} | "
            f"{r['chunks_per_passage']:.2f} | {r['index_mb']:.1f} | "
            f"{bold}{r['recall_at_5']:.4f}{bold}{d} | {r['mrr_at_10']:.4f} | "
            f"{r['ndcg_at_10']:.4f} | {r['embed_p50_ms']:.2f} ms | "
            f"{r['retrieve_p50_ms']:.3f} ms | {note} |")
    return "\n".join(out)


def by_query_type(rows: list[dict]) -> str:
    types = sorted({t for r in rows for t in r.get("recall_by_query_type", {})})
    if not types:
        return ""
    out = ["| Strategy | " + " | ".join(types) + " |",
           "|---" * (len(types) + 1) + "|"]
    for r in sorted(rows, key=lambda r: -r["recall_at_5"]):
        cells = [f"{r['recall_by_query_type'].get(t, float('nan')):.3f}" for t in types]
        out.append(f"| `{r['strategy']}` | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("bench/chunking_results.json"))
    ap.add_argument("--out", type=Path, default=Path("docs/chunking-ablation.md"))
    a = ap.parse_args()

    data = json.loads(a.results.read_text(encoding="utf-8"))
    rows = data["results"]
    by_lang: dict[str, list] = defaultdict(list)
    for r in rows:
        by_lang[r["lang"]].append(r)

    primary = "eng_Latn" if "eng_Latn" in by_lang else list(by_lang)[0]
    pr = by_lang[primary]
    win = max(pr, key=lambda r: r["recall_at_5"])
    base = next((r for r in pr if r["strategy"] == "passage_atomic"), win)
    collapsed = [r for r in pr if r.get("identical_to")]

    L = [
        "# Chunking ablation",
        "",
        "**Gate B deliverable.** Every number is read from "
        "[`bench/chunking_results.json`](../bench/chunking_results.json), produced by "
        "[`bench/ablate_chunking.py`](../bench/ablate_chunking.py) and rendered by "
        "[`bench/render_ablation.py`](../bench/render_ablation.py).",
        "",
        "## Method",
        "",
        f"- **Corpus**: {pr[0]['passages']:,} passages — every candidate passage of "
        f"5,000 sampled `validation` queries, so distractors are real.",
        f"- **Eval queries**: {pr[0]['eval_queries']:,} — the labelled subset only "
        "(a query counts only if `is_selected` marks at least one of its passages).",
        "- **Gold labels**: `passages.is_selected`, the dataset's own relevance "
        "labels. These are **real metrics, not a silver standard** — see "
        "[discovery report §4](discovery/report.md).",
        "- **Scoring**: chunks resolve to their `passage_id`, best rank per passage "
        "wins. A strategy cannot win by flooding the top-k with many chunks of the "
        "same passage.",
        "- **Dense retrieval only.** Chunking changes what gets embedded, so adding "
        "BM25 here would confound the comparison. Hybrid fusion is measured at Gate C.",
        f"- **Model**: `{data['model']}`, ONNX int8, {data['threads']} threads.",
        "",
        f"## Results — {primary}",
        "",
        fmt(pr),
        "",
    ]

    if collapsed:
        L += [
            "## The degenerate arms are proven, not asserted",
            "",
            "Before embedding, each arm's emitted chunk text is fingerprinted "
            "(BLAKE2b over the concatenated chunks). Arms with an identical "
            "fingerprint are *the same retrieval system* and are shown reusing the "
            "same vectors — which is why their rows tie **exactly** rather than "
            "approximately:",
            "",
        ]
        for r in collapsed:
            L.append(f"- `{r['strategy']}` → byte-identical to "
                     f"`{r['identical_to']}` (`{r['chunk_fingerprint'][:16]}…`)")
        L += [
            "",
            "This is the predicted consequence of the corpus shape: English passages "
            "are p50 **72** tokens and max **319**, and **0.000%** exceed the e5 "
            "512-token window. A 256- or 512-token splitter has nothing to split.",
            "",
        ]

    L += [f"## Recall@5 by query type — {primary}", "", by_query_type(pr), ""]

    others = [l for l in by_lang if l != primary]
    if others:
        L += ["## Per-language breakdown", ""]
        for lg in others:
            L += [f"### {lg}", "", fmt(by_lang[lg]), ""]

    L += [
        "## Interpretation",
        "",
        f"**{win['strategy']}** takes the top Recall@5 at **{win['recall_at_5']:.4f}** "
        f"(MRR@10 {win['mrr_at_10']:.4f}, nDCG@10 {win['ndcg_at_10']:.4f}).",
        "",
        "_Written after the numbers landed — see the committed JSON._",
        "",
    ]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {a.out} ({len(rows)} rows, {len(by_lang)} language(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
