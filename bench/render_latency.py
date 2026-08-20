"""
Render bench/results.json into docs/latency-report.md, including the
per-stage stacked chart the README embeds.

The chart is inline SVG on purpose: no chart library, no CDN, renders in a
GitHub README and inside the Space's CSP without a network fetch.

P100 is printed at the same weight as p50 throughout. It is the ugliest number
and the only one that says whether the deadline mechanism actually holds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PCTS = ("p50", "p70", "p90", "p95", "p100")

# Matches web/tokens.css so the doc and the sunline agree on stage colours.
STAGE_COLORS = {
    "normalize": "#4b5c93", "dense": "#FF9E3D", "sparse": "#c9793a",
    "fuse": "#7a6bd8", "guard": "#2FBF8F", "generate": "#2b8f6f",
    "stt": "#3b4a7a",
}


def bar_chart(per_stage: dict, budget: float, width: int = 720) -> str:
    """Horizontal stacked bar: one row per percentile, segments per stage."""
    stages = [s for s in per_stage if per_stage[s].get("p50") is not None]
    if not stages:
        return ""
    rows = []
    totals = {p: sum(per_stage[s].get(p) or 0 for s in stages) for p in PCTS}
    scale_max = max(max(totals.values()), budget) * 1.08
    row_h, gap, left, top = 30, 12, 54, 26

    parts = [
        f'<svg viewBox="0 0 {width} {top + len(PCTS)*(row_h+gap) + 34}" '
        f'width="100%" role="img" aria-label="per-stage latency by percentile" '
        f'xmlns="http://www.w3.org/2000/svg" font-family="ui-monospace,monospace">'
    ]
    # budget line
    bx = left + (budget / scale_max) * (width - left - 90)
    parts.append(
        f'<line x1="{bx:.1f}" y1="{top-14}" x2="{bx:.1f}" '
        f'y2="{top + len(PCTS)*(row_h+gap) - 6}" stroke="#FF5F52" '
        f'stroke-width="1.5" stroke-dasharray="4 3"/>'
        f'<text x="{bx+5:.1f}" y="{top-16}" font-size="11" fill="#FF5F52">'
        f'{budget:.0f}ms budget</text>')

    for i, p in enumerate(PCTS):
        y = top + i * (row_h + gap)
        x = left
        parts.append(f'<text x="0" y="{y+19}" font-size="12" fill="#8A93AD">{p}</text>')
        for s in stages:
            v = per_stage[s].get(p) or 0
            if v <= 0:
                continue
            w = (v / scale_max) * (width - left - 90)
            parts.append(
                f'<rect x="{x:.1f}" y="{y}" width="{max(w,0.7):.1f}" height="{row_h}" '
                f'fill="{STAGE_COLORS.get(s, "#4b5c93")}">'
                f'<title>{s} {p} {v:.2f}ms</title></rect>')
            x += w
        over = totals[p] > budget
        parts.append(
            f'<text x="{x+7:.1f}" y="{y+19}" font-size="12" '
            f'fill="{"#FF5F52" if over else "#2FBF8F"}">{totals[p]:.1f}ms</text>')
        rows.append(p)

    ly = top + len(PCTS) * (row_h + gap) + 16
    lx = left
    for s in stages:
        parts.append(f'<rect x="{lx}" y="{ly-9}" width="9" height="9" '
                     f'fill="{STAGE_COLORS.get(s, "#4b5c93")}"/>')
        parts.append(f'<text x="{lx+13}" y="{ly}" font-size="11" fill="#8A93AD">{s}</text>')
        lx += 22 + 7.2 * len(s)
    parts.append("</svg>")
    return "".join(parts)


def pct_row(name: str, d: dict) -> str:
    cells = " | ".join(
        f"{d[p]:.1f}" if d.get(p) is not None else "—" for p in PCTS)
    return f"| {name} | {cells} | {d.get('n', '—')} |"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("bench/results.json"))
    ap.add_argument("--out", type=Path, default=Path("docs/latency-report.md"))
    a = ap.parse_args()

    d = json.loads(a.results.read_text(encoding="utf-8"))
    s = d["summary"]
    hw, budget = s["hardware"], s["budget_ms"]
    warm, cold = s["CORE_RAG_LOOP_MS"]["warm"], s["CORE_RAG_LOOP_MS"]["cold"]

    verdict = ("**PASS**" if (warm.get("p100") or 1e9) < budget else "**FAIL**")

    L = [
        "# Latency report",
        "",
        "**Gate C deliverable.** Every number is read from "
        "[`bench/results.json`](../bench/results.json), produced by "
        "[`bench/run.py`](../bench/run.py).",
        "",
        "## The claim",
        "",
        f"`CORE_RAG_LOOP_MS` **p100 = {warm.get('p100', float('nan')):.1f} ms** "
        f"against a **{budget:.0f} ms** budget over {s['n_queries']} queries "
        f"→ {verdict}.",
        "",
        "`CORE_RAG_LOOP_MS` is T2–T6: normalize + language ID, query embedding, "
        "dense ∥ sparse retrieval, RRF fusion + MMR, and the input/relevance "
        "guardrails. It excludes STT and generation, which are network-bound on "
        "third-party vendors. Those are reported below at the same prominence — "
        "see the measurement contract in the [README](../README.md).",
        "",
        "## Hardware, because a latency number without a machine is not a number",
        "",
        "| | |",
        "|---|---|",
        f"| Platform | `{hw.get('platform')}` |",
        f"| CPU | {hw.get('processor') or '—'} |",
        f"| Cores | {hw.get('cpu_physical', '—')} physical / "
        f"{hw.get('cpu_count', '—')} logical |",
        f"| RAM | {hw.get('ram_gb', '—')} GB |",
        f"| Python | {hw.get('python')} |",
        f"| ONNX threads | {s['threads']} |",
        f"| Embed batch | {s['embed_batch']} |",
        f"| Index build | {s['index_build_s']} s |",
        f"| Warmup | {s['warmup_ms']} ms |",
        f"| Relevance τ | {s.get('tau')} |",
        "",
        "Index parameters per partition:",
        "",
        "| Language | Vectors | dtype | M | ef_add | ef_search | self-retrieval |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for lg, p in s["index_params"].items():
        L.append(f"| `{lg}` | {p['vectors']:,} | {p['dtype']} | {p['connectivity']} | "
                 f"{p['expansion_add']} | {p['expansion_search']} | "
                 f"{p['self_retrieval']:.4f} |")

    L += [
        "",
        "## CORE_RAG_LOOP_MS",
        "",
        "| Phase | " + " | ".join(PCTS) + " | n |",
        "|---|" + "---:|" * (len(PCTS) + 1),
        pct_row("**warm**", warm),
        pct_row("cold", cold),
        "",
        "Cold runs are **reported, not discarded**. Silently dropping them is how "
        "a p100 gets flattering. In the deployed Space the cold path is paid at "
        "boot, before the readiness probe passes, so live traffic sees the warm "
        "numbers.",
        "",
        "## Per stage",
        "",
        "| Stage | " + " | ".join(PCTS) + " | n |",
        "|---|" + "---:|" * (len(PCTS) + 1),
    ]
    for name, v in s["per_stage_warm"].items():
        L.append(pct_row(f"`{name}`", v))

    L += ["", bar_chart(s["per_stage_warm"], budget), ""]

    L += ["## Per language", "",
          "| Language | " + " | ".join(PCTS) + " | n |",
          "|---|" + "---:|" * (len(PCTS) + 1)]
    for lg, v in s["per_language_warm"].items():
        L.append(pct_row(f"`{lg}`", v))

    L += [
        "",
        "## Budget behaviour",
        "",
        "| | |",
        "|---|---:|",
        f"| Queries over budget | {s['over_budget_warm']} |",
        f"| Queries that degraded | {s['degraded_warm']} |",
        f"| Queries refused | {s['refused_warm']} |",
        "",
        "Degradation is the mechanism that makes the budget real rather than "
        "aspirational: each stage reads `Budget.remaining_ms` and downgrades — "
        "drops `ef_search`, cuts `k`, skips MMR, falls back to sparse-only — "
        "instead of overrunning. Every degradation is logged as a structured "
        "event and appears in `/trace/{request_id}`.",
        "",
        "## The other boundaries",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| `STT_MS` | {s.get('STT_MS') or '**not measured**'} |",
        f"| `TTFT_MS` | {s.get('TTFT_MS') or '**not measured**'} |",
        f"| `E2E_MS` | {s.get('E2E_MS') or '**not measured**'} |",
        "",
        s.get("note", ""),
        "",
    ]

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
