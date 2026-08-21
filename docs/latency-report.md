# Latency report

**Gate C deliverable.** Every number is read from [`bench/results.json`](../bench/results.json), produced by [`bench/run.py`](../bench/run.py).

## The claim

`CORE_RAG_LOOP_MS` **p100 = 14.9 ms** against a **200 ms** budget over 224 queries → **PASS**.

`CORE_RAG_LOOP_MS` is T2–T6: normalize + language ID, query embedding, dense ∥ sparse retrieval, RRF fusion + MMR, and the input/relevance guardrails. It excludes STT and generation, which are network-bound on third-party vendors. Those are reported below at the same prominence — see the measurement contract in the [README](../README.md).

## Hardware, because a latency number without a machine is not a number

| | |
|---|---|
| Platform | `Windows-11-10.0.26200-SP0` |
| CPU | Intel64 Family 6 Model 183 Stepping 1, GenuineIntel |
| Cores | 16 physical / 24 logical |
| RAM | 16.9 GB |
| Python | 3.13.1 |
| ONNX threads | 8 |
| Embed batch | 64 |
| Index build | 40.9 s |
| Warmup | 11.1 ms |
| Relevance τ | per-language (see below) |

Index parameters per partition:

| Language | Vectors | dtype | M | ef_add | ef_search | self-retrieval |
|---|---:|---|---:|---:|---:|---:|
| `eng_Latn` | 49,611 | ScalarKind.F16 | 16 | 128 | 64 | 0.9950 |
| `hin_Deva` | 49,556 | ScalarKind.F16 | 16 | 128 | 64 | 1.0000 |
| `tam_Taml` | 49,235 | ScalarKind.F16 | 16 | 128 | 64 | 1.0000 |

### Relevance gate, per language

| Language | τ | AUC | state |
|---|---:|---:|---|
| `eng_Latn` | 0.88265 | 0.9102 | active |
| `hin_Deva` | 0.88134 | 0.8085 | active |
| `tam_Taml` | — | 0.6895 | **disabled** — AUC below the 0.75 floor |

One global τ calibrated on English refused **72.7% of answerable Tamil queries** — 100% of the Tamil `description` stratum — because the cosine distribution shifts with the language. Per-language τ cut overall false-refusal from **35.4% to 13.8%**.

Tamil's gate is **switched off**, not merely loosened: at AUC 0.6895 its positive and negative score distributions nearly overlap (means 0.8739 vs 0.8605), and a threshold fitted to a curve that cannot discriminate refuses at random while looking principled. The cost is explicit — Tamil now refuses 0% of out-of-corpus queries as well as 0% of good ones — and the output-side groundedness rail carries that language instead.


## CORE_RAG_LOOP_MS

| Phase | p50 | p70 | p90 | p95 | p100 | n |
|---|---:|---:|---:|---:|---:|---:|
| **warm** | 9.9 | 10.5 | 11.5 | 11.9 | 14.9 | 667 |
| cold | 13.9 | 14.6 | 16.2 | 16.7 | 17.2 | 5 |

Cold runs are **reported, not discarded**. Silently dropping them is how a p100 gets flattering. In the deployed Space the cold path is paid at boot, before the readiness probe passes, so live traffic sees the warm numbers.

## Per stage

| Stage | p50 | p70 | p90 | p95 | p100 | n |
|---|---:|---:|---:|---:|---:|---:|
| `dense` | 9.2 | 9.8 | 10.7 | 11.2 | 14.1 | 667 |
| `fuse` | 0.3 | 0.3 | 0.4 | 0.4 | 0.8 | 667 |
| `normalize` | 0.1 | 0.1 | 0.1 | 0.1 | 0.3 | 667 |
| `sparse` | 8.8 | 9.6 | 10.7 | 11.1 | 14.1 | 667 |

<svg viewBox="0 0 720 270" width="100%" role="img" aria-label="per-stage latency by percentile" xmlns="http://www.w3.org/2000/svg" font-family="ui-monospace,monospace"><line x1="587.3" y1="12" x2="587.3" y2="230" stroke="#FF5F52" stroke-width="1.5" stroke-dasharray="4 3"/><text x="592.3" y="10" font-size="11" fill="#FF5F52">200ms budget</text><text x="0" y="45" font-size="12" fill="#8A93AD">p50</text><rect x="54.0" y="26" width="24.6" height="30" fill="#FF9E3D"><title>dense p50 9.23ms</title></rect><rect x="78.6" y="26" width="0.9" height="30" fill="#7a6bd8"><title>fuse p50 0.33ms</title></rect><rect x="79.5" y="26" width="0.7" height="30" fill="#4b5c93"><title>normalize p50 0.07ms</title></rect><rect x="79.7" y="26" width="23.6" height="30" fill="#c9793a"><title>sparse p50 8.85ms</title></rect><text x="110.3" y="45" font-size="12" fill="#2FBF8F">18.5ms</text><text x="0" y="87" font-size="12" fill="#8A93AD">p70</text><rect x="54.0" y="68" width="26.1" height="30" fill="#FF9E3D"><title>dense p70 9.79ms</title></rect><rect x="80.1" y="68" width="0.9" height="30" fill="#7a6bd8"><title>fuse p70 0.34ms</title></rect><rect x="81.0" y="68" width="0.7" height="30" fill="#4b5c93"><title>normalize p70 0.08ms</title></rect><rect x="81.3" y="68" width="25.7" height="30" fill="#c9793a"><title>sparse p70 9.62ms</title></rect><text x="113.9" y="87" font-size="12" fill="#2FBF8F">19.8ms</text><text x="0" y="129" font-size="12" fill="#8A93AD">p90</text><rect x="54.0" y="110" width="28.7" height="30" fill="#FF9E3D"><title>dense p90 10.75ms</title></rect><rect x="82.7" y="110" width="1.0" height="30" fill="#7a6bd8"><title>fuse p90 0.36ms</title></rect><rect x="83.6" y="110" width="0.7" height="30" fill="#4b5c93"><title>normalize p90 0.10ms</title></rect><rect x="83.9" y="110" width="28.5" height="30" fill="#c9793a"><title>sparse p90 10.69ms</title></rect><text x="119.4" y="129" font-size="12" fill="#2FBF8F">21.9ms</text><text x="0" y="171" font-size="12" fill="#8A93AD">p95</text><rect x="54.0" y="152" width="29.8" height="30" fill="#FF9E3D"><title>dense p95 11.16ms</title></rect><rect x="83.8" y="152" width="1.0" height="30" fill="#7a6bd8"><title>fuse p95 0.36ms</title></rect><rect x="84.7" y="152" width="0.7" height="30" fill="#4b5c93"><title>normalize p95 0.11ms</title></rect><rect x="85.0" y="152" width="29.7" height="30" fill="#c9793a"><title>sparse p95 11.13ms</title></rect><text x="121.7" y="171" font-size="12" fill="#2FBF8F">22.8ms</text><text x="0" y="213" font-size="12" fill="#8A93AD">p100</text><rect x="54.0" y="194" width="37.6" height="30" fill="#FF9E3D"><title>dense p100 14.11ms</title></rect><rect x="91.6" y="194" width="2.0" height="30" fill="#7a6bd8"><title>fuse p100 0.75ms</title></rect><rect x="93.6" y="194" width="0.7" height="30" fill="#4b5c93"><title>normalize p100 0.26ms</title></rect><rect x="94.3" y="194" width="37.7" height="30" fill="#c9793a"><title>sparse p100 14.15ms</title></rect><text x="139.0" y="213" font-size="12" fill="#2FBF8F">29.3ms</text><rect x="54" y="243" width="9" height="9" fill="#FF9E3D"/><text x="67" y="252" font-size="11" fill="#8A93AD">dense</text><rect x="112.0" y="243" width="9" height="9" fill="#7a6bd8"/><text x="125.0" y="252" font-size="11" fill="#8A93AD">fuse</text><rect x="162.8" y="243" width="9" height="9" fill="#4b5c93"/><text x="175.8" y="252" font-size="11" fill="#8A93AD">normalize</text><rect x="249.60000000000002" y="243" width="9" height="9" fill="#c9793a"/><text x="262.6" y="252" font-size="11" fill="#8A93AD">sparse</text></svg>

## Per language

| Language | p50 | p70 | p90 | p95 | p100 | n |
|---|---:|---:|---:|---:|---:|---:|
| `eng_Latn` | 9.0 | 9.3 | 10.3 | 10.7 | 13.3 | 191 |
| `hin_Deva` | 10.0 | 10.5 | 11.4 | 11.8 | 13.9 | 285 |
| `tam_Taml` | 10.6 | 11.2 | 11.9 | 12.6 | 14.9 | 191 |

## Budget behaviour

| | |
|---|---:|
| Queries over budget | 0 |
| Queries that degraded | 0 |
| Queries refused | 108 |

Degradation is the mechanism that makes the budget real rather than aspirational: each stage reads `Budget.remaining_ms` and downgrades — drops `ef_search`, cuts `k`, skips MMR, falls back to sparse-only — instead of overrunning. Every degradation is logged as a structured event and appears in `/trace/{request_id}`.

## The other boundaries

| Metric | Value |
|---|---|
| `STT_MS` | **not measured** |
| `TTFT_MS` | **not measured** |
| `E2E_MS` | **not measured** |

STT_MS/TTFT_MS/E2E_MS require live API keys; run with --with-generation and a network path to populate them.
