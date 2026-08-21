# RAG in Goa

**Live:** [huggingface.co/spaces/srg101/raginggoa](https://huggingface.co/spaces/srg101/raginggoa) — frontend only. The retrieval backend is ~15 GB across 15 language partitions and runs separately behind a tunnel, because this account's tier returns `402` for both Docker and ZeroGPU Spaces. If the backend is down the page says so, deliberately. See [`docs/deployment.md`](docs/deployment.md).

Voice-to-grounded-answer RAG over [`ai4bharat/MSMARCO-XI`](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) — English plus 14 Indic languages.
**HH Goa 2026 · Task #02** · `#RAGInGoa`

---

## The measurement contract

A round trip to a hosted STT API is 300 ms – 1 s. A hosted LLM's first token is 200–800 ms.
**No voice-in/answer-out pipeline is under 200 ms end to end, and any submission claiming otherwise is measuring something narrower.** So here is exactly what is measured, and where the boundary sits.

```
T0  mic stop / audio bytes complete
T1  Sarvam transcript returned          → STT_MS       (network-bound, reported separately)
T2  transcript normalized + lang-ID     ─┐
T3  query embedded (local ONNX int8)     │
T4  BM25 + dense retrieved in parallel   ├─ CORE_RAG_LOOP_MS  ← the < 200 ms claim
T5  RRF fused + MMR diversified          │
T6  input + grounding guardrails passed  ─┘
T7  first answer token                  → TTFT_MS
T8  answer complete + citations verified → E2E_MS
```

**The headline claim is `CORE_RAG_LOOP_MS` p100 < 200 ms.** `STT_MS`, `TTFT_MS` and `E2E_MS` are published in the same table, at the same prominence, because hiding them would be the dodge.

### The numbers

Full report: [`docs/latency-report.md`](docs/latency-report.md) · data: [`bench/results.json`](bench/results.json)
224 stratified queries × 3 reps = 667 warm measurements, Intel i7-14650HX, 8 ONNX threads, index F16 M=16 ef_add=256.

| | p50 | p70 | p90 | p95 | **p100** |
|---|---:|---:|---:|---:|---:|
| **`CORE_RAG_LOOP_MS`** warm | 9.9 | 10.5 | 11.5 | 11.9 | **14.9** |
| `CORE_RAG_LOOP_MS` cold | 13.9 | 14.6 | 16.2 | 16.7 | 17.2 |
| `STT_MS` (Sarvam, live) | — | — | — | — | 2,748–2,886 |
| `TTFT_MS` (Groq) | — | — | — | — | ~1,100 |
| `E2E_MS` | — | — | — | — | ~4,000 |

**p100 = 14.9 ms against a 200 ms budget.** 0 of 667 queries went over budget; 0 needed degradation. Cold runs are reported rather than discarded.

Per stage (warm p50): `dense` 9.2 ms ∥ `sparse` 8.9 ms (concurrent), `fuse` 0.33 ms, `normalize` 0.07 ms.

And the honest half of the contract: **STT alone is ~2.8 s and generation ~1.1 s.** End-to-end is ~4 s, dominated entirely by two vendor round trips. The `< 200 ms` claim is about the part this repo controls, which is why the boundary is drawn where it is.

We also go after end-to-end anyway:

- **Streaming STT** over Sarvam's realtime WebSocket, not batch upload-and-wait
- **Speculative retrieval** — retrieval fires on a partial transcript mid-utterance and refreshes only if the final transcript diverges past an edit-distance threshold. Verified live against Sarvam (partials at 832 ms, final at 2,748 ms → 1,364 ms hidden). **Measured worth: ~0.3% of end-to-end**, because the loop it hides costs 10 ms. Kept because it is free and correctness-preserving; not claimed as a headline
- **Local quantized embeddings** — ONNX int8, in-process, never an embedding API call
- **In-process vector index** — usearch HNSW, no network hop
- **Warm boot** before the readiness probe passes, so a cold first request cannot land in the p100 bucket
- **Deadline-aware degradation** — every stage reads the remaining budget and downgrades rather than overrunning

---

## What the discovery changed

Gate A ran before any pipeline code. It did not confirm assumptions; it overturned them. Full report: [`docs/discovery/report.md`](docs/discovery/report.md), reproduced by [`src/ingest/discover.py`](src/ingest/discover.py).

| Assumption | What the data actually shows |
|---|---|
| A flat passage corpus | Nested `passages{English_passages[], Translated_passages[], is_selected[]}` — one row per *query*, ~10 candidate passages each |
| No relevance labels; build a silver set | **`is_selected` is MS MARCO's own qrel.** Real Recall@5 / MRR@10 / nDCG@10, no silver standard needed |
| Chunking will be the main lever | Passages are p50 **72** tokens, max **319**. **0.000%** exceed the e5 512-token window |
| 14 independent language corpora | The 14 shards are *parallel*, aligned by `query_id`. English passages are duplicated **14×** across shards |
| Train is the corpus to use | Train is missing **Telugu** entirely; validation has all 14 languages |

Two consequences shaped everything downstream: relevance labels made real metrics possible, and short passages meant chunking was likely to be a no-op — which is exactly what the ablation went on to measure.

The HF dataset viewer is also permanently broken for this dataset (`ArrowNotImplementedError` on the nested struct), so schema discovery went through the parquet metadata directly.

---

## Chunking: 8 strategies, 13 arms, and a negative result

Full table: [`docs/chunking-ablation.md`](docs/chunking-ablation.md) · data: [`bench/chunking_results.json`](bench/chunking_results.json)

All eight required strategies are implemented behind one `Chunker` interface and measured on 49,611 real passages against 2,707 label-bearing queries.

| Strategy | Recall@5 |
|---|---:|
| **`passage_atomic`** (control) | **0.8814** |
| `fixed_512_o0` · `metadata_aware` | 0.8814 — *byte-identical output* |
| `fixed_256_o0` | 0.8810 |
| `sentence_pack_128` | 0.8792 |
| `late_chunk_96` | 0.8785 |
| `fixed_128_o25` | 0.8777 |
| `doc2query` | 0.8714 |
| `hierarchical_c64` | 0.8689 |
| `fixed_128_o0` | 0.8678 |
| `semantic_p85/p90/p95` | 0.8485–0.8508 |

**The entire chunking design space is worth 3.3 points of Recall@5, and the arm that wins is the one that does nothing.** On a corpus built as retrieval units, re-cutting can only lose context.

The three-way tie at the top is **exact, not approximate**: each arm's emitted chunk text is fingerprinted with BLAKE2b before embedding, so arms with matching fingerprints are proven to be the same retrieval system rather than assumed to be.

A ranking of 25 numbers would be over-claiming, so every gap is tested with a **paired bootstrap** (10,000 resamples, same queries per arm) across all three languages:

> **No chunking strategy beats the passage-atomic control by a statistically significant margin, in any language. Several lose by one.**

The losses sharpen as the language gets lower-resource — `hierarchical_c64` costs −0.0122 on English, −0.0432 on Hindi and **−0.1107 on Tamil**, all significant. A weaker encoder leans harder on surrounding context, so removing context costs it more.

The sharpest finding is one nobody would design for: **`sentence_pack_128` — the script-aware strategy built specifically to respect Indic sentence boundaries — is a significant loss on Tamil** (−0.0163) while being harmless on English and Hindi. The strategy most obviously motivated by this dataset is the one that measurably hurts its hardest language.

### The effect that dwarfs chunking

The 14 shards are *parallel* — same queries, same passages, only the language differs — so this isolates language and nothing else:

| Language | Recall@5 (control) |
|---|---:|
| `eng_Latn` | 0.8814 |
| `hin_Deva` | 0.6786 |
| `tam_Taml` | 0.4972 |

**A 38-point collapse on identical content**, an order of magnitude larger than anything chunking does. That is a property of `multilingual-e5-small`, not of the corpus, and it is the strongest argument for the cross-lingual English fallback in the ADR.

---

## Engineering notes — the bugs worth reading

Each of these was silent, plausible-looking, and would have invalidated a published number.

**1. The deadline mechanism was inert at the scale that needed it.**
`_dense` awaited the embedding in a thread, then called the HNSW search *synchronously* — blocking the event loop. At 49k vectors that was 1–2 ms and invisible. At 953k it is 50–155 ms, and while the loop is blocked `asyncio.wait_for` cannot fire its timeout and the "concurrent" sparse arm cannot be joined. Measured: **dense 155.7 ms against an 80 ms stage timeout that never triggered.**

| | p50 | p95 | p100 |
|---|---:|---:|---:|
| before | 54.7 | 154.2 | 181.4 ms |
| after | 29.1 | 41.3 | 52.6 ms |

**2. Retries were burning the budget.**
`RetryPolicy` defaults to 2 retries. A slow BM25 timed out at 60 ms, backed off, retried, timed out again — **p100 210.65 ms recorded for a stage with a 60 ms timeout.** Retries are for *transient* failures; a local search that takes 200 ms will take 200 ms again. Local compute now uses `NO_RETRY`; network stages keep real retries. Production p100 **214 ms FAIL → 108 ms PASS**.

**3. The relevance gate was fed the wrong score — in production, not just the eval.**
τ is calibrated on dense cosine (~0.89). The orchestrator passed the **RRF fused score** (~0.033). Nothing could ever clear the threshold; the red-team run reported a **100% false-refusal rate**. It only surfaced because the suite has benign controls — a block-rate-only report would have shown a perfect 1.000 and shipped.

**4. One global τ refused 72.7% of answerable Tamil queries.**
100% of the Tamil `description` stratum. Tamil retrieves weaker, so its cosine distribution sits lower. Per-language τ cut overall false-refusal **35.4% → 13.8%**, and Tamil's gate is now *switched off* (AUC 0.6895, below the floor) rather than pretending to discriminate.

**5. The vector index was broken and still returned sensible-looking results.**
`usearch` `ScalarKind.I8` with L2-normalized floats: self-retrieval collapsed and a whole ablation table came out plausible with one inexplicable row. A `self_retrieval_rate` guard now fails the build.

> **And the correction that matters most.** I later claimed I8 was broken *as a dtype*, citing 0.085 self-retrieval. That was measured on **random** vectors and does not transfer: real e5 embeddings are anisotropic, and I8 scores 0.990 on them. I8 is not broken — it costs ~5 points of recall. A synthetic sanity check produced a confident false positive, and the code comment now says so. Separately, I diagnosed three bugs from **curl output that Git Bash had mangled**; the server was correct throughout. Both are recorded in the commit log rather than quietly dropped.

**6. `fuse` cost 260–420 ms of a 200 ms budget.**
MMR fetched candidate vectors by *re-embedding passage text*, one forward pass each, on the critical path. The vectors were already in the index. → 0.2–0.6 ms.

**7. Cold mmap served garbage to the first user.**
An 873 MB memory-mapped index is still on disk at boot. The first live query took 412 ms, **both** retrieval arms blew their timeouts, nothing was retrieved, and the answer was correctly refused as ungrounded. Boot now pages the graph in before the readiness probe passes: **412 ms → 10.1 ms**.

## The harness

Requirement #5 asks for structure around the model, so the structure is the deliverable.

- **Typed contracts end to end** — Pydantic v2 at every boundary: `AudioChunk → Transcript → NormalizedQuery → RetrievalSet → DraftAnswer → VerifiedAnswer`. No dicts cross a stage line. This is what caught the relevance-score bug: `top_score` (RRF) and `relevance_score` (cosine) are different fields because they are different scales.
- **`Stage`** with `name`, `timeout_ms`, `retry`, `fallback`, `breaker`, `optional`. The orchestrator composes; stages know nothing about each other.
- **Async DAG** — dense ∥ sparse, joined at fusion. Both run off the event loop, which is what makes the timeouts real (see below).
- **Deadline propagation** — a `Budget` threads through every stage carrying `remaining_ms`.
- **Retries** with exponential backoff and jitter on *network* dependencies only, plus a per-dependency circuit breaker.
- **Tool-calling generator** with a declared surface: `search_corpus`, `expand_query`, `cite`, `refuse`, `ask_clarification`. The model chooses; the harness executes.
- **Structured output + repair**, capped at one attempt. An unbounded repair loop is a latency bomb.
- **Observability** — JSON-lines logs, a `request_id` on every record, and `/trace/{request_id}` returning the full stage timeline. The UI reads that endpoint.

### Deadline degradation, measured

Every stage reads the remaining budget and downgrades rather than overrunning. Measured on the 953k-vector production partition:

| budget | p100 | degraded | what it did |
|---:|---:|---|---|
| 200 ms | 46.3 ms | 0/25 | nothing — no pressure |
| 50 ms | 45.0 ms | 25/25 | `ef_search→16`, `k→10`, skip MMR |
| 25 ms | 40.6 ms | 25/25 | `sparse_only` — drop the dense arm |
| 10 ms | 0.4 ms | 25/25 | `budget_exhausted` → refuse |

Every degradation is a structured event and appears in the trace.

### Error taxonomy

| Kind | Cause | What the user gets |
|---|---|---|
| `UpstreamError` | Sarvam or the generator failed or timed out | retry with backoff; on an open breaker, the **fallback transcriber**; otherwise a plain "the service did not respond" |
| `BudgetExceeded` | the deadline ran out before retrieval could finish | refusal explaining the request was too slow, with the degradation trail in the trace |
| `GuardrailBlocked` | injection, unsafe content, or an unsupported language | a refusal that **names the category** — never a generic error |
| `ValidationError` | the generator returned JSON that failed the `Answer` schema | one repair attempt with the validation error fed back, then a safe refusal |
| `NoRelevantContext` | relevance gate below τ, or nothing retrieved | *"nothing in the corpus covers this"* + how to rephrase |

Refusal reasons carried to the UI: `off_topic`, `unsafe`, `prompt_injection`, `low_confidence_transcript`, `empty_audio`, `unsupported_language`, `ungrounded`, `not_in_retrieved_set`, `pii_detected`. Each has its own designed copy — a refusal rendered as an error toast wastes the best evidence for requirement #6.


---

## Guardrails

Input and output rails, every decision emitted as a `GuardrailEvent` visible in the UI and in `/trace/{request_id}`. Red-team set: **53 queries across 10 categories** ([`bench/redteam.jsonl`](bench/redteam.jsonl)).

Running it found three real failures, all now fixed and locked in as regression tests:

- **English-only safety patterns.** `"synthesize methamphetamine"` never matched (the pattern ended at `meth\b`), and the Hindi *"how to make a bomb at home"* was not covered at all. An English-only safety layer on an Indic-language product is not a safety layer.
- **Latin script treated as English.** French fell through to `eng_Latn` and was answered instead of refused; CJK and Cyrillic hit the same default.
- **PII leak.** `+91 98765 43210` was logged unredacted because the phone pattern tolerated separators only after the country code.

Patterns stay **verb+object** rather than bare nouns, so *"who invented the atomic bomb"* still answers — over-blocking is a failure too, and the false-refusal rate is reported alongside the block rate.

The relevance gate's τ is calibrated against a real ROC, using the dataset's own **44.92% unanswerable** queries as negatives — plausible, on-domain questions whose retrieved passages genuinely do not contain the answer, which is exactly where an ungated RAG system hallucinates. **An uncalibrated τ fails open and says so in the trace**, rather than silently applying an invented constant.

---

## Running it

```bash
pip install -e .                      # runtime; torch is NOT required
python -m src.ingest.discover         # Gate A: dataset discovery
python -m src.ingest.build_slice      # sample + dedup + parquet
python -m bench.ablate_chunking       # Gate B: chunking ablation
python -m bench.calibrate_tau         # relevance threshold + ROC
python -m bench.run_redteam           # guardrail evaluation
uvicorn src.api.app:app --port 7860
```

GPU index builds use a separate extra (`pip install -e ".[gpu]"`). On an RTX 4050 the fp32 CUDA path runs at **1,018 passages/s** against **156/s** for int8 on 16 CPU threads — 3.9 h versus 25 h for the full 14.3M-passage build. int8 has no usable CUDA kernels and crashes the provider, so GPU builds use fp32; the two agree to 0.990 mean cosine, so the vectors are interchangeable.

`SARVAM_API_KEY` and `GROQ_API_KEY` come from Space secrets. Never in code, never in the repo.

---

## Repo map

| Path | What is in it |
|---|---|
| [`docs/adr/0001-architecture.md`](docs/adr/0001-architecture.md) | every decision, with the alternatives rejected and why |
| [`docs/discovery/report.md`](docs/discovery/report.md) | Gate A discovery report |
| [`docs/chunking-ablation.md`](docs/chunking-ablation.md) | Gate B, all 13 arms |
| `src/chunking/` | 8 strategies behind one interface |
| `src/index/` | dense (HNSW), sparse (BM25), fusion, exact ground truth |
| `src/harness/` | typed contracts, `Stage`, `Budget`, orchestrator, retries, breakers |
| `src/guardrails/` | input/, output/, `thresholds.yaml` |
| `src/api/` | FastAPI, WebSocket, Sarvam streaming client, `/trace/{id}` |
| `web/` | the sunline |
| `bench/` | every number in this README traces to a script here |

Every number in this README comes from a committed script that reproduces it. Where a required approach measured worse than a simpler one, the comparison is published rather than the conclusion quietly dropped.
