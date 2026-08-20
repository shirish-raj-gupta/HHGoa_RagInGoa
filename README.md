# RAG in Goa

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

> **Gate C in progress.** The full ≥200-query benchmark has not been run yet, so no latency table is published here. Nothing goes in this section until `bench/run.py` produces it. An early integration run on a 3,000-passage index measured the core loop at 6.4–10.6 ms, but that is not the benchmark and is not a claim.

We also go after end-to-end anyway:

- **Streaming STT** over Sarvam's realtime WebSocket, not batch upload-and-wait
- **Speculative retrieval** — retrieval fires on a partial transcript mid-utterance and refreshes only if the final transcript diverges past an edit-distance threshold, hiding the core loop inside the time the user is still speaking
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

**The entire chunking design space is worth 3.3 points of Recall@5, and the arm that wins is the one that does nothing.** On a corpus built as retrieval units, re-cutting can only lose context. That is a result, and it is reported as one rather than buried under a strategy picked for looking sophisticated.

The three-way tie at the top is **exact, not approximate**: each arm's emitted chunk text is fingerprinted with BLAKE2b before embedding, and arms with matching fingerprints are proven to be the same retrieval system rather than assumed to be.

---

## Engineering notes — three bugs worth reading

Each of these was silent, plausible-looking, and would have invalidated a published number.

**1. The vector index was broken and still returned sensible-looking results.**
`usearch` `ScalarKind.I8` expects int8-range values, not L2-normalized floats, and does not scale them. Self-retrieval — searching for a vector in an index that *contains* it — was the tell:

| dtype | ef_search=64 | ef_search=256 |
|---|---:|---:|
| F32 | 99.0% | 100.0% |
| F16 | 100.0% | 99.5% |
| I8 | **9.0%** | **1.0%** |

It produced a complete ablation table with one inexplicable row. A `self_retrieval_rate` guard now fails the build instead.

**2. The ablation was measuring HNSW luck, not chunking.**
Even on F16, real e5 embeddings cluster far more tightly than the random vectors a sanity check uses, and the default graph parameters scored 0.84 self-retrieval — dragging one arm to R@5 0.316. The ablation now runs on **exact** brute-force search, so a difference in the table is a difference in chunking. HNSW is tuned separately against exact as ground truth.

**3. `fuse` cost 260–420 ms of a 200 ms budget.**
MMR was fetching candidate vectors by *re-embedding the passage text*, one forward pass per candidate, on the critical path. The vectors were already in the index.

| | fuse | core_rag_loop |
|---|---:|---:|
| before | 260–422 ms | 282–432 ms — over budget |
| after | 0.2–0.6 ms | 6.4–10.6 ms |

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
