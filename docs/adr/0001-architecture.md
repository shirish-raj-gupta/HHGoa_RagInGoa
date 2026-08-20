# ADR 0001 — Architecture

- **Status:** **accepted** at Gate A sign-off, 2026-08-20 (see §11 for what was decided)
- **Date:** 2026-08-20
- **Context:** HH Goa 2026 Task #2 — voice-to-grounded-answer RAG over `ai4bharat/MSMARCO-XI`
- **Evidence:** [`docs/discovery/report.md`](../discovery/report.md), reproduced by
  [`src/ingest/discover.py`](../../src/ingest/discover.py)

Every decision below cites a measured number or a verified API doc. Where a requirement in
the brief measures worse than a simpler option, that is called out rather than hidden.

---

## 1. Corpus construction

**Decision (Gate A sign-off).** Build the **full `validation` split, all 14 languages** —
~14.3M passages, ~7.3 GB index. Deduplicate English **once, globally**. **Partition the index
by language** rather than building one flat 14.3M-vector index.

**Why full validation.**
- `validation` is 14 languages including **Telugu; `train` has only 13** (report §3). Using
  validation costs nothing and avoids a silent language hole.
- Passages are attached to queries, so the ingest unit is the query with its full candidate
  set intact — that is what makes `is_selected` usable as a retrieval label.
- The 14 shards are **row-aligned** (verified element-wise, report §3), so the same query and
  the same gold labels exist in every language. Language is a clean ablation axis with zero
  confound, and at full scale it covers all 14 rather than a chosen 4.

**Why language partitioning is what makes this viable.** A flat 14.3M-vector index would put
the 200 ms claim out of reach. But the corpus is parallel: a Hindi query only needs the Hindi
partition plus English for cross-lingual fallback. That is **~1.9M vectors searched per query,
not 14.3M** — per-query latency stays near single-partition scale while coverage is complete.
The cost moves from query time to **storage and build time**, which is the right place for it.
Metadata-aware chunking (strategy #7) already required this routing layer, so it is not extra
machinery. Partitions are memory-mapped (`usearch.view()`), so resident RAM tracks the hot set
rather than the full 7.3 GB.

**Accepted costs, stated plainly — and they grew after measurement.**

| | at Gate A sign-off | measured 2026-08-20 |
|---|---:|---:|
| Vector dtype | int8, 384 B | **F16, 768 B** (§4 — int8 is broken) |
| Full-14 index | ~7.3 GB | **~12.8 GB** |
| Embed throughput | ~786 psg/s (short strings) | **156 psg/s** (real passages, 16 threads, length-bucketed) |
| Full-14 build time | ~5 h | **~25 h CPU** |

The index must be built once and then *shipped* — a Space is stateless, so it pulls from a HF
dataset repo at boot and needs `startup_duration_timeout` raised. **12.8 GB does not fit in a
`cpu-basic` Space's 16 GB RAM alongside the runtime**, so memory-mapping (`usearch.view()`) is
now load-bearing rather than an optimization, and cold-boot is minutes.

Both numbers moved the wrong way after real measurement. They are recorded here rather than
quietly absorbed, and §12 carries the consequences for the deployment decision.

**Rejected.**
- *Sampled subset (20k queries, 4 languages)* — ~400 MB and comfortable, and it was my
  recommendation. Overruled at Gate A in favour of full coverage.
- *`train` split* — 49 GB, no Telugu, no upside; same kind of labels.
- *English only* — simpler, but this is an **Indic** voice task. Rejected on relevance.
- *Flat single index over all 14 languages* — the version of "full validation" that genuinely
  cannot meet the budget. Partitioning is the difference.
- *Naive ingest of all shards* — would index the identical English corpus **14 times**
  (13.7M rows → 950k unique, **93.1% waste**, report §7).

## 2. Ingest and cleaning

**Decision.** Stream via pyarrow **column projection**, NFC-normalize, strip control chars,
drop passages over **2× the p99 token length** or failing a repetition-ratio check, dedup with
exact hash then MinHash-LSH (Jaccard ≥ 0.8), persist Parquet with stable
`passage_id`/`doc_id`/`lang`/`script`/`token_len`/`dedup_cluster`.

**Why.** MT degeneration is measured at **0.18%** of Indic passages (max 21,390 chars against
a p99 of 708) — real, cheap to filter, and the only genuine long-text in the dataset.
Within-shard redundancy is **3.4%** (2.74% exact + 0.64% near).

**Rejected.**
- *`datasets.load_dataset(streaming=True)`* — the brief suggests it; it is a **trap here**.
  Each shard is a **single row group** of ~98k rows / 1.16 GB, so the first `next()` pulls the
  whole shard. It buys nothing. Column projection is the real lever: labels cost **0.1 MB**
  per language, passage text costs 445 MB (report §8).
- *Aggressive MinHash on everything* — 0.64% near-dup does not justify a heavy pass. Keep it,
  report it, don't oversell it.

## 3. Embeddings

**Decision.** `intfloat/multilingual-e5-small` (384-d), exported to **ONNX, int8-quantized,
in-process**, with the `query:` / `passage:` prefix convention enforced in the typed layer so
it cannot be forgotten. Config-switchable for the ablation.

**Why.** 384-d keeps the index at 384 B/vector int8. Covers all four scripts. **Zero
passages exceed its 512-token window** (max observed 319 EN / p99 219 HI), so truncation is
a non-issue — measured, not assumed.

**Rejected.**
- *`multilingual-e5-base/large`* — 768/1024-d for 2–2.7× the index and latency. The passages
  are ~72 tokens; there is little for a bigger model to exploit. Revisit only if Gate B recall
  is poor.
- *Any hosted embedding API* — a network hop on the critical path makes the 200 ms claim
  impossible. Non-negotiable.
- *BGE-M3* — strong multilingually, but 1024-d and far heavier; wrong point on the curve.

## 4. Index

**Decision.** **`usearch`** HNSW in-process, **F16**, cosine. Tune `M` ∈ {16, 32} and
`ef_search`; publish the recall-vs-latency curve and mark the chosen operating point.
Sparse: **`bm25s`** with per-script tokenization. Dense and sparse run **concurrently**,
fused with **RRF**, then MMR (λ ≈ 0.7).

**Amended 2026-08-20 — int8 vector storage is unusable, and this was measured, not assumed.**
The original decision said int8 (384 B/vector). It is wrong. `usearch`'s `ScalarKind.I8`
expects values in int8 range and does **not** rescale L2-normalized unit vectors, so the
distance function — and with it the HNSW graph — is destroyed. Self-retrieval (search a
vector against an index that contains it; correct answer is rank 0) on 20k random unit
vectors:

| dtype | ef_search=64 | ef_search=256 | bytes/vector |
|---|---:|---:|---:|
| F32 | 99.0% | 100.0% | 1536 |
| **F16 (chosen)** | **100.0%** | **99.5%** | **768** |
| I8 | 9.0% | 1.0% | 384 |

Pre-scaling by 127 does not fix it (9.5%); casting to `int8` is worse (6.5%). The failure is
**not uniform**, which is what made it dangerous: on real embeddings it produced a
plausible-looking ablation table with a single inexplicable row (`fixed_128_o0` at
R@5 **0.474** where its neighbours scored **0.864**), because graph quality became a lottery
per build instead of consistently bad. A uniformly bad index would have been caught
immediately; this one nearly shipped.

`DensePartition.self_retrieval_rate()` now guards against the whole class of bug, and the
ablation records it per arm so a broken index shows up in the table instead of hiding in it.

**This is a different int8 from the ONNX model quantization**, which is validated and kept
(0.990 mean / 0.987 min cosine agreement with fp32 across five scripts, §3).

**Cost of the fix:** vectors double from 384 to 768 B. Full-14 index goes from ~7.3 GB to
**~12.8 GB** (10.98 GB vectors + 1.83 GB graph at `M=16`). See §12 — this makes the Space
sizing materially tighter and is called out there rather than buried here.

**Why.** In-process, no network hop. At ~778k vectors an HNSW query is single-digit ms, which
is what leaves room in the budget for guardrails. Indic morphology hurts pure lexical
matching, so sparse alone is insufficient; English queries with rare entities are exactly
where dense alone fails. Hence both.

**Rejected.**
- *Hosted vector DB (Pinecone/Qdrant Cloud/Weaviate)* — network hop; disqualifying.
- *FAISS* — fine, but heavier wheel and clumsier int8 story on CPU. `usearch` is config-swappable if it disappoints.
- *Flat/brute-force* — exact, but ~778k × 384-d scan blows the budget.
- *Cross-encoder reranker on the critical path* — 20–80 ms for a quality gain the ablation has
  not yet justified. **Excluded by default**, implemented behind the deadline check, and
  enabled only if Gate C shows it fits. This is the single easiest way to lose the 200 ms claim.

## 5. Chunking — the brief's premise does not survive contact with the data

**Decision.** Implement all 8 strategies behind one `Chunker` interface as required, but
**default to passage-atomic** unless Gate B overturns it, and publish the degenerate cases as
a measured finding rather than padding the table with duplicate rows.

**Why.** Report §5: English passages are **p50 72 / p99 176 / max 319 tokens**. **0.000%**
exceed the 512-token window; **92.2%** are under 128.

| # | Strategy | Status on this data |
|---|---|---|
| 1 | Fixed-size + overlap | **256 and 512 are provable no-ops** — every passage yields one chunk identical to passage-atomic. Only 128 splits anything (~7.8% of EN passages), and it can only destroy context. Report all three; expect 256/512 to tie exactly with #4. |
| 2 | Script-aware sentence packing | Real. Devanagari `।`, Tamil/Bengali punctuation, grapheme-cluster safety. Matters most for the 0.18% long tail. |
| 3 | Semantic breakpoint | Real but weak — a 72-token passage has ~3 sentences to find a breakpoint in. |
| 4 | **Passage-atomic** | **Expected winner.** The control arm that the dataset is built for. |
| 5 | Hierarchical parent–child | Largely degenerate: parent ≈ child ≈ passage. |
| 6 | **Late chunking** | Real and promising — document-context-aware vectors, and MS MARCO is pronoun/entity-heavy. |
| 7 | **Metadata-aware** | Real and useful — `lang`/`script` filtering with cross-lingual fallback is a genuine lever given the parallel corpus. |
| 8 | **Query-aware expansion (doc2query-lite)** | Real recall win, paid for at index-build time, not query time. |

The strategies that actually vary here change **the vector or the retrievable surface**
(2, 6, 7, 8), not the split (1, 3, 5). The ablation should be honest about that.

**Rejected.** *Picking a chunk size up front.* Chunk size must be **token-based using the
model tokenizer**, never char-based: Indic scripts cost **1.8–3.2× the UTF-8 bytes** but only
**1.21× the tokens** (report §5). A char-based size would silently make Tamil chunks a third
the semantic size of English ones — a per-language bug that would look like a model failure.

## 6. Speech-to-text — verified against live docs, not recalled

**Decision.** Sarvam **realtime streaming WebSocket**:
`wss://api.sarvam.ai/speech-to-text-realtime/ws`, model **`saaras:v3-realtime`**, auth via
`API-SUBSCRIPTION-KEY` header, `sample_rate: 16000`, `encoding: linear16`,
`endpointing: vad`, `stream_type: fast`.

**Why this specific endpoint.** It emits **`transcript.partial`** events with a `text` field
*before* `transcript.final`. That is the entire basis of speculative retrieval (§8) — without
verified partials, that trick is fiction. Also available: `vad.speech_start` /
`vad.speech_end` for endpointing, and `mode: codemix` for Hinglish.

**Language coverage — checked, 14/14.** The dataset uses FLORES-200 codes (`hin_Deva`);
Sarvam uses BCP-47 (`hi-IN`). A mapping table is required, and every one of our 14 languages
is supported: `as-IN bn-IN gu-IN hi-IN kn-IN ml-IN mr-IN ne-IN or-IN pa-IN sa-IN ta-IN te-IN
ur-IN`. No coverage gap.

**Rejected.**
- *Batch `POST /speech-to-text`* — upload-and-wait, no partials, no speculation. This is what
  a naive implementation uses and it is why naive implementations cannot beat the budget.
- *`saarika:v2.5`* — legacy, deprecation announced.
- *`mode: translate`* — tempting (Indic speech → English text directly, letting us query only
  the English index), but it discards the user's language, which the answer must match. Keep
  `transcribe`; consider `translate` as a *fallback* if in-language retrieval underperforms.

## 7. Generation

**Decision.** **`claude-haiku-4-5`** as the default generator on the critical path, with
`claude-sonnet-5` and `claude-opus-5` config-switchable and **all three benchmarked for TTFT**
in Gate C. Streaming on. Structured output via `output_config.format` + `client.messages.parse()`
against the `Answer` schema. Tool surface declared with `strict: true`. Prompt caching on the
stable system prompt + tool schema.

**Why.** TTFT is a *published, graded* number in this task. Haiku 4.5 is the fastest and
cheapest ($1/$5 per MTok, 200K context) and the generation job here is narrow — summarize 4
short retrieved passages with citations, in-language. That is not a task that needs frontier
reasoning. Prompt caching matters more than model choice for TTFT, since the system prompt and
tool schema are byte-stable across every request.

**Decided at Gate A:** Haiku 4.5 default, all three benchmarked for TTFT. The model choice
becomes a published number rather than an assertion.

**Rejected.**
- *Local quantized generator (Qwen/Llama on CPU)* — no network hop, but CPU-only generation on
  a Space is far slower than a hosted TTFT, and quality on 14 Indic languages is poor.
- *Sarvam's LLM for generation* — plausible and Indic-strong; keeping it as a fallback
  provider for the circuit breaker rather than the default, so a Sarvam outage cannot take out
  both STT *and* generation at once.
- *Non-streaming* — forfeits TTFT entirely.

## 8. The latency contract

**Decision.** Publish the full boundary table, and claim **only** `CORE_RAG_LOOP_MS p100 < 200 ms`.

```
T0  mic stop / audio bytes complete
T1  Sarvam transcript returned            → STT_MS      (network-bound, reported separately)
T2  transcript normalized + lang-ID     ─┐
T3  query embedded (local ONNX int8)     │
T4  BM25 + dense retrieved in parallel   ├─ CORE_RAG_LOOP_MS  ← the < 200 ms claim
T5  RRF fused + MMR diversified          │
T6  input + grounding guardrails passed ─┘
T7  first answer token                    → TTFT_MS
T8  answer complete + citations verified  → E2E_MS
```

**Why.** A hosted STT round trip is 300 ms–1 s and a hosted first token is 200–800 ms. **A
naive voice-in/answer-out E2E under 200 ms is not physically available**, and any submission
claiming it is measuring something narrower without saying so. Stating the boundary plainly
reads as rigor; hiding it reads as a dodge. `STT_MS`, `TTFT_MS`, and `E2E_MS` are published in
the same table, at P50/P70/P90/P95/**P100**, cold and warm separately.

**Levers we actually pull at E2E** (so the honest boundary is not an excuse):
- **Speculative retrieval** — fire retrieval on `transcript.partial` at ~60% of expected
  utterance; refresh if the final transcript diverges past an edit-distance threshold. This
  hides the *entire* core loop inside STT time. Viable specifically because §6 verified that
  partials exist.
- **Warm boot** — ONNX session, tokenizers, BM25, HNSW, thread pools, and a dummy inference at
  startup, before the readiness probe passes.
- **Deadline-aware degradation** — a `Budget` carrying `remaining_ms` through every stage;
  stages downgrade (drop `ef_search`, cut `k`, skip rerank, sparse-only) rather than overrun.
  Every degradation logged as a structured event. This is what makes the number a *guarantee*
  rather than an average.

**Rejected.** *Claiming E2E < 200 ms.* It would be false, and the judges are engineers.

## 9. Evaluation — real metrics, no silver standard

**Decision.** Compute **true Recall@k / MRR@10 / nDCG@10** from `passages.is_selected`. Build
the ≥200-query bench set stratified by language × `query_type` × answerability. Calibrate the
relevance threshold **τ** on a proper ROC using the natural unanswerable set.

**Why — this is the most consequential discovery.** The brief allowed for a silver-standard
eval set if labels were missing. **Labels are present and clean** (report §4):
**55.03%** of queries carry ≥1 positive → **53,898 labeled queries per language**, and
`"No Answer Present."` ⟺ zero positives with **zero exceptions** across 97,941 rows.

That also hands us **43,991 real unanswerable queries** — genuine queries whose genuine
candidate passages do not contain the answer. So τ gets calibrated against real negatives with
a published ROC, instead of the hardcoded `0.5` the brief rightly calls a guess. The
answerable/unanswerable split is ~55/45, so the ROC is well-conditioned.

**Rejected.** *Building a silver-standard eval set* — strictly worse than the real labels, and
would have quietly capped the credibility of every retrieval number in the submission.

## 10. Stack

Python 3.11 · FastAPI · Pydantic v2 · asyncio · Docker Space (not Gradio — §10 of the brief
needs frontend control). Typed contracts at every stage boundary; no dicts crossing stage
lines.

---

## Consequences

**Good.** Real retrieval metrics from day one. A per-language ablation with no confound.
A latency claim with a mechanism behind it. An index that fits in RAM.

**Accepted costs.** 3 of 8 chunking strategies will report as degenerate — we publish that as
a finding. Corpus is a sampled subset, so absolute recall is not comparable to full-MS-MARCO
literature (relative comparisons across our arms remain valid). 10 of 14 languages are indexed
but not benchmarked in depth.

**Risks.**
- Sarvam realtime partial-transcript timing is documented but not yet measured against a real
  mic. **Mitigation:** measure at Gate C; speculative retrieval degrades to sequential if
  partials arrive too late to help.
- If Gate B shows passage-atomic winning outright, the chunking requirement is satisfied by a
  *negative* result. The brief explicitly endorses this ("ship the simpler one and publish the
  comparison"), but it must be presented as measurement, not as an excuse.

## 11. Gate A sign-off — what was decided

| Question | Decision | Note |
|---|---|---|
| Corpus scope | **Full validation, all 14 languages** | Overrides the sampled-subset recommendation. Viability rests on language partitioning (§1). |
| Generator default | **`claude-haiku-4-5`**, all three benchmarked | §7 |
| Repo layout | **`git init` here**, parent `.gitignore` updated | Standalone repo, clean history |

## 12. Open — deployment target (blocks Gate D only)

Verified via `HfApi().whoami()` on account `srg101`: **`isPro: False`, `canPay: False`, no
orgs.** On a free account, **both Gradio and Docker Spaces require a paid plan**; only Static
Spaces and up to 2 **ZeroGPU** Spaces are free, and **ZeroGPU is Gradio-only — it does not
support Docker**. The brief requires a Docker Space (§10, for frontend control).

Options, in order of preference:

1. **Deploy under the teammate account the brief already names** (`ansh123456789/ragingoa`) if
   it carries PRO — zero cost, matches the brief verbatim.
2. **PRO on `srg101`** (~$9/mo) → Docker Space at `cpu-basic` (2 vCPU / 16 GB). Note this is a
   thin machine for a 7.3 GB partitioned index; Gate C measures whether the budget survives it.
3. **ZeroGPU Gradio Space** — free, and the GPU would speed embedding, but Gradio instead of
   Docker means rebuilding the §10 interface as custom Gradio components. Feasible, more work,
   less frontend control.
4. **Static Space frontend + API hosted elsewhere** — preserves the designed UI, splits the
   deployment.

Gates B and C are deployment-agnostic and proceed regardless.

### Hardware of record (for the benchmark)

Local build/bench machine: **Intel i7-14650HX, 16 physical / 24 logical cores, 15.7 GB RAM,
NVIDIA RTX 4050 Laptop (4 GB), Windows 11.** Index build uses the GPU; all latency numbers in
Gate C are reported **CPU-only** with thread count stated, since the Space has no GPU.
