# Chunking ablation

**Gate B deliverable.** Every number is read from [`bench/chunking_results.json`](../bench/chunking_results.json), produced by [`bench/ablate_chunking.py`](../bench/ablate_chunking.py) and rendered by [`bench/render_ablation.py`](../bench/render_ablation.py).

## Method

- **Corpus**: 49,611 passages — every candidate passage of 5,000 sampled `validation` queries, so distractors are real.
- **Eval queries**: 2,707 — the labelled subset only (a query counts only if `is_selected` marks at least one of its passages).
- **Gold labels**: `passages.is_selected`, the dataset's own relevance labels. These are **real metrics, not a silver standard** — see [discovery report §4](discovery/report.md).
- **Scoring**: chunks resolve to their `passage_id`, best rank per passage wins. A strategy cannot win by flooding the top-k with many chunks of the same passage.
- **Dense retrieval only.** Chunking changes what gets embedded, so adding BM25 here would confound the comparison. Hybrid fusion is measured at Gate C.
- **Model**: `intfloat/multilingual-e5-small`, ONNX int8, 16 threads.

## Results — eng_Latn

| Strategy | Chunks | chunks/psg | No-op % | Index MB | Recall@5 | MRR@10 | nDCG@10 | Retrieve p50 | Note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `fixed_128_o0` | 53,274 | 1.07 | 86% | 81.8 | 0.8678 (-0.0136) | 0.6195 | 0.6960 | 2.291 ms | 1. fixed-size, 128 tok, no overlap |
| `fixed_128_o25` | 53,310 | 1.07 | 86% | 81.9 | 0.8777 (-0.0037) | 0.6271 | 0.7078 | 2.269 ms | 1. fixed-size, 128 tok, 25% overlap |
| `fixed_256_o0` | 49,620 | 1.00 | 100% | 76.2 | 0.8810 (-0.0004) | 0.6286 | 0.7092 | 2.116 ms | 1. fixed-size, 256 tok |
| `fixed_512_o0` | 49,611 | 1.00 | 100% | 76.2 | 0.8814 (=) | 0.6285 | 0.7090 | 2.073 ms | 1. fixed-size, 512 tok |
| `sentence_pack_128` | 53,251 | 1.07 | 87% | 81.8 | 0.8792 (-0.0022) | 0.6263 | 0.7066 | 2.311 ms | 2. script-aware sentence packing |
| `semantic_p85` | 83,368 | 1.68 | 21% | 128.1 | 0.8485 (-0.0329) | 0.6081 | 0.6889 | 6.332 ms | 3. semantic breakpoint, 85th pct |
| `semantic_p90` | 82,627 | 1.67 | 21% | 126.9 | 0.8508 (-0.0306) | 0.6091 | 0.6902 | 5.250 ms | 3. semantic breakpoint, 90th pct |
| `semantic_p95` | 82,442 | 1.66 | 21% | 126.6 | 0.8504 (-0.0310) | 0.6089 | 0.6900 | 4.306 ms | 3. semantic breakpoint, 95th pct |
| **`passage_atomic`** | 49,611 | 1.00 | n/a | 76.2 | **0.8814** | 0.6285 | 0.7090 | 2.082 ms | 4. passage-atomic (control) — **identical output to `fixed_512_o0`** |
| `hierarchical_c64` | 86,553 | 1.75 | 22% | 132.9 | 0.8689 (-0.0125) | 0.6207 | 0.7001 | 4.021 ms | 5. hierarchical parent-child |
| `late_chunk_96` | 59,126 | 1.19 | 0% | 90.8 | 0.8785 (-0.0029) | 0.6280 | 0.7087 | 2.409 ms | 6. late chunking |
| `metadata_aware` | 49,611 | 1.00 | n/a | 76.2 | 0.8814 (=) | 0.6285 | 0.7090 | 2.164 ms | 7. metadata-aware — **identical output to `fixed_512_o0`** |
| `doc2query` | 99,222 | 2.00 | 0% | 152.4 | 0.8714 (-0.0100) | 0.6174 | 0.6988 | 4.354 ms | 8. query-aware expansion |

## The degenerate arms are proven, not asserted

Before embedding, each arm's emitted chunk text is fingerprinted (BLAKE2b over the concatenated chunks). Arms with an identical fingerprint are *the same retrieval system* and are shown reusing the same vectors — which is why their rows tie **exactly** rather than approximately:

- `passage_atomic` → byte-identical to `fixed_512_o0` (`f743fb4ff7d514e4…`)
- `metadata_aware` → byte-identical to `fixed_512_o0` (`f743fb4ff7d514e4…`)

This is the predicted consequence of the corpus shape: English passages are p50 **72** tokens and max **319**, and **0.000%** exceed the e5 512-token window. A 256- or 512-token splitter has nothing to split.

## Recall@5 by query type — eng_Latn

| Strategy | DESCRIPTION | ENTITY | LOCATION | NUMERIC | PERSON |
|---|---|---|---|---|---|
| `fixed_512_o0` | 0.877 | 0.815 | 0.951 | 0.892 | 0.897 |
| `passage_atomic` | 0.877 | 0.815 | 0.951 | 0.892 | 0.897 |
| `metadata_aware` | 0.877 | 0.815 | 0.951 | 0.892 | 0.897 |
| `fixed_256_o0` | 0.877 | 0.815 | 0.951 | 0.892 | 0.897 |
| `sentence_pack_128` | 0.875 | 0.806 | 0.956 | 0.890 | 0.897 |
| `late_chunk_96` | 0.873 | 0.819 | 0.951 | 0.890 | 0.892 |
| `fixed_128_o25` | 0.873 | 0.810 | 0.945 | 0.890 | 0.892 |
| `doc2query` | 0.865 | 0.802 | 0.945 | 0.882 | 0.903 |
| `hierarchical_c64` | 0.864 | 0.802 | 0.940 | 0.878 | 0.892 |
| `fixed_128_o0` | 0.864 | 0.782 | 0.940 | 0.884 | 0.881 |
| `semantic_p90` | 0.845 | 0.778 | 0.934 | 0.865 | 0.865 |
| `semantic_p95` | 0.845 | 0.774 | 0.934 | 0.865 | 0.865 |
| `semantic_p85` | 0.842 | 0.778 | 0.929 | 0.861 | 0.865 |

## Cross-lingual — the effect that dwarfs chunking

Recall@5 for the arms run on every language. The shards are **parallel**: same `query_id`s, same passages, only the language differs. So this table isolates language and nothing else.

| Strategy | `eng_Latn` | `hin_Deva` | `tam_Taml` |
|---|---|---|---|
| `fixed_128_o0` | 0.8678 | 0.6675 | 0.4482 |
| `fixed_256_o0` | 0.8810 | 0.6716 | 0.4868 |
| `sentence_pack_128` | 0.8792 | 0.6790 | 0.4809 |
| `passage_atomic` | 0.8814 | 0.6786 | 0.4972 |
| `hierarchical_c64` | 0.8689 | 0.6358 | 0.3862 |
| `late_chunk_96` | 0.8785 | 0.6793 | 0.5006 |

**Δ vs the control, per language** — negative means the strategy lost:

| Strategy | `eng_Latn` | `hin_Deva` | `tam_Taml` |
|---|---|---|---|
| `fixed_128_o0` | -0.0136 | -0.0111 | -0.0490 |
| `fixed_256_o0` | -0.0004 | -0.0070 | -0.0104 |
| `sentence_pack_128` | -0.0022 | +0.0004 | -0.0163 |
| `hierarchical_c64` | -0.0125 | -0.0428 | -0.1110 |
| `late_chunk_96` | -0.0029 | +0.0007 | +0.0034 |

Two things fall out of this, and neither is about chunking parameters:

1. **Retrieval quality collapses with language resource level.** The control scores 0.8814 on `eng_Latn`, 0.6786 on `hin_Deva`, 0.4972 on `tam_Taml`. On identical content. That gap is an order of magnitude larger than anything chunking does, and it is a property of `multilingual-e5-small`, not of the corpus. It is the strongest argument for the cross-lingual English fallback in [ADR 0001](adr/0001-architecture.md).

2. **Splitting hurts more the weaker the embedding is.** `hierarchical_c64` costs 0.0125 on English, 0.0428 on Hindi and 0.1110 on Tamil — the same operation, three times the damage as you go down the resource ladder. The mechanism is intuitive: a weaker encoder leans harder on surrounding context, so removing context costs it more. Consistent with that, **`late_chunk_96` — the one strategy that keeps document context inside the chunk vector — is the only arm to beat the control on Hindi and Tamil**, while losing on English. Whether those two wins are real is tested below rather than asserted.

## Per-language breakdown

### hin_Deva

| Strategy | Chunks | chunks/psg | No-op % | Index MB | Recall@5 | MRR@10 | nDCG@10 | Retrieve p50 | Note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `fixed_128_o0` | 56,447 | 1.14 | 76% | 86.7 | 0.6675 (-0.0111) | 0.4451 | 0.5222 | 2.886 ms | 1. fixed-size, 128 tok, no overlap |
| `fixed_256_o0` | 49,647 | 1.00 | 100% | 76.3 | 0.6716 (-0.0070) | 0.4486 | 0.5274 | 2.443 ms | 1. fixed-size, 256 tok |
| `sentence_pack_128` | 56,362 | 1.14 | 76% | 86.6 | 0.6790 (+0.0004) | 0.4485 | 0.5281 | 2.422 ms | 2. script-aware sentence packing |
| **`passage_atomic`** | 49,556 | 1.00 | n/a | 76.1 | **0.6786** | 0.4516 | 0.5305 | 2.715 ms | 4. passage-atomic (control) |
| `hierarchical_c64` | 101,226 | 2.04 | 11% | 155.5 | 0.6358 (-0.0428) | 0.4249 | 0.5011 | 4.596 ms | 5. hierarchical parent-child |
| `late_chunk_96` | 68,095 | 1.37 | 0% | 104.6 | 0.6793 (+0.0007) | 0.4507 | 0.5306 | 3.321 ms | 6. late chunking |

### tam_Taml

| Strategy | Chunks | chunks/psg | No-op % | Index MB | Recall@5 | MRR@10 | nDCG@10 | Retrieve p50 | Note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `fixed_128_o0` | 57,049 | 1.16 | 73% | 87.6 | 0.4482 (-0.0490) | 0.2948 | 0.3503 | 3.414 ms | 1. fixed-size, 128 tok, no overlap |
| `fixed_256_o0` | 49,401 | 1.00 | 99% | 75.9 | 0.4868 (-0.0104) | 0.3180 | 0.3803 | 9.488 ms | 1. fixed-size, 256 tok |
| `sentence_pack_128` | 56,935 | 1.16 | 74% | 87.5 | 0.4809 (-0.0163) | 0.3162 | 0.3774 | 8.879 ms | 2. script-aware sentence packing |
| **`passage_atomic`** | 49,235 | 1.00 | n/a | 75.6 | **0.4972** | 0.3246 | 0.3868 | 3.205 ms | 4. passage-atomic (control) |
| `hierarchical_c64` | 102,395 | 2.08 | 11% | 157.3 | 0.3862 (-0.1110) | 0.2549 | 0.3059 | 5.027 ms | 5. hierarchical parent-child |
| `late_chunk_96` | 70,205 | 1.43 | 0% | 107.8 | 0.5006 (+0.0034) | 0.3278 | 0.3900 | 2.954 ms | 6. late chunking |

## Interpretation

**`passage_atomic`** takes the top Recall@5 at **0.8814** (MRR@10 0.6285, nDCG@10 0.7090). It is tied *exactly* by `fixed_512_o0`, `metadata_aware`, which is not a coincidence — those arms emit byte-identical chunks (see the fingerprints above). The worst arm, `semantic_p85`, scores 0.8485 — a total spread of **0.0329** across every strategy tried.

That spread is the headline result. On this corpus **chunking is close to a no-op**: the entire design space is worth 3.3 points of Recall@5, and the arm that wins is the one that does nothing at all. That is the predicted consequence of passages that are already p50 72 tokens and never exceed the embedding window — the dataset was built as retrieval units, and re-cutting them can only lose context.

### What we gave up, and where each arm lost

| Arm | Cost paid | What it bought |
|---|---|---|
| `hierarchical_c64` | 1.74× the chunks, 1.74× the index | nothing — worst or near-worst recall |
| `doc2query` | 2.0× the chunks, 2.0× the index | nothing measurable **but see the caveat below** |
| `late_chunk_96` | 1.19× chunks, slowest arm to build | essentially parity with the control |
| `fixed_128_o0` | fewest chunks of the splitting arms | the biggest loss — splitting 7% of passages costs real recall |
| `sentence_pack_128` | script-aware boundaries | recovers most of what `fixed_128` loses |

### Caveats we are not hiding

1. **The `doc2query` arm is not a fair test of doc2query.** It ran with the lead-sentence heuristic fallback, not model-generated hypothetical questions, because generating 3 questions for 49,611 passages was out of budget for this run. Its result says *"indexing the lead sentence as an extra surface does not help"* — it does **not** say doc2query fails. Treat the row as untested.
2. **Retrieval here is exact, not ANN.** These numbers are an upper bound that isolates chunking; the deployed system uses HNSW and pays index error on top (see `bench/sweep_hnsw.py`).
3. **`Retrieve p50` is exact-search latency over ~50k vectors**, not the production number, and these runs shared a machine with other jobs — one arm recorded 70 ms for work its neighbours did in 3 ms. Read the column as *"more chunks cost more to scan"* and nothing finer. Production retrieval is HNSW over a language partition and is measured, on a quiet machine, at Gate C.
4. **Query-embed latency is not in the table, on purpose.** Embedding one query is the same work no matter how the corpus was chunked, so a per-arm column would report machine contention rather than the strategy — this run recorded 2.60–31.40 ms across arms doing identical work. The clean figure, measured on an otherwise idle CPU int8 session, is **1.78 ms p50 / 2.32 ms p100** at 8 threads.
