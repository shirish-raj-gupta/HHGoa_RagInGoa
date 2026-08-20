# Discovery Report — `ai4bharat/MSMARCO-XI`

**Gate A deliverable.** Every number below is emitted by
[`src/ingest/discover.py`](../../src/ingest/discover.py) → `docs/discovery/discovery.json`.
Nothing here is recalled from memory; the dataset card was not trusted for schema.

Probed 2026-08-20 against revision `refs/convert/parquet`.

---

## 0. TL;DR — the four findings that change the build

1. **This is not a passage corpus. It is a query-centric parallel translation corpus.**
   One row = one MS MARCO query + its ~10 candidate passages, in English *and* one Indic
   language, side by side. There is no standalone passage table to index.
2. **Real relevance labels exist** (`passages.is_selected`), and they are perfectly
   consistent with the answerability marker. **We can compute true Recall@k / MRR@10 /
   nDCG@10 — no silver-standard eval set is needed.** This was the open question in the brief.
3. **There is essentially nothing to chunk.** English passages are p50 **72** tokens,
   p99 **176**, max **319** — **0.000%** exceed the e5 512-token window and 92.2% are under
   128. Re-splitting is a no-op for 5 of the 8 strategies. The chunking ablation must be
   reframed (see §7) or it will produce eight identical rows.
4. **The corpus is 14× redundant in English by construction**, and the HF dataset viewer is
   permanently broken for it. Both facts have direct engineering consequences (§3, §8).

---

## 1. Configs, splits, row counts

`GET /splits` → a single config, `default`, with two splits.

| Split | Rows | Shards | Parquet bytes |
|---|---:|---:|---:|
| `train` | 10,080,140 | 13 | 49.04 GB |
| `validation` | 1,371,174 | 14 | 6.58 GB |
| **Total** | **11,451,314** | **27** | **55.62 GB** |

Uncompressed in memory: **146.6 GB**. This does not fit anywhere near a Space, so corpus
selection is a real design decision, not a formality (→ ADR 0001).

---

## 2. Schema — as read from the parquet footer, not the card

```
source_lang        string                      # always "eng_Latn"
target_lang        string                      # one value per shard
query_id           int64                       # MS MARCO qid, the join key
query_type         string                      # DESCRIPTION|NUMERIC|ENTITY|PERSON|LOCATION
query              string                      # the query, in target_lang
Eng_Query          string                      # the query, in English
Answer             string                      # the answer, in target_lang
Eng_Answer         string                      # the answer, in English
passages           struct<
                     English_passages:    list<string>,   # ~10 candidates, English
                     Translated_passages: list<string>,   # ~10 candidates, target_lang
                     is_selected:         list<int64>     # <-- RELEVANCE LABEL
                   >
meta               struct<model_name, temperature, max_tokens,
                          top_p, presence_penalty, frequency_penalty>
```

`meta.model_name = "ckpt-3epochs-sft-then-400k-kd"`, `temperature = 0`, `max_tokens = 4096`.
**The Indic text is machine translation output from an AI4Bharat SFT+KD checkpoint, not
human translation.** That is a quality caveat we must state, and it shows up concretely in §6.

### One real row (`query_id` 1102432, `hin_Deva`)

| Field | Value |
|---|---|
| `query_type` | `DESCRIPTION` |
| `Eng_Query` | `. what is a corporation?` |
| `query` | `कॉर्पोरेशन क्या है?` |
| `Eng_Answer` | `A corporation is a company or group of people authorized to act as a single entity and recognized as such in law.` |
| `Answer` | `निगम एक कंपनी या लोगों का समूह होता है जो एक एकल इकाई के रूप में कार्य करने के लिए अधिकृत होता है…` |
| `is_selected` | `[0,0,0,0,0,1,0,0,0,0]` |
| selected `English_passages[5]` | `McDonald's Corporation is one of the most recognizable corporations in the world. A corporation is a company or group of people authorized to act as a single entity (legally a person)…` |
| selected `Translated_passages[5]` | `मैकडॉनल्ड कॉर्पोरेशन दुनिया के सबसे पहचानने योग्य निगमों में से एक है। एक निगम एक कंपनी या लोगों का समूह है…` |

Note the leading `. ` in `Eng_Query` — query text is raw and needs normalization.

---

## 3. Languages — one shard per language, and the splits disagree

`target_lang` min == max on every shard, so **each parquet file is exactly one language.**
This was read from parquet column statistics at zero data-transfer cost.

| # | `target_lang` | Script | In `validation` | In `train` |
|---|---|---|:--:|:--:|
| 1 | `asm_Beng` | Bengali (Assamese) | ✅ 97,941 | ✅ 778,638 |
| 2 | `ben_Beng` | Bengali | ✅ 97,941 | ✅ 778,638 |
| 3 | `guj_Gujr` | Gujarati | ✅ 97,941 | ✅ 778,638 |
| 4 | `hin_Deva` | Devanagari | ✅ 97,941 | ✅ 778,638 |
| 5 | `kan_Knda` | Kannada | ✅ 97,941 | ✅ 778,638 |
| 6 | `mal_Mlym` | Malayalam | ✅ 97,941 | ✅ 778,638 |
| 7 | `mar_Deva` | Devanagari | ✅ 97,941 | ✅ 765,873 |
| 8 | `npi_Deva` | Devanagari | ✅ 97,941 | ✅ 754,154 |
| 9 | `ory_Orya` | Odia | ✅ 97,941 | ✅ 782,282 |
| 10 | `pan_Guru` | Gurmukhi | ✅ 97,941 | ✅ 778,638 |
| 11 | `san_Deva` | Devanagari (Sanskrit) | ✅ 97,941 | ✅ 778,638 |
| 12 | `tam_Taml` | Tamil | ✅ 97,941 | ✅ 778,638 |
| 13 | **`tel_Telu`** | Telugu | ✅ 97,941 | ❌ **absent** |
| 14 | `urd_Arab` | Arabic (Urdu) | ✅ 97,941 | ✅ 770,089 |

**Finding: Telugu exists in `validation` but not in `train`.** 14 languages vs 13.
Anything that assumes train/val language parity will silently drop Telugu.

**English is not a language shard.** It is carried inside every row as `Eng_Query` /
`English_passages` / `Eng_Answer`. English is therefore replicated across all 14 shards.

### The shards are row-aligned — verified, not assumed

For all 14 validation shards, the full `query_id` sequence and the full `Eng_Query` sequence
are **identical, in order** (checked element-wise across 97,941 rows × 14 shards):

```
asm_Beng  (reference)
ben_Beng  query_id_identical=True  eng_query_identical=True
…
urd_Arab  query_id_identical=True  eng_query_identical=True
```

Corroborated for free from the footer: the `English_passages` column is **314.3 MB
uncompressed in every single shard**, to the decimal.

**Consequences.**
- Row index `i` is a join key across all 14 languages. No matching, no alignment step.
- A perfectly controlled cross-lingual eval comes free: the *same* query, the *same* gold
  labels, in 14 languages. A per-language breakdown is a pure ablation with no confound.
- Naive ingest of all shards would index the English corpus **14 times**.
  13,685,630 English passage instances collapse to **950,721 unique — 93.1% waste.**

---

## 4. Relevance labels — the decisive result

`passages.is_selected` is the MS MARCO relevance flag, present and populated.

Measured on `validation/hin_Deva`, 97,941 rows / 977,545 passage instances:

| Selected passages per query | Queries |
|---:|---:|
| 0 | 44,046 |
| 1 | 50,864 |
| 2 | 2,604 |
| 3 | 339 |
| 4 | 69 |
| 5 | 16 |
| 6 | 3 |

**55.03% of queries carry ≥1 positive label** → **53,898 labeled queries per language**,
and because of §3 the *same* labels apply in all 14 languages. The complement — **44.92%**
marked `"No Answer Present."` — is not waste; see below.

Candidates per query: mean 9.98, p50 10, p90 10, p99 10, min 1, max 27.
Alignment of `English_passages` / `Translated_passages` / `is_selected` list lengths:
**0 mismatches in 97,941 rows.**

### Labels and answerability agree exactly

| | `is_selected` all zero | `is_selected` has a 1 |
|---|---:|---:|
| `Eng_Answer == "No Answer Present."` | **43,991** | **0** |
| otherwise | 55 | 53,895 |

The off-diagonal cell is **empty**. `"No Answer Present."` ⟺ zero positives, with zero
exceptions. (The 55 residual cases are empty-string answers, not a labeling conflict.)

**This is the single most valuable property of the dataset for this task:**

- We get **true retrieval metrics** — Recall@k, MRR@10, nDCG@10 — on 53,898 labeled
  queries per language. The brief's fallback ("build a silver-standard eval set") is **not
  needed**, and a silver set would have been strictly worse.
- We get **43,991 natural, human-curated unanswerable queries** — real queries whose real
  retrieved candidates genuinely do not contain the answer. This is exactly the
  "unanswerable-but-plausible" red-team category in §8 of the brief, except it is real data
  rather than something we invented, and it is large enough to *calibrate* the relevance
  threshold τ with a proper ROC instead of guessing.
- Answerable vs unanswerable is a near 55/45 split — close to balanced, so the ROC is
  well-conditioned.

`query_type` distribution (`validation`, per language):

| Type | Count | Share |
|---|---:|---:|
| DESCRIPTION | 52,912 | 54.0% |
| NUMERIC | 24,741 | 25.3% |
| ENTITY | 8,427 | 8.6% |
| PERSON | 6,206 | 6.3% |
| LOCATION | 5,655 | 5.8% |

This is our stratification key for the ≥200-query benchmark set, alongside language and
answerability.

---

## 5. Length distributions — why the chunking brief has to change

Tokenized with the **actual** `intfloat/multilingual-e5-small` tokenizer
(`XLMRobertaTokenizer`, 250,002 vocab), 40,000 sampled passages, 20,000 queries.

### Tokens

| | passage EN | passage HI | query EN | query HI |
|---|---:|---:|---:|---:|
| mean | 77.8 | 97.6 | 8.6 | 13.1 |
| p50 | **72** | **87** | 8 | 10 |
| p90 | 120 | 141 | 13 | 17 |
| p95 | 139 | 167 | 15 | 20 |
| p99 | 176 | 219 | 21 | 28 |
| max | **319** | 4,756 | 71 | 3,414 |

### Characters

| | passage EN | passage HI |
|---|---:|---:|
| p50 | 295 | 292 |
| p90 | 491 | 472 |
| p99 | 693 | 708 |
| max | 1,391 | **21,390** |

### The headline

| Measure | English | Hindi |
|---|---:|---:|
| Passages exceeding e5's 512-token window | **0.000%** | 0.182% |
| Passages under 128 tokens | **92.2%** | 85.9% |
| Median token inflation vs English | — | **1.21×** |

**Not one English passage in the sample exceeds the embedding window. The maximum is 319
tokens — 62% of the budget.** The entire corpus already fits, whole, into one vector.

This is the "twist" the brief anticipated, now quantified. It means:

- **Fixed-size 256 and 512** chunking are provably no-ops: every passage yields exactly one
  chunk identical to passage-atomic. Three ablation rows would be byte-identical.
- **Fixed-size 128** is the only fixed-size setting that does anything, and it fires on just
  7.8% of English passages — it can only *destroy* context, never add any.
- **Hierarchical parent–child** degenerates: the parent is the passage and the child is
  usually the whole passage too.
- The only genuine chunking pressure in the whole dataset is the **0.182% of Indic passages
  that blow past 512 tokens** — and those are MT failures (§6), not real long documents.

The ablation is still worth running — a measured negative result is evidence — but it must
be reframed around what actually varies here: **sentence packing, semantic breakpoints,
late chunking, doc2query expansion, and metadata filtering**, all of which change the
*vector* or the *retrievable surface* rather than the *split*. See ADR 0001 §5.

### UTF-8 byte expansion per language (footer metadata, zero download)

`English_passages` is 314.3 MB uncompressed in every shard. `Translated_passages`:

| Language | Passage MB | ×EN | Query ×EN |
|---|---:|---:|---:|
| `urd_Arab` | 564.4 | **1.80** | 1.91 |
| `guj_Gujr` | 784.4 | 2.50 | 2.58 |
| `pan_Guru` | 795.4 | 2.53 | 2.76 |
| `hin_Deva` | 808.7 | 2.57 | 2.91 |
| `asm_Beng` | 809.5 | 2.58 | 2.66 |
| `ben_Beng` | 819.9 | 2.61 | 2.66 |
| `ory_Orya` | 824.0 | 2.62 | 2.85 |
| `tel_Telu` | 850.1 | 2.70 | 2.74 |
| `mar_Deva` | 848.0 | 2.70 | 2.87 |
| `npi_Deva` | 854.2 | 2.72 | 2.80 |
| `kan_Knda` | 890.6 | 2.83 | 3.01 |
| `mal_Mlym` | 955.6 | 3.04 | 3.17 |
| `san_Deva` | 988.9 | 3.15 | 3.56 |
| `tam_Taml` | **1,004.7** | **3.20** | 3.54 |

Indic scripts cost **1.8–3.2× the UTF-8 bytes** of English for the same content, but only
**1.21× the tokens**. Byte length is a bad proxy for token budget here, and it is off by
different amounts per language — any char-based chunk size would silently produce Tamil
chunks ~⅓ the semantic size of the English ones. **All chunk sizing must be token-based,
using the actual model tokenizer.**

### Scripts observed (4,000 sampled passages)

- English passages: `LATIN` 4000 / 4000.
- Hindi passages: `DEVANAGARI` 3,988, `LATIN` 11, `ARABIC` 1 — code-switching and
  untranslated fragments do occur, at roughly 0.3%.

---

## 6. Data quality — MT degeneration, with prompt leakage

The Hindi passage char distribution has p99 = 708 but max = 21,390. That tail is not long
documents; it is machine-translation failure.

- **1,778 of 977,545 Hindi passages (0.1819%) exceed 3,000 characters** — this is the real,
  measurable defect.
- **Prompt leakage is rare, not systemic: the literal string `Translated Text:` appears in
  exactly 2 passages out of 977,545 (0.0002%)**, one of which is the longest passage in the
  shard. Worth knowing about, not worth engineering around. (Stated precisely because the
  single example below is vivid and would otherwise imply a much bigger problem than the
  count supports.)
- The longest (21,390 chars) is a repetition loop, and happens to be one of the two leakage
  cases. Its tail:

```
… forwarded to Boone County Sheriff for service with/without date?"

Translated Text: "Vacate the petition. What does the petition to vacate probation
summons issued and forwarded to Boone County Sheriff for service with/without date?"

Translated Text: "Vacate the petition. What does the petition to …
```

Two distinct defects visible in one sample:

1. **Degenerate repetition** — the classic greedy-decoding loop (`temperature = 0`).
   Affects **0.18%** of passages. This one matters.
2. **Prompt leakage** — the scaffold string `Translated Text:` is baked into the text and
   the "translation" is English. Affects **2 passages (0.0002%)**. This one does not.

`query` has the same repetition failure mode: max 3,414 tokens on a field whose p99 is 28.

**Ingest should filter defect 1**, and the filter is cheap: a token-length cap at ~2× p99
plus a repetition-ratio check catches essentially all of it at a cost of 0.18% of the corpus.
That is a real, measured justification for a cleaning step — not hygiene theatre. It is also
the *only* place in this dataset where a chunker would ever be load-bearing, and the right
answer there is to **drop** the passage, not to chunk it.

---

## 7. Duplicates and near-duplicates

Within a single language shard (977,545 English passage instances, exact hashing over the
full column; MinHash-LSH over a 30,000 sample at Jaccard ≥ 0.8):

| Measure | English | Hindi |
|---|---:|---:|
| Instances | 977,545 | 977,545 |
| Exact-unique | 950,721 | 953,398 |
| Exact duplicate rate | **2.74%** | 2.47% |
| Near-duplicate rate (30k sample, Jaccard ≥ 0.8) | **0.64%** | — |
| Most-repeated passage appears | **345×** | — |

Within-shard redundancy totals ~**3.4%** English (2.74% exact + 0.64% near) — *lower* than
the brief assumed for MS MARCO derivatives. MinHash earns its place, but it is not where the
win is. The honest read: the interesting duplication in this dataset is not intra-shard, it
is **structural and cross-shard**.

| Ingest strategy | English passage rows indexed |
|---|---:|
| Naive: all 14 shards × `English_passages` | 13,685,630 |
| Deduplicated | **950,721** |
| **Removed** | **93.1%** |

Dedup here is not a marginal quality win. It is the difference between a 7.3 GB index and a
0.5 GB one, and it comes from understanding the schema rather than from a clever algorithm.

---

## 8. Access-path findings (these cost real time — recording them)

1. **The HF dataset viewer is broken for this dataset and will stay broken.**
   - `GET /first-rows?…&split=train` → **501**, `"Job manager crashed while running this
     job (missing heartbeats)"`
   - `GET /rows?…&offset=0&length=2` → **500**,
     `ArrowNotImplementedError: Nested data conversions not implemented for chunked array outputs`

   The cause is the `passages` struct-of-lists column. **The documented discovery path in
   the brief (`/first-rows`) does not work for this dataset.** `/splits`, `/size`, `/info`
   and `/parquet` all return 200, so the schema is still recoverable — via `/info` — and the
   parquet conversion branch is complete and readable. Anyone reproducing this should skip
   the viewer entirely.

2. **`streaming=True` is a trap here.** Each shard is a **single row group** of ~98k rows
   (~1.16 GB uncompressed). Streaming cannot read less than one row group, so the first
   `next()` pulls the whole shard. It offers no memory or bandwidth advantage on this
   dataset.

3. **Column projection is the real lever.** Per-shard compressed column sizes:

   | Column | Compressed |
   |---|---:|
   | `passages.Translated_passages` | 271.9 MB |
   | `passages.English_passages` | 173.0 MB |
   | `Answer` | 5.7 MB |
   | `query` | 4.3 MB |
   | `Eng_Answer` | 3.6 MB |
   | `Eng_Query` | 2.6 MB |
   | `query_id` | 0.7 MB |
   | `passages.is_selected` | **0.1 MB** |
   | everything else | ~0.0 MB |
   | **total** | **461.9 MB** |

   Labels cost **0.1 MB per language**. Queries + answers + labels cost ~17 MB. Only passage
   *text* is expensive. Reading nested leaves directly (`passages.is_selected`) works fine in
   pyarrow — it is only the datasets-server conversion that fails.

4. **Footer statistics answered the language-layout question for free** — `target_lang`
   min/max per shard mapped all 27 shards without transferring a byte of data.

---

## 9. Corpus sizing for the Space

Derived from measured uniqueness (950,721 unique EN / 953,398 unique Indic per 97,941
queries) and 384-d int8 vectors + an HNSW `M=16` graph.

| Scenario | Queries | Langs | EN psg | Indic psg | Total | Index | Text |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full validation, all 14 | 97,941 | 14 | 950,721 | 13,347,572 | 14,298,293 | 7.3 GB | 11.3 GB |
| Full validation, EN only | 97,941 | 0 | 950,721 | 0 | 950,721 | 487 MB | 306 MB |
| 50k queries, 14 langs | 50,000 | 14 | 485,353 | 6,814,080 | 7,299,433 | 3.7 GB | 5.8 GB |
| 25k queries, 6 langs | 25,000 | 6 | 242,676 | 1,460,160 | 1,702,836 | 872 MB | 1.3 GB |
| **20k queries, EN+hi+ta+bn** | **20,000** | **3+EN** | **194,141** | **584,064** | **778,205** | **398 MB** | **546 MB** |
| 10k queries, EN+hi+ta+bn | 10,000 | 3+EN | 97,070 | 292,032 | 389,102 | 199 MB | 273 MB |

The full multilingual corpus is out of reach for a Space that must answer in under 200 ms.
Recommendation carried into ADR 0001.

---

## 10. Open questions for Gate A sign-off

1. **Corpus scope** — how many languages and how many queries (§9). This sets index size,
   build time, and how much of the ablation is multilingual.
2. **Chunking ablation reframing** (§5) — three of the eight strategies are provably
   degenerate on this data. Report them as measured no-ops, or substitute variants that
   actually vary?
3. **Repository layout** — this directory sits inside the existing
   `Desktop/Project` git repo (root has unrelated `Datathon_2026/`, `HHGoa_Task2/`).
   The submission needs its own repo to push to a HF Space.

---

*Reproduce:* `python src/ingest/discover.py --out docs/discovery`
(≈ 462 MB download for the passage-level passes; `--skip-download` runs structural passes only).
