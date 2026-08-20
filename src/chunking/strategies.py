"""
The eight chunking strategies required by the brief.

Read docs/chunking-ablation.md for which ones actually do anything on this
dataset. Three of them provably cannot (see each class's DEGENERACY note) and
they are kept, run, and reported rather than quietly dropped.
"""
from __future__ import annotations

import numpy as np

from .base import Chunk, Chunker, Passage


class FixedSizeChunker(Chunker):
    """
    1. Fixed-size + overlap. The baseline we exist to beat.

    DEGENERACY: passages are p50 72 / max 319 tokens, so target=256 and
    target=512 return every passage whole - byte-identical to PassageAtomic.
    Only target=128 splits anything, and only ~7.8% of English passages.
    """

    def __init__(self, tokenizer, target: int = 256, overlap_pct: float = 0.0):
        super().__init__(tokenizer)
        self.target, self.overlap_pct = target, overlap_pct
        self.name = f"fixed_{target}_o{int(overlap_pct * 100)}"

    def chunk(self, p: Passage) -> list[Chunk]:
        spans = self.tok.token_spans(p.text)
        if len(spans) <= self.target:
            return [self._whole(p, degenerate=True, reason="shorter_than_target")]

        stride = max(1, self.target - int(self.target * self.overlap_pct))
        out = []
        for k, i in enumerate(range(0, len(spans), stride)):
            window = spans[i:i + self.target]
            if not window:
                break
            cs, ce = window[0][0], window[-1][1]
            out.append(Chunk(
                chunk_id=f"{p.passage_id}#{k}", passage_id=p.passage_id, doc_id=p.doc_id,
                text=p.text[cs:ce], lang=p.lang, script=p.script,
                token_len=len(window), char_start=cs, char_end=ce,
            ))
            if i + self.target >= len(spans):
                break
        return out


class SentencePackingChunker(Chunker):
    """
    2. Script-aware sentence packing.

    Packs whole sentences up to a token target instead of cutting mid-word.
    Boundaries respect Devanagari danda, Arabic-script stops, and Latin
    periods; grapheme clusters are never split (see base.split_sentences).
    """

    def __init__(self, tokenizer, target: int = 128):
        super().__init__(tokenizer)
        self.target = target
        self.name = f"sentence_pack_{target}"

    def chunk(self, p: Passage) -> list[Chunk]:
        sents = self.split_sentences(p.text)
        if len(sents) == 1:
            return [self._whole(p, degenerate=True, reason="single_sentence")]

        out, buf, buf_tok, k = [], [], 0, 0
        for cs, ce in sents:
            n = len(self.tok.encode(p.text[cs:ce]))
            if buf and buf_tok + n > self.target:
                a, b = buf[0][0], buf[-1][1]
                out.append(Chunk(
                    chunk_id=f"{p.passage_id}#{k}", passage_id=p.passage_id,
                    doc_id=p.doc_id, text=p.text[a:b], lang=p.lang, script=p.script,
                    token_len=buf_tok, char_start=a, char_end=b))
                k, buf, buf_tok = k + 1, [], 0
            buf.append((cs, ce))
            buf_tok += n
        if buf:
            a, b = buf[0][0], buf[-1][1]
            out.append(Chunk(
                chunk_id=f"{p.passage_id}#{k}", passage_id=p.passage_id,
                doc_id=p.doc_id, text=p.text[a:b], lang=p.lang, script=p.script,
                token_len=buf_tok, char_start=a, char_end=b))
        if len(out) == 1:
            out[0].degenerate = True
            out[0].extra["reason"] = "packed_to_one"
        return out


class SemanticBreakpointChunker(Chunker):
    """
    3. Semantic breakpoint. Cut where adjacent-sentence cosine similarity
    drops below a percentile threshold.

    DEGENERACY RISK: a 72-token passage has ~3 sentences, so there is very
    little signal to find a breakpoint in. Reported, not hidden.
    """

    def __init__(self, tokenizer, embed_fn, percentile: int = 90):
        super().__init__(tokenizer)
        self.embed_fn, self.percentile = embed_fn, percentile
        self.name = f"semantic_p{percentile}"

    def chunk(self, p: Passage) -> list[Chunk]:
        sents = self.split_sentences(p.text)
        if len(sents) < 3:
            return [self._whole(p, degenerate=True, reason="too_few_sentences")]

        texts = [p.text[a:b] for a, b in sents]
        V = self.embed_fn(texts)
        V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
        sims = np.sum(V[:-1] * V[1:], axis=1)
        # cut where similarity is unusually LOW -> low percentile of sims
        thresh = np.percentile(sims, 100 - self.percentile)
        cuts = [i + 1 for i, s in enumerate(sims) if s <= thresh]

        out, prev, k = [], 0, 0
        for c in cuts + [len(sents)]:
            if c <= prev:
                continue
            a, b = sents[prev][0], sents[c - 1][1]
            out.append(Chunk(
                chunk_id=f"{p.passage_id}#{k}", passage_id=p.passage_id,
                doc_id=p.doc_id, text=p.text[a:b], lang=p.lang, script=p.script,
                token_len=len(self.tok.encode(p.text[a:b])),
                char_start=a, char_end=b))
            prev, k = c, k + 1
        if len(out) == 1:
            out[0].degenerate = True
            out[0].extra["reason"] = "no_breakpoint_found"
        return out


class PassageAtomicChunker(Chunker):
    """
    4. Passage-atomic. The passage is the indivisible unit.

    The control arm, and the expected winner: MS MARCO passages were built by
    humans as self-contained answer-bearing units, and 100% of them already
    fit inside the embedding window.
    """

    name = "passage_atomic"

    def chunk(self, p: Passage) -> list[Chunk]:
        # not "degenerate" - being whole is the point of this strategy
        return [self._whole(p, degenerate=False)]


class HierarchicalChunker(Chunker):
    """
    5. Hierarchical parent-child. Index fine children for precision, return
    the parent passage for context. Retrieve small, answer big.

    DEGENERACY: with p50 = 72 tokens and a 64-token child target, most
    passages yield a single child that IS the parent.
    """

    def __init__(self, tokenizer, child_target: int = 64):
        super().__init__(tokenizer)
        self.child_target = child_target
        self.name = f"hierarchical_c{child_target}"

    def chunk(self, p: Passage) -> list[Chunk]:
        sents = self.split_sentences(p.text)
        out, buf, buf_tok, k = [], [], 0, 0

        def flush():
            nonlocal buf, buf_tok, k
            if not buf:
                return
            a, b = buf[0][0], buf[-1][1]
            out.append(Chunk(
                chunk_id=f"{p.passage_id}#{k}", passage_id=p.passage_id,
                doc_id=p.doc_id, text=p.text[a:b], lang=p.lang, script=p.script,
                token_len=buf_tok, char_start=a, char_end=b,
                parent_text=p.text))          # <- answer big
            k, buf, buf_tok = k + 1, [], 0

        for cs, ce in sents:
            n = len(self.tok.encode(p.text[cs:ce]))
            if buf and buf_tok + n > self.child_target:
                flush()
            buf.append((cs, ce))
            buf_tok += n
        flush()

        if len(out) == 1:
            out[0].degenerate = True
            out[0].extra["reason"] = "child_equals_parent"
        return out


class LateChunker(Chunker):
    """
    6. Late chunking. Embed the FULL passage first, then mean-pool token spans
    into per-chunk vectors, so every chunk vector carries document-level
    context.

    Genuinely different from the others: it changes the VECTOR, not the split.
    Expected to help on the pronoun/entity-heavy queries MS MARCO is full of.
    The pooling itself lives in the embedder; this class emits the spans and
    flags them so the indexer knows to use late-pooled vectors.
    """

    def __init__(self, tokenizer, target: int = 96):
        super().__init__(tokenizer)
        self.target = target
        self.name = f"late_chunk_{target}"

    def chunk(self, p: Passage) -> list[Chunk]:
        spans = self.tok.token_spans(p.text)
        n = len(spans)
        if n <= self.target:
            c = self._whole(p, degenerate=False, late=True, tok_lo=0, tok_hi=n)
            c.extra["late_chunk"] = True
            return [c]

        out = []
        for k, i in enumerate(range(0, n, self.target)):
            w = spans[i:i + self.target]
            if not w:
                break
            cs, ce = w[0][0], w[-1][1]
            out.append(Chunk(
                chunk_id=f"{p.passage_id}#{k}", passage_id=p.passage_id,
                doc_id=p.doc_id, text=p.text[cs:ce], lang=p.lang, script=p.script,
                token_len=len(w), char_start=cs, char_end=ce,
                parent_text=p.text,
                # the indexer mean-pools full-passage token vectors over this range
                extra={"late_chunk": True, "tok_lo": i, "tok_hi": i + len(w)}))
        return out


class MetadataAwareChunker(Chunker):
    """
    7. Metadata-aware. Passage-atomic, but every filterable field is attached
    as payload so the retriever can filter at query time.

    On this corpus that is the difference between searching ~1.9M vectors
    (one language partition + English) and 14.3M - the mechanism that makes
    the full 14-language index viable at all. See ADR 0001 section 1.
    """

    name = "metadata_aware"

    def __init__(self, tokenizer, dedup_cluster: dict[str, int] | None = None):
        super().__init__(tokenizer)
        self.dedup_cluster = dedup_cluster or {}

    def chunk(self, p: Passage) -> list[Chunk]:
        c = self._whole(p, degenerate=False)
        c.extra.update({
            "lang": p.lang, "script": p.script, "token_len": c.token_len,
            "doc_id": p.doc_id,
            "dedup_cluster": self.dedup_cluster.get(p.passage_id),
            "partition": p.lang,      # the routing key
        })
        return [c]


class QueryExpansionChunker(Chunker):
    """
    8. Query-aware expansion (doc2query-lite). Generate 2-3 hypothetical
    questions per passage at INDEX time and index them alongside the text.

    Costs index build time, buys recall. Query-time cost is zero, which is why
    it is the one "clever" strategy that is safe on a latency-bound system.

    gen_fn(text, lang, n) -> list[str]; when absent, falls back to a cheap
    lead-sentence heuristic so the arm still runs without a generator.
    """

    name = "doc2query"

    def __init__(self, tokenizer, gen_fn=None, n_questions: int = 3):
        super().__init__(tokenizer)
        self.gen_fn, self.n = gen_fn, n_questions

    def chunk(self, p: Passage) -> list[Chunk]:
        base = self._whole(p, degenerate=False)
        qs: list[str] = []
        if self.gen_fn is not None:
            try:
                qs = list(self.gen_fn(p.text, p.lang, self.n))[: self.n]
            except Exception:
                qs = []
        if not qs:
            # heuristic fallback: the lead sentence carries the topic in MS MARCO
            sents = self.split_sentences(p.text)
            qs = [p.text[a:b] for a, b in sents[:1]]

        base.extra["expansions"] = qs
        # expansions are indexed as extra retrievable surfaces pointing at the
        # SAME passage, so a hit on a question resolves to the real passage
        out = [base]
        for k, q in enumerate(qs, start=1):
            out.append(Chunk(
                chunk_id=f"{p.passage_id}#q{k}", passage_id=p.passage_id,
                doc_id=p.doc_id, text=q, lang=p.lang, script=p.script,
                token_len=len(self.tok.encode(q)),
                char_start=0, char_end=len(p.text),
                parent_text=p.text,
                extra={"is_expansion": True}))
        return out


def build_registry(tokenizer, embed_fn=None, gen_fn=None) -> dict[str, Chunker]:
    """Every arm the ablation runs, keyed by name."""
    reg: list[Chunker] = [
        FixedSizeChunker(tokenizer, 128, 0.0),
        FixedSizeChunker(tokenizer, 128, 0.25),
        FixedSizeChunker(tokenizer, 256, 0.0),
        FixedSizeChunker(tokenizer, 512, 0.0),
        SentencePackingChunker(tokenizer, 128),
        PassageAtomicChunker(tokenizer),
        HierarchicalChunker(tokenizer, 64),
        LateChunker(tokenizer, 96),
        MetadataAwareChunker(tokenizer),
        QueryExpansionChunker(tokenizer, gen_fn=gen_fn),
    ]
    if embed_fn is not None:
        for pct in (85, 90, 95):
            reg.append(SemanticBreakpointChunker(tokenizer, embed_fn, pct))
    return {c.name: c for c in reg}
