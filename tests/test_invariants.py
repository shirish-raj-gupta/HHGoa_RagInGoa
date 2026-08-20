"""
Tests for the invariants that actually caught bugs in this build.

Each test here exists because something silently broke, not because a coverage
target needed filling. The comments name the failure.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.chunking.base import Chunker, Passage
from src.chunking.strategies import (FixedSizeChunker, PassageAtomicChunker,
                                     SentencePackingChunker)
from src.index.exact import ExactPartition
from src.index.fusion import mmr, rrf


class FakeTok:
    """Whitespace tokenizer with real char offsets - enough for chunker logic."""

    def encode(self, text):
        return text.split()

    def token_spans(self, text):
        spans, i = [], 0
        for w in text.split():
            j = text.index(w, i)
            spans.append((j, j + len(w)))
            i = j + len(w)
        return spans


TOK = FakeTok()


def P(text, pid="p1"):
    return Passage(pid, "d1", text, "eng_Latn", "LATIN")


# --------------------------------------------------------------- chunking
def test_unsplit_passages_are_byte_identical_to_atomic():
    """
    The bug: a 38-point recall drop was blamed on chunking when only ~7% of
    passages were split. Proving the other 93% are untouched is what let us
    rule chunking out and find the real cause (a broken index dtype).
    """
    text = " ".join(f"w{i}" for i in range(40))          # under any target
    fixed = FixedSizeChunker(TOK, 128, 0.0).chunk(P(text))
    atomic = PassageAtomicChunker(TOK).chunk(P(text))
    assert len(fixed) == 1
    assert fixed[0].text == atomic[0].text == text


def test_fixed_size_splits_only_when_over_target():
    short = " ".join(f"w{i}" for i in range(10))
    long = " ".join(f"w{i}" for i in range(30))
    c = FixedSizeChunker(TOK, 16, 0.0)
    assert len(c.chunk(P(short))) == 1
    assert len(c.chunk(P(long))) > 1


def test_chunks_never_lose_text():
    """Every chunk must be a real substring - no mangled offsets."""
    text = " ".join(f"word{i}" for i in range(60))
    for ch in FixedSizeChunker(TOK, 16, 0.25).chunk(P(text)):
        assert ch.text in text
        assert text[ch.char_start:ch.char_end] == ch.text


@pytest.mark.parametrize("text,expected_min", [
    ("First sentence. Second sentence. Third one.", 3),
    ("पहला वाक्य। दूसरा वाक्य। तीसरा वाक्य।", 3),          # devanagari danda
    ("پہلا جملہ۔ دوسرا جملہ۔", 2),                           # urdu full stop
])
def test_sentence_splitting_is_script_aware(text, expected_min):
    """Indic punctuation is not optional - a Latin-only splitter loses recall."""
    assert len(Chunker.split_sentences(text)) >= expected_min


def test_decimal_is_not_a_sentence_boundary():
    """'3.5' must not split - the terminator needs trailing whitespace."""
    spans = Chunker.split_sentences("The value is 3.5 metres today.")
    assert len(spans) == 1


def test_sentence_packing_preserves_all_content():
    text = "Alpha one. Beta two. Gamma three. Delta four."
    joined = "".join(c.text for c in SentencePackingChunker(TOK, 4).chunk(P(text)))
    for token in ("Alpha", "Beta", "Gamma", "Delta"):
        assert token in joined


# ------------------------------------------------------------------ index
def test_exact_search_self_retrieval_is_perfect():
    """
    The bug: usearch ScalarKind.I8 returned a vector's own row only 1-9% of
    the time, which silently destroyed every retrieval metric. Exact search is
    the ground truth this is measured against.
    """
    rng = np.random.default_rng(0)
    V = rng.normal(size=(500, 32)).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    part = ExactPartition("eng_Latn", dim=32)
    part.add(V, [f"c{i}" for i in range(500)], [f"p{i}" for i in range(500)])
    for i in (0, 137, 499):
        assert part.search(V[i], k=1)[0].chunk_id == f"c{i}"


def test_exact_batch_matches_single():
    rng = np.random.default_rng(1)
    V = rng.normal(size=(200, 16)).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    part = ExactPartition("eng_Latn", dim=16)
    part.add(V, [f"c{i}" for i in range(200)], [f"p{i}" for i in range(200)])
    batch = part.search_batch(V[:5], k=3)
    for i in range(5):
        single = [part.chunk_ids.index(h.chunk_id) for h in part.search(V[i], k=3)]
        assert list(batch[i]) == single


# ----------------------------------------------------------------- fusion
class H:
    def __init__(self, pid, score=1.0):
        self.passage_id, self.score = pid, score


def test_rrf_rewards_agreement_between_arms():
    """A passage both arms rank highly must beat one only a single arm likes."""
    dense = [H("a"), H("b"), H("c")]
    sparse = [H("c"), H("a"), H("z")]
    out = rrf(dense, sparse, top_k=4)
    assert out[0].passage_id in ("a", "c")
    assert {f.passage_id for f in out[:2]} == {"a", "c"}


def test_rrf_records_both_provenances():
    out = rrf([H("a")], [H("a")], top_k=1)
    assert out[0].dense_rank == 0 and out[0].sparse_rank == 0
    assert set(out[0].sources) == {"dense", "sparse"}


def test_mmr_keeps_hits_without_vectors():
    """
    A missing vector must not drop a hit - that would be a silent recall bug
    dressed up as diversification.
    """
    hits = rrf([H("a"), H("b"), H("c")], [], top_k=3)
    out = mmr(hits, vectors={}, lam=0.7, k=3)
    assert len(out) == 3


def test_mmr_demotes_a_near_duplicate():
    v = {"a": np.array([1.0, 0.0]), "dup": np.array([0.999, 0.044]),
         "far": np.array([0.0, 1.0])}
    hits = rrf([H("a"), H("dup"), H("far")], [], top_k=3)
    out = [h.passage_id for h in mmr(hits, v, lam=0.5, k=2)]
    assert out[0] == "a" and out[1] == "far"
