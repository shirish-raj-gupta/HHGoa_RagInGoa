"""
One Chunker interface, eight strategies behind it.

Design constraint from discovery (docs/discovery/report.md section 5):
MS MARCO-XI passages are p50 72 / p99 176 / max 319 tokens in English, and
0.000% exceed the e5 512-token window. Several classic strategies therefore
degenerate to "return the passage unchanged" on this data. That is a result,
not a bug - the interface reports it via Chunk.degenerate so the ablation can
say so out loud instead of silently emitting duplicate rows.

All sizing is TOKEN-based using the real model tokenizer. Never characters:
Indic scripts cost 1.8-3.2x the UTF-8 bytes of English but only 1.21x the
tokens (report section 5), so a char-based size would make Tamil chunks about
a third the semantic size of the English ones.
"""
from __future__ import annotations

import abc
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Sequence

# Sentence terminators by script. Devanagari/Bengali/Gujarati/Gurmukhi/Odia all
# use danda + double danda; Tamil/Telugu/Kannada/Malayalam use Latin stops;
# Urdu (Arabic script) uses its own full stop and question mark.
DANDA = "।"          # devanagari danda
DOUBLE_DANDA = "॥"   # devanagari double danda
ARABIC_FULL_STOP = "۔"
ARABIC_QUESTION = "؟"
SENTENCE_ENDS = frozenset({".", "!", "?", DANDA, DOUBLE_DANDA,
                           ARABIC_FULL_STOP, ARABIC_QUESTION, "\n"})


@dataclass(slots=True)
class Passage:
    """A corpus passage as it arrives from ingest."""
    passage_id: str
    doc_id: str
    text: str
    lang: str          # FLORES-200 code, e.g. hin_Deva
    script: str        # e.g. DEVANAGARI


@dataclass(slots=True)
class Chunk:
    """A retrievable unit produced by a Chunker."""
    chunk_id: str
    passage_id: str
    doc_id: str
    text: str
    lang: str
    script: str
    token_len: int
    char_start: int
    char_end: int
    # parent text to return when this chunk is retrieved (small-to-big).
    # None means "the chunk is its own context".
    parent_text: str | None = None
    # True when the strategy could not do anything to this passage and simply
    # returned it whole. Aggregated per-strategy by the ablation.
    degenerate: bool = False
    extra: dict = field(default_factory=dict)


class Tokenizer(abc.ABC):
    """Minimal surface the chunkers need. Backed by the real e5 tokenizer."""

    @abc.abstractmethod
    def encode(self, text: str) -> list[int]: ...

    @abc.abstractmethod
    def token_spans(self, text: str) -> list[tuple[int, int]]:
        """(char_start, char_end) per token, so chunks cut on real boundaries."""


class Chunker(abc.ABC):
    """Turn one passage into >= 1 retrievable chunks."""

    name: str = "base"

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tok = tokenizer

    @abc.abstractmethod
    def chunk(self, p: Passage) -> list[Chunk]: ...

    def chunk_many(self, ps: Iterable[Passage]) -> list[Chunk]:
        out: list[Chunk] = []
        for p in ps:
            out.extend(self.chunk(p))
        return out

    # -- helpers shared by subclasses -------------------------------------

    def _whole(self, p: Passage, *, degenerate: bool, **extra) -> Chunk:
        """Emit the passage unchanged as a single chunk."""
        return Chunk(
            chunk_id=f"{p.passage_id}#0", passage_id=p.passage_id, doc_id=p.doc_id,
            text=p.text, lang=p.lang, script=p.script,
            token_len=len(self.tok.encode(p.text)),
            char_start=0, char_end=len(p.text),
            degenerate=degenerate, extra=extra,
        )

    @staticmethod
    def _normalize(text: str) -> str:
        """NFC + strip control chars, preserving script and combining marks."""
        text = unicodedata.normalize("NFC", text)
        return "".join(c for c in text
                       if c in "\n\t" or unicodedata.category(c)[0] != "C")

    @staticmethod
    def split_sentences(text: str) -> list[tuple[int, int]]:
        """
        Script-aware sentence spans as (start, end) char offsets.

        Never splits a grapheme cluster: a boundary is only taken when the
        terminator is followed by whitespace or end-of-text, so decimals
        ("3.5"), abbreviations mid-token, and combining marks stay intact.
        """
        spans, start = [], 0
        n = len(text)
        for i, ch in enumerate(text):
            if ch not in SENTENCE_ENDS:
                continue
            nxt = text[i + 1] if i + 1 < n else ""
            if nxt and not nxt.isspace():
                continue  # e.g. "3.5" - not a boundary
            j = i + 1
            while j < n and text[j].isspace():
                j += 1
            if i + 1 > start:
                spans.append((start, i + 1))
            start = j
        if start < n:
            spans.append((start, n))
        return spans or [(0, n)]
