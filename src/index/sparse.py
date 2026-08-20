"""
Sparse retrieval (BM25) with per-script tokenization.

Why per-script: a single Latin-centric analyzer silently under-tokenizes every
Indic script. Devanagari and friends terminate sentences with danda; Urdu uses
Arabic-script punctuation; none of them are served by English stemming. Getting
this wrong does not raise - it just loses recall on exactly the languages this
task is about.

Sparse exists here because dense alone fails on rare entities and numerals,
which is 34% of this corpus by query_type (NUMERIC + ENTITY).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import bm25s
import numpy as np

# Script-specific punctuation that must be stripped before tokenizing.
DANDA, DOUBLE_DANDA = "।", "॥"
ARABIC_PUNCT = "،؛؟۔"          # , ; ? .
EXTRA_PUNCT = DANDA + DOUBLE_DANDA + ARABIC_PUNCT

# Unicode-aware word tokenizer. \w in Python's re is Unicode-aware by default,
# so this handles Devanagari, Tamil, Bengali, Arabic and Latin alike.
WORD = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)

# Latin stopwords only. There is no reliable, license-clean stopword list for
# all 14 languages here, and a wrong one costs more than none - so Indic terms
# are left intact and BM25's IDF does the down-weighting instead.
EN_STOP = frozenset("""a an the and or but if of to in on at for with by from as is are was
were be been being it its this that these those i you he she they we what which who whom how
when where why do does did not no nor so than then there here have has had can could will
would should may might must about into over under again further once""".split())


def normalize_for_bm25(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    for ch in EXTRA_PUNCT:
        text = text.replace(ch, " ")
    return text


def tokenize(text: str, lang: str = "eng_Latn") -> list[str]:
    toks = [t.lower() for t in WORD.findall(normalize_for_bm25(text))]
    if lang == "eng_Latn":
        toks = [t for t in toks if t not in EN_STOP]
    return toks


@dataclass(slots=True)
class SparseHit:
    passage_id: str
    score: float
    rank: int


class SparsePartition:
    """BM25 over one language partition."""

    def __init__(self, lang: str):
        self.lang = lang
        self.passage_ids: list[str] = []
        self._bm25: bm25s.BM25 | None = None

    def build(self, texts: list[str], passage_ids: list[str]) -> None:
        self.passage_ids = passage_ids
        corpus = [tokenize(t, self.lang) for t in texts]
        vocab: dict[str, int] = {}
        ids = [[vocab.setdefault(t, len(vocab)) for t in doc] for doc in corpus]
        self._vocab = vocab
        self._bm25 = bm25s.BM25()
        self._bm25.index(bm25s.tokenization.Tokenized(ids=ids, vocab=vocab), show_progress=False)

    def search(self, query: str, k: int = 10) -> list[SparseHit]:
        if self._bm25 is None or not self.passage_ids:
            return []
        toks = [self._vocab[t] for t in tokenize(query, self.lang) if t in self._vocab]
        if not toks:
            return []
        q = bm25s.tokenization.Tokenized(ids=[toks], vocab=self._vocab)
        idx, sc = self._bm25.retrieve(q, k=min(k, len(self.passage_ids)),
                                      show_progress=False)
        return [SparseHit(self.passage_ids[int(i)], float(s), r)
                for r, (i, s) in enumerate(zip(idx[0], sc[0]))]


class SparseIndex:
    """Language-partitioned BM25, mirroring DenseIndex's routing."""

    def __init__(self) -> None:
        self.partitions: dict[str, SparsePartition] = {}

    def build(self, lang: str, texts: list[str], passage_ids: list[str]) -> None:
        p = SparsePartition(lang)
        p.build(texts, passage_ids)
        self.partitions[lang] = p

    def search(self, query: str, lang: str, k: int = 10,
               fallback: str = "eng_Latn") -> list[SparseHit]:
        langs = [lang] if lang in self.partitions else []
        if fallback in self.partitions and fallback != lang:
            langs.append(fallback)
        hits: list[SparseHit] = []
        for lg in langs or list(self.partitions):
            hits.extend(self.partitions[lg].search(query, k))
        hits.sort(key=lambda h: -h.score)
        for r, h in enumerate(hits):
            h.rank = r
        return hits[:k]
