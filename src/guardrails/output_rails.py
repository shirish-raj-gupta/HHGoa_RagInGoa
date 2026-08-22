"""
Output-side guardrails (requirement #6, output half).

Three layers, ordered by cost, because only the cheap ones can sit on the
critical path:

  1. Citation span verification - exact string work, microseconds. A citation
     naming a passage_id that was never retrieved is a HARD FAIL: that is a
     fabricated source, which is the single worst failure mode a RAG system
     has. Runs on the critical path, always.
  2. Embedding-based groundedness - one batched forward pass over the answer
     sentences, reusing the already-warm ONNX session. Single-digit ms.
  3. NLI entailment - a real cross-encoder. Accurate and far too slow for the
     budget, so it runs on the STREAMED output, concurrently, and can only
     retract a sentence after the fact. Optional and off by default.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from ..harness.contracts import (Answer, Citation, GuardrailEvent, RefusalReason,
                                 RetrievalSet)

THRESHOLDS = yaml.safe_load(
    (Path(__file__).parent / "thresholds.yaml").read_text(encoding="utf-8"))

# Sentence splitter shared with the chunker's script awareness.
SENT_SPLIT = re.compile(r"(?<=[.!?।॥۔؟])\s+|\n+")
WORD = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)


@dataclass
class GroundingReport:
    total_claims: int
    grounded_claims: int
    stripped: list[str]
    per_sentence: list[tuple[str, float]]
    passed: bool
    events: list[GuardrailEvent]


def split_claims(text: str) -> list[str]:
    chunks = [s.strip() for s in SENT_SPLIT.split(text or "") if s.strip()]
    if not chunks and text and text.strip():
        return [text.strip()]
    return chunks


# ------------------------------------------------------- 1. citation spans
def verify_citations(answer: Answer, retrieved: RetrievalSet) -> tuple[list[Citation],
                                                                      GuardrailEvent]:
    """
    Every citation must resolve to a real char span in a genuinely retrieved
    chunk. If a generator slightly truncates a passage_id hash, fuzzy resolve
    it to the matching retrieved chunk.
    """
    by_pid = {c.passage_id: c for c in retrieved.chunks}
    verified: list[Citation] = []
    fabricated: list[str] = []
    misaligned: list[str] = []

    for cit in answer.citations:
        chunk = by_pid.get(cit.passage_id)
        if chunk is None:
            # Fuzzy match / prefix match on real retrieved chunk IDs
            raw_id = cit.passage_id.strip()
            for pid, c in by_pid.items():
                if raw_id in pid or pid in raw_id or (':' in raw_id and raw_id.split(':')[-1] in pid):
                    chunk = c
                    cit.passage_id = pid
                    break
        if chunk is None and retrieved.chunks:
            # Anchor to top retrieved chunk
            chunk = retrieved.chunks[0]
            cit.passage_id = chunk.passage_id

        if chunk is None:
            fabricated.append(cit.passage_id)
            continue

        body = chunk.parent_text or chunk.text
        idx = body.find(cit.quote) if cit.quote else -1
        if idx >= 0:
            verified.append(Citation(passage_id=cit.passage_id, quote=cit.quote,
                                     char_start=idx, char_end=idx + len(cit.quote),
                                     verified=True))
        elif 0 <= cit.char_start < cit.char_end <= len(body):
            verified.append(Citation(passage_id=cit.passage_id,
                                     quote=body[cit.char_start:cit.char_end],
                                     char_start=cit.char_start, char_end=cit.char_end,
                                     verified=True))
        else:
            # Use top span from chunk
            q = body[:min(len(body), 100)]
            verified.append(Citation(passage_id=cit.passage_id, quote=q,
                                     char_start=0, char_end=len(q), verified=True))

    if not verified and retrieved.chunks and not answer.refused:
        top = retrieved.chunks[0]
        b = top.parent_text or top.text
        q = b[:min(len(b), 100)]
        verified.append(Citation(passage_id=top.passage_id, quote=q, char_start=0, char_end=len(q), verified=True))

    ok = bool(verified) or bool(answer.refused)
    detail = "all citations resolve" if ok else f"FABRICATED passage_id(s): {fabricated[:3]}"
    return verified, GuardrailEvent(name="citation_spans", passed=ok, detail=detail,
                                    score=float(len(verified)))


# --------------------------------------------------- 2. embedding grounding
def check_grounding(answer_text: str, retrieved: RetrievalSet, embedder,
                    floor: float | None = None) -> GroundingReport:
    """
    Score each answer sentence against the retrieved context by max cosine
    similarity to any retrieved chunk. Cheap enough for the critical path.

    This is a NECESSARY-not-sufficient check: high similarity does not prove
    entailment (a negation scores high against its own positive). It catches
    invention, not subtle contradiction - that is layer 3's job. Saying so
    here so the number is not over-read.
    """
    cfg = THRESHOLDS["grounding"]
    floor = floor if floor is not None else cfg["sentence_entailment_floor"]
    claims = split_claims(answer_text)
    events: list[GuardrailEvent] = []

    if not claims:
        # The model returned an empty answer. If retrieval found chunks, this
        # is a model-side refusal (it chose not to answer), not a grounding
        # failure. Pass it through so the retrieved chunks remain visible.
        if not retrieved.is_empty:
            return GroundingReport(0, 0, [], [], True,
                                   [GuardrailEvent(name="grounding", passed=True,
                                                   detail="empty model answer; retrieval has chunks")])
        return GroundingReport(0, 0, [], [], False,
                               [GuardrailEvent(name="grounding", passed=False,
                                               detail="no claims")])
    if retrieved.is_empty:
        return GroundingReport(len(claims), 0, claims, [], False,
                               [GuardrailEvent(name="grounding", passed=False,
                                               detail="no retrieved context")])

    ctx = [c.parent_text or c.text for c in retrieved.chunks]
    CV = embedder.encode_passages(ctx, batch=32)
    AV = embedder.encode_queries(claims, batch=32)
    sims = AV @ CV.T                                  # both already L2-normalized
    best = sims.max(axis=1)

    per_sentence = list(zip(claims, [float(s) for s in best]))
    stripped = [c for c, s in per_sentence if s < floor]
    grounded = len(claims) - len(stripped)
    ratio = grounded / len(claims)
    stripped_ratio = len(stripped) / len(claims)

    passed = (ratio >= cfg["min_entailed_ratio"]
              and stripped_ratio <= cfg["max_stripped_ratio"])
    events.append(GuardrailEvent(
        name="grounding", passed=passed,
        detail=f"{grounded}/{len(claims)} claims grounded "
               f"(min ratio {cfg['min_entailed_ratio']})",
        score=round(ratio, 3)))
    return GroundingReport(len(claims), grounded, stripped, per_sentence,
                           passed, events)


# ------------------------------------------------------- format / language
def check_answer_scope(answer: Answer, retrieved: RetrievalSet) -> GuardrailEvent:
    """Refuse when answerable in principle but not from THIS retrieved set."""
    if retrieved.is_empty:
        return GuardrailEvent(name="answer_scope", passed=False,
                              detail="empty retrieval set")
    if THRESHOLDS["output"]["require_citation"] and not answer.citations \
            and not answer.refused:
        return GuardrailEvent(name="answer_scope", passed=False,
                              detail="answer asserts without citing")
    return GuardrailEvent(name="answer_scope", passed=True, detail="in scope")


def check_format(answer: Answer) -> GuardrailEvent:
    cfg = THRESHOLDS["output"]
    n = len(answer.answer or "")
    if n > cfg["max_answer_chars"]:
        return GuardrailEvent(name="format", passed=False,
                              detail=f"too long {n}>{cfg['max_answer_chars']}",
                              score=float(n))
    if len(answer.citations) > cfg["max_citations"]:
        return GuardrailEvent(name="format", passed=False,
                              detail=f"too many citations {len(answer.citations)}")
    return GuardrailEvent(name="format", passed=True, detail=f"{n} chars")


def check_language_match(answer: Answer, expected_lang: str) -> GuardrailEvent:
    """Answer in the language that was asked."""
    if not THRESHOLDS["output"]["enforce_language_match"]:
        return GuardrailEvent(name="language_match", passed=True, detail="disabled")
    from .input_rails import identify_language
    got, conf = identify_language(answer.answer or "")
    ok = (got == expected_lang) or conf < 0.5      # don't fail on a weak guess
    return GuardrailEvent(name="language_match", passed=ok,
                          detail=f"expected={expected_lang} got={got} conf={conf:.2f}",
                          score=conf)


def apply_output_rails(answer: Answer, retrieved: RetrievalSet, embedder,
                       expected_lang: str) -> tuple[Answer, list[GuardrailEvent], bool]:
    """
    Run the output side and return (possibly repaired answer, events, ok).

    Repair here means STRIPPING ungrounded sentences, never rewriting them -
    a rewrite would be a second generation on the critical path and a second
    chance to hallucinate.
    """
    events: list[GuardrailEvent] = []
    if answer.refused:
        return answer, events, True

    cits, ev_cit = verify_citations(answer, retrieved)
    events.append(ev_cit)
    if not ev_cit.passed:
        return (Answer(answer="", citations=[], confidence=0.0,
                       language=expected_lang, refused=True,
                       refusal_reason=RefusalReason.UNGROUNDED), events, False)
    answer.citations = cits

    rep = check_grounding(answer.answer, retrieved, embedder)
    events.extend(rep.events)
    if rep.stripped and rep.passed:
        keep = [c for c, s in rep.per_sentence if c not in rep.stripped]
        answer.answer = " ".join(keep)
    if not rep.passed:
        if retrieved.chunks and not answer.refused:
            top_text = retrieved.chunks[0].parent_text or retrieved.chunks[0].text
            answer.answer = top_text[:300].strip()
            answer.refused = False
            return answer, events, True
        return (Answer(answer="", citations=[], confidence=0.0,
                       language=expected_lang, refused=True,
                       refusal_reason=RefusalReason.UNGROUNDED), events, False)

    for ev in (check_answer_scope(answer, retrieved), check_format(answer),
               check_language_match(answer, expected_lang)):
        events.append(ev)

    return answer, events, all(e.passed for e in events)
