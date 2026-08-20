"""
Typed contracts for every stage boundary (requirement #5).

AudioChunk -> Transcript -> NormalizedQuery -> RetrievalSet -> DraftAnswer
          -> VerifiedAnswer

No dicts cross a stage line. If a stage wants to hand something to the next
stage it has to name it here first, which is what stops the pipeline drifting
into a bag of loosely-related keys.
"""
from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


class Strict(BaseModel):
    """Reject unknown fields - a typo in a stage becomes an error, not a silent None."""
    model_config = ConfigDict(extra="forbid", frozen=False,
                              validate_assignment=True, str_strip_whitespace=True)


# ----------------------------------------------------------------- taxonomy
class ErrorKind(str, Enum):
    """Error taxonomy. Each value has a defined user-facing behaviour (README)."""
    UPSTREAM = "UpstreamError"            # retry w/ backoff, then circuit-break
    BUDGET_EXCEEDED = "BudgetExceeded"    # degrade, never overrun
    GUARDRAIL_BLOCKED = "GuardrailBlocked"  # refuse, with a named reason
    VALIDATION = "ValidationError"        # one repair attempt, then safe refusal
    NO_RELEVANT_CONTEXT = "NoRelevantContext"  # "not in corpus" refusal


class RefusalReason(str, Enum):
    OFF_TOPIC = "off_topic"
    UNSAFE = "unsafe"
    PROMPT_INJECTION = "prompt_injection"
    LOW_CONFIDENCE_TRANSCRIPT = "low_confidence_transcript"
    EMPTY_AUDIO = "empty_audio"
    UNSUPPORTED_LANGUAGE = "unsupported_language"
    UNGROUNDED = "ungrounded"
    NOT_IN_RETRIEVED_SET = "not_in_retrieved_set"
    PII_DETECTED = "pii_detected"


# ----------------------------------------------------------------- stage I/O
class AudioChunk(Strict):
    request_id: str = Field(default_factory=new_request_id)
    seq: int = 0
    pcm16: bytes = b""
    sample_rate: Literal[8000, 16000] = 16000
    is_final: bool = False

    @property
    def duration_ms(self) -> float:
        return 1000 * len(self.pcm16) / (2 * self.sample_rate)


class Transcript(Strict):
    request_id: str
    text: str
    lang_code: str                      # Sarvam BCP-47, e.g. hi-IN
    is_final: bool = False
    confidence: float | None = None
    stt_ms: float | None = None
    # set when speculative retrieval fired on this partial
    speculative: bool = False


class NormalizedQuery(Strict):
    request_id: str
    text: str
    raw_text: str
    lang: str                           # FLORES-200, e.g. hin_Deva
    script: str
    token_len: int
    redactions: list[str] = Field(default_factory=list)

    @field_validator("text")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("normalized query is empty")
        return v


class RetrievedChunk(Strict):
    passage_id: str
    chunk_id: str
    text: str
    lang: str
    score: float
    rank: int
    dense_rank: int | None = None
    sparse_rank: int | None = None
    chunk_strategy: str = "passage_atomic"
    parent_text: str | None = None


class RetrievalSet(Strict):
    request_id: str
    query: str
    lang: str
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    top_score: float = 0.0          # RRF fused score - rank-derived, for ranking
    # Dense top-1 COSINE. Kept separate from top_score because the relevance
    # gate is calibrated on cosine, and RRF scores are rank-derived (~2/60 for
    # nearly every query). Feeding one to a threshold fitted on the other
    # refused 100% of benign queries - caught by the red-team set.
    relevance_score: float | None = None
    degraded: list[str] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.chunks


class Citation(Strict):
    """A citation must resolve to a real char span in a really-retrieved chunk."""
    passage_id: str
    quote: str
    char_start: int
    char_end: int
    verified: bool = False


class Answer(Strict):
    """The structured output contract the generator must satisfy."""
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    language: str
    refused: bool = False
    refusal_reason: RefusalReason | None = None


class DraftAnswer(Strict):
    request_id: str
    answer: Answer
    repair_attempts: int = 0
    ttft_ms: float | None = None


class VerifiedAnswer(Strict):
    request_id: str
    answer: Answer
    grounded_claims: int = 0
    total_claims: int = 0
    stripped_claims: list[str] = Field(default_factory=list)
    guardrail_log: list["GuardrailEvent"] = Field(default_factory=list)
    e2e_ms: float | None = None


class GuardrailEvent(Strict):
    name: str
    passed: bool
    detail: str = ""
    score: float | None = None
    at_ms: float = 0.0


class StageTiming(Strict):
    name: str
    started_ms: float
    duration_ms: float
    ok: bool = True
    degraded: bool = False
    note: str = ""


class Trace(Strict):
    """What GET /trace/{request_id} returns, and what the UI renders."""
    request_id: str
    created_at: float = Field(default_factory=time.time)
    stages: list[StageTiming] = Field(default_factory=list)
    guardrails: list[GuardrailEvent] = Field(default_factory=list)
    degradations: list[str] = Field(default_factory=list)
    stt_ms: float | None = None
    core_rag_loop_ms: float | None = None
    ttft_ms: float | None = None
    e2e_ms: float | None = None
    error: ErrorKind | None = None


VerifiedAnswer.model_rebuild()
