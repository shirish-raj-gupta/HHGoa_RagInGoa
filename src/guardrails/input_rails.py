"""
Input-side guardrails (requirement #6, input half).

Every check returns a GuardrailEvent so the decision is visible in the trace
and in the UI. A guardrail that blocks silently is worse than no guardrail -
the whole point is showing the system knowing when NOT to answer.

Order matters and is cheapest-first: audio checks before transcription,
transcript checks before embedding, relevance gate after retrieval. There is
no reason to embed a query we are going to refuse.
"""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from ..harness.contracts import GuardrailEvent, RefusalReason

THRESHOLDS = yaml.safe_load(
    (Path(__file__).parent / "thresholds.yaml").read_text(encoding="utf-8"))

# Script -> FLORES-200 language, for language identification without a model.
# Unambiguous for 9 of 15; Devanagari and Bengali are shared, so those fall
# through to the STT-reported language rather than being guessed here.
SCRIPT_TO_LANG = {
    "GUJARATI": "guj_Gujr", "KANNADA": "kan_Knda", "MALAYALAM": "mal_Mlym",
    "ORIYA": "ory_Orya", "GURMUKHI": "pan_Guru", "TAMIL": "tam_Taml",
    "TELUGU": "tel_Telu", "ARABIC": "urd_Arab", "LATIN": "eng_Latn",
}
AMBIGUOUS_SCRIPTS = {"DEVANAGARI": ("hin_Deva", "mar_Deva", "npi_Deva", "san_Deva"),
                     "BENGALI": ("ben_Beng", "asm_Beng")}

# Sarvam BCP-47 -> FLORES-200. The dataset and the STT vendor disagree on
# language codes; this table is the only place that knows it.
SARVAM_TO_FLORES = {
    "en-IN": "eng_Latn", "as-IN": "asm_Beng", "bn-IN": "ben_Beng",
    "gu-IN": "guj_Gujr", "hi-IN": "hin_Deva", "kn-IN": "kan_Knda",
    "ml-IN": "mal_Mlym", "mr-IN": "mar_Deva", "ne-IN": "npi_Deva",
    "or-IN": "ory_Orya", "pa-IN": "pan_Guru", "sa-IN": "san_Deva",
    "ta-IN": "tam_Taml", "te-IN": "tel_Telu", "ur-IN": "urd_Arab",
}
FLORES_TO_SARVAM = {v: k for k, v in SARVAM_TO_FLORES.items()}

# Prompt-injection patterns, screened on the TRANSCRIPT. Multilingual because
# the attack will not always arrive in English.
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(your\s+|the\s+|previous\s+|prior\s+)*instructions?",
    r"disregard\s+(all\s+|the\s+|your\s+)?(previous|prior|above|system)",
    r"forget\s+(everything|all|your\s+instructions?|what\s+you)",
    r"you\s+are\s+now\s+(a|an|no\s+longer)",
    r"(reveal|show|print|repeat|tell\s+me)\s+(your\s+|the\s+)?(system\s+)?"
    r"(prompt|instructions?|rules)",
    r"pretend\s+(to\s+be|you\s+are)",
    r"act\s+as\s+(if|a|an)\b",
    r"developer\s+mode|jailbreak|DAN\s+mode",
    r"new\s+instructions?\s*:",
    r"</?(system|instruction|prompt)>",
    r"अपने\s+निर्देश.*(भूल|अनदेखा)",       # hindi: forget/ignore your instructions
    r"अब\s+तुम\s+हो",                       # hindi: you are now
    r"உங்கள்\s+வழிமுறைகளை.*(மறந்து|புறக்கணி)",  # tamil
]
INJECTION_RE = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in INJECTION_PATTERNS]

# PII redacted BEFORE anything is logged.
PII_PATTERNS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "phone_in": re.compile(r"\b(?:\+?91[\s-]?)?[6-9]\d{9}\b"),
    "aadhaar": re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
    "pan": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "card": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
}

UNSAFE_CATEGORIES = {
    "weapons": [r"\b(build|make|construct)\s+(a\s+)?(bomb|explosive|ied)\b",
                r"\bpipe\s+bomb\b", r"\bnerve\s+agent\b"],
    "self_harm": [r"\b(how\s+to\s+)?(kill|hurt)\s+myself\b", r"\bsuicide\s+method"],
    "illicit": [r"\b(synthesi[sz]e|cook|manufacture)\s+(meth|fentanyl|heroin)\b"],
    "malware": [r"\bwrite\s+(a\s+)?(ransomware|keylogger|botnet)\b"],
}
UNSAFE_RE = {k: [re.compile(p, re.I) for p in v] for k, v in UNSAFE_CATEGORIES.items()}


@dataclass
class RailResult:
    passed: bool
    event: GuardrailEvent
    reason: RefusalReason | None = None
    payload: dict | None = None


def _ev(name, passed, detail="", score=None) -> GuardrailEvent:
    return GuardrailEvent(name=name, passed=passed, detail=detail, score=score)


# ------------------------------------------------------------------- audio
def check_audio(pcm16: bytes, sample_rate: int = 16000) -> RailResult:
    """Empty / silent / sub-threshold audio -> ask the user to repeat."""
    cfg = THRESHOLDS["audio"]
    dur_ms = 1000 * len(pcm16) / (2 * sample_rate)
    if dur_ms < THRESHOLDS["transcript"]["min_duration_ms"]:
        return RailResult(False, _ev("audio", False, f"too_short {dur_ms:.0f}ms",
                                     dur_ms), RefusalReason.EMPTY_AUDIO)
    x = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
    if rms < cfg["min_rms"]:
        return RailResult(False, _ev("audio", False, f"silence rms={rms:.4f}", rms),
                          RefusalReason.EMPTY_AUDIO)
    return RailResult(True, _ev("audio", True, f"{dur_ms:.0f}ms rms={rms:.3f}", rms))


# -------------------------------------------------------------- transcript
def check_transcript_confidence(text: str, confidence: float | None) -> RailResult:
    """Below threshold -> confirm the transcript before acting on it."""
    cfg = THRESHOLDS["transcript"]
    if not text or len(text.strip()) < cfg["min_chars"]:
        return RailResult(False, _ev("transcript", False, "empty"),
                          RefusalReason.EMPTY_AUDIO)
    if confidence is not None and confidence < cfg["min_confidence"]:
        return RailResult(False, _ev("transcript", False,
                                     f"low_confidence {confidence:.2f}", confidence),
                          RefusalReason.LOW_CONFIDENCE_TRANSCRIPT)
    return RailResult(True, _ev("transcript", True,
                                f"conf={confidence if confidence is not None else 'n/a'}",
                                confidence))


def detect_script(text: str) -> str:
    from collections import Counter
    c: Counter = Counter()
    for ch in text[:400]:
        if not ch.isalpha():
            continue
        try:
            c[unicodedata.name(ch).split()[0]] += 1
        except ValueError:
            pass
    return c.most_common(1)[0][0] if c else "LATIN"


def identify_language(text: str, stt_lang: str | None = None) -> tuple[str, float]:
    """
    Script-based language ID, deferring to the STT's own answer where script
    alone cannot decide (Devanagari covers hi/mr/ne/sa; Bengali covers bn/as).
    """
    script = detect_script(text)
    if script in AMBIGUOUS_SCRIPTS:
        cands = AMBIGUOUS_SCRIPTS[script]
        if stt_lang and SARVAM_TO_FLORES.get(stt_lang) in cands:
            return SARVAM_TO_FLORES[stt_lang], 0.9
        return cands[0], 0.5                      # most likely of the group
    lang = SCRIPT_TO_LANG.get(script)
    if lang:
        return lang, 0.95
    if stt_lang and stt_lang in SARVAM_TO_FLORES:
        return SARVAM_TO_FLORES[stt_lang], 0.7
    return "eng_Latn", 0.3


def check_language(lang: str) -> RailResult:
    sup = THRESHOLDS["language"]["supported"]
    if lang not in sup:
        return RailResult(False, _ev("language", False, f"unsupported:{lang}"),
                          RefusalReason.UNSUPPORTED_LANGUAGE)
    return RailResult(True, _ev("language", True, lang))


# -------------------------------------------------------- injection / safety
def check_injection(text: str) -> RailResult:
    hits = [p.pattern[:38] for p in INJECTION_RE if p.search(text)]
    if len(hits) >= THRESHOLDS["injection"]["min_hits_to_block"]:
        return RailResult(False, _ev("injection", False, f"{len(hits)} pattern(s)",
                                     float(len(hits))),
                          RefusalReason.PROMPT_INJECTION, {"patterns": hits})
    return RailResult(True, _ev("injection", True, "clean"))


def check_unsafe(text: str) -> RailResult:
    for cat, pats in UNSAFE_RE.items():
        if any(p.search(text) for p in pats):
            # the refusal names the category rather than being generic
            return RailResult(False, _ev("unsafe", False, f"category={cat}"),
                              RefusalReason.UNSAFE, {"category": cat})
    return RailResult(True, _ev("unsafe", True, "clean"))


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Redact BEFORE logging. Returns (clean_text, kinds_found)."""
    found = []
    out = text
    for kind, pat in PII_PATTERNS.items():
        if pat.search(out):
            found.append(kind)
            out = pat.sub(f"[{kind.upper()}]", out)
    return out, found


# ------------------------------------------------------------ relevance gate
def check_relevance(top_score: float, tau: float | None = None) -> RailResult:
    """
    Off-topic / out-of-domain gate.

    tau comes from thresholds.yaml, where it is written by
    bench/calibrate_tau.py. If it has never been calibrated we FAIL OPEN and
    say so in the trace, rather than silently applying an invented constant.
    """
    cfg = THRESHOLDS["relevance"]
    tau = tau if tau is not None else cfg["tau"]
    if tau is None:
        return RailResult(True, _ev("relevance", True,
                                    f"UNCALIBRATED tau; top={top_score:.3f}",
                                    top_score))
    if top_score < tau - cfg["ambiguous_band"]:
        return RailResult(False, _ev("relevance", False,
                                     f"top={top_score:.3f} < tau={tau:.3f}", top_score),
                          RefusalReason.OFF_TOPIC)
    if top_score < tau:
        return RailResult(False, _ev("relevance", False,
                                     f"ambiguous top={top_score:.3f}", top_score),
                          RefusalReason.NOT_IN_RETRIEVED_SET)
    return RailResult(True, _ev("relevance", True, f"top={top_score:.3f}", top_score))
