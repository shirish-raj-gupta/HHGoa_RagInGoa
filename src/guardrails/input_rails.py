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

# Assamese and Bengali share a script but not an alphabet: Assamese writes
# ro as ৰ (U+09F0) and wo as ৱ (U+09F1), where Bengali uses র and ব. Those two
# characters are frequent enough in ordinary Assamese text to identify it.
# Without this, every Assamese query was routed to the Bengali partition -
# observed live against the asm_Beng index, which is exactly the kind of
# silent mis-routing that costs recall without raising anything.
ASSAMESE_ONLY = "ৰৱ"


def disambiguate_bengali(text: str) -> str:
    return "asm_Beng" if any(c in text for c in ASSAMESE_ONLY) else "ben_Beng"

# Scripts that are definitely not in this corpus. Without these, CJK and
# Cyrillic fell through to the eng_Latn default and were answered instead of
# refused - caught by the red-team set (unsupported_language scored 0/2).
UNSUPPORTED_SCRIPTS = {
    "CJK": "zho_Hans", "HIRAGANA": "jpn_Jpan", "KATAKANA": "jpn_Jpan",
    "HANGUL": "kor_Hang", "CYRILLIC": "rus_Cyrl", "GREEK": "ell_Grek",
    "THAI": "tha_Thai", "HEBREW": "heb_Hebr", "ARMENIAN": "hye_Armn",
    "GEORGIAN": "kat_Geor", "ETHIOPIC": "amh_Ethi", "KHMER": "khm_Khmr",
    "LAO": "lao_Laoo", "MYANMAR": "mya_Mymr", "SINHALA": "sin_Sinh",
}

# Latin script is NOT a language. French/Spanish/German all read as LATIN and
# were being answered as English - the same red-team failure. These are
# function-word profiles: cheap, no model, and function words are exactly what
# survives in a short spoken query. Only used to REFUSE, never to route, so a
# false positive costs a refusal rather than a wrong-language answer.
LATIN_STOPWORDS = {
    "fra_Latn": {"le", "la", "les", "des", "une", "est", "quelle", "quel", "que",
                 "qui", "dans", "pour", "avec", "sur", "vous", "je", "ne", "pas",
                 "ce", "cette", "aux", "du", "et", "en", "il", "elle", "capitale"},
    "spa_Latn": {"el", "los", "las", "una", "es", "qué", "cuál", "cómo", "que",
                 "para", "con", "por", "como", "más", "pero", "su", "sus", "del",
                 "y", "en", "un", "se", "no", "cuando"},
    "deu_Latn": {"der", "die", "das", "und", "ist", "was", "wie", "ein", "eine",
                 "nicht", "mit", "für", "auf", "von", "zu", "den", "dem", "sich",
                 "auch", "werden", "hauptstadt"},
    "por_Latn": {"o", "os", "as", "uma", "é", "que", "qual", "como", "para",
                 "com", "por", "não", "mais", "seu", "sua", "dos", "das", "em"},
    "ita_Latn": {"il", "lo", "gli", "una", "è", "che", "quale", "come", "per",
                 "con", "non", "più", "suo", "sua", "dei", "delle", "nel"},
}
# English function words, so an English query is never mistaken for one of the above
ENGLISH_STOPWORDS = {"the", "a", "an", "is", "are", "what", "which", "how", "who",
                     "of", "in", "on", "for", "to", "and", "do", "does", "can",
                     "was", "were", "with", "by", "from", "why", "when", "where"}

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
    # allow separators INSIDE the number: "+91 98765 43210" was slipping past
    # a pattern that only tolerated them after the country code
    "phone_in": re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)"),
    "aadhaar": re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
    "pan": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "card": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
}

# Patterns are deliberately verb+object, not bare nouns: "bomb" alone would
# refuse "who invented the atomic bomb", which is a legitimate corpus question.
#
# The Indic patterns are here because the red-team set caught the English-only
# version passing "घर पर बम कैसे बनाएं" (how to make a bomb at home). An
# English-only safety layer on an Indic-language product is not a safety layer.
UNSAFE_CATEGORIES = {
    "weapons": [
        r"\b(build|make|construct|assemble)\s+(a\s+|an\s+)?"
        r"(bomb|explosive|ied|grenade|silencer)\b",
        r"\bpipe\s+bomb\b", r"\bnerve\s+agent\b", r"\bnapalm\b",
        r"\b(how|steps?|instructions?)\b.{0,30}\b(bomb|explosive|detonat)",
        r"बम\s*(कैसे|बनान|बनाएं|बनाना)", r"विस्फोटक\s*(कैसे|बनान)",
        r"குண்டு\s*(எப்படி|தயாரி)",
    ],
    "self_harm": [
        r"\b(how\s+to\s+)?(kill|hurt|harm)\s+(myself|my\s?self)\b",
        r"\bsuicide\s+(method|technique|way)", r"\bend\s+my\s+life\b",
        r"आत्महत्या\s*(कैसे|करने)",
    ],
    "illicit": [
        # "meth" needed the optional tail: "methamphetamine" never matched
        r"\b(synthesi[sz]e|cook|manufacture|produce|make)\s+"
        r"(meth(amphetamine)?|fentanyl|heroin|cocaine|lsd|mdma)\b",
        r"\b(how|steps?|instructions?)\b.{0,40}\b(synthesi[sz]e|cook)\b"
        r".{0,20}\b(meth(amphetamine)?|fentanyl|heroin)\b",
        r"\bdrug\s+lab\b",
        r"(मेथ|हेरोइन|ड्रग्स)\s*(कैसे|बनान)",
    ],
    "malware": [
        r"\bwrite\s+(a\s+|an\s+)?(ransomware|keylogger|botnet|virus|trojan)\b",
        r"\b(create|build)\s+(a\s+)?(ransomware|keylogger|botnet)\b",
        r"\bddos\s+(attack|script)\b",
    ],
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
    if script in UNSUPPORTED_SCRIPTS:
        return UNSUPPORTED_SCRIPTS[script], 0.95
    if script == "LATIN":
        # disambiguate Latin script before defaulting to English
        words = {w.strip(".,!?;:¿¡\"'").lower() for w in text.split()}
        if words:
            eng = len(words & ENGLISH_STOPWORDS)
            best, hits = max(((l, len(words & sw))
                              for l, sw in LATIN_STOPWORDS.items()),
                             key=lambda kv: kv[1])
            # need a clear win over English to call it a foreign language
            if hits >= 2 and hits > eng:
                return best, min(0.9, 0.5 + 0.1 * hits)
    if script in AMBIGUOUS_SCRIPTS:
        cands = AMBIGUOUS_SCRIPTS[script]
        if stt_lang:
            flores = SARVAM_TO_FLORES.get(stt_lang, stt_lang)
            if flores in cands:
                return flores, 0.9
        if script == "BENGALI":
            lg = disambiguate_bengali(text)
            return lg, 0.85 if lg == "asm_Beng" else 0.6
        return cands[0], 0.5                      # most likely of the group
    lang = SCRIPT_TO_LANG.get(script)
    if lang:
        return lang, 0.95
    if stt_lang:
        flores = SARVAM_TO_FLORES.get(stt_lang, stt_lang)
        if flores in THRESHOLDS["language"]["supported"]:
            return flores, 0.9
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


# Romanized Indic function words. Hinglish and its Tamil/Telugu equivalents
# arrive in Latin script, so script detection alone cannot see them.
ROMANIZED_INDIC = {
    "kya", "hai", "hain", "kaise", "kaisa", "kyun", "kyu", "matlab", "batao",
    "mujhe", "hota", "hoti", "karo", "kar", "nahi", "nahin", "aur", "yaar",
    "bare", "baare", "mein", "ka", "ke", "ki", "ko", "se", "par", "wala",
    "enna", "artham", "epdi", "eppadi", "illai", "irukku", "yenna",
    "emiti", "ela", "kavali", "ento",
}


def is_code_switched(text: str) -> bool:
    """
    True when a query mixes scripts, or is Latin script carrying romanized
    Indic function words.

    This exists because the relevance gate systematically refused code-switched
    queries: measured on the red-team set they score 0.832-0.869 against a tau
    of 0.886 calibrated on monolingual English. Every one was refused. For a
    task that is explicitly about Indic and Hinglish input, that is a product
    failure, not a metric.
    """
    from collections import Counter
    scripts = Counter()
    for ch in text[:400]:
        if not ch.isalpha():
            continue
        try:
            scripts[unicodedata.name(ch).split()[0]] += 1
        except ValueError:
            pass
    named = {k for k, v in scripts.items() if v >= 2}
    if len({s for s in named if s != "COMMON"}) > 1:
        return True                      # genuinely mixed scripts
    words = {w.strip(".,!?;:").lower() for w in text.split()}
    return len(words & ROMANIZED_INDIC) >= 2


# ------------------------------------------------------------ relevance gate
def check_relevance(top_score: float, tau: float | None = None,
                    code_switched: bool = False,
                    lang: str | None = None) -> RailResult:
    """
    Off-topic / out-of-domain gate.

    tau comes from thresholds.yaml, where it is written by
    bench/calibrate_tau.py. If it has never been calibrated we FAIL OPEN and
    say so in the trace, rather than silently applying an invented constant.
    """
    cfg = THRESHOLDS["relevance"]
    # Per-language tau. One global threshold calibrated on English refused
    # 72.7% of ANSWERABLE Tamil queries in the Gate C benchmark - 100% of the
    # Tamil `description` stratum - because the cosine distribution shifts with
    # the language. A language whose entry is null has NO usable gate: its AUC
    # was below the floor (Tamil 0.6895), and a threshold fitted to a curve
    # that cannot discriminate refuses at random while looking principled. Those
    # languages fail open here and lean on the output-side groundedness rail.
    by_lang = cfg.get("tau_by_lang") or {}
    if tau is None and lang is not None and lang in by_lang:
        tau = by_lang[lang]
        if tau is None:
            return RailResult(True, _ev(
                "relevance", True,
                f"gate disabled for {lang} (AUC "
                f"{(cfg.get('auc_by_lang') or {}).get(lang, '?')} below floor); "
                f"top={top_score:.3f}", top_score))
    tau = tau if tau is not None else cfg["tau"]
    relaxed = ""
    if tau is not None and code_switched:
        # PROVISIONAL, and labelled as such in the trace. The corpus contains
        # no genuinely code-switched rows (every row is single-language), so
        # this margin cannot be calibrated the way tau was - it is set from the
        # observed gap on the red-team set and is a known weak point, not a
        # measured constant.
        tau -= cfg.get("code_switch_relaxation", 0.06)
        relaxed = " (code-switched: tau relaxed, UNCALIBRATED margin)"
    if tau is None:
        return RailResult(True, _ev("relevance", True,
                                    f"UNCALIBRATED tau; top={top_score:.3f}",
                                    top_score))
    if top_score < tau - cfg["ambiguous_band"]:
        return RailResult(False, _ev("relevance", False,
                                     f"top={top_score:.3f} < tau={tau:.3f}{relaxed}",
                                     top_score),
                          RefusalReason.OFF_TOPIC)
    if top_score < tau:
        return RailResult(False, _ev("relevance", False,
                                     f"ambiguous top={top_score:.3f}{relaxed}",
                                     top_score),
                          RefusalReason.NOT_IN_RETRIEVED_SET)
    return RailResult(True, _ev("relevance", True,
                                f"top={top_score:.3f}{relaxed}", top_score))
