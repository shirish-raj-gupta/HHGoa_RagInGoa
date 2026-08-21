"""
Sarvam realtime streaming STT client.

Endpoint and message schema verified against the live docs on 2026-08-20, not
recalled:

    wss://api.sarvam.ai/speech-to-text-realtime/ws
    model        saaras:v3-realtime   (the only model on this endpoint)
    auth         API-SUBSCRIPTION-KEY header, or the
                 "api-subscription-key.<key>" websocket subprotocol
    encoding     linear16 | linear32 | mulaw | alaw
    sample_rate  8000 | 16000
    stream_type  fast | balanced | simulated
    endpointing  vad | manual
    mode         transcribe | translate | verbatim | translit | codemix

    client -> {"event": "audio_input", "audio": "<base64>"}
              speech_start / speech_end / flush / ping / end / config.update
    server -> session.begin | vad.speech_start | vad.speech_end
              transcript.partial | transcript.final | config.updated
              pong | session.end | error

The decisive property is `transcript.partial`. It arrives DURING the utterance,
which is what lets retrieval fire before the user stops speaking (see
speculative_retrieval below). Without verified partials that trick is fiction,
which is exactly why this was checked rather than assumed.

The batch POST /speech-to-text endpoint is deliberately not used: it is
upload-and-wait, so it cannot overlap retrieval with speech at all.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable

import websockets

from ..config import get as cfg_get
from ..harness.stage import CircuitBreaker, RetryPolicy, UpstreamError

log = logging.getLogger("sarvam")

WS_URL = "wss://api.sarvam.ai/speech-to-text-realtime/ws"
MODEL = "saaras:v3-realtime"

# FLORES-200 (dataset) <-> BCP-47 (Sarvam). All 14 corpus languages are
# supported by the realtime endpoint; verified against the published list.
FLORES_TO_SARVAM = {
    "eng_Latn": "en-IN", "asm_Beng": "as-IN", "ben_Beng": "bn-IN",
    "guj_Gujr": "gu-IN", "hin_Deva": "hi-IN", "kan_Knda": "kn-IN",
    "mal_Mlym": "ml-IN", "mar_Deva": "mr-IN", "npi_Deva": "ne-IN",
    "ory_Orya": "or-IN", "pan_Guru": "pa-IN", "san_Deva": "sa-IN",
    "tam_Taml": "ta-IN", "tel_Telu": "te-IN", "urd_Arab": "ur-IN",
}
SARVAM_TO_FLORES = {v: k for k, v in FLORES_TO_SARVAM.items()}


@dataclass
class SttEvent:
    kind: str                 # "partial" | "final" | "vad" | "error" | "session"
    text: str = ""
    lang_code: str | None = None
    confidence: float | None = None
    at_ms: float = 0.0
    raw: dict = field(default_factory=dict)


class SarvamSTT:
    def __init__(self, api_key: str | None = None, *, language_code: str = "unknown",
                 sample_rate: int = 16000, stream_type: str = "fast",
                 mode: str = "transcribe", endpointing: str = "vad",
                 breaker: CircuitBreaker | None = None):
        self.api_key = api_key or cfg_get("SARVAM_API_KEY")
        self.language_code = language_code
        self.sample_rate = sample_rate
        self.stream_type = stream_type
        self.mode = mode
        self.endpointing = endpointing
        self.retry = RetryPolicy(max_retries=2, base_ms=100, max_ms=800)
        self.breaker = breaker or CircuitBreaker("sarvam_stt", threshold=4,
                                                 cooldown_s=30.0)

    # ------------------------------------------------------------------ url
    def _url(self) -> str:
        from urllib.parse import urlencode
        q = urlencode({
            "model": MODEL,
            "language_code": self.language_code,
            "encoding": "linear16",
            "sample_rate": str(self.sample_rate),
            "stream_type": self.stream_type,
            "endpointing": self.endpointing,
            "mode": self.mode,
        })
        return f"{WS_URL}?{q}"

    # --------------------------------------------------------------- stream
    async def stream(self, audio_chunks: AsyncIterator[bytes],
                     *, timeout_s: float = 30.0) -> AsyncIterator[SttEvent]:
        """
        Feed PCM16 chunks in, get SttEvents out. Partials arrive mid-utterance.

        The API key never appears in a URL - it goes in a header, so it cannot
        leak into a proxy log or a screenshot of the address bar.
        """
        if not self.api_key:
            yield SttEvent("error", text="SARVAM_API_KEY is not set")
            return
        if self.breaker.is_open:
            yield SttEvent("error", text="sarvam circuit open")
            return

        t0 = time.perf_counter_ns()

        def ms() -> float:
            return (time.perf_counter_ns() - t0) / 1e6

        last_exc: Exception | None = None
        for attempt in range(self.retry.max_retries + 1):
            try:
                async with websockets.connect(
                        self._url(),
                        additional_headers={"API-SUBSCRIPTION-KEY": self.api_key},
                        open_timeout=5, close_timeout=2,
                        max_size=4 * 1024 * 1024) as ws:

                    async def pump() -> None:
                        try:
                            async for chunk in audio_chunks:
                                await ws.send(json.dumps({
                                    "event": "audio_input",
                                    "audio": base64.b64encode(chunk).decode(),
                                }))
                            await ws.send(json.dumps({"event": "end"}))
                        except Exception as e:                # pragma: no cover
                            log.debug("audio pump ended: %s", e)

                    task = asyncio.create_task(pump())
                    try:
                        while True:
                            raw = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
                            msg = json.loads(raw)
                            ev = msg.get("event") or msg.get("type") or ""

                            if ev == "transcript.partial":
                                yield SttEvent("partial", msg.get("text", ""),
                                               msg.get("language_code"),
                                               msg.get("language_probability"),
                                               ms(), msg)
                            elif ev == "transcript.final":
                                yield SttEvent("final", msg.get("text", ""),
                                               msg.get("language_code"),
                                               msg.get("language_probability"),
                                               ms(), msg)
                            elif ev in ("vad.speech_start", "vad.speech_end"):
                                yield SttEvent("vad", ev, at_ms=ms(), raw=msg)
                            elif ev == "session.begin":
                                yield SttEvent("session", ev, at_ms=ms(), raw=msg)
                            elif ev == "session.end":
                                yield SttEvent("session", ev, at_ms=ms(), raw=msg)
                                break
                            elif ev == "error":
                                self.breaker.record(False)
                                yield SttEvent("error", str(msg), at_ms=ms(), raw=msg)
                                break
                    finally:
                        task.cancel()
                self.breaker.record(True)
                return
            except (websockets.WebSocketException, asyncio.TimeoutError, OSError) as e:
                last_exc = e
                self.breaker.record(False)
                if attempt < self.retry.max_retries:
                    await asyncio.sleep(self.retry.delay_ms(attempt) / 1000)
                    continue
        yield SttEvent("error", text=f"sarvam unreachable: {last_exc}")


# --------------------------------------------------------- speculative layer
def edit_distance(a: str, b: str) -> int:
    """Levenshtein, iterative, O(min(len)) memory."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@dataclass
class SpeculationDecision:
    fired: bool
    reason: str
    at_ms: float = 0.0
    partial: str = ""


class SpeculativeRetriever:
    """
    Fire retrieval on a PARTIAL transcript, then refresh only if the final
    transcript diverged enough to matter.

    This is the single biggest end-to-end latency lever available here: it
    hides the entire core RAG loop inside the time the user is still speaking.
    It is only honest because `transcript.partial` was verified to exist.

    `divergence_ratio` is normalized edit distance. Below the threshold the
    speculative result is kept; above it, retrieval re-runs on the final text.
    Re-running is correctness-preserving - speculation can only ever save time,
    never change the answer.
    """

    def __init__(self, min_chars: int = 12, min_words: int = 3,
                 divergence_ratio: float = 0.25):
        self.min_chars = min_chars
        self.min_words = min_words
        self.divergence_ratio = divergence_ratio
        self._fired_on: str | None = None

    def should_fire(self, partial: str, at_ms: float = 0.0) -> SpeculationDecision:
        """Fire once, when the partial looks like a real query rather than noise."""
        if self._fired_on is not None:
            return SpeculationDecision(False, "already_fired", at_ms, partial)
        t = (partial or "").strip()
        if len(t) < self.min_chars:
            return SpeculationDecision(False, "too_short", at_ms, t)
        if len(t.split()) < self.min_words:
            return SpeculationDecision(False, "too_few_words", at_ms, t)
        self._fired_on = t
        return SpeculationDecision(True, "fired", at_ms, t)

    def needs_refresh(self, final: str) -> tuple[bool, float]:
        """After the final lands, decide whether the speculative result stands."""
        if self._fired_on is None:
            return True, 1.0
        f, p = (final or "").strip(), self._fired_on
        if not f:
            return True, 1.0
        ratio = edit_distance(f.lower(), p.lower()) / max(len(f), 1)
        return ratio > self.divergence_ratio, ratio

    def reset(self) -> None:
        self._fired_on = None


# ------------------------------------------------------- fallback transcriber
WHISPER_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
# whisper-large-v3, NOT the turbo variant. Measured on the same Hindi
# utterance: turbo returned "نگم کیا ہے؟" - correct words, URDU SCRIPT - while
# large-v3 returned "निगम क्या है?" in Devanagari. Script drives language
# identification, which drives which index partition is searched, so a
# wrong-script transcript silently routes the whole query to the wrong corpus.
# Turbo is ~100ms faster and unusable for this pipeline.
WHISPER_MODEL = "whisper-large-v3"


def pcm16_to_wav(pcm: bytes, sample_rate: int = 16000) -> bytes:
    """Whisper wants a container; the WS path speaks raw PCM."""
    import io
    import wave
    buf = io.BytesIO()
    w = wave.open(buf, "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sample_rate)
    w.writeframes(pcm)
    w.close()
    return buf.getvalue()


class WhisperFallback:
    """
    Batch STT, used only when Sarvam's circuit breaker is open.

    Sarvam stays primary: it is the vendor the task specifies, and it streams,
    which is what makes speculative retrieval possible at all. This path is
    strictly the degraded one - it cannot emit partials, so the UI shows no
    live transcript and speculation is skipped. Losing a feature is the right
    trade against losing the request.

    Post-audio latency is comparable: Sarvam's final landed ~806ms after the
    audio ended (2,886ms wall on 2.08s of speech); Whisper returns in ~770ms
    on the whole file.
    """

    def __init__(self, api_key: str | None = None, model: str = WHISPER_MODEL):
        self.api_key = api_key or cfg_get("GROQ_API_KEY")
        self.model = model

    async def transcribe(self, pcm: bytes, *, sample_rate: int = 16000,
                         lang_hint: str | None = None) -> SttEvent:
        if not self.api_key:
            return SttEvent("error", text="GROQ_API_KEY is not set")
        import httpx
        data = {"model": self.model, "response_format": "json"}
        if lang_hint:
            # Whisper wants ISO-639-1; Sarvam codes are BCP-47 like "hi-IN"
            data["language"] = lang_hint.split("-")[0]
        t0 = time.perf_counter_ns()
        try:
            async with httpx.AsyncClient(timeout=60) as c:
                r = await c.post(
                    WHISPER_URL,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    files={"file": ("audio.wav", pcm16_to_wav(pcm, sample_rate),
                                    "audio/wav")},
                    data=data)
            ms = (time.perf_counter_ns() - t0) / 1e6
            if r.status_code != 200:
                return SttEvent("error", text=f"whisper {r.status_code}: "
                                              f"{r.text[:160]}", at_ms=ms)
            return SttEvent("final", (r.json().get("text") or "").strip(),
                            lang_code=lang_hint, at_ms=ms,
                            raw={"provider": "groq_whisper", "model": self.model})
        except Exception as e:
            return SttEvent("error", text=f"whisper unreachable: {e}",
                            at_ms=(time.perf_counter_ns() - t0) / 1e6)


async def transcribe_with_fallback(pcm: bytes, *, language_code: str = "unknown",
                                   sample_rate: int = 16000,
                                   stt: SarvamSTT | None = None
                                   ) -> tuple[SttEvent, str]:
    """
    Sarvam first; Whisper only if Sarvam's breaker is open or it errors.

    Returns (event, provider) so the trace records WHICH transcriber answered -
    a fallback that is invisible in the trace is indistinguishable from the
    primary working.
    """
    # Sarvam speaks BCP-47 ("hi-IN"); the rest of this codebase speaks
    # FLORES-200 ("hin_Deva"). Passing FLORES straight through got
    # "Unsupported language_code 'hin_Deva'" and silently demoted every request
    # to the fallback transcriber - a working fallback masking a broken primary
    # is the worst version of this bug, because nothing looks wrong.
    sarvam_code = FLORES_TO_SARVAM.get(language_code, language_code)
    stt = stt or SarvamSTT(language_code=sarvam_code, sample_rate=sample_rate)

    if not stt.breaker.is_open:
        async def one_shot():
            step = int(sample_rate * 0.04) * 2
            for i in range(0, len(pcm), step):
                yield pcm[i:i + step]

        final: SttEvent | None = None
        err: SttEvent | None = None
        async for ev in stt.stream(one_shot()):
            if ev.kind == "final":
                final = ev
            elif ev.kind == "error":
                err = ev
        if final and final.text.strip():
            return final, "sarvam"
        log.warning("sarvam produced no transcript (%s); falling back",
                    err.text[:120] if err else "no final")

    ev = await WhisperFallback().transcribe(pcm, sample_rate=sample_rate,
                                            lang_hint=sarvam_code)
    return ev, "groq_whisper"
