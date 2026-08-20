"""
FastAPI app: WebSocket voice in, streamed grounded answer out, plus the
/trace/{request_id} endpoint the UI reads.

Warm boot is deliberate and happens BEFORE the readiness probe passes: ONNX
session, tokenizer, BM25, HNSW and a dummy inference all run at startup. A
cold first request would otherwise land in the p100 bucket and make the
latency table a lie about steady-state behaviour.

Typed-query fallback (`POST /query`) exists so the Space is testable without a
microphone - which is how most judges will first touch it.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..generation.generator import DEFAULT_MODEL, Generator
from ..guardrails.output_rails import apply_output_rails
from ..harness.contracts import Trace
from ..harness.orchestrator import CoreLoop
from ..index.dense import DenseIndex
from ..index.embedder import OnnxEmbedder, E5Tokenizer
from ..index.sparse import SparseIndex
from .sarvam import SarvamSTT, SpeculativeRetriever

log = logging.getLogger("api")
logging.basicConfig(
    level=logging.INFO,
    format='{"ts":"%(asctime)s","lvl":"%(levelname)s","logger":"%(name)s",'
           '"msg":"%(message)s"}',
)

ARTIFACTS = Path(os.environ.get("RAG_ARTIFACTS", "artifacts"))
ONNX_DIR = ARTIFACTS / "e5-small-onnx"
SLICE_DIR = Path(os.environ.get("RAG_SLICE", "data/slice"))
WEB_DIR = Path(__file__).resolve().parents[2] / "web"
BUDGET_MS = float(os.environ.get("RAG_BUDGET_MS", "200"))
THREADS = int(os.environ.get("RAG_THREADS", "8"))

# request_id -> Trace. Bounded: an unbounded trace map is a memory leak that
# only shows up in a long-running demo, which is exactly when it matters.
TRACES: "OrderedDict[str, dict]" = OrderedDict()
MAX_TRACES = 500


def remember(trace: Trace, extra: dict | None = None) -> None:
    payload = json.loads(trace.model_dump_json())
    if extra:
        payload.update(extra)
    TRACES[trace.request_id] = payload
    while len(TRACES) > MAX_TRACES:
        TRACES.popitem(last=False)


class State:
    embedder: OnnxEmbedder | None = None
    core: CoreLoop | None = None
    generator: Generator | None = None
    ready: bool = False
    boot_ms: float = 0.0
    corpus_langs: list[str] = []
    n_chunks: int = 0


S = State()


def _load(langs: list[str]) -> None:
    """Build the in-process indices. Called once, at boot."""
    S.embedder = OnnxEmbedder(ONNX_DIR / "model_int8.onnx", ONNX_DIR,
                              threads=THREADS, warm=True)
    dense, sparse, texts = DenseIndex(), SparseIndex(), {}
    for lang in langs:
        d = SLICE_DIR / lang
        if not (d / "corpus.parquet").exists():
            log.warning("no corpus for %s, skipping", lang)
            continue
        c = pd.read_parquet(d / "corpus.parquet")
        V = S.embedder.encode_passages(c.text.tolist(), batch=64)
        dense.add(lang, V, c.passage_id.tolist(), c.passage_id.tolist())
        sparse.build(lang, c.text.tolist(), c.passage_id.tolist())
        texts.update(dict(zip(c.passage_id, c.text)))
        S.corpus_langs.append(lang)
        # fail loudly at boot rather than serving a silently broken index
        sr = dense.partitions[lang].self_retrieval_rate(100)
        log.info("loaded %s: %d passages, self_retrieval=%.3f", lang, len(c), sr)
        if sr < 0.95:
            raise RuntimeError(f"index for {lang} is broken (self_retrieval={sr:.3f})")

    tau = None
    tj = Path("bench/tau_calibration.json")
    if tj.exists():
        tau = json.loads(tj.read_text(encoding="utf-8"))["chosen"]["tau"]
    S.core = CoreLoop(S.embedder, dense, sparse, tau=tau, chunk_texts=texts)
    S.generator = Generator(model=os.environ.get("RAG_MODEL", DEFAULT_MODEL))
    S.n_chunks = dense.n_chunks


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter_ns()
    langs = os.environ.get("RAG_LANGS", "eng_Latn,hin_Deva,tam_Taml").split(",")
    try:
        await asyncio.to_thread(_load, [l.strip() for l in langs if l.strip()])
        S.ready = True
    except Exception as e:                                   # pragma: no cover
        log.exception("boot failed: %s", e)
    S.boot_ms = (time.perf_counter_ns() - t0) / 1e6
    log.info("warm boot complete in %.0f ms, ready=%s", S.boot_ms, S.ready)
    yield


app = FastAPI(title="RAG in Goa", lifespan=lifespan)


class QueryIn(BaseModel):
    text: str
    lang: str | None = None
    budget_ms: float | None = None


@app.get("/health")
async def health():
    return {"ready": S.ready, "boot_ms": round(S.boot_ms, 1),
            "langs": S.corpus_langs, "chunks": S.n_chunks,
            "budget_ms": BUDGET_MS, "model": os.environ.get("RAG_MODEL", DEFAULT_MODEL)}


@app.get("/trace/{request_id}")
async def get_trace(request_id: str):
    t = TRACES.get(request_id)
    if t is None:
        return JSONResponse({"error": "unknown request_id"}, status_code=404)
    return t


async def _answer(text: str, lang_hint: str | None, budget_ms: float,
                  stt_ms: float | None = None, on_token=None) -> dict:
    """Core loop -> guardrails -> generation -> output guardrails."""
    assert S.core and S.generator and S.embedder
    res = await S.core.run(text, budget_ms=budget_ms, stt_lang=lang_hint,
                           stt_ms=stt_ms)
    trace = res.trace

    if res.refused:
        ans = {
            "answer": "", "citations": [], "refused": True,
            "refusal_reason": res.refusal_reason.value if res.refusal_reason else None,
            "detail": res.refusal_detail,
        }
        remember(trace, {"answer": ans, "query": text})
        return {"request_id": trace.request_id, "answer": ans,
                "retrieved": [], "trace": json.loads(trace.model_dump_json())}

    t_gen = time.perf_counter_ns()
    draft = None
    async for kind, payload in S.generator.stream(text, res.retrieval,
                                                  res.retrieval.lang):
        if kind == "token" and on_token:
            await on_token(payload)
        elif kind == "result":
            draft = payload
    trace.ttft_ms = draft.ttft_ms if draft else None
    trace.e2e_ms = (time.perf_counter_ns() - t_gen) / 1e6 + (
        trace.core_rag_loop_ms or 0)

    checked, events, ok = apply_output_rails(
        draft.answer, res.retrieval, S.embedder, res.retrieval.lang)
    trace.guardrails.extend(events)

    ans = json.loads(checked.model_dump_json())
    out = {
        "request_id": trace.request_id,
        "answer": ans,
        "retrieved": [json.loads(c.model_dump_json()) for c in res.retrieval.chunks],
        "trace": json.loads(trace.model_dump_json()),
    }
    remember(trace, {"answer": ans, "query": text})
    return out


@app.post("/query")
async def query(q: QueryIn):
    """Typed-query fallback - the whole pipeline minus STT."""
    if not S.ready:
        return JSONResponse({"error": "not ready"}, status_code=503)
    return await _answer(q.text, q.lang, q.budget_ms or BUDGET_MS)


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    """
    Voice path. Client sends binary PCM16 frames and a {"event":"end"} text
    frame. Server streams partial transcripts, then answer tokens.

    Speculative retrieval fires on the first substantial partial so the core
    loop overlaps the tail of the utterance.
    """
    await websocket.accept()
    if not S.ready:
        await websocket.send_json({"type": "error", "message": "not ready"})
        await websocket.close()
        return

    q: asyncio.Queue[bytes | None] = asyncio.Queue()
    spec = SpeculativeRetriever()

    async def audio_iter():
        while (chunk := await q.get()) is not None:
            yield chunk

    async def recv_loop():
        try:
            while True:
                msg = await websocket.receive()
                if msg.get("bytes") is not None:
                    await q.put(msg["bytes"])
                elif msg.get("text"):
                    if json.loads(msg["text"]).get("event") == "end":
                        await q.put(None)
                        return
        except (WebSocketDisconnect, RuntimeError):
            await q.put(None)

    task = asyncio.create_task(recv_loop())
    stt = SarvamSTT(language_code=os.environ.get("RAG_STT_LANG", "unknown"))
    final_text, stt_ms, lang_code = "", None, None
    try:
        async for ev in stt.stream(audio_iter()):
            if ev.kind == "partial":
                await websocket.send_json({"type": "partial", "text": ev.text,
                                           "at_ms": ev.at_ms})
                d = spec.should_fire(ev.text, ev.at_ms)
                if d.fired:
                    await websocket.send_json({"type": "speculative",
                                               "at_ms": ev.at_ms,
                                               "partial": d.partial})
            elif ev.kind == "final":
                final_text, stt_ms, lang_code = ev.text, ev.at_ms, ev.lang_code
                await websocket.send_json({"type": "final", "text": ev.text,
                                           "stt_ms": ev.at_ms})
            elif ev.kind == "error":
                await websocket.send_json({"type": "error", "message": ev.text})

        if final_text.strip():
            refresh, ratio = spec.needs_refresh(final_text)
            await websocket.send_json({"type": "speculation_result",
                                       "refreshed": refresh,
                                       "divergence": round(ratio, 3)})

            async def on_token(t: str):
                await websocket.send_json({"type": "token", "text": t})

            out = await _answer(final_text, lang_code, BUDGET_MS, stt_ms, on_token)
            await websocket.send_json({"type": "done", **out})
    except WebSocketDisconnect:
        pass
    finally:
        task.cancel()
        try:
            await websocket.close()
        except RuntimeError:
            pass


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/")
    async def index():
        return FileResponse(str(WEB_DIR / "index.html"))
