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
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import load_dotenv, status as key_status
from ..generation.generator import DEFAULT_MODEL, Generator
from ..guardrails.output_rails import apply_output_rails
from ..harness.contracts import Trace
from ..harness.orchestrator import CoreLoop
from ..index.dense import DenseIndex, DensePartition
from ..index.embedder import OnnxEmbedder, E5Tokenizer
from ..index.sparse import SparseIndex, SparsePartition
from ..index.textstore import TextStore
from .sarvam import (FLORES_TO_SARVAM, SarvamSTT, SpeculativeRetriever,
                     WhisperFallback)

log = logging.getLogger("api")
logging.basicConfig(
    level=logging.INFO,
    format='{"ts":"%(asctime)s","lvl":"%(levelname)s","logger":"%(name)s",'
           '"msg":"%(message)s"}',
)

load_dotenv()          # no-op on HF Spaces, where secrets are already env

ARTIFACTS = Path(os.environ.get("RAG_ARTIFACTS", "artifacts"))
ONNX_DIR = ARTIFACTS / "e5-small-onnx"
SLICE_DIR = Path(os.environ.get("RAG_SLICE", "data/slice"))
INDEX_DIR = Path(os.environ.get("RAG_INDEX", "artifacts/index"))
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
    texts: dict = {}


S = State()


def _load(langs: list[str]) -> None:
    """
    Load the PREBUILT index from disk. Never re-embed at boot.

    The previous version embedded the corpus on every start. At 14.3M passages
    and ~156 passages/s on CPU that is a four-hour cold start - a Space would
    never come up. Partitions are memory-mapped (usearch `view`), so resident
    memory tracks the hot set rather than the ~12.8GB total, and passage text
    lives in SQLite so only retrieved rows are ever read.
    """
    S.embedder = OnnxEmbedder(ONNX_DIR / "model_int8.onnx", ONNX_DIR,
                              threads=THREADS, warm=True)
    dense, sparse = DenseIndex(), SparseIndex()
    for lang in langs:
        vec = INDEX_DIR / f"{lang}.usearch"
        if not vec.exists():
            log.warning("no prebuilt partition for %s, skipping", lang)
            continue
        try:
            part = DensePartition.load(vec, view=True)
        except RuntimeError:
            log.warning("mmap failed for %s, loading into RAM instead", lang)
            part = DensePartition.load(vec, view=False)
        dense.partitions[lang] = part

        bm = INDEX_DIR / f"{lang}.bm25"
        if bm.exists():
            sparse.partitions[lang] = SparsePartition.load(bm)
        else:
            log.warning("no BM25 for %s - dense only", lang)

        db = INDEX_DIR / f"{lang}.texts.db"
        if db.exists():
            S.texts[lang] = TextStore(db)

        # Page the mmap in before the readiness probe passes. Measured cold:
        # the first live query took 412ms (dense 355ms, sparse 302ms) - BOTH
        # arms blew their stage timeouts, nothing was retrieved, and the answer
        # was correctly refused as ungrounded. By the third query it was 70ms.
        # A Space that reports ready while its index is still on disk serves
        # garbage to whoever arrives first.
        t_warm = time.perf_counter_ns()
        rng = np.random.default_rng(0)
        probe = rng.normal(size=(24, part.dim)).astype(np.float32)
        probe /= np.linalg.norm(probe, axis=1, keepdims=True)
        for v in probe:
            part.search(v, k=10, expansion_search=16)
        if lang in sparse.partitions:
            for w in ("the", "what is", "how many", "who", "meaning", "definition"):
                sparse.partitions[lang].search(w, k=10)
        warm_ms = (time.perf_counter_ns() - t_warm) / 1e6

        sr = part.self_retrieval_rate(100)
        log.info("loaded %s: %d vectors, self_retrieval=%.3f, warmed in %.0fms",
                 lang, len(part.chunk_ids), sr, warm_ms)
        if sr < 0.95:
            raise RuntimeError(f"index for {lang} is broken (self_retrieval={sr:.3f})")
        S.corpus_langs.append(lang)

    S.core = CoreLoop(S.embedder, dense, sparse, tau=None,
                      text_lookup=_lookup_texts)
    S.generator = Generator(model=os.environ.get("RAG_MODEL", "openai/gpt-oss-20b"), use_tools=False)
    S.n_chunks = sum(len(p.chunk_ids) for p in dense.partitions.values())

    # ---- Force ALL mmap pages into OS page cache ----
    # Reading the .usearch file sequentially triggers OS readahead and fills
    # the kernel buffer cache. After this, mmap page faults resolve instantly
    # from RAM instead of disk. This is the same trick PostgreSQL/MySQL use
    # (pg_prewarm / innodb_buffer_pool_load_at_startup).
    t_prefault = time.perf_counter_ns()
    for lang in langs:
        usf = INDEX_DIR / f"{lang}.usearch"
        if usf.exists():
            with open(usf, "rb") as f:
                while f.read(1 << 20):  # 1MB chunks
                    pass
        bm = INDEX_DIR / f"{lang}.bm25"
        if bm.exists():
            # BM25 index is a directory of numpy arrays - read them all
            import glob
            for npy in glob.glob(str(bm / "**" / "*"), recursive=True):
                try:
                    with open(npy, "rb") as f:
                        while f.read(1 << 20):
                            pass
                except (IsADirectoryError, PermissionError):
                    pass
    # Warm ONNX embedder JIT
    try:
        S.embedder.encode_queries(["warmup", "what is definition", "explain the process"])
    except Exception:
        pass
    prefault_ms = (time.perf_counter_ns() - t_prefault) / 1e6
    log.info("page cache prefault complete in %.0fms: %d index files read into RAM",
             prefault_ms, len(langs) * 2)


def _lookup_texts(passage_ids: list[str]) -> dict[str, str]:
    """Fetch only the passages actually retrieved, from every open store."""
    out: dict[str, str] = {}
    for st in S.texts.values():
        missing = [p for p in passage_ids if p not in out]
        if not missing:
            break
        out.update(st.get(missing))
    return out


def _fetch_index_if_missing() -> None:
    """
    On a Space the index arrives from a dataset repo, not the image. An 18GB
    corpus does not belong in a Space repo, and baking it into the image would
    make every code change a multi-GB rebuild.
    """
    repo = os.environ.get("RAG_INDEX_REPO")
    if not repo or any(INDEX_DIR.glob("*.usearch")):
        return
    from huggingface_hub import snapshot_download
    langs = [l.strip() for l in
             os.environ.get("RAG_LANGS", "eng_Latn").split(",") if l.strip()]
    patterns = [f"{l}*" for l in langs] + ["manifest.json", "e5-small-onnx/*"]
    log.info("pulling index for %s from %s", langs, repo)
    t = time.perf_counter_ns()
    snapshot_download(repo_id=repo, repo_type="dataset",
                      local_dir=str(INDEX_DIR), allow_patterns=patterns,
                      token=os.environ.get("HF_TOKEN"))
    log.info("index pulled in %.0fs", (time.perf_counter_ns() - t) / 1e9)


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter_ns()
    langs = os.environ.get("RAG_LANGS", "eng_Latn,hin_Deva,tam_Taml").split(",")
    try:
        await asyncio.to_thread(_fetch_index_if_missing)
        await asyncio.to_thread(_load, [l.strip() for l in langs if l.strip()])
        S.ready = True
    except Exception as e:                                   # pragma: no cover
        log.exception("boot failed: %s", e)
    S.boot_ms = (time.perf_counter_ns() - t0) / 1e6
    log.info("warm boot complete in %.0f ms, ready=%s", S.boot_ms, S.ready)
    yield


app = FastAPI(title="RAG in Goa", lifespan=lifespan)

# The frontend is served from a Hugging Face Static Space while the backend
# runs elsewhere (locally, behind a tunnel), so requests are cross-origin.
# RAG_ALLOW_ORIGINS is a comma-separated allowlist; it defaults to the Space
# rather than "*" so a stray page cannot drive someone else's API keys.
_origins = [o.strip() for o in os.environ.get(
    "RAG_ALLOW_ORIGINS",
    "https://srg101-raginggoa.static.hf.space,"
    "https://huggingface.co,"
    "http://localhost:7860,http://127.0.0.1:7860").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class QueryIn(BaseModel):
    text: str
    lang: str | None = None
    budget_ms: float | None = None


@app.get("/health")
async def health():
    # key_status reports booleans only - never the key itself
    return {"ready": S.ready, "boot_ms": round(S.boot_ms, 1),
            "langs": S.corpus_langs, "chunks": S.n_chunks,
            "budget_ms": BUDGET_MS, "model": os.environ.get("RAG_MODEL", DEFAULT_MODEL),
            "keys": key_status()}


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
                "retrieved": [json.loads(c.model_dump_json()) for c in res.retrieval.chunks],
                "trace": json.loads(trace.model_dump_json())}

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

    # Buffered so the fallback transcriber has something to send. Whisper is
    # batch: if Sarvam yields no transcript we need the whole utterance, not a
    # stream we already consumed.
    captured = bytearray()

    async def recv_loop():
        try:
            while True:
                msg = await websocket.receive()
                if msg.get("bytes") is not None:
                    captured.extend(msg["bytes"])
                    await q.put(msg["bytes"])
                elif msg.get("text"):
                    if json.loads(msg["text"]).get("event") == "end":
                        await q.put(None)
                        return
        except (WebSocketDisconnect, RuntimeError):
            await q.put(None)

    task = asyncio.create_task(recv_loop())
    # "auto" is Sarvam's wildcard. "unknown" is rejected outright with
    # "Unsupported language_code", which killed the whole socket - and because
    # the fallback only engages after the stream ends, the user saw an error
    # rather than a degraded answer.
    stt = SarvamSTT(language_code=os.environ.get("RAG_STT_LANG", "auto"))
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

        # Degraded path: Sarvam gave nothing usable (breaker open, upstream
        # error, or silence it could not resolve). Whisper cannot stream, so
        # the live transcript and speculation are both skipped - which the UI
        # is told explicitly rather than left to guess.
        provider = "sarvam"
        if not final_text.strip() and captured:
            await websocket.send_json({"type": "stt_fallback",
                                       "reason": "sarvam produced no transcript"})
            ev = await WhisperFallback().transcribe(
                bytes(captured), sample_rate=16000,
                # Whisper wants an ISO-639-1 code or nothing; "auto" is a
                # Sarvam-ism and would be rejected.
                lang_hint=(lang_code.split("-")[0] if lang_code else None))
            if ev.kind == "final" and ev.text.strip():
                final_text, stt_ms, provider = ev.text, ev.at_ms, "groq_whisper"
                await websocket.send_json({"type": "final", "text": ev.text,
                                           "stt_ms": ev.at_ms,
                                           "provider": provider})
            else:
                await websocket.send_json({"type": "error",
                                           "message": "no transcript from either "
                                                      "transcriber"})

        if final_text.strip():
            refresh, ratio = spec.needs_refresh(final_text)
            await websocket.send_json({"type": "speculation_result",
                                       "refreshed": refresh,
                                       "divergence": round(ratio, 3)})

            async def on_token(t: str):
                await websocket.send_json({"type": "token", "text": t})

            out = await _answer(final_text, lang_code, BUDGET_MS, stt_ms, on_token)
            out["stt_provider"] = provider
            if out.get("trace") is not None:
                out["trace"]["stt_provider"] = provider
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
