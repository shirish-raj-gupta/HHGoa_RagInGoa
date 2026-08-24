"""
Generator interface implementation for rag-local-eval-loop.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from src.config import load_dotenv
from src.generation.generator import Generator, _extractive_fallback
from src.harness.contracts import RetrievedChunk, RetrievalSet, new_request_id

load_dotenv()


@dataclass
class GeneratedAnswer:
    text: str
    grounded: bool
    generation_ms: float
    model: str


_GEN: Generator | None = None


def _get_gen() -> Generator:
    global _GEN
    if _GEN is None:
        _GEN = Generator()
    return _GEN


def generate_answer(query: str, results: list) -> GeneratedAnswer:
    t0 = time.perf_counter()
    if not results:
        return GeneratedAnswer(
            text="The provided documents don't contain information about this.",
            grounded=False,
            generation_ms=0.1,
            model="openai/gpt-oss-120b",
        )

    chunks = [
        RetrievedChunk(
            passage_id=getattr(r, "source", str(i)),
            chunk_id=f"chunk_{i}",
            text=getattr(r, "text", ""),
            score=1.0 / (i + 1),
            rank=i + 1,
            lang="eng_Latn",
            chunk_strategy="passage_atomic",
        )
        for i, r in enumerate(results)
    ]
    rs = RetrievalSet(
        request_id=new_request_id(),
        query=query,
        lang="eng_Latn",
        chunks=chunks,
        relevance_score=0.85,
    )

    gen = _get_gen()

    # Synchronous wrapper around the async generator stream
    async def _run():
        final_res = None
        async for kind, payload in gen.stream(query, rs, "eng_Latn"):
            if kind == "result":
                final_res = payload
        return final_res

    try:
        res = asyncio.run(_run())
        if res and res.answer:
            ans = res.answer
            elapsed = (time.perf_counter() - t0) * 1000.0
            return GeneratedAnswer(
                text=ans.answer or _extractive_fallback(query, rs, "eng_Latn").answer,
                grounded=not ans.refused,
                generation_ms=elapsed,
                model=res.model or "openai/gpt-oss-120b",
            )
    except Exception:
        pass

    # Safe fallback
    fb = _extractive_fallback(query, rs, "eng_Latn")
    return GeneratedAnswer(
        text=fb.answer,
        grounded=True,
        generation_ms=(time.perf_counter() - t0) * 1000.0,
        model="extractive_fallback",
    )
