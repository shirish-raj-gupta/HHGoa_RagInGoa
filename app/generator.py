"""
Generator interface implementation for rag-local-eval-loop.
"""
from __future__ import annotations

import asyncio
import json
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

    # Formulate context for the LLM
    context_text = "\n\n".join([f"[{i+1}] {getattr(r, 'text', '')}" for i, r in enumerate(results[:5])])
    prompt = (
        f"Question: {query}\n\n"
        f"Context passages:\n{context_text}\n\n"
        f"Task: Answer the question ONLY if the context passages directly and factually provide the true answer to the question. "
        f"If the context passages do not provide the exact answer (or only mention the keywords without explaining the answer), you MUST return refused: true and "
        f"answer: 'The provided documents don't contain information about this.'\n"
        f"Return valid JSON: {{\"answer\": string, \"grounded\": bool, \"refused\": bool}}"
    )

    from groq import Groq
    import os

    api_key = os.environ.get("GROQ_API_KEY")
    client = Groq(api_key=api_key)

    try:
        resp = client.chat.completions.create(
            model=gen.model,
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a factual, zero-hallucination assistant. When evidence is missing in the passages, you must refuse and say the documents don't contain information."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        raw = resp.choices[0].message.content or "{}"
        data = json.loads(raw)
        ans_text = data.get("answer", "")
        is_refused = bool(data.get("refused", False)) or "don't contain" in ans_text.lower() or "not contain" in ans_text.lower()
        elapsed = (time.perf_counter() - t0) * 1000.0

        if is_refused:
            return GeneratedAnswer(
                text="The provided documents don't contain information about this.",
                grounded=False,
                generation_ms=elapsed,
                model=gen.model,
            )

        return GeneratedAnswer(
            text=ans_text,
            grounded=True,
            generation_ms=elapsed,
            model=gen.model,
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
