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
        f"Instruction:\n"
        f"1. Determine if the context passages directly and explicitly contain the answer to the question.\n"
        f"2. If the context passages do NOT contain the answer, return:\n"
        f"   {{\"grounded\": false, \"refused\": true, \"answer\": \"The provided documents don't contain information about this.\"}}\n"
        f"3. If the context passages DO contain the answer, synthesize the answer using ONLY facts directly from those passages:\n"
        f"   {{\"grounded\": true, \"refused\": false, \"answer\": \"<factual answer>\"}}\n"
    )

    from groq import Groq
    import os

    api_key = os.environ.get("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    model_name = os.environ.get("RAG_MODEL", "openai/gpt-oss-120b")

    try:
        resp = client.chat.completions.create(
            model=model_name,
            temperature=0,
            max_tokens=400,
            messages=[
                {"role": "system", "content": "You are a strict zero-hallucination assistant. Return your response as a JSON object."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        raw = resp.choices[0].message.content or "{}"
        data = json.loads(raw)
        ans_text = str(data.get("answer", "")).strip()
        is_grounded_flag = bool(data.get("grounded", True))
        is_refused_flag = bool(data.get("refused", False))
        
        # Check text signals
        lower_ans = ans_text.lower()
        refusal_phrases = ("don't contain", "do not contain", "not contain", "does not contain", "no information", "insufficient evidence", "cannot be answered")
        has_refusal_phrase = any(p in lower_ans for p in refusal_phrases)

        # Check if the answer has factual grounding in the context passages
        ctx_lower = context_text.lower()
        ans_words = [w for w in lower_ans.split() if len(w) > 4]
        overlap_ratio = (sum(1 for w in ans_words if w in ctx_lower) / len(ans_words)) if ans_words else 1.0

        is_refused = is_refused_flag or (not is_grounded_flag) or has_refusal_phrase or (overlap_ratio < 0.35)
        elapsed = (time.perf_counter() - t0) * 1000.0

        if is_refused:
            return GeneratedAnswer(
                text="The provided documents don't contain information about this.",
                grounded=False,
                generation_ms=elapsed,
                model=model_name,
            )

        return GeneratedAnswer(
            text=ans_text,
            grounded=True,
            generation_ms=elapsed,
            model=model_name,
        )
    except Exception as exc:
        print("[generator exception]:", type(exc), exc)

    # Safe fallback
    fb = _extractive_fallback(query, rs, "eng_Latn")
    return GeneratedAnswer(
        text=fb.answer,
        grounded=True,
        generation_ms=(time.perf_counter() - t0) * 1000.0,
        model="extractive_fallback",
    )
