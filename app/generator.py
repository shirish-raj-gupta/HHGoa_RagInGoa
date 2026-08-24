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
_GROQ_CLIENT = None


def _get_groq_client():
    global _GROQ_CLIENT
    if _GROQ_CLIENT is None:
        from groq import Groq
        import os
        api_key = os.environ.get("GROQ_API_KEY")
        _GROQ_CLIENT = Groq(api_key=api_key)
    return _GROQ_CLIENT


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
            model="openai/gpt-oss-20b",
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

    # Formulate context for the LLM
    context_text = "\n\n".join([f"[{i+1}] {getattr(r, 'text', '')}" for i, r in enumerate(results[:5])])
    prompt = (
        f"Question: {query}\n\n"
        f"Context passages:\n{context_text}\n\n"
        f"Instruction:\n"
        f"1. Determine if the context passages directly and explicitly contain the answer to the question.\n"
        f"2. If the context passages do NOT contain the answer, return JSON:\n"
        f"   {{\"grounded\": false, \"refused\": true, \"answer\": \"The provided documents don't contain information about this.\"}}\n"
        f"3. If the context passages DO contain the answer, return a concise 1-2 sentence factual answer in JSON:\n"
        f"   {{\"grounded\": true, \"refused\": false, \"answer\": \"<concise answer>\"}}\n"
        f"Respond with a single JSON object."
    )

    import os
    import re

    client = _get_groq_client()
    model_name = os.environ.get("RAG_MODEL", "openai/gpt-oss-120b")

    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                temperature=0,
                max_tokens=300,
                messages=[
                    {"role": "system", "content": "You are a concise zero-hallucination assistant. Always respond with a single valid JSON object."},
                    {"role": "user", "content": prompt}
                ],
            )
            raw = resp.choices[0].message.content or ""
            cleaned = re.sub(r'<thought>.*?</thought>', '', raw, flags=re.DOTALL).strip()
            match = re.search(r'\{.*\}', cleaned, flags=re.DOTALL)
            data = None
            if match:
                try:
                    data = json.loads(match.group(0))
                except Exception:
                    pass
            if data is None:
                try:
                    data = json.loads(cleaned)
                except Exception:
                    pass
            if not data:
                is_ref = "don't contain" in cleaned.lower() or "not contain" in cleaned.lower() or "no information" in cleaned.lower()
                ans_m = re.search(r'"answer"\s*:\s*"([^"]+)', cleaned)
                ans_str = ans_m.group(1) if ans_m else cleaned
                data = {"answer": ans_str, "grounded": not is_ref, "refused": is_ref}

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
            if attempt == 0:
                time.sleep(1.0)
            else:
                print(f"[generator API fallback]: {exc}")

    # Safe fallback with strict unanswerable refusal check
    fb = _extractive_fallback(query, rs, "eng_Latn")
    q_words = {w.lower() for w in re.findall(r"\w+", query) if len(w) > 3}
    ctx_words = {w.lower() for w in re.findall(r"\w+", context_text)}
    overlap_count = len(q_words & ctx_words)
    is_safe = (overlap_count >= 2) and (not fb.refused)

    if not is_safe:
        return GeneratedAnswer(
            text="The provided documents don't contain information about this.",
            grounded=False,
            generation_ms=(time.perf_counter() - t0) * 1000.0,
            model="safe_refusal",
        )

    return GeneratedAnswer(
        text=fb.answer,
        grounded=True,
        generation_ms=(time.perf_counter() - t0) * 1000.0,
        model="extractive_fallback",
    )
