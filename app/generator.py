"""
Generator interface implementation for rag-local-eval-loop.

Optimized for:
- Token efficiency (compact context to stay well below Groq TPM limits)
- 0% False Refusal Rate (accurately answer all answerable queries from context)
- 0% False Confidence Rate (properly refuse unanswerable queries)
- Sub-1500ms generation latency (using openai/gpt-oss-20b)
- Full compatibility with LLM-as-a-judge (Groq / OpenAI / Anthropic)
"""
from __future__ import annotations

import json
import os
import re
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
            model="no_results",
        )

    # Compact context: top 4 passages, max 350 chars each (saves 65% tokens against TPM limits)
    context_blocks = []
    for i, r in enumerate(results[:4]):
        txt = getattr(r, 'text', '')
        if len(txt) > 350:
            txt = txt[:350].rsplit(' ', 1)[0]
        context_blocks.append(f"[{i+1}] {txt}")
    context_text = "\n".join(context_blocks)

    prompt = (
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\n\n"
        f"Instructions:\n"
        f"1. If the context contains facts or formulas to answer the question, return JSON:\n"
        f'   {{"grounded": true, "refused": false, "answer": "<concise 1-sentence answer>"}}\n'
        f"2. If the context lacks facts to answer the question, return JSON:\n"
        f'   {{"grounded": false, "refused": true, "answer": "The provided documents don\'t contain information about this."}}\n'
        f"Respond with a single valid JSON object."
    )

    client = _get_groq_client()
    model_name = os.environ.get("RAG_MODEL", "openai/gpt-oss-20b")

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                temperature=0,
                max_tokens=150,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a concise zero-hallucination assistant. You answer questions strictly based on the provided context passages. "
                            "Always respond with a single valid JSON object."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            raw = resp.choices[0].message.content or ""
            cleaned = re.sub(r'<thought>.*?</thought>', '', raw, flags=re.DOTALL).strip()
            cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL).strip()

            data = None
            match = re.search(r'\{.*\}', cleaned, flags=re.DOTALL)
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
                is_ref = any(p in cleaned.lower() for p in ("don't contain", "do not contain", "not contain", '"refused": true', '"grounded": false'))
                ans_m = re.search(r'"answer"\s*:\s*"([^"]+)', cleaned)
                ans_str = ans_m.group(1) if ans_m else cleaned[:150]
                if ans_str.startswith('{"') or '":' in ans_str:
                    ans_str = "The provided documents don't contain information about this."
                    is_ref = True
                data = {"answer": ans_str, "grounded": not is_ref, "refused": is_ref}

            ans_text = str(data.get("answer", "")).strip()
            is_grounded = bool(data.get("grounded", False)) and not bool(data.get("refused", False))

            # Clean up partial JSON leaks
            if ans_text.startswith('{') or '":' in ans_text:
                is_grounded = False
                ans_text = "The provided documents don't contain information about this."

            # Check for refusal phrases inside answer text
            refusal_phrases = (
                "don't contain", "do not contain", "not contain",
                "does not contain", "no information", "cannot be answered",
                "doesn't contain", "no answer", "not addressed",
            )
            if any(p in ans_text.lower() for p in refusal_phrases) or len(ans_text) < 5:
                is_grounded = False

            elapsed = (time.perf_counter() - t0) * 1000.0

            if not is_grounded:
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
            if "429" in str(exc) or "rate" in str(exc).lower():
                time.sleep(2.0 * (attempt + 1))
            else:
                time.sleep(0.5)

    # Fallback if API fails
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
    fb = _extractive_fallback(query, rs, "eng_Latn")
    elapsed = (time.perf_counter() - t0) * 1000.0

    if fb.refused or not fb.answer:
        return GeneratedAnswer(
            text="The provided documents don't contain information about this.",
            grounded=False,
            generation_ms=elapsed,
            model="safe_refusal",
        )

    return GeneratedAnswer(
        text=fb.answer,
        grounded=True,
        generation_ms=elapsed,
        model="extractive_fallback",
    )
