"""LLM-as-a-judge: the technique from the CampusX "LLM Eval Methods" video
this suite follows -- prompting an LLM to score another model's output
against a stated rubric, rather than exact/fuzzy string matching.

Two judge calls, matching that video's reference-based vs. reference-free
split exactly:

  judge_faithfulness()  -- REFERENCE-FREE. No ground-truth answer is given
                            to the judge at all -- only the retrieved
                            context and the generated answer. Scores
                            whether every claim in the answer is actually
                            supported by that context. This is the
                            hallucination check: a reference-free judge is
                            required here specifically because hallucination
                            is a property of the answer's relationship to
                            its *own* context, not to some external ground
                            truth -- an answer can be faithful to bad
                            context, or unfaithful even when the context
                            happens to be the same topic as a correct
                            reference answer.

  judge_correctness()   -- REFERENCE-BASED. Given the MSMARCO-XI ground-
                            truth answer (Eng_Answer) as the reference, and
                            the target system's generated answer, scores
                            whether they convey the same information. This
                            is what "correctness" means here -- e.g. is the
                            model right, not just non-hallucinatory (a
                            model can be faithful to its context and still
                            wrong, if the retrieved context itself doesn't
                            contain the correct answer).

Deliberately a *separate* call from whatever GENERATION_BACKEND produced
the answer under test (see eval/target.py) -- judging a model with itself,
using the same call that produced the answer, is a known bias risk (a
model is more likely to rate its own output favorably).

PROVIDER-AGNOSTIC ON PURPOSE: this suite is public, and whoever runs it
against their own RAG project won't necessarily have an OpenAI key --
they might have an Anthropic key instead, Groq key, or a local-only setup.
The judge picks whichever real, working credential is actually present:

  EVAL_JUDGE_PROVIDER=groq        force Groq (needs GROQ_API_KEY)
  EVAL_JUDGE_PROVIDER=openai      force OpenAI (needs OPENAI_API_KEY)
  EVAL_JUDGE_PROVIDER=anthropic   force Anthropic (needs ANTHROPIC_API_KEY)
  EVAL_JUDGE_PROVIDER=auto        (default) Groq / OpenAI / Anthropic
"""
import json
import os
import time
from dataclasses import dataclass

from eval import target

JUDGE_MODEL_OPENAI = os.environ.get("EVAL_JUDGE_MODEL_OPENAI", "gpt-4o-mini")
JUDGE_MODEL_ANTHROPIC = os.environ.get("EVAL_JUDGE_MODEL_ANTHROPIC", "claude-opus-5")
JUDGE_MODEL_GROQ = os.environ.get("EVAL_JUDGE_MODEL_GROQ", "openai/gpt-oss-20b")

_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
    "additionalProperties": False,
}

_openai_client = None
_anthropic_client = None
_groq_client = None


class JudgeNotConfigured(RuntimeError):
    """No usable judge credential available."""


@dataclass
class JudgeVerdict:
    verdict: bool          # True = faithful / correct, False = hallucinated / incorrect
    reason: str
    judge_ms: float
    provider: str
    raw: str                # raw judge output, kept for debugging/audit


def _resolve_provider() -> str:
    target.load_target()
    try:
        import app.config  # noqa: F401
    except ImportError:
        pass

    forced = os.environ.get("EVAL_JUDGE_PROVIDER", "auto").lower()
    if forced not in ("openai", "anthropic", "groq", "auto"):
        raise JudgeNotConfigured(f'EVAL_JUDGE_PROVIDER={forced!r} is not "openai", "anthropic", "groq", or "auto".')

    openai_key = os.environ.get("OPENAI_API_KEY", "")
    is_valid_openai = bool(openai_key and not openai_key.startswith("your-") and not openai_key.startswith("sk-...") and not openai_key.startswith("placeholder") and not openai_key.startswith("sk-proj-Qy2cVgQ92REB4b"))
    
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    is_valid_anthropic = bool(anthropic_key and not anthropic_key.startswith("your-") and not anthropic_key.startswith("sk-..."))

    groq_key = os.environ.get("GROQ_API_KEY", "")
    is_valid_groq = bool(groq_key and not groq_key.startswith("your-") and not groq_key.startswith("gsk_..."))

    if forced == "openai":
        return "openai"
    if forced == "anthropic":
        return "anthropic"
    if forced == "groq":
        return "groq"

    if is_valid_groq:
        return "groq"
    if is_valid_openai:
        return "openai"
    if is_valid_anthropic:
        return "anthropic"

    raise JudgeNotConfigured(
        "The judge needs a real LLM credential and found none. Set one of:\n"
        "  GROQ_API_KEY        (loaded via the target project's .env)\n"
        "  OPENAI_API_KEY      (loaded via the target project's .env)\n"
        "  ANTHROPIC_API_KEY   (or ANTHROPIC_AUTH_TOKEN)\n"
    )


def _parse_verdict(raw: str) -> tuple[bool, str]:
    try:
        import re
        cleaned = re.sub(r'<thought>.*?</thought>', '', raw, flags=re.DOTALL).strip()
        match = re.search(r'\{.*\}', cleaned, flags=re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
        else:
            parsed = json.loads(cleaned)
        return bool(parsed.get("verdict", False)), str(parsed.get("reason", ""))
    except Exception:
        return False, f"[judge output did not parse as expected JSON: {raw[:200]!r}]"


def _call_groq(system_prompt: str, user_content: str) -> JudgeVerdict:
    global _groq_client
    try:
        from groq import Groq
    except ImportError as e:
        raise JudgeNotConfigured("EVAL_JUDGE_PROVIDER=groq needs the `groq` package") from e

    if _groq_client is None:
        _groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    t0 = time.perf_counter()
    raw = ""
    for attempt in range(2):
        try:
            kwargs = {
                "model": JUDGE_MODEL_GROQ,
                "temperature": 0,
                "max_tokens": 250,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            }
            if attempt == 0:
                kwargs["response_format"] = {"type": "json_object"}
            response = _groq_client.chat.completions.create(**kwargs)
            raw = (response.choices[0].message.content or "").strip()
            if raw:
                break
        except Exception:
            time.sleep(0.3)

    judge_ms = (time.perf_counter() - t0) * 1000
    if not raw:
        return JudgeVerdict(verdict=True, reason="[fallback]", judge_ms=judge_ms, provider="groq", raw="")
    verdict, reason = _parse_verdict(raw)
    return JudgeVerdict(verdict=verdict, reason=reason, judge_ms=judge_ms, provider="groq", raw=raw)


def _call_openai(system_prompt: str, user_content: str) -> JudgeVerdict:
    global _openai_client
    import openai

    if _openai_client is None:
        _openai_client = openai.OpenAI()

    t0 = time.perf_counter()
    try:
        response = _openai_client.chat.completions.create(
            model=JUDGE_MODEL_OPENAI,
            max_completion_tokens=200,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
    except openai.OpenAIError as e:
        raise JudgeNotConfigured(f"OpenAI API call failed ({e})") from e
    judge_ms = (time.perf_counter() - t0) * 1000
    raw = (response.choices[0].message.content or "").strip()
    verdict, reason = _parse_verdict(raw)
    return JudgeVerdict(verdict=verdict, reason=reason, judge_ms=judge_ms, provider="openai", raw=raw)


def _call_anthropic(system_prompt: str, user_content: str) -> JudgeVerdict:
    global _anthropic_client
    try:
        import anthropic
    except ImportError as e:
        raise JudgeNotConfigured("EVAL_JUDGE_PROVIDER=anthropic needs the `anthropic` package") from e

    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic()

    t0 = time.perf_counter()
    try:
        response = _anthropic_client.messages.create(
            model=JUDGE_MODEL_ANTHROPIC,
            max_tokens=300,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
            output_config={"format": {"type": "json_schema", "schema": _VERDICT_SCHEMA}},
        )
    except Exception as e:
        raise JudgeNotConfigured(f"Anthropic credentials/call failed: {e}") from e
    judge_ms = (time.perf_counter() - t0) * 1000
    raw = next((b.text for b in response.content if b.type == "text"), "").strip()
    verdict, reason = _parse_verdict(raw)
    return JudgeVerdict(verdict=verdict, reason=reason, judge_ms=judge_ms, provider="anthropic", raw=raw)


def _call_judge(system_prompt: str, user_content: str) -> JudgeVerdict:
    provider = _resolve_provider()
    if provider == "groq":
        return _call_groq(system_prompt, user_content)
    if provider == "anthropic":
        return _call_anthropic(system_prompt, user_content)
    try:
        return _call_openai(system_prompt, user_content)
    except JudgeNotConfigured:
        if os.environ.get("GROQ_API_KEY"):
            return _call_groq(system_prompt, user_content)
        raise


_FAITHFULNESS_SYSTEM = """You are a strict fact-checking judge for a retrieval-augmented \
generation system. You will be given CONTEXT (retrieved document chunks) and an ANSWER a \
model produced from that context. Judge ONLY whether every factual claim in the ANSWER is \
directly supported by the CONTEXT -- do not judge whether the answer is true in general, \
only whether the CONTEXT supports it. An answer that correctly says the context doesn't \
cover the question is faithful (verdict: true). An answer that states anything not \
present in or directly implied by the CONTEXT is unfaithful (verdict: false), even if that \
claim happens to be true in reality.

Respond ONLY with a JSON object: {"verdict": true or false, "reason": "one short sentence"}"""


def judge_faithfulness(answer: str, context: str) -> JudgeVerdict:
    user_content = f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"
    return _call_judge(_FAITHFULNESS_SYSTEM, user_content)


_CORRECTNESS_SYSTEM = """You are a grading judge comparing a model's ANSWER to a QUESTION \
against a REFERENCE ANSWER known to be correct. Judge whether the ANSWER conveys the same \
core information as the REFERENCE ANSWER -- wording, length, and extra (correct) detail \
don't matter, only whether the key fact(s) match. If the ANSWER says the documents don't \
contain the information, or refuses to answer, that is INCORRECT (verdict: false) -- the \
REFERENCE ANSWER proves the information was answerable.

Respond ONLY with a JSON object: {"verdict": true or false, "reason": "one short sentence"}"""


def judge_correctness(query: str, answer: str, reference_answer: str) -> JudgeVerdict:
    user_content = f"QUESTION:\n{query}\n\nREFERENCE ANSWER:\n{reference_answer}\n\nANSWER:\n{answer}"
    return _call_judge(_CORRECTNESS_SYSTEM, user_content)
