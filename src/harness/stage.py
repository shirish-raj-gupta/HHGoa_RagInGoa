"""
Stage abstraction, retries, circuit breaker, and the async DAG orchestrator.

A Stage knows its own name, timeout, retry policy and fallback, and nothing
whatsoever about its neighbours. The Orchestrator composes them. That is what
makes this a harness rather than a function that calls an LLM.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Generic, TypeVar

from .budget import Budget, BudgetExceeded
from .contracts import ErrorKind, StageTiming, Trace

log = logging.getLogger("harness")

I = TypeVar("I")
O = TypeVar("O")


class UpstreamError(Exception):
    """A dependency failed in a way that is worth retrying."""


class CircuitOpen(Exception):
    """The breaker is open; fail fast instead of piling onto a sick dependency."""


@dataclass
class RetryPolicy:
    max_retries: int = 2
    base_ms: float = 20.0
    max_ms: float = 200.0
    jitter: float = 0.3

    def delay_ms(self, attempt: int) -> float:
        raw = min(self.max_ms, self.base_ms * (2 ** attempt))
        return raw * (1 + random.uniform(-self.jitter, self.jitter))


@dataclass
class CircuitBreaker:
    """
    Per-dependency breaker. Opens after N consecutive failures and short-circuits
    to the degraded path until a cooldown elapses.
    """
    name: str
    threshold: int = 5
    cooldown_s: float = 20.0
    _fails: int = 0
    _opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if time.time() - self._opened_at > self.cooldown_s:
            # half-open: let one request through to probe recovery
            self._opened_at, self._fails = None, self.threshold - 1
            return False
        return True

    def record(self, ok: bool) -> None:
        if ok:
            self._fails, self._opened_at = 0, None
        else:
            self._fails += 1
            if self._fails >= self.threshold and self._opened_at is None:
                self._opened_at = time.time()
                log.warning("circuit_open dependency=%s fails=%d", self.name, self._fails)


@dataclass
class Stage(Generic[I, O]):
    """One unit of pipeline work."""
    name: str
    fn: Callable[..., Awaitable[O]]
    timeout_ms: float = 1000.0
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    fallback: Callable[..., Awaitable[O]] | None = None
    breaker: CircuitBreaker | None = None
    # if False, a failure here fails the whole request
    optional: bool = False

    async def run(self, payload: I, *, budget: Budget, trace: Trace,
                  **kw: Any) -> O:
        started = budget.elapsed_ms
        slice_ms = min(self.timeout_ms, max(1.0, budget.remaining_ms))

        if self.breaker and self.breaker.is_open:
            trace.stages.append(StageTiming(
                name=self.name, started_ms=started, duration_ms=0.0,
                ok=False, degraded=True, note="circuit_open"))
            budget.note(self.name, "circuit_open")
            if self.fallback:
                return await self.fallback(payload, budget=budget, **kw)
            raise CircuitOpen(self.name)

        last: Exception | None = None
        for attempt in range(self.retry.max_retries + 1):
            if budget.over():
                break
            try:
                out = await asyncio.wait_for(
                    self.fn(payload, budget=budget, **kw),
                    timeout=max(0.001, min(slice_ms, budget.remaining_ms) / 1000),
                )
                if self.breaker:
                    self.breaker.record(True)
                trace.stages.append(StageTiming(
                    name=self.name, started_ms=started,
                    duration_ms=budget.elapsed_ms - started, ok=True,
                    note=f"attempt={attempt}" if attempt else ""))
                return out
            except (asyncio.TimeoutError, UpstreamError, ConnectionError) as e:
                last = e
                if self.breaker:
                    self.breaker.record(False)
                if attempt < self.retry.max_retries and not budget.over():
                    d = self.retry.delay_ms(attempt)
                    if d < budget.remaining_ms:
                        await asyncio.sleep(d / 1000)
                        continue
                break
            except Exception as e:                      # non-retryable
                last = e
                break

        trace.stages.append(StageTiming(
            name=self.name, started_ms=started,
            duration_ms=budget.elapsed_ms - started, ok=False,
            degraded=bool(self.fallback), note=type(last).__name__ if last else "failed"))

        if self.fallback:
            budget.note(self.name, "fallback")
            return await self.fallback(payload, budget=budget, **kw)
        if self.optional:
            return None                                  # type: ignore[return-value]
        trace.error = ErrorKind.UPSTREAM
        raise last or UpstreamError(self.name)


async def gather_stages(payload: Any, stages: list[Stage], *,
                        budget: Budget, trace: Trace, **kw) -> list[Any]:
    """
    Run stages concurrently and join. This is the dense || sparse fan-out.

    return_exceptions=True so one arm failing degrades the result instead of
    taking the whole request down - a dead BM25 shard should still let dense
    answer.
    """
    return await asyncio.gather(
        *(s.run(payload, budget=budget, trace=trace, **kw) for s in stages),
        return_exceptions=True,
    )
