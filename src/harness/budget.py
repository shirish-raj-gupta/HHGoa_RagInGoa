"""
Deadline propagation.

This is the mechanism that turns "p100 < 200ms" from an average into a
guarantee. Every stage receives the same Budget object, reads remaining_ms,
and DEGRADES rather than overrunning. A stage that cannot finish inside its
slice must return something worse, not something late.

Every degradation is recorded, so the trace can show exactly which corner was
cut on the slowest requests instead of leaving p100 unexplained.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


class BudgetExceeded(Exception):
    """Raised only when a stage cannot degrade any further."""


@dataclass
class Degradation:
    stage: str
    action: str
    remaining_ms: float


@dataclass
class Budget:
    """A wall-clock deadline threaded through the whole core loop."""
    total_ms: float = 200.0
    started_ns: int = field(default_factory=time.perf_counter_ns)
    degradations: list[Degradation] = field(default_factory=list)
    # reserve for stages that must always get a slice (guardrails, assembly)
    reserve_ms: float = 4.0

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter_ns() - self.started_ns) / 1e6

    @property
    def remaining_ms(self) -> float:
        return self.total_ms - self.elapsed_ms

    @property
    def spendable_ms(self) -> float:
        """Remaining budget minus the reserve the tail stages still need."""
        return self.remaining_ms - self.reserve_ms

    def over(self) -> bool:
        return self.remaining_ms <= 0

    def note(self, stage: str, action: str) -> None:
        self.degradations.append(Degradation(stage, action, round(self.remaining_ms, 2)))

    # ---- the actual degradation ladder, cheapest quality loss first --------

    def retrieval_params(self, *, base_k: int = 10, base_ef: int = 16,
                         stage: str = "retrieve") -> dict:
        """
        Pick retrieval parameters that fit the remaining budget.

        The ladder is ordered by how much quality each step costs:
          >40ms  full quality
          >25ms  ef=16, base_k
          >12ms  cut k as well (less for MMR to work with)
          >3ms   sparse-only (BM25 needs no embedding forward pass)
          <=2ms  nothing survives - caller must refuse rather than guess

        Note: if the embedding was already done before this call (the new
        flow), the caller upgrades sparse-only back to hybrid for free.
        """
        r = self.spendable_ms
        if r > 40:
            return {"k": base_k, "ef_search": base_ef, "mode": "hybrid"}
        if r > 25:
            return {"k": base_k, "ef_search": base_ef, "mode": "hybrid"}
        if r > 12:
            self.note(stage, "ef_search->16,k->8")
            return {"k": 8, "ef_search": 16, "mode": "hybrid"}
        if r > 3:
            self.note(stage, "sparse_only")
            return {"k": 8, "ef_search": 16, "mode": "sparse"}
        self.note(stage, "budget_exhausted")
        raise BudgetExceeded(f"{r:.1f}ms left at {stage}")

    def allow_rerank(self, est_ms: float = 40.0, stage: str = "rerank") -> bool:
        """Cross-encoder reranking only runs if it demonstrably fits."""
        if self.spendable_ms > est_ms:
            return True
        self.note(stage, "skip_rerank")
        return False

    def allow_mmr(self, est_ms: float = 5.0, stage: str = "mmr") -> bool:
        if self.spendable_ms > est_ms:
            return True
        self.note(stage, "skip_mmr")
        return False

    def child(self, ms: float) -> "Budget":
        """A sub-budget that can never outlive its parent."""
        return Budget(total_ms=min(ms, self.remaining_ms),
                      degradations=self.degradations, reserve_ms=0.0)
