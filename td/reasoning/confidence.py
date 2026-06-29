"""Confidence scoring — multi-factor decision confidence.

Combines router confidence, MHN retrieval similarity, and Z3
satisfiability into a single actionable confidence score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConfidenceScore:
    """Multi-factor confidence for routing decisions.

    Attributes:
        router_confidence: Product of 3 router softmax scores.
        mhn_similarity: Cosine similarity of best MHN retrieval.
        z3_satisfiability: 1.0 if SAT, 0.0 if UNSAT, 0.5 if UNKNOWN.
        combined: Weighted combination of all factors.
    """
    router_confidence: float
    mhn_similarity: float
    z3_satisfiability: float
    combined: float

    @property
    def decision(self) -> str:
        """Derive action from confidence level.

        > 0.9  → "execute" (autonomous)
        0.7-0.9 → "confirm" (ask user first)
        < 0.7  → "escalate" (send to TD Pro or ask user)

        Returns:
            One of "execute", "confirm", "escalate".
        """
        if self.combined >= 0.9:
            return "execute"
        elif self.combined >= 0.7:
            return "confirm"
        return "escalate"

    def __repr__(self) -> str:
        return (f"ConfidenceScore({self.combined:.3f} → {self.decision}, "
                f"router={self.router_confidence:.3f}, "
                f"mhn={self.mhn_similarity:.3f}, "
                f"z3={self.z3_satisfiability:.1f})")


def compute_confidence(
    routing_result,
    mhn_results: list[tuple[np.ndarray, float, dict]] | None,
    z3_result=None,
    weights: dict[str, float] | None = None,
) -> ConfidenceScore:
    """Compute combined confidence score.

    Default weights:
        router: 0.4
        mhn:    0.3
        z3:     0.3

    If any component is None (not invoked), its weight is
    redistributed proportionally to the remaining components.

    Args:
        routing_result: RoutingResult from HierarchicalRouter.
        mhn_results: MHN retrieval results (list of tuples) or None.
        z3_result: Z3Result from Z3Bridge or None.
        weights: Optional custom weights dict.

    Returns:
        ConfidenceScore.
    """
    # Copy to avoid mutating caller's dict (bug fix from external review #2)
    w = dict(weights) if weights else {"router": 0.4, "mhn": 0.3, "z3": 0.3}

    router_conf = routing_result.combined_confidence

    if mhn_results and len(mhn_results) > 0:
        mhn_sim = mhn_results[0][1]  # similarity of best match
    else:
        mhn_sim = 0.0
        # Redistribute mhn weight to router
        w["router"] += w.pop("mhn", 0.3)

    if z3_result is not None:
        z3_sat = z3_result.satisfiability_score
    else:
        z3_sat = 0.0  # No Z3 validation — don't inflate confidence with neutral score
        # Redistribute z3 weight proportionally
        remaining = w.pop("z3", 0.3)
        total_remaining = sum(w.values())
        for k in w:
            w[k] += remaining * (w[k] / total_remaining) if total_remaining > 0 else 0

    # Normalize weights
    total_w = sum(w.values())
    if total_w > 0:
        w = {k: v / total_w for k, v in w.items()}

    combined = (
        router_conf * w.get("router", 0) +
        mhn_sim * w.get("mhn", 0) +
        z3_sat * w.get("z3", 0)
    )

    return ConfidenceScore(
        router_confidence=router_conf,
        mhn_similarity=mhn_sim,
        z3_satisfiability=z3_sat,
        combined=combined,
    )
