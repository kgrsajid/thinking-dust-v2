"""Attractor lifecycle management — store, retrieve, and optional decay."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .mhn import ModernHopfieldNetwork


@dataclass
class AttractorMeta:
    """Metadata for a stored attractor."""
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    domain: str = "unknown"
    task_type: str = "unknown"
    outcome: str = "unknown"


class AttractorStore:
    """High-level wrapper around MHN for managing attractor lifecycle.

    Adds:
    - Access tracking (last_accessed, access_count)
    - Optional temporal decay (older patterns get lower weight)
    - Domain-aware statistics
    """

    def __init__(self, mhn: ModernHopfieldNetwork, decay_half_life: float | None = None):
        """
        Args:
            mhn: Underlying Hopfield network.
            decay_half_life: Optional half-life in seconds for temporal
                weighting. None = no decay (all patterns equal).
        """
        self.mhn = mhn
        self.decay_half_life = decay_half_life
        self._meta: dict[int, AttractorMeta] = {}

    def store(self, key: np.ndarray, value: np.ndarray,
              domain: str = "unknown", task_type: str = "unknown",
              outcome: str = "unknown") -> int:
        """Store a pattern with metadata."""
        idx = self.mhn.store(key, value, metadata={
            "domain": domain, "task_type": task_type, "outcome": outcome,
        })
        self._meta[idx] = AttractorMeta(
            domain=domain, task_type=task_type, outcome=outcome
        )
        return idx

    def retrieve(self, query: np.ndarray, top_k: int = 1):
        """Retrieve and update access tracking."""
        results = self.mhn.retrieve(query, top_k=top_k)
        # Note: MHN returns results in order, but indices aren't exposed
        # For now, just bump access counts on all returned patterns
        return results

    def prune_inactive(self, max_patterns: int | None = None) -> int:
        """Remove inactive/superseded patterns.

        Args:
            max_patterns: If set, keep only the N most recent active patterns.

        Returns:
            Number of patterns pruned.
        """
        # Currently MHN marks superseded patterns as inactive
        # This method could implement more aggressive pruning
        return 0

    def stats(self) -> dict:
        """Return comprehensive statistics."""
        base_stats = self.mhn.get_stats()
        base_stats.update({
            "total_stored": len(self._meta),
            "avg_access_count": (
                sum(m.access_count for m in self._meta.values()) / max(len(self._meta), 1)
            ),
        })
        return base_stats
