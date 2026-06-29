"""Attractor lifecycle management — store, retrieve, and optional decay.

Note: Metadata is stored in StoredPattern.metadata (inside MHN).
AttractorStore is now a thin convenience wrapper — no parallel metadata dict.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .mhn import ModernHopfieldNetwork


class AttractorStore:
    """High-level wrapper around MHN for managing attractor lifecycle.

    Provides domain-aware convenience methods. All metadata lives in
    StoredPattern.metadata — no parallel tracking dict (previous design
    desynced from MHN indices).
    """

    def __init__(self, mhn: ModernHopfieldNetwork):
        """
        Args:
            mhn: Underlying Hopfield network.
        """
        self.mhn = mhn

    def store(self, key: np.ndarray, value: np.ndarray,
              domain: str = "unknown", task_type: str = "unknown",
              outcome: str = "unknown",
              **extra_meta) -> int:
        """Store a pattern with metadata.

        Args:
            key: Situation HDC vector.
            value: Action HDC vector.
            domain: Domain label.
            task_type: Task type label.
            outcome: Outcome label ("success", "failure", "partial").
            **extra_meta: Additional metadata to merge in.
        """
        meta = {
            "domain": domain,
            "task_type": task_type,
            "outcome": outcome,
            **extra_meta,
        }
        return self.mhn.store(key, value, metadata=meta)

    def retrieve(self, query: np.ndarray, top_k: int = 1):
        """Retrieve patterns from MHN.

        Args:
            query: Query HDC vector.
            top_k: Number of patterns to retrieve.
        """
        return self.mhn.retrieve(query, top_k=top_k)

    def stats(self) -> dict:
        """Return statistics from the underlying MHN."""
        return self.mhn.get_stats()
