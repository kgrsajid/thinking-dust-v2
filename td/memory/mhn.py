"""Modern Hopfield Network (MHN) with Input-Dependent Plasticity (IDP).

Based on Ramsauer et al. (2020) "Hopfield Networks is All You Need"
and Betteti et al. (2025) IDP from Science Advances.

Storage: Patterns stored as rows in a matrix. No gradient training.
Capacity: ~0.14 × dim × exp(dim) — effectively unlimited for D=10K.
Retrieval: Energy descent via log-sum-exp attention.
IDP: Energy landscape dynamically reshaped by query vector.

Key property: Zero catastrophic forgetting. Old patterns are never
overwritten or degraded by new ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MHNConfig:
    """Configuration for Modern Hopfield Network.

    Attributes:
        dim: Vector dimensionality (matches HDC dim).
        beta: Base inverse temperature (sharpness of retrieval).
            Higher = sharper retrieval but less noise tolerance.
        max_iters: Maximum energy descent steps.
        tol: Convergence tolerance (L2 norm of change between steps).
        idp_enabled: Whether to use Input-Dependent Plasticity.
        idp_threshold: Center of sigmoid for IDP β modulation.
            Patterns with similarity above this get sharper β.
        min_similarity: Minimum similarity to return a retrieval result.
            Below this, returns None (no matching pattern).
    """
    dim: int = 10_000
    beta: float = 1.0
    max_iters: int = 100
    tol: float = 1e-4
    idp_enabled: bool = True
    idp_threshold: float = 0.3
    min_similarity: float = 0.3


@dataclass
class StoredPattern:
    """A single stored pattern with metadata."""
    key: np.ndarray          # Situation HDC vector (used for retrieval matching)
    value: np.ndarray        # Action/value HDC vector (returned on retrieval)
    composite: np.ndarray    # bind(key, value) — kept for algebraic queries
    metadata: dict           # Domain, task type, outcome, timestamp
    active: bool = True      # False if superseded by correction


class ModernHopfieldNetwork:
    """Modern Hopfield Network with Input-Dependent Plasticity.

    Uses log-sum-exp energy function from Ramsauer et al. (2020):
        E(x) = -logsumexp(β · <x, ξ_i>) / β + ||x||²/2 + const

    where ξ_i are stored patterns.

    Retrieval is one-step with softmax attention:
        retrieved = Σ_i softmax(β · <x, ξ_i>)_i · ξ_i

    IDP from Betteti et al. (2025) modulates β per-pattern based on
    similarity to the query, preventing cross-domain interference.
    """

    def __init__(self, config: MHNConfig = MHNConfig()):
        self.config = config
        self.patterns: list[StoredPattern] = []
        self._pattern_matrix: np.ndarray | None = None  # Cache for speed
        self._dirty: bool = True  # Cache needs rebuild

    def _rebuild_cache(self) -> None:
        """Rebuild the pattern matrix cache for vectorized retrieval."""
        if not self.patterns:
            self._pattern_matrix = None
            self._dirty = False
            return
        active = [p for p in self.patterns if p.active]
        if not active:
            self._pattern_matrix = None
            self._dirty = False
            return
        # Store KEY vectors for similarity-based retrieval (not composites)
        self._pattern_matrix = np.stack(
            [p.key.astype(np.float32) for p in active]
        )
        self._active_indices = [i for i, p in enumerate(self.patterns) if p.active]
        self._dirty = False

    def store(self, key: np.ndarray, value: np.ndarray,
              metadata: dict | None = None) -> int:
        """Store a key-value pattern as an attractor.

        The composite is bind(key, value), which allows algebraic
        value retrieval: composite ⊗ key ≈ value.

        Args:
            key: Situation HDC vector.
            value: Action/value HDC vector.
            metadata: Optional metadata dict.

        Returns:
            Pattern index (int).
        """
        from ..perception.hdc import bind
        composite = bind(key, value)
        pattern = StoredPattern(
            key=key.astype(np.int8).copy(),
            value=value.astype(np.int8).copy(),
            composite=composite.astype(np.int8).copy(),
            metadata=metadata or {},
        )
        self.patterns.append(pattern)
        self._dirty = True
        return len(self.patterns) - 1

    def retrieve(self, query: np.ndarray,
                 top_k: int = 1) -> list[tuple[np.ndarray, float, dict]]:
        """Retrieve nearest stored pattern(s) via energy descent.

        Uses softmax attention (one-step approximation of energy descent):
            α_i = exp(β_i · sim(query, pattern_i))
            retrieved = Σ α_i / Σ α_j · pattern_i

        With IDP, β_i is modulated per-pattern based on similarity.

        Args:
            query: Query HDC vector.
            top_k: Number of top patterns to return.

        Returns:
            List of (pattern_vector, similarity_score, metadata) tuples,
            sorted by similarity descending. Length = min(top_k, num_patterns).
            Returns empty list if no patterns stored.
        """
        if not self.patterns:
            return []

        if self._dirty:
            self._rebuild_cache()

        if self._pattern_matrix is None:
            return []

        # Compute similarities
        query_f = query.astype(np.float32)
        # Cosine similarity = dot product for bipolar normalized vectors
        sims = self._pattern_matrix @ query_f / self.config.dim

        # IDP: modulate β per-pattern
        if self.config.idp_enabled:
            betas = self._compute_idp_betas(sims)
        else:
            betas = np.full(len(sims), self.config.beta, dtype=np.float32)

        # Softmax attention
        logits = betas * sims * self.config.dim  # Scale up for sharper softmax
        logits -= logits.max()  # Numerical stability
        exp_logits = np.exp(logits)
        weights = exp_logits / (exp_logits.sum() + 1e-12)

        # Get top_k indices by weight
        top_indices = np.argsort(weights)[::-1][:top_k]

        results = []
        active_patterns = [self.patterns[i] for i in self._active_indices]
        for idx in top_indices:
            pattern = active_patterns[idx]
            sim = float(sims[idx])
            if sim >= self.config.min_similarity:
                results.append((
                    pattern.value.copy(),
                    sim,
                    pattern.metadata.copy(),
                ))

        return results

    def retrieve_value(self, key: np.ndarray) -> np.ndarray | None:
        """Retrieve only the value part for a given key.

        Uses binding algebra: stored composite = bind(key, value).
        Retrieved composite ⊗ key ≈ value.

        Args:
            key: Key HDC vector.

        Returns:
            Value HDC vector, or None if no pattern above similarity threshold.
        """
        from ..perception.hdc import bind
        results = self.retrieve(key, top_k=1)
        if not results:
            return None
        retrieved_value, sim, _ = results[0]
        return retrieved_value

    def update_pattern(self, index: int, key: np.ndarray,
                       value: np.ndarray, metadata: dict | None = None):
        """Update an existing pattern (for online correction learning).

        Does NOT remove old pattern (no catastrophic forgetting).
        Marks old pattern as inactive and stores a new one.

        Args:
            index: Index of pattern to supersede.
            key: New key HDC vector.
            value: New value HDC vector.
            metadata: New metadata.
        """
        if 0 <= index < len(self.patterns):
            self.patterns[index].active = False
        new_idx = self.store(key, value, metadata)
        self._dirty = True
        return new_idx

    def _compute_idp_betas(self, sims: np.ndarray) -> np.ndarray:
        """Compute per-pattern β values using IDP.

        Betteti et al. (2025): modulate the energy landscape's curvature
        based on the query. Patterns similar to the query get sharper β
        (deeper wells), dissimilar patterns get damped β (shallower wells).

        Implementation:
            β_i = β_base × sigmoid((sim_i - threshold) × gain)

        This prevents spurious attractors from unrelated domains.

        Args:
            sims: Similarity scores for each active pattern.

        Returns:
            Per-pattern β values.
        """
        gain = 10.0  # Steepness of sigmoid
        sigmoid = 1.0 / (1.0 + np.exp(-gain * (sims - self.config.idp_threshold)))
        return self.config.beta * (0.1 + 0.9 * sigmoid)

    def get_stats(self) -> dict:
        """Return memory statistics."""
        active_count = sum(1 for p in self.patterns if p.active)
        total_count = len(self.patterns)
        memory_mb = total_count * self.config.dim * 3 / (1024 * 1024)  # 3 vectors per pattern

        domain_counts: dict[str, int] = {}
        for p in self.patterns:
            if p.active:
                domain = p.metadata.get("domain", "unknown")
                domain_counts[domain] = domain_counts.get(domain, 0) + 1

        return {
            "num_patterns": total_count,
            "num_active": active_count,
            "memory_mb": round(memory_mb, 2),
            "domains": domain_counts,
        }

    def __len__(self) -> int:
        return sum(1 for p in self.patterns if p.active)

    def __repr__(self) -> str:
        return f"ModernHopfieldNetwork(patterns={len(self)}, dim={self.config.dim})"
