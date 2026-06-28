"""Hierarchical Router — full 3-level cascade.

RouterA (domain) → RouterB (task type) → RouterC (strategy)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .router_a import RouterA, DOMAINS
from .router_b import RouterB, TASK_TYPES
from .router_c import RouterC, STRATEGIES


@dataclass
class RoutingResult:
    """Result of the full routing cascade.

    Contains all classification outputs and confidence scores
    for debugging, logging, and confidence computation.
    """
    domain: str
    domain_confidence: float
    task_type: str
    task_type_confidence: float
    strategy: str
    strategy_confidence: float
    combined_confidence: float

    def __repr__(self) -> str:
        return (f"RoutingResult({self.domain}/{self.task_type}/"
                f"{self.strategy} conf={self.combined_confidence:.3f})")


class HierarchicalRouter:
    """Full 3-level router cascade.

    Pipeline: HDC → RouterA (domain) → RouterB (task type) → RouterC (strategy)

    Total parameters:
        RouterA: ~5K (after ternary sparsity)
        RouterB × 5: ~10K
        RouterC: ~1K
        Total: ~16K (before pruning) → ~5K effective

    Total inference time: ~0.03ms (3 sequential ternary matmuls).
    """

    def __init__(self, input_dim: int = 10_000, device: str = "cpu"):
        """Initialize all router levels.

        Args:
            input_dim: HDC vector dimensionality.
            device: torch device for computation.
        """
        self.input_dim = input_dim
        self.device = device

        self.router_a = RouterA(input_dim=input_dim).to(device)
        self.router_b_instances = {
            domain: RouterB(domain, input_dim=input_dim).to(device)
            for domain in DOMAINS
        }
        self.router_c = RouterC(input_dim=input_dim).to(device)

        self._all_modules = (
            [self.router_a] +
            list(self.router_b_instances.values()) +
            [self.router_c]
        )

    def route(self, hdc_vector: np.ndarray,
              mhn_retrieval_vector: np.ndarray | None = None) -> RoutingResult:
        """Classify input through full cascade.

        Args:
            hdc_vector: HDC vector of shape (input_dim,).
            mhn_retrieval_vector: Optional retrieved MHN pattern.
                If provided, bundled with query for RouterC input.

        Returns:
            RoutingResult with all classification decisions.
        """
        with torch.no_grad():
            x = torch.from_numpy(hdc_vector.astype(np.float32)).to(self.device)

            # Level 1: Domain detection
            domain_probs = self.router_a(x.unsqueeze(0)).squeeze(0)
            domain_idx = domain_probs.argmax().item()
            domain = DOMAINS[domain_idx]
            domain_conf = float(domain_probs[domain_idx])

            # Level 2: Task type detection (only for selected domain)
            router_b = self.router_b_instances[domain]
            task_probs = router_b(x.unsqueeze(0)).squeeze(0)
            task_idx = task_probs.argmax().item()
            task_type = router_b.task_types[task_idx]
            task_conf = float(task_probs[task_idx])

            # Level 3: Strategy selection
            if mhn_retrieval_vector is not None:
                from ..perception.hdc import bundle
                router_c_input = bundle(hdc_vector, mhn_retrieval_vector)
            else:
                router_c_input = hdc_vector

            x_c = torch.from_numpy(router_c_input.astype(np.float32)).to(self.device)
            strategy_probs = self.router_c(x_c.unsqueeze(0)).squeeze(0)
            strategy_idx = strategy_probs.argmax().item()
            strategy = STRATEGIES[strategy_idx]
            strategy_conf = float(strategy_probs[strategy_idx])

            combined = domain_conf * task_conf * strategy_conf

            return RoutingResult(
                domain=domain,
                domain_confidence=domain_conf,
                task_type=task_type,
                task_type_confidence=task_conf,
                strategy=strategy,
                strategy_confidence=strategy_conf,
                combined_confidence=combined,
            )

    def parameters(self):
        """Yield all trainable parameters (for unified training)."""
        for module in self._all_modules:
            yield from module.parameters()

    def train(self, mode: bool = True):
        """Set training mode for all sub-modules."""
        for module in self._all_modules:
            module.train(mode)

    def eval(self):
        """Set eval mode for all sub-modules."""
        self.train(False)

    def save(self, path: str):
        """Save all router weights."""
        torch.save({
            "router_a": self.router_a.state_dict(),
            "router_b": {d: rb.state_dict() for d, rb in self.router_b_instances.items()},
            "router_c": self.router_c.state_dict(),
        }, path)

    def load(self, path: str):
        """Load all router weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.router_a.load_state_dict(checkpoint["router_a"])
        for domain, rb in self.router_b_instances.items():
            rb.load_state_dict(checkpoint["router_b"][domain])
        self.router_c.load_state_dict(checkpoint["router_c"])
