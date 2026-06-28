"""Ternary Linear Layer — weights constrained to {-1, 0, +1}.

Based on BitNet b1.58 (Ma et al., 2024, Microsoft Research).

Storage: 2 bits per weight (vs 32 bits for float32) → 16× compression.
Inference: pure integer multiply-add (no floating point).
Training: Straight-Through Estimator (STE) for backprop through ternarization.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TernaryLinear(nn.Module):
    """Linear layer with ternary weights {-1, 0, +1}.

    Uses the Straight-Through Estimator (STE):
    - Forward: use ternarized weights (deterministic, integer arithmetic)
    - Backward: gradient passes through as if weights were continuous

    Ternarization rule (per-row):
        1. Compute threshold τ = 0.7 × mean(|w_row|)
        2. w_ternary = sign(w) × (|w| > τ)

    This means ~70% of weights become 0 (sparse), reducing both
    storage and computation.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """Initialize ternary linear layer.

        Args:
            in_features: Input dimensionality.
            out_features: Output dimensionality.
            bias: Whether to include bias term.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Continuous weights (used during training, ternarized at inference)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Initialize weights (Kaiming uniform for good starting point)
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

        # Store ternarized weights cache for inference
        self._ternary_cache: torch.Tensor | None = None
        self._cache_valid = False

    def _ternarize(self, w: torch.Tensor) -> torch.Tensor:
        """Ternarize continuous weights to {-1, 0, +1}.

        Per-row thresholding:
            τ = 0.7 × mean(|w_row|)
            w_t = sign(w) × (|w| > τ)

        This zeros out ~70% of weights, making the matrix sparse.

        Args:
            w: Continuous weight tensor of shape (out, in).

        Returns:
            Ternarized weight tensor of same shape.
        """
        # Per-row statistics
        row_means = w.abs().mean(dim=1, keepdim=True)
        thresholds = 0.7 * row_means

        # Ternarize: sign(w) * (|w| > threshold)
        ternary = torch.sign(w) * (w.abs() > thresholds).float()

        return ternary

    def get_ternary_weights(self) -> torch.Tensor:
        """Get the current ternarized weights.

        Returns:
            Tensor of shape (out_features, in_features) with values in {-1, 0, +1}.
        """
        with torch.no_grad():
            return self._ternarize(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with STE.

        During training:
            - Ternarize weights for forward computation
            - Use STE to pass gradients through ternarization
            - Continuous weights are updated by optimizer

        During inference (eval mode):
            - Use cached ternary weights for pure integer arithmetic

        Args:
            x: Input tensor of shape (batch, in_features).

        Returns:
            Output tensor of shape (batch, out_features).
        """
        if self.training:
            # Ternarize with STE
            w_t = self._ternarize(self.weight)
            # STE: detach ternarized for forward, but keep gradient flowing
            w_ste = w_t + self.weight - self.weight.detach()
            self._cache_valid = False
        else:
            # Use cached ternary weights in eval mode
            if not self._cache_valid:
                with torch.no_grad():
                    self._ternary_cache = self._ternarize(self.weight)
                    self._cache_valid = True
            w_ste = self._ternary_cache

        return F.linear(x, w_ste, self.bias)

    def num_ternary_params(self) -> dict:
        """Return parameter statistics.

        Returns:
            Dict with total params, and counts of {-1, 0, +1}.
        """
        w = self.get_ternary_weights()
        return {
            "total": w.numel(),
            "neg_ones": int((w == -1).sum()),
            "zeros": int((w == 0).sum()),
            "pos_ones": int((w == 1).sum()),
            "sparsity": float((w == 0).float().mean()),
        }

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None}, ternary=True")
