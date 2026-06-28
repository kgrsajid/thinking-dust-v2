"""Router A — Domain Detector.

Classifies input HDC vector into 5 domains.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .ternary_linear import TernaryLinear

DOMAINS = ["Web", "API", "File", "Monitor", "Unknown"]


class RouterA(nn.Module):
    """Domain Detector — classifies input into 5 domains.

    Architecture:
        TernaryLinear(10_000, 128) → ReLU → TernaryLinear(128, 5) → Softmax

    Parameters: 10_000×128 + 128×5 = 1,280,640 weights
    After ternarization + sparsity: ~5K effective non-zero weights.

    Input: HDC vector (10K-dim, bipolar → converted to float)
    Output: softmax probabilities over 5 domains
    Time: ~0.01ms (integer matmul on CPU after ternarization)
    """

    def __init__(self, input_dim: int = 10_000, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = TernaryLinear(input_dim, hidden_dim)
        self.fc2 = TernaryLinear(hidden_dim, len(DOMAINS))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, hdc_vector: torch.Tensor) -> torch.Tensor:
        """Classify HDC vector into domain probabilities.

        Args:
            hdc_vector: HDC vector tensor of shape (batch, 10_000)
                        or (10_000,) for single input.

        Returns:
            Softmax probabilities of shape (batch, 5) or (5,).
        """
        if hdc_vector.dim() == 1:
            hdc_vector = hdc_vector.unsqueeze(0)

        x = self.fc1(hdc_vector.float())
        x = torch.relu(x)
        x = self.fc2(x)
        return self.softmax(x)

    def classify(self, hdc_vector_numpy) -> "RoutingResult":
        """Convenience: classify numpy HDC vector and return domain name + confidence.

        Returns:
            Tuple-like: (domain_name, confidence, all_probs)
        """
        with torch.no_grad():
            x = torch.from_numpy(hdc_vector_numpy.astype(np.float32))
            probs = self.forward(x).squeeze(0)
            idx = probs.argmax().item()
            return DOMAINS[idx], float(probs[idx]), probs.numpy()


# Avoid circular import at module level — import inside method
import numpy as np  # noqa: E402
