"""Router A — Domain Detector.

Classifies input HDC vector into 5 domains.
Uses BitNet b1.58-style ternary layers with LayerNorm (SubLN).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .ternary_linear import TernaryLinear

DOMAINS = ["Web", "API", "File", "Monitor", "Unknown"]


class RouterA(nn.Module):
    """Domain Detector — classifies input into 5 domains.

    Architecture:
        HDC normalize → TernaryLinear(10K, 128) → LayerNorm → ReLU
        → TernaryLinear(128, 5) → Softmax

    The LayerNorm between layers prevents logit saturation (BitNet uses
    SubLN for the same purpose). Without it, ternary matmuls produce
    logits of magnitude ~80+ which saturate softmax.
    """

    def __init__(self, input_dim: int = 10_000, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.fc1 = TernaryLinear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = TernaryLinear(hidden_dim, len(DOMAINS))

    def forward(self, hdc_vector: torch.Tensor) -> torch.Tensor:
        if hdc_vector.dim() == 1:
            hdc_vector = hdc_vector.unsqueeze(0)

        # Normalize HDC input: ±1 in D dims → divide by √D for O(1) activations
        x = hdc_vector.float() / (hdc_vector.shape[-1] ** 0.5)
        x = self.fc1(x)
        x = self.ln1(x)  # SubLN: prevents intermediate activation explosion
        x = torch.relu(x)
        x = self.fc2(x)
        return torch.softmax(x, dim=-1)

    def classify(self, hdc_vector_numpy: np.ndarray) -> tuple[str, float, np.ndarray]:
        with torch.no_grad():
            x = torch.from_numpy(hdc_vector_numpy.astype(np.float32))
            probs = self.forward(x).squeeze(0)
            idx = probs.argmax().item()
            return DOMAINS[idx], float(probs[idx]), probs.numpy()
