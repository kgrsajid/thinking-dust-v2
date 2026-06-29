"""Router C — Strategy Selector.

Picks execution pathway: MEMORY_ONLY, MEMORY_THEN_VALIDATE, or ESCALATE.
Uses BitNet b1.58-style ternary layers with LayerNorm (SubLN).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .ternary_linear import TernaryLinear

STRATEGIES = ["MEMORY_ONLY", "MEMORY_THEN_VALIDATE", "ESCALATE"]


class RouterC(nn.Module):
    """Strategy Selector — picks execution pathway.

    Architecture:
        HDC normalize → TernaryLinear(10K, 32) → LayerNorm → ReLU
        → TernaryLinear(32, 3) → Softmax
    """

    def __init__(self, input_dim: int = 10_000, hidden_dim: int = 32):
        super().__init__()
        self.fc1 = TernaryLinear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = TernaryLinear(hidden_dim, len(STRATEGIES))

    def forward(self, hdc_vector: torch.Tensor) -> torch.Tensor:
        if hdc_vector.dim() == 1:
            hdc_vector = hdc_vector.unsqueeze(0)

        x = hdc_vector.float() / (hdc_vector.shape[-1] ** 0.5)
        x = self.fc1(x)
        x = self.ln1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return torch.softmax(x, dim=-1)

    def classify(self, hdc_vector_numpy: np.ndarray):
        with torch.no_grad():
            x = torch.from_numpy(hdc_vector_numpy.astype(np.float32))
            probs = self.forward(x).squeeze(0)
            idx = probs.argmax().item()
            return STRATEGIES[idx], float(probs[idx]), probs.numpy()
