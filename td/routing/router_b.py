"""Router B — Task Type Detector.

One instance per domain. Classifies task type within a domain.
Uses BitNet b1.58-style ternary layers with LayerNorm (SubLN).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .ternary_linear import TernaryLinear


TASK_TYPES = {
    "Web":     ["Form", "Navigation", "Extraction", "Interaction"],
    "API":     ["Sequential", "Parallel", "ErrorHandling", "Retry"],
    "File":    ["Parse", "Transform", "Validate", "Generate"],
    "Monitor": ["Threshold", "LogAnalysis", "AlertRouting", "Routine"],
    "Unknown": ["Complex", "Proof", "Novel", "Ambiguous"],
}


class RouterB(nn.Module):
    """Task Type Detector — one instance per domain.

    Architecture:
        HDC normalize → TernaryLinear(10K, 128) → LayerNorm → ReLU
        → TernaryLinear(128, 4) → Softmax
    """

    def __init__(self, domain: str, input_dim: int = 10_000, hidden_dim: int = 128):
        super().__init__()
        assert domain in TASK_TYPES, f"Unknown domain: {domain}"
        self.domain = domain
        self.task_types = TASK_TYPES[domain]

        self.fc1 = TernaryLinear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = TernaryLinear(hidden_dim, len(self.task_types))

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
            return self.task_types[idx], float(probs[idx]), probs.numpy()
