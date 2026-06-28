"""Router B — Task Type Detector.

One instance per domain. Classifies task type within a domain.
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
        TernaryLinear(10_000, 64) → ReLU → TernaryLinear(64, 4) → Softmax

    Only the relevant domain's RouterB is invoked (gated by RouterA output).

    Input: HDC vector (10K-dim)
    Output: softmax probabilities over 4 task types (for the selected domain)
    """

    def __init__(self, domain: str, input_dim: int = 10_000, hidden_dim: int = 64):
        """Initialize RouterB for a specific domain.

        Args:
            domain: One of DOMAINS ("Web", "API", "File", "Monitor", "Unknown").
            input_dim: HDC dimensionality.
            hidden_dim: Hidden layer size.
        """
        super().__init__()
        assert domain in TASK_TYPES, f"Unknown domain: {domain}"
        self.domain = domain
        self.task_types = TASK_TYPES[domain]

        self.fc1 = TernaryLinear(input_dim, hidden_dim)
        self.fc2 = TernaryLinear(hidden_dim, len(self.task_types))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, hdc_vector: torch.Tensor) -> torch.Tensor:
        """Classify HDC vector into task type probabilities.

        Args:
            hdc_vector: HDC vector tensor of shape (batch, input_dim).

        Returns:
            Softmax probabilities of shape (batch, 4).
        """
        if hdc_vector.dim() == 1:
            hdc_vector = hdc_vector.unsqueeze(0)

        x = self.fc1(hdc_vector.float())
        x = torch.relu(x)
        x = self.fc2(x)
        return self.softmax(x)

    def classify(self, hdc_vector_numpy: np.ndarray):
        """Classify numpy HDC vector.

        Returns:
            (task_type_name, confidence, all_probs)
        """
        with torch.no_grad():
            x = torch.from_numpy(hdc_vector_numpy.astype(np.float32))
            probs = self.forward(x).squeeze(0)
            idx = probs.argmax().item()
            return self.task_types[idx], float(probs[idx]), probs.numpy()
