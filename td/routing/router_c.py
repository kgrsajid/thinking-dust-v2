"""Router C — Strategy Selector.

Picks execution pathway: MEMORY_ONLY, MEMORY_THEN_VALIDATE, or ESCALATE.
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
        TernaryLinear(10_000, 32) → ReLU → TernaryLinear(32, 3) → Softmax

    Input: HDC vector + retrieved MHN pattern (bundled together)
    Output: softmax probabilities over 3 strategies

    Decision logic:
        MEMORY_ONLY:         high router confidence + high MHN similarity
        MEMORY_THEN_VALIDATE: high router confidence + medium MHN similarity
        ESCALATE:            low router confidence OR low MHN similarity
    """

    def __init__(self, input_dim: int = 10_000, hidden_dim: int = 32):
        super().__init__()
        self.fc1 = TernaryLinear(input_dim, hidden_dim)
        self.fc2 = TernaryLinear(hidden_dim, len(STRATEGIES))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, hdc_vector: torch.Tensor) -> torch.Tensor:
        """Classify HDC vector into strategy probabilities.

        Args:
            hdc_vector: HDC vector (query bundled with MHN retrieval).
                        Shape: (batch, input_dim) or (input_dim,).

        Returns:
            Softmax probabilities of shape (batch, 3) or (3,).
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
            (strategy_name, confidence, all_probs)
        """
        with torch.no_grad():
            x = torch.from_numpy(hdc_vector_numpy.astype(np.float32))
            probs = self.forward(x).squeeze(0)
            idx = probs.argmax().item()
            return STRATEGIES[idx], float(probs[idx]), probs.numpy()
