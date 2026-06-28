"""System metrics → HDC vector encoder.

Discretizes continuous metric values into categorical bins for HDC encoding.
"""

from __future__ import annotations

import numpy as np

from .hdc import ConceptVocabulary, bind, bundle, generate_hypervector


def _discretize(value: float) -> str:
    """Discretize a 0-100 metric value into a concept.

    Args:
        value: Metric value (typically 0-100 percentage).

    Returns:
        One of "low", "medium", "high", "critical".
    """
    if value >= 90:
        return "critical"
    elif value >= 70:
        return "high"
    elif value >= 30:
        return "medium"
    else:
        return "low"


class MetricsEncoder:
    """Encodes system metrics into HDC vectors.

    Values are discretized into: "low" (<30), "medium" (30-70),
    "high" (70-90), "critical" (>90). Exact values are not preserved
    in HDC — only the pattern matters.

    Example:
        {"cpu": 92.5, "memory": 78.0, "disk": 45.0, "service": "nginx"}
        → bundle(
            bind("cpu", "critical"),
            bind("memory", "high"),
            bind("disk", "medium"),
            bind("service", "nginx")
          )
    """

    def __init__(self, vocabulary: ConceptVocabulary):
        self.vocab = vocabulary

    def encode(self, metrics: dict[str, float | str]) -> np.ndarray:
        """Encode system metrics into HDC vector.

        Args:
            metrics: Dict of metric name → value.
                Numeric values are discretized.
                String values are used as-is (concept lookup).

        Returns:
            np.ndarray[int8] of shape (dim,).
        """
        parts: list[np.ndarray] = []

        for key, value in metrics.items():
            if not self.vocab.has(key):
                self.vocab.add_concept(key)

            if isinstance(value, (int, float)):
                category = _discretize(float(value))
            else:
                category = str(value)

            if not self.vocab.has(category):
                self.vocab.add_concept(category)

            parts.append(bind(self.vocab.get(key), self.vocab.get(category)))

        if not parts:
            return generate_hypervector(self.vocab.dim)

        result = parts[0]
        for p in parts[1:]:
            result = bundle(result, p)
        return result
