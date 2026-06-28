"""JSON/API response → HDC vector encoder."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from .hdc import ConceptVocabulary, bind, bundle


class APIEncoder:
    """Encodes API responses and structured data into HDC vectors.

    Pipeline: JSON/dict → schema extraction → key-value binding → HDC

    Example:
        {"user": {"id": 123, "name": "Alice"}, "orders": [...]}
        → bundle(
            bind("entity", "user"), bind("id", "number"),
            bind("name", "string"), bind("entity", "orders"),
            bind("count", "multiple")
          )
    """

    def __init__(self, vocabulary: ConceptVocabulary):
        self.vocab = vocabulary

    def _type_to_concept(self, value: Any) -> str:
        """Map a Python value to a concept name."""
        if isinstance(value, bool):
            return "valid" if value else "invalid"
        elif isinstance(value, int):
            return "number"
        elif isinstance(value, float):
            return "number"
        elif isinstance(value, str):
            return "string"
        elif isinstance(value, list):
            return "multiple"
        elif isinstance(value, dict):
            return "entity"
        elif value is None:
            return "absent"
        return "unknown"

    def _encode_dict(self, data: dict, depth: int = 0, max_depth: int = 3) -> list[np.ndarray]:
        """Recursively encode a dictionary into HDC vectors.

        Args:
            data: Dictionary to encode.
            depth: Current recursion depth.
            max_depth: Maximum depth to prevent deep nesting.

        Returns:
            List of HDC vectors to be bundled.
        """
        parts: list[np.ndarray] = []
        if depth > max_depth:
            return parts

        for key, value in data.items():
            # Ensure key concept exists
            if not self.vocab.has(key):
                self.vocab.add_concept(key)

            if isinstance(value, dict):
                # Recurse into nested dict
                nested = self._encode_dict(value, depth + 1, max_depth)
                parts.extend(nested)
                type_concept = "entity"
            elif isinstance(value, list):
                type_concept = "multiple"
                if value and isinstance(value[0], dict):
                    nested = self._encode_dict(value[0], depth + 1, max_depth)
                    parts.extend(nested)
            else:
                type_concept = self._type_to_concept(value)

            if not self.vocab.has(type_concept):
                self.vocab.add_concept(type_concept)
            parts.append(bind(self.vocab.get(key), self.vocab.get(type_concept)))

        return parts

    def encode(self, data: dict | str, schema: dict | None = None) -> np.ndarray:
        """Encode JSON/dict into HDC vector.

        Args:
            data: Dictionary or JSON string.
            schema: Optional schema for type hints (not yet used).

        Returns:
            np.ndarray[int8] of shape (dim,).
        """
        if isinstance(data, str):
            data = json.loads(data)
        if not isinstance(data, dict):
            raise ValueError(f"APIEncoder expects dict or JSON string, got {type(data)}")

        parts = self._encode_dict(data)
        if not parts:
            from .hdc import generate_hypervector
            return generate_hypervector(self.vocab.dim)

        result = parts[0]
        for p in parts[1:]:
            result = bundle(result, p)
        return result
