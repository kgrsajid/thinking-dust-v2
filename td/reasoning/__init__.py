"""Reasoning module — confidence scoring, constraint schemas, contradiction detection.

Z3 is handled directly in td/thinking.py (entity-driven, no bridge needed).
"""

from .confidence import ConfidenceScore, compute_confidence
from .contradiction_detector import (
    ContradictionDetector,
    ContradictionWarning,
    RELATION_SCHEMA,
    DISJOINT_TYPES,
    TYPE_HIERARCHY,
)

__all__ = [
    "ConfidenceScore", "compute_confidence",
    "ContradictionDetector", "ContradictionWarning",
    "RELATION_SCHEMA", "DISJOINT_TYPES", "TYPE_HIERARCHY",
]
