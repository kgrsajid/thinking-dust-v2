from .confidence import ConfidenceScore, compute_confidence
from .constraint_schemas import (
    WEB_FORM_CONSTRAINTS, API_SEQUENTIAL_CONSTRAINTS,
    FILE_PARSE_CONSTRAINTS, MONITOR_THRESHOLD_CONSTRAINTS,
)

# Z3Bridge imported lazily to avoid crashing if z3 native lib is unavailable
from .z3_bridge import Z3Result

def get_z3_bridge(template_dir=None):
    """Factory for Z3Bridge — safe to call even if z3 is unavailable."""
    from .z3_bridge import Z3Bridge
    return Z3Bridge(template_dir)

__all__ = [
    "Z3Result", "get_z3_bridge", "ConfidenceScore", "compute_confidence",
    "WEB_FORM_CONSTRAINTS", "API_SEQUENTIAL_CONSTRAINTS",
    "FILE_PARSE_CONSTRAINTS", "MONITOR_THRESHOLD_CONSTRAINTS",
]
