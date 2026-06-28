"""Hyperdimensional Computing (HDC) core operations.

Bipolar Sparse Codes (BSC) with {-1, +1} vectors.
All operations are deterministic, O(dim), and require zero training.

Properties:
    - bind(x, y) is self-inverse: bind(bind(x, y), y) == x
    - bundle() is commutative and associative
    - bind distributes over bundle
    - Random vectors are approximately orthogonal (similarity ≈ 0)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class HDCConfig:
    """Configuration for HDC operations.

    Attributes:
        dim: Hypervector dimensionality. 10,000 is standard for HDC.
        dtype: numpy dtype for storage. int8 = 1 byte per element.
        seed: Random seed for reproducible concept vectors.
    """
    dim: int = 10_000
    dtype: type = np.int8
    seed: int = 42


HDC_CONFIG = HDCConfig()


# ---------------------------------------------------------------------------
# Core HDC Operations
# ---------------------------------------------------------------------------

def generate_hypervector(
    dim: int = 10_000,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a random bipolar hypervector from {-1, +1}^D.

    Each element is independently sampled from {-1, +1} with equal
    probability. The resulting vector is approximately orthogonal
    to all other random vectors (similarity ≈ 0 ± 2/√D).

    Args:
        dim: Dimensionality of the vector.
        seed: Optional random seed for reproducibility.

    Returns:
        np.ndarray[int8] of shape (dim,).

    Time: O(dim), ~0.05ms for D=10K.
    """
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=dim).astype(np.int8)


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """HDC binding (association) via element-wise multiplication.

    BSC binding is self-inverse: bind(bind(x, y), y) == x.
    This enables algebraic queries: given bind(concept, role),
    multiply by role to recover concept.

    Args:
        a: Bipolar hypervector.
        b: Bipolar hypervector.

    Returns:
        Bipolar hypervector, same shape as inputs.

    Time: O(dim).
    """
    return (a * b).astype(np.int8)


def bundle(*vectors: np.ndarray) -> np.ndarray:
    """HDC bundling (superposition) via majority vote.

    Computes sign(sum(vectors)). Ties (sum == 0) are resolved as +1.
    Works with a single vector (returns it unchanged).

    Args:
        *vectors: One or more bipolar hypervectors of the same shape.

    Returns:
        Bipolar hypervector.

    Time: O(n * dim) where n = number of vectors.
    """
    if len(vectors) == 1:
        return vectors[0].astype(np.int8)
    if len(vectors) < 1:
        raise ValueError("bundle requires at least 1 vector")
    stacked = np.stack(vectors, axis=0)
    total = stacked.sum(axis=0)
    total[total == 0] = 1  # Deterministic tie-breaking
    return np.sign(total).astype(np.int8)


def permute(v: np.ndarray, shift: int = 1) -> np.ndarray:
    """Cyclic permutation (rotation) of a hypervector.

    permute(x, k)[i] = x[(i - k) % D]

    This encodes sequence/position information. Permutation is
    distance-preserving: similarity(permute(x, k), permute(y, k))
    = similarity(x, y).

    Args:
        v: Bipolar hypervector.
        shift: Number of positions to rotate. Positive = right shift.

    Returns:
        Bipolar hypervector, same shape as input.

    Time: O(dim).
    """
    return np.roll(v, shift=shift).astype(np.int8)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for bipolar vectors.

    For bipolar vectors, cosine similarity = dot(a, b) / dim
    = (number of matching bits - number of differing bits) / dim.

    Range: [-1.0, 1.0]
        > 0.5  → same concept
        ~ 0.0  → unrelated
        < -0.3 → anti-correlated

    Args:
        a: Bipolar hypervector.
        b: Bipolar hypervector.

    Returns:
        Similarity score in [-1.0, 1.0].

    Time: O(dim).
    """
    return float(np.dot(a.astype(np.float32), b.astype(np.float32)) / len(a))


def inverse(v: np.ndarray) -> np.ndarray:
    """Multiplicative inverse for BSC binding.

    For bipolar vectors, the inverse is the vector itself:
    bind(x, y) ⊗ y⁻¹ = x, and y⁻¹ = y.

    This exists for API completeness and clarity of intent.

    Args:
        v: Bipolar hypervector.

    Returns:
        Same vector (BSC is self-inverse).
    """
    return v.copy()


# ---------------------------------------------------------------------------
# Concept Vocabulary
# ---------------------------------------------------------------------------

class ConceptVocabulary:
    """Manages pre-encoded concept hypervectors.

    A vocabulary of ~1000 concepts, each mapped to a random 10K-dim
    bipolar vector. Concepts cover: web actions, API operations,
    file operations, monitoring terms, scheduling terms, common objects.

    The vocabulary is loaded from a JSON file at startup and can be
    extended at runtime. All vectors are reproducible from a fixed seed.

    Attributes:
        dim: Dimensionality of each concept vector.
        concepts: Dict mapping concept name → hypervector.
    """

    def __init__(self, path: str | Path | None = None, dim: int = 10_000):
        """Initialize vocabulary, optionally loading from file.

        Args:
            path: Path to JSON file with concept vectors.
                  Format: {"concept_name": [-1, 1, 1, -1, ...], ...}
            dim: Dimensionality (used if no file or for new concepts).
        """
        self.dim = dim
        self.concepts: dict[str, np.ndarray] = {}
        self._counter = 0
        if path is not None:
            self.load(path)

    def get(self, concept: str) -> np.ndarray:
        """Retrieve hypervector for a concept name.

        Args:
            concept: Concept name (case-sensitive).

        Returns:
            Bipolar hypervector.

        Raises:
            KeyError: If concept not in vocabulary.
        """
        return self.concepts[concept]

    def has(self, concept: str) -> bool:
        """Check if a concept exists in the vocabulary."""
        return concept in self.concepts

    def add_concept(self, name: str, vector: np.ndarray | None = None) -> np.ndarray:
        """Add a new concept to the vocabulary.

        If no vector is provided, a random one is generated.
        The random seed is derived from the concept name hash
        for reproducibility.

        Args:
            name: Concept name.
            vector: Optional pre-defined hypervector.

        Returns:
            The concept's hypervector.
        """
        if name in self.concepts:
            return self.concepts[name]
        if vector is None:
            # Reproducible seed from name
            seed = hash(name) % (2**32)
            vector = generate_hypervector(self.dim, seed=seed)
        vector = vector.astype(np.int8)
        self.concepts[name] = vector
        self._counter += 1
        return vector

    def encode_record(self, **kwargs: str) -> np.ndarray:
        """Encode a key-value record as a single hypervector.

        Uses role-filler binding: for each (key, value) pair,
        binds the key concept with the value concept, then bundles
        all pairs together.

        Example:
            vocab.encode_record(action="click", target="submit_button",
                               context="login_form")

        Implementation:
            bundle(
                bind(get("action"), get("click")),
                bind(get("target"), get("submit_button")),
                bind(get("context"), get("login_form"))
            )

        Args:
            **kwargs: Key-value pairs to encode.

        Returns:
            Bipolar hypervector of shape (dim,).
        """
        if not kwargs:
            raise ValueError("encode_record requires at least one key-value pair")
        parts = []
        for role, filler in kwargs.items():
            role_vec = self.get(role) if self.has(role) else self.add_concept(role)
            filler_vec = self.get(filler) if self.has(filler) else self.add_concept(filler)
            parts.append(bind(role_vec, filler_vec))
        return bundle(*parts)

    def encode_sequence(self, *concepts: str) -> np.ndarray:
        """Encode an ordered sequence using permutation for position.

        position 0: bind(concept_0, permute(identity, 0))
        position 1: bind(concept_1, permute(identity, 1))
        etc.

        The identity vector is a fixed random vector stored as "_seq_identity".

        Args:
            *concepts: Concept names in order.

        Returns:
            Bipolar hypervector encoding the sequence.
        """
        if not concepts:
            raise ValueError("encode_sequence requires at least one concept")
        if not self.has("_seq_identity"):
            self.add_concept("_seq_identity")
        identity = self.get("_seq_identity")
        parts = []
        for i, concept_name in enumerate(concepts):
            c_vec = self.get(concept_name) if self.has(concept_name) else self.add_concept(concept_name)
            parts.append(bind(c_vec, permute(identity, shift=i)))
        return bundle(*parts)

    def save(self, path: str | Path) -> None:
        """Save vocabulary to JSON file."""
        data = {
            name: vec.tolist() for name, vec in self.concepts.items()
            if not name.startswith("_")  # Skip internal concepts
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str | Path) -> None:
        """Load vocabulary from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        for name, vec_list in data.items():
            self.concepts[name] = np.array(vec_list, dtype=np.int8)
        self._counter = len(data)

    def __len__(self) -> int:
        return len(self.concepts)

    def __contains__(self, name: str) -> bool:
        return name in self.concepts

    def __repr__(self) -> str:
        return f"ConceptVocabulary(dim={self.dim}, concepts={len(self.concepts)})"


# ---------------------------------------------------------------------------
# Default Vocabulary Builder
# ---------------------------------------------------------------------------

DEFAULT_CONCEPTS = [
    # Actions
    "click", "type", "scroll", "submit", "select", "hover", "drag",
    "navigate", "extract", "fill", "check", "uncheck", "toggle",
    "open", "close", "wait", "refresh", "back", "forward",
    # API actions
    "fetch", "post_data", "put_data", "delete_data", "patch_data", "call", "retry",
    "authenticate", "authorize", "parse_response", "transform_data",
    # File actions
    "read", "write", "create", "delete_file", "copy", "move",
    "validate", "convert", "import_data", "export_data",
    # Monitor actions
    "alert", "restart", "scale", "notify", "log", "diagnose",
    "threshold", "escalate_action", "monitor_action", "health_check",
    # Targets
    "button", "form", "field", "input_field", "dropdown", "checkbox",
    "link", "image", "table", "modal", "tab_elem", "menu",
    "submit_button", "login_form", "contact_form", "search_bar",
    # API targets
    "endpoint", "auth_token", "api_key", "response", "request",
    "header", "body_param", "query_param", "status_code",
    # File targets
    "csv", "json", "xml", "yaml", "tsv", "txt", "config",
    "schema", "column", "row", "record",
    # Monitor targets
    "cpu", "memory", "disk", "network", "service", "process",
    "nginx", "postgres", "redis", "docker",
    # Contexts / domains
    "web", "api", "file", "monitor", "unknown_domain",
    # Task types
    "form_type", "navigation", "extraction", "interaction",
    "sequential", "parallel", "error_handling",
    "parse_type", "transform", "generate",
    "log_analysis", "alert_routing", "routine",
    "complex", "proof_type", "novel", "ambiguous",
    # Strategies
    "memory_only", "memory_then_validate", "escalate",
    # Roles (for encode_record)
    "action", "target", "context", "value", "condition",
    "domain", "task_type", "strategy", "priority",
    # Outcomes
    "success", "failure", "partial", "pending",
    # Properties
    "visible", "hidden", "enabled", "disabled",
    "required", "optional", "valid", "invalid",
    "present", "absent", "empty", "full",
    # Values (discretized)
    "high", "medium", "low", "critical", "normal",
    # Common objects
    "user", "order", "product", "customer", "invoice",
    "name", "email", "phone", "address", "password",
    "username", "date", "time", "id_field",
    # Logic / Z3
    "and_op", "or_op", "not_op", "implies_op", "constraint",
    "satisfied", "violated", "sat", "unsat", "unknown_result",
    # Result
    "result", "plan", "decision", "confidence", "proof",
    "validated", "rejected",
]


def build_default_vocabulary(dim: int = 10_000, seed: int = 42) -> ConceptVocabulary:
    """Build the default concept vocabulary with ~200 core concepts.

    Each concept gets a reproducible random hypervector derived from
    a master seed. Additional concepts can be added at runtime.

    Args:
        dim: Hypervector dimensionality.
        seed: Master random seed.

    Returns:
        Populated ConceptVocabulary.
    """
    vocab = ConceptVocabulary(dim=dim)
    rng = np.random.default_rng(seed)
    for i, concept in enumerate(DEFAULT_CONCEPTS):
        vec = rng.choice([-1, 1], size=dim).astype(np.int8)
        vocab.concepts[concept] = vec
    vocab._counter = len(DEFAULT_CONCEPTS)
    return vocab
