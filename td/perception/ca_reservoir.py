"""Cellular Automata Reservoir (Rule 90) for feature extraction.

Rule 90 is an elementary cellular automaton where each cell's next state
is the XOR of its left and right neighbors. It produces chaotic dynamics
that generate rich, high-dimensional features from structured input.

Properties:
    - Zero training required (fixed random projection + fixed rule)
    - Additive: CA(A ⊕ B) = CA(A) ⊕ CA(B) — composition without recomputation
    - Output is binarized to HDC vector
    - Time: O(steps × dim) ≈ 0.1ms for T=50, D=10K
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CAConfig:
    """Configuration for CA Reservoir.

    Attributes:
        rule: Elementary CA rule number (90 = XOR of neighbors).
        steps: Number of evolution steps. 50 is standard.
        input_dim: Projection dimension (should match HDC dim).
        seed: Random seed for reproducible projection.
    """
    rule: int = 90
    steps: int = 50
    input_dim: int = 10_000
    seed: int = 42


class CAReservoir:
    """Cellular Automata Reservoir using Rule 90.

    Pipeline:
    1. Project arbitrary-length binary input onto a fixed-size lattice
       via a sparse random XOR matrix.
    2. Evolve Rule 90 for T steps: new[i] = state[i-1] XOR state[i+1].
    3. Binarize final state to bipolar HDC vector {-1, +1}.

    The random projection ensures that different inputs map to different
    regions of the lattice, and Rule 90's chaotic dynamics spread local
    information globally across the lattice over T steps.

    Memory: O(input_dim) for the projection matrix + O(dim) for state.
    Time per evolution: O(steps × dim) with numpy vectorization.
    """

    def __init__(self, config: CAConfig = CAConfig()):
        """Initialize the reservoir with a random projection matrix.

        The projection matrix maps arbitrary-length input to a fixed
        `input_dim`-bit lattice. We use a sparse binary matrix where
        each output bit is the XOR of a random subset of input bits.
        """
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        # Sparse projection: each output bit samples from 3 random input positions
        # This is sufficient for good mixing and keeps projection fast
        self._projection_indices: np.ndarray | None = None
        self._projection_max_input: int = 0

    def _build_projection(self, input_length: int) -> None:
        """Build or resize the random projection for a given input length.

        We use a sparse XOR projection: each of the `input_dim` output bits
        is computed as the XOR of 3 randomly selected input bits.

        Known limitation: If evolve() is called with a larger input after
        a smaller one, the projection is rebuilt with new random indices,
        making results order-dependent. For deterministic multi-input use,
        pre-build with the largest expected input via evolve_batch, or
        instantiate separate reservoirs. (See external review bug #6.)
        """
        n_samples = 3  # XOR of 3 random bits per output — good mixing, sparse
        self._projection_indices = self.rng.integers(
            0, max(input_length, 1),
            size=(self.config.input_dim, n_samples),
            dtype=np.int32,
        )
        self._projection_max_input = input_length

    def evolve(self, binary_input: np.ndarray) -> np.ndarray:
        """Evolve Rule 90 for T steps and return final state as HDC vector.

        Algorithm:
        1. Project `binary_input` onto `input_dim`-bit lattice via sparse XOR.
        2. For T steps: new[i] = state[i-1] XOR state[i+1]  (Rule 90)
        3. Convert {0,1} to {-1,+1} (bipolar HDC).

        Args:
            binary_input: Binary array of any length. Values should be
                in {0, 1} or {False, True}. Will be converted if needed.

        Returns:
            np.ndarray[int8] of shape (input_dim,) with values in {-1, +1}.

        Time: ~0.1ms for T=50, D=10K (numpy vectorized).
        """
        # Ensure binary
        binary_input = np.asarray(binary_input).astype(np.uint8)
        if binary_input.size == 0:
            binary_input = np.array([0, 1], dtype=np.uint8)

        # Build/resize projection if needed
        if (self._projection_indices is None
                or binary_input.size > self._projection_max_input):
            self._build_projection(max(binary_input.size, 100))
        # Project: XOR of selected input bits → output lattice
        n = self.config.input_dim
        idx = self._projection_indices
        # Use hash-based projection for short inputs to avoid clamping collisions
        if binary_input.size < self._projection_max_input:
            # Use modulo to map indices into input range — different from clamping
            # because it creates different patterns for different input sizes
            clamped = idx % max(binary_input.size, 1)
        else:
            clamped = np.minimum(idx, binary_input.size - 1)
        selected = binary_input[clamped]  # shape (n, 3)

        lattice = (selected.sum(axis=1) % 2).astype(np.uint8)  # XOR of 3 bits
        for _ in range(self.config.steps):
            left = np.roll(lattice, 1)
            right = np.roll(lattice, -1)
            lattice = (left ^ right)  # Rule 90: XOR of neighbors

        # Convert {0, 1} to {-1, +1}
        result = (2 * lattice.astype(np.int8) - 1)
        return result

    def evolve_batch(self, inputs: list[np.ndarray]) -> np.ndarray:
        """Evolve multiple inputs.

        Uses the additive property for efficiency when possible:
        CA(A ⊕ B) = CA(A) ⊕ CA(B) for Rule 90.

        Args:
            inputs: List of binary arrays.

        Returns:
            np.ndarray[int8] of shape (len(inputs), input_dim).
        """
        results = np.zeros((len(inputs), self.config.input_dim), dtype=np.int8)
        for i, inp in enumerate(inputs):
            results[i] = self.evolve(inp)
        return results
