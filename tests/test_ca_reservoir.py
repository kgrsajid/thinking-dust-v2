"""Tests for CA Reservoir."""

import numpy as np
import pytest

from td.perception.ca_reservoir import CAReservoir, CAConfig


class TestCAReservoir:
    def test_output_dimension(self):
        ca = CAReservoir(CAConfig(input_dim=1000, steps=10))
        result = ca.evolve(np.array([1, 0, 1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8))
        assert result.shape == (1000,)

    def test_output_values_bipolar(self):
        ca = CAReservoir(CAConfig(input_dim=500, steps=5))
        result = ca.evolve(np.array([1, 0, 1], dtype=np.uint8))
        assert set(np.unique(result)).issubset({-1, 1})

    def test_different_inputs_different_outputs(self):
        ca = CAReservoir(CAConfig(input_dim=1000, steps=20, seed=42))
        # Use genuinely different (non-complementary) inputs
        out1 = ca.evolve(np.array([1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1], dtype=np.uint8))
        out2 = ca.evolve(np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0], dtype=np.uint8))
        from td.perception.hdc import similarity
        sim = similarity(out1, out2)
        assert sim < 0.5  # Different inputs should not be identical

    def test_empty_input(self):
        ca = CAReservoir(CAConfig(input_dim=100, steps=5))
        result = ca.evolve(np.array([], dtype=np.uint8))
        assert result.shape == (100,)

    def test_evolve_batch(self):
        ca = CAReservoir(CAConfig(input_dim=500, steps=10))
        inputs = [
            np.array([1, 0, 1, 0], dtype=np.uint8),
            np.array([0, 1, 0, 1], dtype=np.uint8),
        ]
        results = ca.evolve_batch(inputs)
        assert results.shape == (2, 500)

    def test_reproducible(self):
        ca1 = CAReservoir(CAConfig(input_dim=1000, steps=10, seed=42))
        ca2 = CAReservoir(CAConfig(input_dim=1000, steps=10, seed=42))
        inp = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
        assert np.array_equal(ca1.evolve(inp), ca2.evolve(inp))
