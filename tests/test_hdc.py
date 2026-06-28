"""Tests for HDC core operations."""

import numpy as np
import pytest

from td.perception.hdc import (
    HDCConfig, generate_hypervector, bind, bundle, permute,
    similarity, inverse, ConceptVocabulary, build_default_vocabulary,
    DEFAULT_CONCEPTS,
)


class TestHDCOperations:
    def test_generate_shape_and_values(self):
        v = generate_hypervector(dim=1000, seed=42)
        assert v.shape == (1000,)
        assert set(np.unique(v)).issubset({-1, 1})

    def test_generate_reproducible(self):
        a = generate_hypervector(dim=1000, seed=42)
        b = generate_hypervector(dim=1000, seed=42)
        assert np.array_equal(a, b)

    def test_bind_self_inverse(self):
        x = generate_hypervector(dim=10000, seed=1)
        y = generate_hypervector(dim=10000, seed=2)
        bound = bind(x, y)
        recovered = bind(bound, y)
        assert np.array_equal(recovered, x)

    def test_bundle_commutative(self):
        a = generate_hypervector(dim=10000, seed=1)
        b = generate_hypervector(dim=10000, seed=2)
        ab = bundle(a, b)
        ba = bundle(b, a)
        assert similarity(ab, ba) > 0.95

    def test_bind_distributive_over_bundle(self):
        a = generate_hypervector(dim=10000, seed=1)
        b = generate_hypervector(dim=10000, seed=2)
        c = generate_hypervector(dim=10000, seed=3)
        left = bind(bundle(a, b), c)
        right = bundle(bind(a, c), bind(b, c))
        assert similarity(left, right) > 0.8

    def test_similarity_identity(self):
        x = generate_hypervector(dim=10000, seed=42)
        assert similarity(x, x) == pytest.approx(1.0, abs=1e-6)

    def test_similarity_orthogonal(self):
        a = generate_hypervector(dim=10000, seed=1)
        b = generate_hypervector(dim=10000, seed=2)
        sim = similarity(a, b)
        assert -0.1 < sim < 0.1  # Approximately orthogonal

    def test_permute_changes_vector(self):
        x = generate_hypervector(dim=10000, seed=42)
        shifted = permute(x, shift=1)
        assert not np.array_equal(x, shifted)

    def test_permute_preserves_similarity(self):
        a = generate_hypervector(dim=10000, seed=1)
        b = generate_hypervector(dim=10000, seed=2)
        sim_orig = similarity(a, b)
        sim_shifted = similarity(permute(a, 3), permute(b, 3))
        assert sim_orig == pytest.approx(sim_shifted, abs=0.01)

    def test_inverse_is_identity(self):
        x = generate_hypervector(dim=1000, seed=42)
        assert np.array_equal(inverse(x), x)

    def test_noise_robustness(self):
        x = generate_hypervector(dim=10000, seed=42)
        noisy = x.copy()
        flip_idx = np.random.default_rng(1).choice(10000, 500, replace=False)
        noisy[flip_idx] *= -1
        assert similarity(x, noisy) > 0.9  # 5% noise → sim > 0.9


class TestConceptVocabulary:
    def test_build_default(self):
        vocab = build_default_vocabulary(dim=1000)
        assert len(vocab) >= len(DEFAULT_CONCEPTS)
        assert vocab.dim == 1000

    def test_get_concept(self):
        vocab = build_default_vocabulary(dim=1000)
        vec = vocab.get("click")
        assert vec.shape == (1000,)

    def test_add_concept(self):
        vocab = ConceptVocabulary(dim=1000)
        v = vocab.add_concept("test_concept")
        assert v.shape == (1000,)
        assert vocab.has("test_concept")

    def test_add_concept_reproducible(self):
        vocab = ConceptVocabulary(dim=1000)
        v1 = vocab.add_concept("my_concept")
        vocab2 = ConceptVocabulary(dim=1000)
        v2 = vocab2.add_concept("my_concept")
        assert np.array_equal(v1, v2)

    def test_encode_record(self):
        vocab = build_default_vocabulary(dim=10000)
        vec = vocab.encode_record(action="click", target="button")
        assert vec.shape == (10000,)
        # Should have positive similarity to both "click" and "button"
        assert similarity(vec, vocab.get("click")) > 0.1
        assert similarity(vec, vocab.get("button")) > 0.1

    def test_encode_sequence(self):
        vocab = build_default_vocabulary(dim=10000)
        vec = vocab.encode_sequence("click", "submit", "login")
        assert vec.shape == (10000,)
