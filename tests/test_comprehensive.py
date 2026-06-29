"""Comprehensive edge-case and integration tests.
Based on external code review findings (Grok Code-Fast-1).
"""

import json
import os
import tempfile

import numpy as np
import pytest

from td.perception.hdc import (
    HDCConfig, generate_hypervector, bind, bundle, permute,
    similarity, inverse, ConceptVocabulary, build_default_vocabulary,
)
from td.perception.ca_reservoir import CAReservoir, CAConfig
from td.memory.mhn import ModernHopfieldNetwork, MHNConfig
from td.routing.hierarchical_router import HierarchicalRouter
from td.pipeline import TDPipeline, TDDecision


# =========================================================================
# Reproducibility Tests (Bug #1: hash(name) was salted per-process)
# =========================================================================

class TestReproducibility:
    def test_add_concept_reproducible_across_instances(self):
        """Same concept name → same vector in different ConceptVocabulary instances."""
        v1 = ConceptVocabulary(dim=1000)
        v2 = ConceptVocabulary(dim=1000)
        vec1 = v1.add_concept("test_concept_xyz")
        vec2 = v2.add_concept("test_concept_xyz")
        assert np.array_equal(vec1, vec2), (
            "add_concept must be reproducible across instances (bug #1)"
        )

    def test_add_concept_different_names_different_vectors(self):
        """Different concept names → different vectors."""
        vocab = ConceptVocabulary(dim=1000)
        v1 = vocab.add_concept("alpha")
        v2 = vocab.add_concept("beta")
        assert not np.array_equal(v1, v2)

    def test_build_default_vocabulary_reproducible(self):
        """build_default_vocabulary produces same vectors every time."""
        v1 = build_default_vocabulary(dim=1000)
        v2 = build_default_vocabulary(dim=1000)
        assert np.array_equal(v1.get("click"), v2.get("click"))


# =========================================================================
# Dimension Validation Tests (Bug #2, #5, #6)
# =========================================================================

class TestDimensionValidation:
    def test_add_concept_wrong_dim_raises(self):
        """Adding a vector with wrong dimension should assert."""
        vocab = ConceptVocabulary(dim=1000)
        wrong_vec = generate_hypervector(dim=500)
        with pytest.raises(ValueError):
            vocab.add_concept("bad", wrong_vec)

    def test_bind_shape_mismatch_raises(self):
        """bind() with mismatched shapes should raise."""
        a = generate_hypervector(dim=1000)
        b = generate_hypervector(dim=500)
        with pytest.raises(ValueError):
            bind(a, b)

    def test_load_clears_existing(self):
        """load() should clear existing concepts first (bug #5)."""
        vocab = ConceptVocabulary(dim=100)
        vocab.add_concept("old_concept")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"new_concept": [1, -1] * 50}, f)
            f.flush()
            path = f.name
        try:
            vocab.load(path)
            assert "old_concept" not in vocab
            assert "new_concept" in vocab
        finally:
            os.unlink(path)

    def test_load_validates_dimensions(self):
        """load() should reject vectors with wrong dimension."""
        vocab = ConceptVocabulary(dim=100)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"bad": [1, -1] * 25}, f)  # dim=50, not 100
            f.flush()
            path = f.name
        try:
            with pytest.raises(ValueError):
                vocab.load(path)
        finally:
            os.unlink(path)


# =========================================================================
# Edge Case Tests
# =========================================================================

class TestEdgeCases:
    def test_similarity_empty_vectors(self):
        """similarity on empty arrays returns nan (division by zero)."""
        empty = np.array([], dtype=np.int8)
        result = similarity(empty, empty)
        assert np.isnan(result)

    def test_bundle_single_vector(self):
        """bundle with one vector should return that vector."""
        v = generate_hypervector(dim=1000, seed=42)
        result = bundle(v)
        assert np.array_equal(result, v)

    def test_bundle_many_vectors(self):
        """bundle should handle many vectors without overflow."""
        vectors = [generate_hypervector(dim=1000, seed=i) for i in range(100)]
        result = bundle(*vectors)
        assert result.shape == (1000,)
        assert set(np.unique(result)).issubset({-1, 1})

    def test_permute_full_rotation(self):
        """permute by dim should equal original."""
        v = generate_hypervector(dim=1000, seed=42)
        assert np.array_equal(permute(v, shift=1000), v)

    def test_permute_zero_shift(self):
        """permute by 0 should equal original."""
        v = generate_hypervector(dim=1000, seed=42)
        assert np.array_equal(permute(v, shift=0), v)

    def test_ca_empty_input_produces_valid_vector(self):
        """CA reservoir with empty input should not crash."""
        ca = CAReservoir(CAConfig(input_dim=500, steps=5))
        result = ca.evolve(np.array([], dtype=np.uint8))
        assert result.shape == (500,)
        assert set(np.unique(result)).issubset({-1, 1})

    def test_ca_single_bit_input(self):
        """CA reservoir with single bit should not crash."""
        ca = CAReservoir(CAConfig(input_dim=500, steps=5))
        result = ca.evolve(np.array([1], dtype=np.uint8))
        assert result.shape == (500,)

    def test_mhn_store_many_patterns(self):
        """MHN should handle 100+ patterns without degradation."""
        mhn = ModernHopfieldNetwork(MHNConfig(dim=2000, min_similarity=0.1, beta=2.0))
        keys = []
        for i in range(100):
            k = generate_hypervector(dim=2000, seed=i * 7)
            v = generate_hypervector(dim=2000, seed=i * 7 + 1)
            mhn.store(k, v, {"id": i})
            keys.append(k)

        # Verify retrieval for random sample
        for idx in [0, 25, 50, 75, 99]:
            results = mhn.retrieve(keys[idx], top_k=1)
            assert len(results) > 0
            assert results[0][2]["id"] == idx

    def test_mhn_retrieve_top_k(self):
        """retrieve with top_k=5 should return up to 5 results."""
        mhn = ModernHopfieldNetwork(MHNConfig(dim=1000, min_similarity=0.0, beta=0.5))
        for i in range(20):
            k = generate_hypervector(dim=1000, seed=i * 3)
            v = generate_hypervector(dim=1000, seed=i * 3 + 100)
            mhn.store(k, v, {"id": i})

        query = generate_hypervector(dim=1000, seed=0)
        results = mhn.retrieve(query, top_k=5)
        assert len(results) <= 5

    def test_router_all_domains_classifiable(self):
        """Router should be able to classify all 5 domains without crashing."""
        router = HierarchicalRouter(input_dim=1000)
        vocab = build_default_vocabulary(dim=1000)
        from td.perception.nl_parser import NLParser
        parser = NLParser(vocab)

        test_inputs = [
            "Click the submit button",           # Web
            "Fetch user data from the API",       # API
            "Parse the CSV file",                 # File
            "Alert if CPU exceeds 90 percent",    # Monitor
            "Design a marketing strategy",        # Unknown
        ]

        for text in test_inputs:
            vec = parser.parse(text)
            result = router.route(vec)
            assert result.domain in ["Web", "API", "File", "Monitor", "Unknown"]


# =========================================================================
# Mathematical Property Tests
# =========================================================================

class TestHDCAlgebra:
    def test_bind_associative(self):
        """bind(bind(a,b),c) == bind(a, bind(b,c))."""
        a = generate_hypervector(dim=5000, seed=1)
        b = generate_hypervector(dim=5000, seed=2)
        c = generate_hypervector(dim=5000, seed=3)
        left = bind(bind(a, b), c)
        right = bind(a, bind(b, c))
        assert np.array_equal(left, right)

    def test_bundle_associative(self):
        """bundle(a,b,c) has same similarity regardless of grouping."""
        a = generate_hypervector(dim=5000, seed=1)
        b = generate_hypervector(dim=5000, seed=2)
        c = generate_hypervector(dim=5000, seed=3)

        result1 = bundle(bundle(a, b), c)
        result2 = bundle(a, bundle(b, c))

        # With deterministic tie-breaking, should be identical
        assert np.array_equal(result1, result2)

    def test_double_bind_cancels(self):
        """bind(bind(x, y), bind(x, y)) == all-ones (since x⊗x = +1s)."""
        x = generate_hypervector(dim=1000, seed=1)
        y = generate_hypervector(dim=1000, seed=2)
        double = bind(bind(x, y), bind(x, y))
        assert np.all(double == 1)

    def test_similarity_bound(self):
        """Similarity is always in [-1, 1]."""
        for seed_a in range(10):
            for seed_b in range(10):
                a = generate_hypervector(dim=1000, seed=seed_a)
                b = generate_hypervector(dim=1000, seed=seed_b)
                sim = similarity(a, b)
                assert -1.01 <= sim <= 1.01

    def test_permute_similarity_invariance(self):
        """Permutation preserves similarity."""
        a = generate_hypervector(dim=3000, seed=1)
        b = generate_hypervector(dim=3000, seed=2)
        sim_orig = similarity(a, b)
        sim_shifted = similarity(permute(a, 7), permute(b, 7))
        assert abs(sim_orig - sim_shifted) < 0.05


# =========================================================================
# Integration Tests
# =========================================================================

class TestIntegration:
    @pytest.fixture(scope="class")
    def pipeline(self):
        return TDPipeline(dim=1000)

    def test_full_cycle_learn_decide_learn(self, pipeline):
        """Learn → decide → learn correction → decide again."""
        situation = "Fill out the registration form with name and email"

        # First decision (no memory)
        d1 = pipeline.decide(situation)
        assert isinstance(d1, TDDecision)

        # Learn a successful pattern
        pipeline.learn(
            situation,
            [{"action": "click", "target": "name_field"},
             {"action": "click", "target": "submit"}],
            "success",
            {"domain": "Web"},
        )

        # Second decision (with memory)
        d2 = pipeline.decide(situation)
        assert isinstance(d2, TDDecision)
        assert len(pipeline.mhn) > 0

    def test_pipeline_with_all_input_types(self, pipeline):
        """Pipeline handles NL, DOM, API, and metrics inputs."""
        # NL
        d1 = pipeline.decide("Click submit button")
        assert d1.routing.domain is not None

        # Metrics
        vec = pipeline.perceive(
            {"cpu": 95.0, "memory": 80.0, "service": "nginx"},
            "metrics",
        )
        assert vec.shape == (1000,)

        # API
        vec2 = pipeline.perceive(
            {"user": "alice", "id": 123},
            "api",
        )
        assert vec2.shape == (1000,)

    def test_save_load_roundtrip(self, pipeline, tmp_path):
        """Save and load preserves router + vocabulary state."""
        pipeline.save_state(str(tmp_path / "state"))

        # Create new pipeline and load
        p2 = TDPipeline(dim=1000)
        p2.load_state(str(tmp_path / "state"))

        # Both should produce same routing for same input
        test_input = "Click the submit button"
        from td.perception.nl_parser import NLParser
        parser1 = NLParser(pipeline.vocab)
        parser2 = NLParser(p2.vocab)
        vec1 = parser1.parse(test_input)
        vec2 = parser2.parse(test_input)
        assert np.array_equal(vec1, vec2)


# =========================================================================
# Stress Tests
# =========================================================================

class TestStress:
    def test_mhn_large_capacity(self):
        """MHN should handle 500 patterns with good retrieval."""
        dim = 3000
        mhn = ModernHopfieldNetwork(MHNConfig(dim=dim, min_similarity=0.1, beta=2.0))
        keys = []
        for i in range(500):
            k = generate_hypervector(dim=dim, seed=i * 13)
            v = generate_hypervector(dim=dim, seed=i * 13 + 7)
            mhn.store(k, v, {"idx": i})
            keys.append(k)

        # Check retrieval accuracy on sample
        correct = 0
        sample = [0, 100, 250, 499]
        for idx in sample:
            results = mhn.retrieve(keys[idx], top_k=1)
            if results and results[0][2]["idx"] == idx:
                correct += 1
        assert correct >= 3  # At least 3/4 correct

    def test_router_high_dimensional(self):
        """Router should work with 10K-dim vectors (production size)."""
        router = HierarchicalRouter(input_dim=10000)
        vocab = build_default_vocabulary(dim=10000)
        from td.perception.nl_parser import NLParser
        parser = NLParser(vocab)

        vec = parser.parse("Click the submit button on the login form")
        result = router.route(vec)
        assert result.domain is not None
        assert 0 <= result.combined_confidence <= 1.0

    def test_rapid_sequential_decisions(self):
        """Pipeline handles many rapid decisions without state corruption."""
        pipeline = TDPipeline(dim=500)
        for i in range(50):
            decision = pipeline.decide(f"Click button number {i}")
            assert isinstance(decision, TDDecision)
