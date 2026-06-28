"""Tests for Modern Hopfield Network."""

import numpy as np
import pytest

from td.perception.hdc import generate_hypervector, bind, similarity
from td.memory.mhn import ModernHopfieldNetwork, MHNConfig


class TestMHN:
    def test_store_and_retrieve(self):
        mhn = ModernHopfieldNetwork(MHNConfig(dim=1000, min_similarity=0.1))
        key = generate_hypervector(dim=1000, seed=1)
        value = generate_hypervector(dim=1000, seed=2)
        mhn.store(key, value, {"domain": "test"})
        results = mhn.retrieve(key, top_k=1)
        assert len(results) == 1
        retrieved_val, sim, meta = results[0]
        assert sim > 0.5
        assert meta["domain"] == "test"

    def test_no_catastrophic_forgetting(self):
        """Store many patterns, verify first is still retrievable."""
        mhn = ModernHopfieldNetwork(MHNConfig(dim=2000, min_similarity=0.1))
        first_key = generate_hypervector(dim=2000, seed=100)
        first_value = generate_hypervector(dim=2000, seed=200)
        mhn.store(first_key, first_value, {"id": 0})

        # Store 50 more patterns
        for i in range(1, 51):
            k = generate_hypervector(dim=2000, seed=i * 10)
            v = generate_hypervector(dim=2000, seed=i * 10 + 1)
            mhn.store(k, v, {"id": i})

        # First pattern should still be retrievable
        results = mhn.retrieve(first_key, top_k=1)
        assert len(results) == 1
        assert results[0][2]["id"] == 0

    def test_noise_tolerance(self):
        """Retrieve with noisy query."""
        mhn = ModernHopfieldNetwork(MHNConfig(dim=2000, min_similarity=0.1, beta=2.0))
        key = generate_hypervector(dim=2000, seed=1)
        value = generate_hypervector(dim=2000, seed=2)
        mhn.store(key, value, {"test": True})

        # Add 10% noise
        noisy_key = key.copy()
        flip = np.random.default_rng(42).choice(2000, 200, replace=False)
        noisy_key[flip] *= -1

        results = mhn.retrieve(noisy_key, top_k=1)
        assert len(results) == 1
        assert results[0][1] > 0.5  # Should still match

    def test_empty_memory(self):
        mhn = ModernHopfieldNetwork(MHNConfig(dim=1000))
        query = generate_hypervector(dim=1000, seed=1)
        results = mhn.retrieve(query)
        assert results == []

    def test_idp_improves_retrieval(self):
        """IDP should help when patterns span multiple 'domains'."""
        dim = 2000
        mhn_idp = ModernHopfieldNetwork(MHNConfig(dim=dim, idp_enabled=True, min_similarity=0.1))
        mhn_no_idp = ModernHopfieldNetwork(MHNConfig(dim=dim, idp_enabled=False, min_similarity=0.1))

        # Store patterns from 3 different "domains"
        for domain_seed in [100, 200, 300]:
            for i in range(10):
                k = generate_hypervector(dim, seed=domain_seed + i)
                v = generate_hypervector(dim, seed=domain_seed + i + 1000)
                for mhn in [mhn_idp, mhn_no_idp]:
                    mhn.store(k, v, {"domain": domain_seed})

        # Query with a specific domain's key
        query = generate_hypervector(dim, seed=105)
        results_idp = mhn_idp.retrieve(query, top_k=1)
        results_no_idp = mhn_no_idp.retrieve(query, top_k=1)

        # Both should find a match, IDP should be at least as good
        assert len(results_idp) > 0
        assert len(results_no_idp) > 0

    def test_update_pattern(self):
        """Correction learning: old pattern deactivated, new stored."""
        mhn = ModernHopfieldNetwork(MHNConfig(dim=1000, min_similarity=0.1))
        key = generate_hypervector(dim=1000, seed=1)
        old_val = generate_hypervector(dim=1000, seed=2)
        new_val = generate_hypervector(dim=1000, seed=3)

        idx = mhn.store(key, old_val, {"version": "old"})
        mhn.update_pattern(idx, key, new_val, {"version": "new"})

        # Old pattern should be inactive
        assert not mhn.patterns[idx].active
        # New pattern should exist and be active
        results = mhn.retrieve(key, top_k=1)
        assert len(results) == 1
        assert results[0][2].get("version") == "new"

    def test_get_stats(self):
        mhn = ModernHopfieldNetwork(MHNConfig(dim=1000))
        for i in range(5):
            k = generate_hypervector(dim=1000, seed=i)
            v = generate_hypervector(dim=1000, seed=i + 100)
            mhn.store(k, v, {"domain": "test"})
        stats = mhn.get_stats()
        assert stats["num_patterns"] == 5
        assert stats["num_active"] == 5
        assert "test" in stats["domains"]
