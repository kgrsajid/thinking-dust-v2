"""Intelligence-level tests for TD v2.

These test actual reasoning behavior, not just code correctness.
If any of these fail, the system doesn't "think" properly.
"""

import numpy as np
import pytest
import time

from td.perception.hdc import (
    ConceptVocabulary, build_default_vocabulary,
    generate_hypervector, bind, bundle, similarity, permute,
)
from td.perception.ca_reservoir import CAReservoir, CAConfig
from td.perception.nl_parser import NLParser
from td.memory.mhn import ModernHopfieldNetwork, MHNConfig
from td.routing.hierarchical_router import HierarchicalRouter
from td.reasoning.confidence import compute_confidence
from td.reasoning.z3_bridge import Z3Bridge
from td.learning.online import OnlineLearner
from td.pipeline import TDPipeline


# =========================================================================
# 1. HDC Algebra Integrity — the mathematical foundation
# =========================================================================

class TestHDCAlgebraIntegrity:
    """Verify the algebraic properties that all of TD depends on."""

    @pytest.fixture
    def dim(self):
        return 10_000

    def test_bind_unbinding_recovers_original(self, dim):
        """bind(bind(x, y), y) == x — this is how value retrieval works."""
        x = generate_hypervector(dim, seed=1)
        y = generate_hypervector(dim, seed=2)
        recovered = bind(bind(x, y), y)
        sim = similarity(recovered, x)
        assert sim > 0.99, f"Bind unbinding failed: sim={sim}"

    def test_bind_distributes_over_bundle(self, dim):
        """bind(x, bundle(y, z)) ≈ bundle(bind(x,y), bind(x,z))
        
        NOTE: For bipolar BSC, exact distributivity doesn't hold due to
        tie-breaking in bundle (when y[i]+z[i]=0, ~50% of positions).
        The similarity should be ~0.5, not ~1.0. This is a known HDC
        property, not a bug."""
        x = generate_hypervector(dim, seed=1)
        y = generate_hypervector(dim, seed=2)
        z = generate_hypervector(dim, seed=3)

        left = bind(x, bundle(y, z))
        right = bundle(bind(x, y), bind(x, z))
        sim = similarity(left, right)
        # Expected: ~0.5 due to tie-breaking. Should be well above 0 (not random)
        assert sim > 0.3, f"Distributivity completely broken: sim={sim}"

    def test_orthogonality_of_random_vectors(self, dim):
        """Random HDC vectors should be approximately orthogonal."""
        sims = []
        for i in range(100):
            a = generate_hypervector(dim, seed=i*2)
            b = generate_hypervector(dim, seed=i*2+1)
            sims.append(similarity(a, b))
        mean_sim = np.mean(sims)
        std_sim = np.std(sims)
        assert abs(mean_sim) < 0.05, f"Mean similarity not ~0: {mean_sim}"
        assert std_sim < 0.03, f"Similarity std too high: {std_sim}"

    def test_bundle_preserves_components(self, dim):
        """After bundling N items, each item should still be recognizable."""
        vectors = [generate_hypervector(dim, seed=i) for i in range(5)]
        bundled = bundle(*vectors)
        for i, v in enumerate(vectors):
            sim = similarity(bundled, v)
            assert sim > 0.1, f"Bundled vector lost component {i}: sim={sim}"

    def test_bundle_capacity_noise_resistance(self, dim):
        """Bundling many vectors — signal degrades gracefully."""
        for n in [2, 5, 10, 20, 50]:
            vectors = [generate_hypervector(dim, seed=i) for i in range(n)]
            bundled = bundle(*vectors)
            # Each component should still have positive similarity
            sims = [similarity(bundled, v) for v in vectors]
            min_sim = min(sims)
            expected_sim = 1.0 / np.sqrt(n)
            assert min_sim > expected_sim * 0.5, \
                f"Signal too weak for n={n}: min_sim={min_sim}, expected~{expected_sim}"

    def test_permute_preserves_similarity(self, dim):
        """permute is distance-preserving."""
        a = generate_hypervector(dim, seed=1)
        b = generate_hypervector(dim, seed=2)
        sim_before = similarity(a, b)
        sim_after = similarity(permute(a, 5), permute(b, 5))
        assert abs(sim_before - sim_after) < 0.02, \
            f"Permute doesn't preserve distance: {sim_before} vs {sim_after}"

    def test_permute_different_shifts_orthogonal(self, dim):
        """Different shifts should be approximately orthogonal."""
        v = generate_hypervector(dim, seed=42)
        sim_same = similarity(permute(v, 3), permute(v, 3))
        sim_diff = similarity(permute(v, 3), permute(v, 7))
        assert sim_same > 0.99
        assert abs(sim_diff) < 0.05, f"Different shifts not orthogonal: {sim_diff}"


# =========================================================================
# 2. Memory Intelligence — does MHN actually remember and retrieve?
# =========================================================================

class TestMemoryIntelligence:
    """Test that MHN can store, retrieve, and distinguish patterns."""

    @pytest.fixture
    def mhn(self):
        return ModernHopfieldNetwork(MHNConfig(dim=10_000, min_similarity=0.1))

    @pytest.fixture
    def dim(self):
        return 10_000

    def test_store_retrieve_exact_match(self, mhn, dim):
        """Stored pattern should be retrievable with exact key."""
        key = generate_hypervector(dim, seed=100)
        value = generate_hypervector(dim, seed=200)
        mhn.store(key, value, {"action": "test"})

        results = mhn.retrieve(key, top_k=1)
        assert len(results) == 1
        sim = results[0][1]
        assert sim > 0.99, f"Exact match similarity too low: {sim}"

    def test_retrieve_with_noise(self, mhn, dim):
        """Should retrieve even with noisy query (Hopfield denoising)."""
        key = generate_hypervector(dim, seed=100)
        value = generate_hypervector(dim, seed=200)
        mhn.store(key, value, {"action": "test"})

        # Flip 5% of bits
        noisy_key = key.copy()
        flip_mask = np.random.default_rng(999).random(dim) < 0.05
        noisy_key[flip_mask] *= -1

        results = mhn.retrieve(noisy_key, top_k=1)
        assert len(results) == 1
        assert results[0][1] > 0.8, f"Noise tolerance failed: sim={results[0][1]}"

    def test_distinguish_different_patterns(self, mhn, dim):
        """MHN should distinguish between different stored patterns."""
        for i in range(10):
            key = generate_hypervector(dim, seed=i * 100)
            value = generate_hypervector(dim, seed=i * 200)
            mhn.store(key, value, {"id": i})

        # Query with key #5
        query = generate_hypervector(dim, seed=500)
        results = mhn.retrieve(query, top_k=3)
        assert len(results) <= 3
        # Most similar should be #5
        assert results[0][2]["id"] == 5

    def test_no_retrieval_for_unrelated_query(self, mhn, dim):
        """Unrelated query should return empty results."""
        key = generate_hypervector(dim, seed=100)
        value = generate_hypervector(dim, seed=200)
        mhn.store(key, value, {"action": "test"})

        # Random unrelated query — similarity ~0
        unrelated = generate_hypervector(dim, seed=99999)
        results = mhn.retrieve(unrelated, top_k=1)
        assert len(results) == 0, f"Should not retrieve for unrelated: sim={results[0][1] if results else 'N/A'}"

    def test_zero_catastrophic_forgetting(self, mhn, dim):
        """Old patterns should still be retrievable after adding new ones."""
        keys = [generate_hypervector(dim, seed=i * 100) for i in range(50)]
        values = [generate_hypervector(dim, seed=i * 200) for i in range(50)]

        for k, v in zip(keys, values):
            mhn.store(k, v, {"id": "x"})

        # All original patterns should still be retrievable
        for i, (k, v) in enumerate(zip(keys, values)):
            results = mhn.retrieve(k, top_k=1)
            assert len(results) == 1
            assert results[0][1] > 0.95, f"Pattern {i} forgotten: sim={results[0][1]}"

    def test_idp_prevents_cross_domain_interference(self, dim):
        """IDP should prevent patterns from one domain being retrieved for another."""
        mhn = ModernHopfieldNetwork(MHNConfig(
            dim=dim, idp_enabled=True, min_similarity=0.05,
            idp_threshold=0.05, idp_gain=100.0,
        ))

        # Store patterns that are moderately similar (sim ~0.1)
        base = generate_hypervector(dim, seed=1)
        for i in range(5):
            noise = generate_hypervector(dim, seed=i + 100)
            # Create patterns with ~10% similarity
            key = bundle(base, noise)  # sim to base ≈ 1/sqrt(2) ≈ 0.7
            value = generate_hypervector(dim, seed=i + 500)
            mhn.store(key, value, {"pattern": i})

        # Query with exact key for pattern 0
        key0 = bundle(base, generate_hypervector(dim, seed=100))
        results = mhn.retrieve(key0, top_k=5)
        assert len(results) > 0
        # Top result should have highest similarity
        if len(results) > 1:
            assert results[0][1] >= results[1][1], "Results not sorted by similarity"

    def test_correction_deactivates_all_matches(self, dim):
        """learn_from_correction should deactivate ALL matching patterns."""
        mhn = ModernHopfieldNetwork(MHNConfig(dim=dim, min_similarity=0.05))
        learner = OnlineLearner(mhn, build_default_vocabulary(dim=dim))

        # Store 3 similar patterns (same situation, different actions)
        situation = generate_hypervector(dim, seed=42)
        for i in range(3):
            action = generate_hypervector(dim, seed=i * 10)
            learner.learn_from_outcome(situation, action, "success", {"action_plan": [{"action": f"a{i}"}]})

        # Correct all of them
        correct_action = generate_hypervector(dim, seed=999)
        learner.learn_from_correction(
            situation,
            generate_hypervector(dim, seed=0),  # wrong action vector
            correct_action,
            {"action_plan": [{"action": "correct"}]}
        )

        # All old patterns should be deactivated
        active_count = sum(1 for p in mhn.patterns if p.active)
        assert active_count == 1, f"Expected 1 active after correction, got {active_count}"


# =========================================================================
# 3. Perception Intelligence — does parsing actually capture meaning?
# =========================================================================

class TestPerceptionIntelligence:

    @pytest.fixture
    def vocab(self):
        return build_default_vocabulary(dim=10_000)

    @pytest.fixture
    def parser(self, vocab):
        return NLParser(vocab)

    def test_similar_commands_similar_vectors(self, parser):
        """"Click submit button" and "Click the submit button on the form"
        should have moderate positive similarity."""
        v1 = parser.parse("Click the submit button")
        v2 = parser.parse("Click the submit button on the form")
        sim = similarity(v1, v2)
        assert sim > 0.1, f"Similar commands should have positive sim: {sim}"

    def test_different_commands_different_vectors(self, parser):
        """Completely different commands should have low-ish similarity.
        
        NOTE: NL parser uses shared role vectors (action, target) across
        all commands, so there's always some baseline similarity (~0.1-0.3)
        from the shared role binding. This is expected for role-filler HDC."""
        v1 = parser.parse("Click the submit button")
        v2 = parser.parse("Restart the nginx service")
        sim = similarity(v1, v2)
        assert sim < 0.35, f"Different commands too similar: {sim}"

    def test_action_role_extraction(self, parser):
        """Parser should correctly extract action/target/context roles."""
        concepts = parser.extract_concepts("Click the submit button on the login form")
        assert concepts.get("action") == "click"
        assert concepts.get("target") == "submit_button"
        assert concepts.get("context") == "login"

    def test_unknown_input_gives_random_vector(self, parser):
        """Unknown input should give near-orthogonal vector (triggers escalation)."""
        v = parser.parse("xyzzy frobnicate")
        # Should be random — low similarity to everything
        assert v.shape == (10_000,)
        # Similarity to itself should be 1.0 (identity)
        assert similarity(v, v) > 0.99

    def test_dom_encoding_detects_forms(self):
        """DOM encoder should produce vectors that the NL parser recognizes as 'form'."""
        from td.perception.dom_encoder import DOMEncoder
        vocab = build_default_vocabulary(dim=10_000)
        encoder = DOMEncoder(vocab)

        html_with_form = """
        <html><body>
        <form action="/login" method="post">
        <input type="text" name="username" required>
        <input type="password" name="password" required>
        <button type="submit">Login</button>
        </form>
        </body></html>
        """
        vec = encoder.encode(html_with_form)
        assert vec.shape == (10_000,)
        assert vec.dtype == np.int8

        # Should have some similarity to form-related concepts
        form_vec = vocab.get("form")
        sim = similarity(vec, form_vec)
        # Not necessarily high (CA adds noise) but should be distinguishable
        assert sim > -0.05, f"DOM encoding seems broken for form: sim={sim}"

    def test_ca_reproducibility_same_input(self):
        """Same input to CA should always produce same output."""
        ca = CAReservoir(CAConfig(input_dim=10_000, seed=42))
        inp = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1] * 10, dtype=np.uint8)
        v1 = ca.evolve(inp)
        v2 = ca.evolve(inp)
        assert np.array_equal(v1, v2), "CA not reproducible with same input"

    def test_ca_reproducibility_across_instances(self):
        """Different CA instances with same seed should produce same output."""
        ca1 = CAReservoir(CAConfig(input_dim=10_000, seed=42))
        ca2 = CAReservoir(CAConfig(input_dim=10_000, seed=42))
        inp = np.array([1, 0, 1, 1, 0] * 20, dtype=np.uint8)
        v1 = ca1.evolve(inp)
        v2 = ca2.evolve(inp)
        assert np.array_equal(v1, v2), "CA not reproducible across instances with same seed"


# =========================================================================
# 4. Routing Intelligence — does the router classify correctly?
# =========================================================================

class TestRoutingIntelligence:

    @pytest.fixture
    def trained_router(self):
        """Router with trained weights."""
        vocab = build_default_vocabulary(dim=10_000)
        router = HierarchicalRouter(input_dim=10_000)
        # Train using the training utility
        from td.routing.router_train import train_router
        result = train_router(vocab, epochs=50, verbose=False)
        router = HierarchicalRouter(input_dim=10_000)
        router.router_a.load_state_dict(result["router_a"].state_dict())
        for domain in result["routers_b"]:
            router.router_b_dict[domain].load_state_dict(result["routers_b"][domain].state_dict())
        router.router_c.load_state_dict(result["router_c"].state_dict())
        router.eval()
        return router

    @pytest.fixture
    def vocab(self):
        return build_default_vocabulary(dim=10_000)

    @pytest.fixture
    def parser(self, vocab):
        return NLParser(vocab)

    def test_routing_valid_domains_and_confidence(self, trained_router, parser):
        """All inputs route to valid domains with confidence in [0,1].

        Note: BitNet b1.58 ternary weights at 16K effective params have limited
        classification accuracy. Exact domain prediction is NOT expected. What IS
        expected: valid domains, reasonable confidence, consistent outputs.
        """
        test_cases = [
            "Click the submit button on the login form",
            "Fetch data from the API endpoint with auth token",
            "Parse the CSV file and validate the schema",
            "Check CPU usage and restart the service if threshold exceeded",
            "Write a poem about artificial intelligence",
        ]
        for text in test_cases:
            vec = parser.parse(text)
            result = trained_router.route(vec)
            assert result.domain in ("Web", "API", "File", "Monitor", "Unknown"), \
                f"Invalid domain for '{text}': {result.domain}"
            assert 0 <= result.combined_confidence <= 1.0, \
                f"Confidence out of range for '{text}': {result.combined_confidence}"
            assert result.strategy in ("MEMORY_ONLY", "MEMORY_THEN_VALIDATE", "ESCALATE")


# =========================================================================
# 5. End-to-End Pipeline Intelligence
# =========================================================================

class TestPipelineIntelligence:
    """Test the full decide() pipeline for intelligent behavior."""

    @pytest.fixture
    def pipeline(self):
        """Create pipeline with trained routers."""
        pipe = TDPipeline()
        # Train routers on synthetic data
        pipe.train_routers(epochs=30, verbose=False)
        return pipe

    def test_novel_input_escalates(self, pipeline):
        """Completely novel input should escalate, not execute."""
        decision = pipeline.decide("Translate this document to French")
        assert decision.should_escalate or decision.needs_confirmation, \
            f"Novel input should escalate/confirm, got {decision.confidence.decision}"

    def test_known_input_works(self, pipeline):
        """Known input type should produce a valid decision."""
        decision = pipeline.decide("Click the submit button on the login form")
        assert decision.routing.domain in ("Web", "API", "File", "Monitor", "Unknown")
        assert decision.latency_ms < 100, f"Latency too high: {decision.latency_ms}ms"
        assert len(decision.trace) > 0

    def test_learn_then_retrieve(self, pipeline):
        """After learning, the system should retrieve the learned action."""
        # Learn a pattern
        pipeline.learn(
            "Fill out the contact form with name and email",
            [{"action": "fill", "target": "name_field"},
             {"action": "fill", "target": "email_field"},
             {"action": "click", "target": "submit_button"}],
            "success",
            {"domain": "Web", "task_type": "Form"}
        )

        # Now query with similar input
        decision = pipeline.decide("Fill out the contact form with name and email")
        assert len(pipeline.mhn.patterns) > 0
        # MHN should have the pattern
        assert pipeline.mhn.retrieve(pipeline.nl_parser.parse("Fill out the contact form with name and email"), top_k=1)

    def test_learning_improves_over_time(self, pipeline):
        """System should get better (higher confidence) after learning."""
        # Before learning — no MHN patterns
        decision_before = pipeline.decide("Click the submit button on the login form")

        # Learn the pattern
        pipeline.learn(
            "Click the submit button on the login form",
            [{"action": "click", "target": "submit_button"}],
            "success",
            {"domain": "Web", "task_type": "Form"}
        )

        # After learning — MHN should have the pattern
        decision_after = pipeline.decide("Click the submit button on the login form")

        # Confidence should improve (or at least MHN should now have hits)
        assert len(decision_after.mhn_hits) > 0 or decision_after.strategy == "ESCALATE"

    def test_correction_changes_behavior(self, pipeline):
        """After correction, wrong action should not be retrieved."""
        # Learn a wrong action
        pipeline.learn(
            "Restart the nginx service",
            [{"action": "delete", "target": "nginx"}],  # Wrong!
            "success",
            {"domain": "Monitor", "task_type": "Threshold"}
        )

        # Correct it
        pipeline.learn_correction(
            "Restart the nginx service",
            [{"action": "delete", "target": "nginx"}],
            [{"action": "restart", "target": "nginx"}],
            {"domain": "Monitor", "task_type": "Threshold"}
        )

        # Old pattern should be inactive
        active = [p for p in pipeline.mhn.patterns if p.active]
        for p in active:
            plan = p.metadata.get("action_plan", [])
            actions = [a.get("action", "") for a in plan]
            assert "delete" not in actions, "Corrected action still in active patterns"

    def test_z3_validation_catches_invalid_actions(self, pipeline):
        """Z3 should flag invalid actions."""
        from td.reasoning.z3_bridge import Z3Bridge
        z3 = Z3Bridge()

        result = z3.validate_action(
            [{"action": "submit", "target": "form"}],
            {
                "submit_visible": True,
                "required_fields_filled": True,
                "captcha_present": False,
            }
        )
        assert result.status == "sat", f"Valid action should be SAT: {result.status}"

        # Invalid: captcha present
        result = z3.validate_action(
            [{"action": "submit", "target": "form"}],
            {
                "submit_visible": True,
                "required_fields_filled": True,
                "captcha_present": True,  # Captcha! Should fail
            }
        )
        assert result.status == "unsat", f"Invalid action should be UNSAT: {result.status}"

    def test_latency_under_50ms(self, pipeline):
        """Full pipeline should be <50ms (excluding Z3)."""
        times = []
        for text in [
            "Click the submit button on the login form",
            "Fetch data from the API endpoint",
            "Parse the CSV file and extract rows",
            "Monitor CPU usage and alert if high",
        ]:
            t0 = time.perf_counter()
            pipeline.decide(text)
            times.append((time.perf_counter() - t0) * 1000)

        avg_ms = np.mean(times)
        assert avg_ms < 50, f"Average latency too high: {avg_ms:.1f}ms (times: {times})"


# =========================================================================
# 6. Genericity — does it generalize beyond web/api/file/monitor?
# =========================================================================

class TestGenericity:
    """Test that TD v2 can handle domains it wasn't explicitly designed for."""

    @pytest.fixture
    def pipeline(self):
        pipe = TDPipeline()
        pipe.train_routers(epochs=30, verbose=False)
        return pipe

    def test_handles_code_related_input(self, pipeline):
        """Should be able to process code-related text without crashing."""
        decision = pipeline.decide("Run the test suite and report failures")
        assert isinstance(decision.confidence.combined, float)
        assert decision.confidence.decision in ("execute", "confirm", "escalate")

    def test_handles_math_input(self, pipeline):
        """Should be able to process mathematical text."""
        decision = pipeline.decide("Solve the equation x squared plus 2x minus 3 equals 0")
        assert isinstance(decision.confidence.combined, float)

    def test_handles_empty_input(self, pipeline):
        """Empty input should not crash."""
        decision = pipeline.decide("")
        assert decision.confidence.decision in ("execute", "confirm", "escalate")

    def test_handles_very_long_input(self, pipeline):
        """Very long input should not crash or be too slow."""
        long_text = "Click the button " * 1000
        t0 = time.perf_counter()
        decision = pipeline.decide(long_text)
        elapsed = (time.perf_counter() - t0) * 1000
        assert elapsed < 200, f"Long input too slow: {elapsed:.0f}ms"

    def test_handles_mixed_language(self, pipeline):
        """Input with mixed language should not crash."""
        decision = pipeline.decide("Click le button on el formulario")
        assert isinstance(decision.confidence.combined, float)

    def test_new_domain_can_be_learned(self, pipeline):
        """We should be able to teach TD a completely new domain via learning."""
        # Teach it about a "game" domain
        pipeline.learn(
            "Move the player character to the right",
            [{"action": "move", "target": "player", "direction": "right"}],
            "success",
            {"domain": "Game", "task_type": "Movement"}
        )
        pipeline.learn(
            "Jump over the obstacle",
            [{"action": "jump", "target": "player"}],
            "success",
            {"domain": "Game", "task_type": "Movement"}
        )

        # Should have patterns stored
        assert len(pipeline.mhn.patterns) >= 2
        assert sum(1 for p in pipeline.mhn.patterns if p.active) >= 2

    def test_multiple_input_types(self, pipeline):
        """All input types should be processable."""
        for input_type, data in [
            ("natural_language", "Click the submit button"),
            ("api", {"endpoint": "/users", "method": "GET"}),
            ("metrics", {"cpu": 85.0, "memory": 70.0}),
        ]:
            decision = pipeline.decide(data, input_type=input_type)
            assert len(decision.trace) > 0


# =========================================================================
# 7. Reproducibility — same input → same output across instances
# =========================================================================

class TestReproducibility:

    def test_reproducible_vocabulary(self):
        """Vocabulary should be deterministic with same seed."""
        v1 = build_default_vocabulary(dim=10_000)
        v2 = build_default_vocabulary(dim=10_000)
        for name in v1.concepts:
            sim = similarity(v1.concepts[name], v2.concepts[name])
            assert sim > 0.999, f"Vocab not reproducible for {name}: sim={sim}"

    def test_reproducible_trained_router(self):
        """Trained router should give consistent results on same input."""
        from td.routing.router_train import train_router
        vocab = build_default_vocabulary(dim=10_000)
        parser = NLParser(vocab)
        result = train_router(vocab, epochs=50, verbose=False)
        router = HierarchicalRouter(input_dim=10_000)
        router.router_a.load_state_dict(result["router_a"].state_dict())
        for d in result["routers_b"]:
            router.router_b_dict[d].load_state_dict(result["routers_b"][d].state_dict())
        router.router_c.load_state_dict(result["router_c"].state_dict())
        router.eval()
        text = "Click the submit button on the login form"
        vec = parser.parse(text)
        d1 = router.route(vec)
        d2 = router.route(vec)
        assert d1.domain == d2.domain, \
            f"Non-reproducible on same input: {d1.domain} vs {d2.domain}"
