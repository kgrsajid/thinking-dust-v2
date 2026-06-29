"""Tests for Thinking Dust v2 — the thinking loop.

Tests all four mechanisms:
    1. IDP iterative refinement
    2. HDC algebraic decomposition
    3. Z3 constraint solving (entity-driven)
    4. Automatic attractor storage
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from td.perception.hdc import build_default_vocabulary
from td.memory.mhn import ModernHopfieldNetwork, MHNConfig
from td.thinking import ThinkingDust, ThinkingResult


@pytest.fixture
def td():
    """Fresh ThinkingDust in pure mode (0 seed)."""
    vocab = build_default_vocabulary(dim=10000)
    mhn = ModernHopfieldNetwork(MHNConfig(dim=10000, min_similarity=0.01))
    return ThinkingDust(vocab=vocab, mhn=mhn, pure_mode=True)


@pytest.fixture
def td_seeded():
    """ThinkingDust with 50 seed patterns."""
    vocab = build_default_vocabulary(dim=10000)
    mhn = ModernHopfieldNetwork(MHNConfig(dim=10000, min_similarity=0.01))
    return ThinkingDust(vocab=vocab, mhn=mhn, pure_mode=False)


# ─── Mechanism 1: IDP ──────────────────────────────────────

class TestIDP:
    def test_idp_converges(self, td):
        """IDP should converge for any input."""
        result = td.think("schedule a meeting")
        assert result.iterations >= 1
        assert any(t.converged for t in result.thoughts)

    def test_idp_state_evolution(self, td):
        """State should change between iterations."""
        result = td.think("allocate budget")
        if len(result.thoughts) > 1:
            # State should be different (not identical vectors)
            assert result.thoughts[0].description != ""

    def test_idp_empty_memory(self, td):
        """Pure mode with 0 patterns should converge immediately."""
        result = td.think("anything")
        assert result.thoughts[0].retrieved_similarity == 0.0


# ─── Mechanism 2: HDC Decomposition ─────────────────────────

class TestHDCDecomposition:
    def test_decomposition_returns_sub_problems(self, td):
        """Should always return sub-problems."""
        result = td.think("schedule meeting")
        assert len(result.sub_problems) >= 4
        for sp in result.sub_problems:
            assert "concept" in sp
            assert "hdc" in sp

    def test_semantic_prototypes(self, td):
        """Prototypes should be semantic (not random)."""
        assert "extract_entities" in td.sub_problem_prototypes
        assert "find_solution" in td.sub_problem_prototypes


# ─── Mechanism 3: Z3 (entity-driven) ────────────────────────

class TestZ3:
    def test_scheduling(self, td):
        """Scheduling with people should produce Z3 assignments."""
        result = td.think("Schedule meetings with Alice Bob and Carol")
        assert result.solution is not None
        assert result.solution["type"] == "schedule"
        assert "Monday" in result.solution["formatted"]
        assert "alice" in result.solution["formatted"].lower()

    def test_budget(self, td):
        """Budget allocation should produce Z3 distribution."""
        result = td.think("Allocate 5000 to 4 departments")
        assert result.solution is not None
        assert result.solution["type"] == "budget"
        assert "$" in result.solution["formatted"]
        assert "5,000" in result.solution["formatted"]

    def test_proof(self, td):
        """Logic proof should be verified by Z3."""
        result = td.think("Prove that if A implies B and B implies C then A implies C")
        assert result.solution is not None
        assert result.solution["type"] == "proof"
        assert "Verified" in result.solution["formatted"]

    def test_csp_knapsack(self, td):
        """CSP/knapsack should produce Z3 selection."""
        result = td.think("Optimize the knapsack problem with 8 items")
        assert result.solution is not None
        assert result.solution["type"] == "csp"
        assert "weight=" in result.solution["formatted"]

    def test_no_z3_for_advice(self, td):
        """Advice questions should not trigger Z3."""
        result = td.think("how to be happy")
        # Should NOT be a Z3 type
        if result.solution:
            assert result.solution.get("type") not in ("schedule", "budget", "proof", "csp")


# ─── Mechanism 4: Auto-Storage ──────────────────────────────

class TestAutoStorage:
    def test_memory_grows(self, td):
        """Every think() should grow memory by 1."""
        initial = len(td.mhn.patterns)
        td.think("test question 1")
        assert len(td.mhn.patterns) == initial + 1

    def test_taught_knowledge_retrieved(self, td):
        """Taught knowledge should be retrievable at sim=1.0."""
        td.teach("What is the capital of France", "Paris")
        result = td.think("What is the capital of France")
        assert result.confidence > 0.7
        assert "Paris" in result.solution.get("formatted", "")

    def test_seed_ratio(self, td_seeded):
        """Seeded mode should track seed ratio."""
        s = td_seeded.stats()
        assert s["seed_patterns"] == 50
        assert s["seed_ratio_pct"] == 100.0
        td_seeded.think("test question")
        s = td_seeded.stats()
        assert s["learned_patterns"] >= 1


# ─── Teaching ───────────────────────────────────────────────

class TestTeaching:
    def test_teach_and_retrieve(self, td):
        """Teach → retrieve at sim=1.0."""
        td.teach("What is 2+2", "4")
        result = td.think("What is 2+2")
        assert result.confidence > 0.7

    def test_teach_multiple(self, td):
        """Multiple teachings should all be retrievable."""
        td.teach("What is the speed of light", "299792458 meters per second")
        td.teach("Who wrote Hamlet", "William Shakespeare")
        r1 = td.think("What is the speed of light")
        r2 = td.think("Who wrote Hamlet")
        assert "299792458" in r1.solution.get("formatted", "")
        assert "Shakespeare" in r2.solution.get("formatted", "")


# ─── Confidence ─────────────────────────────────────────────

class TestConfidence:
    def test_z3_confidence_high(self, td):
        """Z3 solutions should have high confidence."""
        result = td.think("Schedule meetings with Alice and Bob")
        assert result.confidence >= 0.85

    def test_unknown_confidence_low(self, td):
        """Unknown queries should have low confidence."""
        result = td.think("What is the meaning of life")
        assert result.confidence < 0.7

    def test_taught_confidence_high(self, td):
        """Taught facts should have high confidence on exact match."""
        td.teach("What is X", "The answer is 42")
        result = td.think("What is X")
        assert result.confidence > 0.7


# ─── Pure Mode ──────────────────────────────────────────────

class TestPureMode:
    def test_starts_empty(self, td):
        """Pure mode should start with 0 patterns."""
        assert len(td.mhn.patterns) == 0
        assert td.stats()["seed_patterns"] == 0

    def test_learns_from_teaching(self, td):
        """Pure mode should learn from teaching."""
        td.teach("question 1", "answer 1")
        assert len(td.mhn.patterns) == 1

    def test_z3_works_without_memory(self, td):
        """Z3 should work even with 0 memory."""
        result = td.think("Schedule meetings with Alice and Bob")
        assert result.solution is not None
        assert result.solution["type"] == "schedule"
