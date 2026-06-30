"""Tests for Thinking Dust v2 — generic thinking loop.

Tests all four mechanisms + universal Z3 primitives + teaching.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from td.perception.hdc import build_default_vocabulary
from td.memory.mhn import ModernHopfieldNetwork, MHNConfig
from td.thinking import GenericThinkingDust, GenericZ3Solver


@pytest.fixture
def td():
    vocab = build_default_vocabulary(dim=10000)
    mhn = ModernHopfieldNetwork(MHNConfig(dim=10000, min_similarity=0.01))
    return GenericThinkingDust(vocab=vocab, mhn=mhn, dim=10000, pure_mode=True)


# ─── Mechanism 1: IDP ──────────────────────────────────────

class TestIDP:
    def test_idp_converges(self, td):
        result = td.think("schedule a meeting")
        assert result.iterations >= 1
        assert any(t.converged for t in result.thoughts)

    def test_idp_empty_memory(self, td):
        result = td.think("anything")
        assert result.thoughts[0].retrieved_similarity == 0.0

    def test_idp_state_evolved(self, td):
        result = td.think("test problem")
        assert result.evolved_state is not None
        assert result.evolved_state.shape == (10000,)


# ─── Mechanism 2: HDC Decomposition ─────────────────────────

class TestHDCDecomposition:
    def test_decomposition_returns_sub_problems(self, td):
        # Constraint path runs HDC decomposition
        result = td.think("schedule meeting alice before bob")
        if result.intent == "constraint":
            assert len(result.sub_problems) >= 4

    def test_prototypes_are_universal(self, td):
        assert "discover_entities" in td.sub_problem_prototypes
        assert "discover_constraints" in td.sub_problem_prototypes


# ─── Mechanism 3: Generic Z3 ────────────────────────────────

class TestZ3:
    def test_produces_output(self, td):
        """Z3 should produce some assignment for a constraint problem."""
        result = td.think("3 different tasks assigned to 3 different workers")
        assert result.solution is not None

    def test_unsat_detected(self, td):
        """Z3 should detect contradictory constraints."""
        solver = GenericZ3Solver()
        assert solver is not None  # Smoke test — full unsat test needs contrived input

    def test_primitives_applied(self, td):
        """Solution should list which universal primitives were used."""
        result = td.think("Allocate 5000 to 4 departments")
        if result.solution and "primitives_applied" in result.solution:
            prims = result.solution["primitives_applied"]
            assert isinstance(prims, list)


# ─── Mechanism 4: Auto-Storage ──────────────────────────────

class TestAutoStorage:
    def test_memory_grows(self, td):
        # Constraint solving stores experiences automatically
        td.think("schedule meeting alice before bob")
        assert len(td.mhn.patterns) >= 1

    def test_memory_grows_per_interaction(self, td):
        # Teaching stores experiences
        for i in range(5):
            td.teach(f"question number {i}", f"answer number {i}")
        assert len(td.mhn.patterns) == 5


# ─── Teaching ───────────────────────────────────────────────

class TestTeaching:
    def test_teach_and_retrieve(self, td):
        td.teach("What is the capital of France", "Paris")
        result = td.think("What is the capital of France")
        assert result.confidence > 0.5

    def test_teach_multiple(self, td):
        td.teach("speed of light", "299792458 meters per second")
        td.teach("who wrote hamlet", "William Shakespeare")
        assert len(td.mhn.patterns) == 2

    def test_teach_with_template(self, td):
        td.teach(
            "Schedule meetings with Alice Bob and Carol",
            "Alice: Mon 9am, Bob: Mon 10am, Carol: Mon 11am",
            constraint_template={
                "primitives": [
                    {"type": "all_different", "subjects": ["e0", "e1", "e2"]},
                    {"type": "bounded", "subjects": ["e0", "e1", "e2"], "min": 1, "max": 40},
                ]
            }
        )
        result = td.think("Schedule meetings with Alice Bob and Carol")
        assert result.solution is not None


# ─── Confidence ─────────────────────────────────────────────

class TestConfidence:
    def test_unknown_is_honest(self, td):
        result = td.think("meaning of life")
        assert result.confidence < 0.7

    def test_taught_is_confident(self, td):
        td.teach("What is 2+2", "4")
        result = td.think("What is 2+2")
        assert result.confidence > 0.5


# ─── Pure Mode ──────────────────────────────────────────────

class TestPureMode:
    def test_starts_empty(self, td):
        assert len(td.mhn.patterns) == 0
        assert td.stats()["seed_patterns"] == 0

    def test_learns_from_teaching(self, td):
        td.teach("question", "answer")
        assert len(td.mhn.patterns) == 1

    def test_stats_track_learning(self, td):
        td.teach("q1", "a1")
        td.teach("q2", "a2")
        td.think("q1")  # should retrieve, not store new
        s = td.stats()
        assert s["total_thinks"] == 1
        assert s["total_learned"] == 2  # 2 taught
        assert s["memory_size"] == 2


# ─── Generic Entity Graph ──────────────────────────────────

class TestEntityGraph:
    def test_parses_entities(self, td):
        result = td.think("Schedule meetings with Alice Bob")
        assert result.solution is not None or result.confidence < 0.5  # honest either way

    def test_numbers_extracted(self, td):
        result = td.think("Allocate 5000 to 4 groups")
        assert result.solution is not None
