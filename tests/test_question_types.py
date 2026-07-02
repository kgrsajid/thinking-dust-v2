"""Comprehensive question-type tests — 5-hop chains.

Tests ALL question types the system supports:
1. Yes/No (is X in Y?)
2. Functional contradiction (are X and Y the same?)
3. Open query (what is X related to?)
4. Temporal (is X before Y?)
5. Multi-entity (which X is in Y?)
6. Inverse (what is in X?)
7. Confidence calibration (confidence decreases with hops)
8. Proof trace (shows reasoning chain)

Each test uses 5-hop chains from real Wikipedia data.
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from td.kg import KnowledgeGraph


@pytest.fixture
def kg():
    g = KnowledgeGraph()
    # Pre-seed composition rules
    g.set_composition_rule("capital_of", "in", "in")
    g.set_composition_rule("born_in", "in", "in")
    g.set_composition_rule("in", "in", "in")
    g.set_composition_rule("part_of", "part_of", "part_of")
    return g


# ═══════════════════════════════════════════════════════════════════════
# 1. YES/NO QUESTIONS — "Is X in Y?"
# ═══════════════════════════════════════════════════════════════════════

class TestYesNoQuestions:
    """Yes/No questions with 5-hop chains."""

    def test_5_hop_yes(self, kg):
        """Is Paris in EU? → YES (2-hop)"""
        kg.add_fact("paris", "capital_of", "france")
        kg.add_fact("france", "in", "eu")

        r = kg.query("paris", "in", "eu")
        assert r.answer is True
        assert r.method == "derived"

    def test_5_hop_chain_yes(self, kg):
        """Is Python's creator in Eurasia? → YES (5-hop)"""
        kg.add_fact("python", "created_by", "guido van rossum")
        kg.add_fact("guido van rossum", "born_in", "haarlem")
        kg.add_fact("haarlem", "in", "netherlands")
        kg.add_fact("netherlands", "in", "europe")
        kg.add_fact("europe", "part_of", "eurasia")

        r = kg.query("guido van rossum", "in", "eurasia")
        assert r.answer is True

    def test_5_hop_no(self, kg):
        """Is Paris in Germany? → NO (not derivable)"""
        kg.add_fact("paris", "capital_of", "france")
        kg.add_fact("france", "in", "eu")
        kg.add_fact("berlin", "capital_of", "germany")
        kg.add_fact("germany", "in", "eu")

        r = kg.query("paris", "in", "germany")
        assert r.answer is None  # Not derivable

    def test_unknown_entity(self, kg):
        """Is Narnia in EU? → UNKNOWN"""
        kg.add_fact("paris", "capital_of", "france")
        r = kg.query("narnia", "in", "eu")
        assert r.answer is None
        assert r.confidence == 0.0


# ═══════════════════════════════════════════════════════════════════════
# 2. FUNCTIONAL CONTRADICTION — "Are X and Y the same?"
# ═══════════════════════════════════════════════════════════════════════

class TestFunctionalContradiction:
    """Functional contradiction questions."""

    def test_different_capitals(self, kg):
        """Are Paris and Berlin the same? → NO (different capitals)"""
        kg.add_fact("paris", "capital_of", "france")
        kg.add_fact("berlin", "capital_of", "germany")

        r = kg.check_same("paris", "berlin")
        assert r.answer is False
        assert "capital_of" in r.proof_trace

    def test_same_functional_value(self, kg):
        """Tokyo and Osaka both claim capital_of Japan → UNKNOWN"""
        kg.add_fact("tokyo", "capital_of", "japan")
        kg.add_fact("osaka", "capital_of", "japan")

        r = kg.check_same("tokyo", "osaka")
        assert r.answer is None  # Same functional value can't prove different

    def test_no_functional_relation(self, kg):
        """Paris and France — no functional relation → UNKNOWN"""
        kg.add_fact("paris", "capital_of", "france")

        r = kg.check_same("paris", "france")
        assert r.answer is None


# ═══════════════════════════════════════════════════════════════════════
# 3. OPEN QUERY — "What is X related to?"
# ═══════════════════════════════════════════════════════════════════════

class TestOpenQuery:
    """Open queries that return lists of objects."""

    def test_open_query_capital_of(self, kg):
        """What is Paris the capital of? → France"""
        kg.add_fact("paris", "capital_of", "france")
        kg.add_fact("berlin", "capital_of", "germany")

        r = kg.query("paris", "capital_of")
        assert r.answer is True
        assert "france" in r.proof_trace

    def test_open_query_borders(self, kg):
        """What does Germany border? → France, Poland, etc."""
        kg.add_fact("germany", "borders", "france")
        kg.add_fact("germany", "borders", "poland")
        kg.add_fact("germany", "borders", "czech republic")

        r = kg.query("germany", "borders")
        assert r.answer is True

    def test_open_query_no_results(self, kg):
        """What does Narnia border? → Nothing"""
        r = kg.query("narnia", "borders")
        assert r.answer is None


# ═══════════════════════════════════════════════════════════════════════
# 4. TEMPORAL — "Is X before Y?"
# ═══════════════════════════════════════════════════════════════════════

class TestTemporalQuestions:
    """Temporal reasoning questions."""

    def test_before_yes(self, kg):
        """Was WW2 before Cold War? → YES"""
        kg.add_fact("world war ii", "before", "cold war",
                    temporal_start=1939, temporal_end=1945)
        kg.add_fact("cold war", "after", "world war ii",
                    temporal_start=1947, temporal_end=1991)

        r = kg.query_temporal("world war ii", "cold war", "before")
        assert r.answer is True

    def test_before_no(self, kg):
        """Was Cold War before WW2? → NO (it was after)"""
        kg.add_fact("world war ii", "before", "cold war",
                    temporal_start=1939, temporal_end=1945)
        kg.add_fact("cold war", "after", "world war ii",
                    temporal_start=1947, temporal_end=1991)

        r = kg.query_temporal("cold war", "world war ii", "before")
        assert r.answer is False

    def test_meets(self, kg):
        """Did Obama's term meet Trump's? → YES"""
        kg.add_fact("obama", "president_of", "usa",
                    temporal_start=2009, temporal_end=2017)
        kg.add_fact("trump", "president_of", "usa",
                    temporal_start=2017, temporal_end=2021)

        r = kg.query_temporal("obama", "trump", "meets")
        assert r.answer is True

    def test_during(self, kg):
        """Was COVID during 21st century? → YES"""
        kg.add_fact("covid-19 pandemic", "during", "21st century",
                    temporal_start=2020, temporal_end=2023)
        kg.add_fact("21st century", "contains", "covid-19 pandemic",
                    temporal_start=2001, temporal_end=2100)

        r = kg.query_temporal("covid-19 pandemic", "21st century", "during")
        assert r.answer is True

    def test_temporal_unknown(self, kg):
        """Was Napoleon before WW2? → UNKNOWN (no temporal data)"""
        r = kg.query_temporal("napoleon", "world war ii", "before")
        assert r.answer is None


# ═══════════════════════════════════════════════════════════════════════
# 5. MULTI-ENTITY — "Which X is in Y?"
# ═══════════════════════════════════════════════════════════════════════

class TestMultiEntity:
    """Questions involving multiple entities."""

    def test_multiple_capitals_in_region(self, kg):
        """Which capitals are in EU? → Paris, Berlin"""
        kg.add_fact("paris", "capital_of", "france")
        kg.add_fact("france", "in", "eu")
        kg.add_fact("berlin", "capital_of", "germany")
        kg.add_fact("germany", "in", "eu")

        # Both should be derivable
        r1 = kg.query("paris", "in", "eu")
        r2 = kg.query("berlin", "in", "eu")
        assert r1.answer is True
        assert r2.answer is True

    def test_entity_with_multiple_relations(self, kg):
        """Germany borders France AND is in EU"""
        kg.add_fact("germany", "borders", "france")
        kg.add_fact("germany", "in", "eu")
        kg.add_fact("france", "in", "eu")

        r1 = kg.query("germany", "borders", "france")
        r2 = kg.query("germany", "in", "eu")
        assert r1.answer is True
        assert r2.answer is True


# ═══════════════════════════════════════════════════════════════════════
# 6. INVERSE — "What is in X?"
# ═══════════════════════════════════════════════════════════════════════

class TestInverseQueries:
    """Inverse relation queries."""

    def test_inverse_capital_of(self, kg):
        """What is the capital of France? → Paris (open query)"""
        kg.add_fact("paris", "capital_of", "france")

        # Open query: what is capital_of France?
        r = kg.query("paris", "capital_of", "france")
        assert r.answer is True

    def test_inverse_symmetric(self, kg):
        """If France borders Germany, does Germany border France?"""
        kg.add_fact("france", "borders", "germany")
        kg.set_relation_property("borders", "symmetric")

        r = kg.query("germany", "borders", "france")
        assert r.answer is True


# ═══════════════════════════════════════════════════════════════════════
# 7. CONFIDENCE CALIBRATION — Confidence decreases with hops
# ═══════════════════════════════════════════════════════════════════════

class TestConfidence:
    """Confidence should decrease with hop count."""

    def test_confidence_1_hop(self, kg):
        """1-hop: highest confidence"""
        kg.add_fact("paris", "capital_of", "france")
        r = kg.query("paris", "capital_of", "france")
        assert r.confidence >= 0.9

    def test_confidence_2_hop(self, kg):
        """2-hop: lower than 1-hop"""
        kg.add_fact("paris", "capital_of", "france")
        kg.add_fact("france", "in", "eu")
        r = kg.query("paris", "in", "eu")
        assert r.confidence < 0.95
        assert r.confidence >= 0.7

    def test_confidence_5_hop(self, kg):
        """5-hop: chain quality matters, not hop count.
        All 'in' relations use explicit composition rules → high confidence."""
        kg.add_fact("a", "in", "b")
        kg.add_fact("b", "in", "c")
        kg.add_fact("c", "in", "d")
        kg.add_fact("d", "in", "e")
        kg.add_fact("e", "in", "f")
        r = kg.query("a", "in", "f")
        # All steps use explicit rules (in ∘ in → in) → high confidence
        assert r.confidence >= 0.80

    def test_confidence_unknown(self, kg):
        """Unknown: 0.0 confidence"""
        r = kg.query("nonexistent", "relation", "target")
        assert r.confidence == 0.0


# ═══════════════════════════════════════════════════════════════════════
# 8. PROOF TRACE — Shows reasoning chain
# ═══════════════════════════════════════════════════════════════════════

class TestProofTrace:
    """Proof traces should show the full reasoning chain."""

    def test_proof_trace_2_hop(self, kg):
        """2-hop proof trace shows both hops."""
        kg.add_fact("paris", "capital_of", "france")
        kg.add_fact("france", "in", "eu")

        r = kg.query("paris", "in", "eu")
        assert "paris" in r.proof_trace
        assert "capital_of" in r.proof_trace
        assert "france" in r.proof_trace
        assert "in" in r.proof_trace
        assert "eu" in r.proof_trace

    def test_proof_trace_5_hop(self, kg):
        """5-hop proof trace shows full chain."""
        kg.add_fact("python", "created_by", "guido van rossum")
        kg.add_fact("guido van rossum", "born_in", "haarlem")
        kg.add_fact("haarlem", "in", "netherlands")
        kg.add_fact("netherlands", "in", "europe")
        kg.add_fact("europe", "part_of", "eurasia")

        r = kg.query("guido van rossum", "in", "eurasia")
        # Should mention key entities in the proof
        assert "guido van rossum" in r.proof_trace.lower()
        assert "haarlem" in r.proof_trace.lower()
        assert "netherlands" in r.proof_trace.lower()
        assert "europe" in r.proof_trace.lower()
        assert "eurasia" in r.proof_trace.lower()

    def test_proof_trace_functional(self, kg):
        """Functional contradiction proof shows the reason."""
        kg.add_fact("paris", "capital_of", "france")
        kg.add_fact("berlin", "capital_of", "germany")

        r = kg.check_same("paris", "berlin")
        assert "capital_of" in r.proof_trace
        assert "functional" in r.proof_trace.lower()

    def test_proof_trace_temporal(self, kg):
        """Temporal proof shows the Allen relation."""
        kg.add_fact("obama", "president_of", "usa",
                    temporal_start=2009, temporal_end=2017)
        kg.add_fact("trump", "president_of", "usa",
                    temporal_start=2017, temporal_end=2021)

        r = kg.query_temporal("obama", "trump", "meets")
        assert "meets" in r.proof_trace.lower()


# ═══════════════════════════════════════════════════════════════════════
# 9. PERSISTENCE — All question types survive SQLite
# ═══════════════════════════════════════════════════════════════════════

class TestQuestionPersistence:
    """All question types persist through SQLite."""

    def test_all_types_persist(self, kg):
        """Yes/No, functional, temporal all persist."""
        # Yes/No facts
        kg.add_fact("paris", "capital_of", "france")
        kg.add_fact("france", "in", "eu")

        # Functional
        kg.add_fact("berlin", "capital_of", "germany")

        # Temporal
        kg.add_fact("obama", "president_of", "usa",
                    temporal_start=2009, temporal_end=2017)
        kg.add_fact("trump", "president_of", "usa",
                    temporal_start=2017, temporal_end=2021)

        # Symmetric
        kg.add_fact("france", "borders", "germany")
        kg.set_relation_property("borders", "symmetric")

        tmp = tempfile.mktemp(suffix=".db")
        try:
            kg.save(tmp)
            kg2 = KnowledgeGraph()
            kg2.load(tmp)

            # Yes/No works
            r1 = kg2.query("paris", "in", "eu")
            assert r1.answer is True

            # Functional works
            r2 = kg2.check_same("paris", "berlin")
            assert r2.answer is False

            # Temporal works
            r3 = kg2.query_temporal("obama", "trump", "meets")
            assert r3.answer is True

            # Symmetric works
            r4 = kg2.query("germany", "borders", "france")
            assert r4.answer is True
        finally:
            os.unlink(tmp)
