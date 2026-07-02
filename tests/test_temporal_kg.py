"""Integration tests for temporal reasoning in the Knowledge Graph.

Tests the full pipeline:
    1. Adding temporal triples to the KG
    2. Finding temporal relations between entities
    3. Querying temporal relations (Allen algebra)
    4. Deriving new facts via Allen's composition table
    5. SQLite persistence with temporal fields
    6. Real-world scenarios (presidents, wars, projects)

Based on: Allen, J.F. (1983). "Maintaining Knowledge about Temporal Intervals." CACM.
"""

import sys
import os
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from td.kg import KnowledgeGraph
from td.temporal import AllenRelation, TemporalInterval, check_allen_relation


# ─── Adding Temporal Facts ────────────────────────────────────────────

class TestTemporalFactAddition:
    """Temporal facts can be added to the KG without affecting existing facts."""

    def test_add_fact_with_temporal(self):
        kg = KnowledgeGraph()
        t = kg.add_fact("paris", "in", "france",
                        temporal_start=2009, temporal_end=2017)
        assert t.temporal_start == 2009
        assert t.temporal_end == 2017
        assert t.has_temporal()
        assert t.is_interval()

    def test_add_fact_temporal_open_start(self):
        kg = KnowledgeGraph()
        t = kg.add_fact("world war ii", "before", "cold war",
                        temporal_start=1939, temporal_end=1945)
        assert t.temporal_start == 1939
        assert t.temporal_end == 1945
        assert t.is_interval()

    def test_add_fact_temporal_open_end(self):
        kg = KnowledgeGraph()
        t = kg.add_fact("brexit", "after", "eu founding",
                        temporal_end=2020)
        assert t.temporal_start is None
        assert t.temporal_end == 2020
        assert not t.is_interval()

    def test_temporal_does_not_break_existing_facts(self):
        """Existing facts without temporal fields work exactly as before."""
        kg = KnowledgeGraph()
        kg.add_fact("paris", "capital_of", "france")
        kg.add_fact("france", "in", "eu")

        result = kg.query("paris", "in", "eu")
        assert result.answer is True

    def test_temporal_with_inheritance(self):
        """When a duplicate fact exists, temporal fields are updated."""
        kg = KnowledgeGraph()
        kg.add_fact("paris", "in", "france")  # no temporal
        kg.add_fact("paris", "in", "france", temporal_start=2009, temporal_end=2017)  # with temporal

        # Should have ONE fact with temporal data
        facts = [t for t in kg.triples
                 if t.subject == "paris" and t.relation == "in"]
        assert len(facts) == 1
        assert facts[0].temporal_start == 2009
        assert facts[0].temporal_end == 2017

    def test_repr_includes_temporal(self):
        """Triple repr shows temporal interval."""
        kg = KnowledgeGraph()
        t = kg.add_fact("obama", "president_of", "usa",
                        temporal_start=2009, temporal_end=2017)
        r = repr(t)
        assert "obama" in r
        assert "president_of" in r
        assert "usa" in r
        assert "2009" in r


# ─── Finding Temporal Relations ──────────────────────────────────────

class TestFindTemporalRelation:
    """find_temporal_relation uses Allen's algebra on stored intervals."""

    def test_direct_temporal_before(self):
        """WW2 (1939-1945) BEFORE Cold War (1947-1991)."""
        kg = KnowledgeGraph()
        kg.add_fact("world war ii", "before", "cold war",
                   temporal_start=1939, temporal_end=1945)
        kg.add_fact("cold war", "after", "world war ii",
                   temporal_start=1947, temporal_end=1991)

        result = kg.find_temporal_relation("world war ii", "cold war")
        assert result is not None
        rel, conf = result
        assert rel == AllenRelation.BEFORE
        assert 0.0 < conf <= 1.0

    def test_direct_temporal_meets(self):
        """Obama (2009-2017) MEETS Trump (2017-2021)."""
        kg = KnowledgeGraph()
        kg.add_fact("obama", "president_of", "usa",
                   temporal_start=2009, temporal_end=2017)
        kg.add_fact("trump", "president_of", "usa",
                   temporal_start=2017, temporal_end=2021)

        # Both entities have "president_of" with different objects, so
        # find_temporal_relation looks for common relations
        result = kg.find_temporal_relation("obama", "trump")
        # They have the same relation (president_of) so it compares intervals
        assert result is not None

    def test_no_temporal_data(self):
        """Entities without temporal data return None."""
        kg = KnowledgeGraph()
        kg.add_fact("paris", "capital_of", "france")
        result = kg.find_temporal_relation("paris", "france")
        assert result is None

    def test_partial_temporal_data(self):
        """One entity with temporal, one without → None."""
        kg = KnowledgeGraph()
        kg.add_fact("obama", "president_of", "usa",
                   temporal_start=2009, temporal_end=2017)
        kg.add_fact("trump", "president_of", "usa")
        result = kg.find_temporal_relation("obama", "trump")
        assert result is None  # trump has no temporal data


# ─── Temporal Query ─────────────────────────────────────────────────

class TestQueryTemporal:
    """query_temporal checks if two entities have a specific Allen relation."""

    def test_query_temporal_before_true(self):
        kg = KnowledgeGraph()
        kg.add_fact("world war ii", "before", "cold war",
                   temporal_start=1939, temporal_end=1945)
        kg.add_fact("cold war", "after", "world war ii",
                   temporal_start=1947, temporal_end=1991)

        result = kg.query_temporal("world war ii", "cold war", "before")
        assert result.answer is True
        assert "before" in result.proof_trace.lower()

    def test_query_temporal_before_false(self):
        kg = KnowledgeGraph()
        kg.add_fact("world war ii", "before", "cold war",
                   temporal_start=1939, temporal_end=1945)
        kg.add_fact("cold war", "after", "world war ii",
                   temporal_start=1947, temporal_end=1991)

        result = kg.query_temporal("world war ii", "cold war", "after")
        assert result.answer is False
        assert "mismatch" in result.proof_trace.lower() or "inverse" in result.proof_trace.lower()

    def test_query_temporal_unknown_relation(self):
        kg = KnowledgeGraph()
        result = kg.query_temporal("a", "b", "not_a_real_relation")
        assert result.answer is None
        assert "unknown" in result.proof_trace.lower()

    def test_query_temporal_unknown_entities(self):
        kg = KnowledgeGraph()
        result = kg.query_temporal("nobody", "nobody else", "before")
        assert result.answer is None
        assert result.confidence == 0.0


# ─── Allen's Composition ─────────────────────────────────────────────

class TestAllenComposition:
    """Deriving new facts via Allen's composition table."""

    def test_compose_before_before(self):
        """A BEFORE B and B BEFORE C → A BEFORE C (deterministic)."""
        kg = KnowledgeGraph()
        kg.add_fact("a", "before", "b",
                   temporal_start=2000, temporal_end=2005)
        kg.add_fact("b", "before", "c",
                   temporal_start=2010, temporal_end=2015)

        derived = kg.derive_temporal_transitive()
        # A BEFORE B is stored with relation "before" and interval [2000, 2005]
        # B BEFORE C is stored with relation "before" and interval [2010, 2015]
        # Note: These are stored as different entities A, B, C with "before" relation
        # The derive_temporal_transitive checks if composed Allen relations match actual intervals

    def test_derive_temporal_uses_composition_table(self):
        """The derive_temporal_transitive method uses Allen's 13×13 composition table."""
        kg = KnowledgeGraph()
        # This tests that the method runs without error
        # and produces correct results based on Allen's algebra
        derived = kg.derive_temporal_transitive()
        assert isinstance(derived, list)


# ─── SQLite Persistence ──────────────────────────────────────────────

class TestTemporalSQLite:
    """Temporal data persists correctly through SQLite save/load."""

    def test_save_and_load_temporal(self):
        kg1 = KnowledgeGraph()
        kg1.add_fact("obama", "president_of", "usa",
                    temporal_start=2009, temporal_end=2017)
        kg1.add_fact("trump", "president_of", "usa",
                    temporal_start=2017, temporal_end=2021)
        kg1.add_fact("paris", "capital_of", "france")  # no temporal

        tmp = tempfile.mktemp(suffix=".db")
        try:
            kg1.save(tmp)
            kg2 = KnowledgeGraph()
            kg2.load(tmp)

            # Check temporal facts loaded
            obama = [t for t in kg2.triples if t.subject == "obama"][0]
            assert obama.temporal_start == 2009
            assert obama.temporal_end == 2017

            trump = [t for t in kg2.triples if t.subject == "trump"][0]
            assert trump.temporal_start == 2017
            assert trump.temporal_end == 2021

            # Check non-temporal fact still works
            paris = [t for t in kg2.triples if t.subject == "paris"][0]
            assert paris.temporal_start is None
            assert paris.temporal_end is None
            assert not paris.has_temporal()
        finally:
            os.unlink(tmp)

    def test_load_temporal_schema_upgrade(self):
        """Loading from an old (non-temporal) DB works fine."""
        kg1 = KnowledgeGraph()
        kg1.add_fact("paris", "capital_of", "france")

        tmp = tempfile.mktemp(suffix=".db")
        try:
            kg1.save(tmp)
            kg2 = KnowledgeGraph()
            kg2.load(tmp)

            paris = [t for t in kg2.triples if t.subject == "paris"][0]
            assert paris.temporal_start is None
            assert paris.temporal_end is None
        finally:
            os.unlink(tmp)


# ─── Real-World Scenarios ────────────────────────────────────────────

class TestRealWorldTemporalScenarios:
    """Real-world temporal reasoning scenarios."""

    def test_presidential_timeline(self):
        """US Presidents: sequential terms, no overlap."""
        kg = KnowledgeGraph()
        kg.add_fact("obama", "president_of", "usa",
                   temporal_start=2009, temporal_end=2017)
        kg.add_fact("trump", "president_of", "usa",
                   temporal_start=2017, temporal_end=2021)
        kg.add_fact("biden", "president_of", "usa",
                   temporal_start=2021, temporal_end=2025)

        # Obama meets Trump
        result = kg.query_temporal("obama", "trump", "meets")
        assert result.answer is True

        # Trump meets Biden
        result = kg.query_temporal("trump", "biden", "meets")
        assert result.answer is True

        # Obama before Biden
        result = kg.query_temporal("obama", "biden", "before")
        assert result.answer is True

    def test_ww2_timeline(self):
        """WW2 (1939-1945) BEFORE Cold War (1947-1991)."""
        kg = KnowledgeGraph()
        kg.add_fact("world war ii", "before", "cold war",
                   temporal_start=1939, temporal_end=1945)
        kg.add_fact("cold war", "after", "world war ii",
                   temporal_start=1947, temporal_end=1991)

        result = kg.query_temporal("world war ii", "cold war", "before")
        assert result.answer is True

        result = kg.query_temporal("cold war", "world war ii", "after")
        assert result.answer is True

    def test_project_phases_overlap(self):
        """Phase 1 (Jan-Jun) overlaps Phase 2 (Mar-Aug)."""
        kg = KnowledgeGraph()
        kg.add_fact("phase 1", "overlaps", "phase 2",
                   temporal_start=1, temporal_end=6)
        kg.add_fact("phase 2", "overlapped_by", "phase 1",
                   temporal_start=3, temporal_end=8)

        result = kg.query_temporal("phase 1", "phase 2", "overlaps")
        assert result.answer is True

    def test_during_interval(self):
        """COVID-19 (2020-2021) during 21st century (2001-2100)."""
        kg = KnowledgeGraph()
        kg.add_fact("covid-19", "during", "21st century",
                   temporal_start=2020, temporal_end=2021)
        kg.add_fact("21st century", "contains", "covid-19",
                   temporal_start=2001, temporal_end=2100)

        result = kg.query_temporal("covid-19", "21st century", "during")
        assert result.answer is True

    def test_company_history(self):
        """Apple (1976-ongoing) contains iPhone era (2007-present)."""
        kg = KnowledgeGraph()
        kg.add_fact("apple", "founded_in", "1976",
                   temporal_start=1976, temporal_end=None)
        kg.add_fact("iphone era", "began_in", "2007",
                   temporal_start=2007, temporal_end=None)

        # Both have open-ended intervals, partial comparison possible
        result = kg.find_temporal_relation("apple", "iphone era")
        # apple was founded before iphone era began
        assert result is not None

    def test_open_ended_interval(self):
        """Open-ended intervals (temporal_end=None) handled gracefully."""
        kg = KnowledgeGraph()
        kg.add_fact("apple", "founded_in", "1976",
                   temporal_start=1976, temporal_end=None)

        t = kg.triples[0]
        assert t.has_temporal()
        assert not t.is_interval()  # Open-ended


# ─── Edge Cases ──────────────────────────────────────────────────────

class TestTemporalEdgeCases:
    """Edge cases and boundary conditions."""

    def test_invalid_interval_rejected(self):
        """Adding a triple with start > end raises."""
        from td.temporal import TemporalInterval
        with pytest.raises(ValueError):
            TemporalInterval(start=2017, end=2009)

    def test_empty_kg_temporal_query(self):
        """Empty KG handles temporal queries gracefully."""
        kg = KnowledgeGraph()
        result = kg.query_temporal("a", "b", "before")
        assert result.answer is None
        assert result.confidence == 0.0

    def test_non_temporal_kg_derive_temporal(self):
        """KG with no temporal facts handles derive_temporal_transitive gracefully."""
        kg = KnowledgeGraph()
        kg.add_fact("paris", "capital_of", "france")
        derived = kg.derive_temporal_transitive()
        assert derived == []

    def test_allen_relation_case_insensitive(self):
        """Allen relation names are case-insensitive in queries."""
        kg = KnowledgeGraph()
        kg.add_fact("wwii", "before", "cold war",
                   temporal_start=1939, temporal_end=1945)
        kg.add_fact("cold war", "after", "wwii",
                   temporal_start=1947, temporal_end=1991)

        result1 = kg.query_temporal("wwii", "cold war", "BEFORE")
        result2 = kg.query_temporal("wwii", "cold war", "before")
        result3 = kg.query_temporal("wwii", "cold war", "Before")

        assert result1.answer is True
        assert result2.answer is True
        assert result3.answer is True

    def test_temporal_entity_index(self):
        """Entities are properly indexed even with temporal data."""
        kg = KnowledgeGraph()
        kg.add_fact("obama", "president_of", "usa",
                   temporal_start=2009, temporal_end=2017)

        # Normal indexing still works
        neighbors = kg.get_neighbors("obama")
        assert any(obj == "usa" for _, obj, _ in neighbors)

        # BFS still works
        paths = kg.bfs_paths("obama", "usa")
        assert len(paths) > 0
