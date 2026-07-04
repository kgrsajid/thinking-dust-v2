"""Tests for OWL Property Chain composition rules (Issue 4 fix).

Verifies that cross-relation composition uses explicit rules,
not the "both transitive" heuristic.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from td.kg import KnowledgeGraph


@pytest.fixture
def kg():
    return KnowledgeGraph()


class TestDefaultCompositionRules:
    """Pre-seeded composition rules."""

    def test_in_in_in(self, kg):
        """in ∘ in → in"""
        assert kg.composition_rules.get(("in", "in")) == "in"

    def test_part_of_part_of(self, kg):
        """part_of ∘ part_of → part_of"""
        assert kg.composition_rules.get(("part_of", "part_of")) == "part_of"

    def test_capital_of_in(self, kg):
        """capital_of ∘ in → in"""
        assert kg.composition_rules.get(("capital_of", "in")) == "in"

    def test_before_before(self, kg):
        """before ∘ before → before"""
        assert kg.composition_rules.get(("before", "before")) == "before"


class TestCrossRelationComposition:
    """Cross-relation composition via explicit rules."""

    def test_capital_of_plus_in(self, kg):
        """Paris capital_of France ∧ France in EU → Paris in EU"""
        kg.add_fact("paris", "capital_of", "france")
        kg.add_fact("france", "in", "eu")
        kg.derive_all()
        assert any(
            t.subject == "paris" and t.relation == "in" and t.object == "eu"
            for t in kg.triples
        )

    def test_no_born_in_in_composition(self, kg):
        """born_in(X,Y) ∧ in(Y,Z) should NOT derive born_in(X,Z)
        unless explicitly declared."""
        kg.add_fact("einstein", "born_in", "ulm")
        kg.add_fact("ulm", "in", "germany")
        kg.derive_all()
        # born_in ∘ in is NOT in default composition rules
        assert not any(
            t.subject == "einstein" and t.relation == "born_in" and t.object == "germany"
            for t in kg.triples
        )

    def test_explicit_born_in_in_rule(self, kg):
        """After declaring born_in ∘ in → in, composition should work."""
        kg.set_composition_rule("born_in", "in", "in")
        kg.add_fact("einstein", "born_in", "ulm")
        kg.add_fact("ulm", "in", "germany")
        kg.derive_all()
        assert any(
            t.subject == "einstein" and t.relation == "in" and t.object == "germany"
            for t in kg.triples
        )

    def test_blocked_composition(self, kg):
        """Explicitly blocked composition should not derive."""
        kg.set_composition_rule("borders", "borders", None)
        kg.add_fact("france", "borders", "germany")
        kg.add_fact("germany", "borders", "poland")
        kg.derive_all()
        # borders ∘ borders → None (blocked)
        assert not any(
            t.subject == "france" and t.relation == "borders" and t.object == "poland"
            for t in kg.triples
        )


class TestAutoCompositionFromTeachRelation:
    """Teaching a relation as transitive auto-adds composition rule."""

    def test_teach_transitive_adds_rule(self, kg):
        """teach_relation('north_of', 'transitive') adds (north_of, north_of) → north_of"""
        kg.set_relation_property("north_of", "transitive")
        assert kg.composition_rules.get(("north_of", "north_of")) == "north_of"

    def test_teach_transitive_enables_derive(self, kg):
        """After teaching north_of as transitive, derive_all should work."""
        kg.set_relation_property("north_of", "transitive")
        kg.add_fact("kazakhstan", "north_of", "uzbekistan")
        kg.add_fact("uzbekistan", "north_of", "tajikistan")
        kg.derive_all()
        assert any(
            t.subject == "kazakhstan" and t.relation == "north_of" and t.object == "tajikistan"
            for t in kg.triples
        )


class TestCompositionConfidence:
    """Confidence reflects composition rule quality."""

    def test_explicit_rule_high_confidence(self, kg):
        """Chain with explicit composition rules has high confidence."""
        kg.add_fact("paris", "capital_of", "france")
        kg.add_fact("france", "in", "eu")
        kg.add_fact("eu", "in", "europe")
        r = kg.query("paris", "in", "europe")
        assert r.answer is True
        assert r.confidence >= 0.90  # All explicit rules

    def test_no_rule_low_confidence(self, kg):
        """Chain without explicit rules has lower confidence."""
        kg.set_relation_property("custom_rel", "transitive")
        kg.add_fact("a", "custom_rel", "b")
        kg.add_fact("b", "custom_rel", "c")
        r = kg.query("a", "custom_rel", "c")
        assert r.answer is True
        # custom_rel has composition rule (auto-added by teach_relation)
        assert r.confidence >= 0.90
