"""Tests for spaCy-based triple extraction (Issue 5 fix).

Tests non-copular patterns, passive voice, and compound relations
that the regex fallback couldn't handle.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from td.kg import KnowledgeGraph
from td.perception.hdc import build_default_vocabulary
from td.memory.mhn import ModernHopfieldNetwork, MHNConfig
from td.thinking import GenericThinkingDust


@pytest.fixture
def td():
    vocab = build_default_vocabulary(dim=10000)
    mhn = ModernHopfieldNetwork(MHNConfig(dim=10000, min_similarity=0.01, idp_enabled=False))
    return GenericThinkingDust(vocab=vocab, mhn=mhn, dim=10000, pure_mode=True)


class TestCopularExtraction:
    """Copular constructions: 'X is Y'"""

    def test_is_the_capital_of(self, td):
        triples = td._extract_triples("Paris is the capital of France", "")
        assert ("paris", "capital_of", "france") in triples

    def test_is_in(self, td):
        triples = td._extract_triples("France is in the EU", "")
        assert ("france", "in", "eu") in triples

    def test_is_part_of(self, td):
        triples = td._extract_triples("EU is part of Europe", "")
        assert ("eu", "part_of", "europe") in triples

    def test_is_before(self, td):
        triples = td._extract_triples("Germany is before Austria", "")
        assert ("germany", "before", "austria") in triples

    def test_is_married_to(self, td):
        triples = td._extract_triples("Alice is married to Bob", "")
        assert ("alice", "married_to", "bob") in triples

    def test_is_sibling_of(self, td):
        triples = td._extract_triples("Bob is sibling of Carol", "")
        assert ("bob", "sibling_of", "carol") in triples


class TestNonCopularExtraction:
    """Non-copular patterns (no 'is'): 'X R Y'"""

    def test_noun_compound_prep(self, td):
        """'Paris capital of France' → (paris, capital_of, france)"""
        triples = td._extract_triples("Paris capital of France", "")
        assert ("paris", "capital_of", "france") in triples

    def test_noun_prep(self, td):
        """'France in the EU' → (france, in, eu)"""
        triples = td._extract_triples("France in the EU", "")
        assert ("france", "in", "eu") in triples

    def test_noun_appos_prep(self, td):
        """'Kazakhstan north of Uzbekistan' → (kazakhstan, north_of, uzbekistan)"""
        triples = td._extract_triples("Kazakhstan north of Uzbekistan", "")
        assert ("kazakhstan", "north_of", "uzbekistan") in triples


class TestPassiveVoiceExtraction:
    """Passive voice: 'X is V-ed by Y'"""

    def test_made_by(self, td):
        """'iPhone is made by Apple' → (iphone, made_by, apple)"""
        triples = td._extract_triples("iPhone is made by Apple", "")
        assert ("iphone", "made_by", "apple") in triples

    def test_directed_by(self, td):
        """'The movie is directed by Spielberg' → (movie, directed_by, spielberg)"""
        triples = td._extract_triples("The movie is directed by Spielberg", "")
        assert ("movie", "directed_by", "spielberg") in triples


class TestVerbExtraction:
    """Verb-based: 'X evolved from Y', 'X treats Y'"""

    def test_evolved_from(self, td):
        triples = td._extract_triples("X evolved from Y", "")
        assert ("x", "evolved_from", "y") in triples

    def test_simple_verb(self, td):
        triples = td._extract_triples("Alice called Bob", "")
        assert ("alice", "called", "bob") in triples


class TestSpacyEndToEnd:
    """End-to-end: teach facts via spaCy extraction, query via KG."""

    def test_teach_and_query_non_copular(self, td):
        """Teach without 'is', query via KG."""
        td.teach("Paris capital of France", "Paris")
        td.teach("France in the EU", "France is in the EU")
        result = td.think("is Paris in the EU")
        assert result.solution is not None
        assert result.solution["type"] == "inferred"

    def test_teach_and_query_passive(self, td):
        """Teach passive voice, query via KG."""
        td.teach("iPhone is made by Apple", "Apple")
        td.teach("Apple was founded by Steve Jobs", "Steve Jobs")
        result = td.think("who makes iPhone")
        # Should at least find Apple via MHN or KG
        assert result.solution is not None
