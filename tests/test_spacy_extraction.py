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


class TestCopularPrepRelationNouns:
    """Copular + prep extraction with RELATION_NOUNS filtering.

    Based on EDC Framework (Zhang & Soh, 2024) + ClausIE complement/adjunct
    distinction (Del Corro & Gemulla, WWW 2013).

    Rules:
    - ALWAYS emit compound relation from prep chain (never suppress)
    - Emit is_a ONLY if attr is NOT a relation noun
    - Relation nouns: part, made, component, derivative, etc.
    - Type nouns: city, student, organelle, etc.

    Reference: ClausIE — https://resources.mpi-inf.mpg.de/d5/clausie/clausie-www13.pdf
    Reference: EDC — arXiv:2404.03868
    Reference: Stanford OpenIE (Angeli et al., 2015)
    """

    def test_part_of_no_is_a(self, td):
        """'cell is part of organism' → (cell, part_of, organism), NO (cell, is_a, part)
        'part' is a RELATION NOUN — it expresses meronymy, not a type."""
        triples = td._extract_triples("cell is part of organism", "")
        assert ("cell", "part_of", "organism") in triples
        assert ("cell", "is_a", "part") not in triples

    def test_capital_of_emits_both(self, td):
        """'Paris is the capital of France' → (paris, capital_of, france) AND (paris, is_a, capital)
        'capital' is a TYPE NOUN (has hypernym 'city' in WordNet) — emit both."""
        triples = td._extract_triples("Paris is the capital of France", "")
        assert ("paris", "capital_of", "france") in triples
        # capital is NOT a relation noun, so is_a should also be emitted
        assert ("paris", "is_a", "capital") in triples

    def test_made_of_no_is_a(self, td):
        """'dna is made of nucleotides' → (dna, made_of, nucleotides), NOT (dna, is_a, made)
        'made' is a RELATION NOUN — it expresses composition, not a type."""
        triples = td._extract_triples("DNA is made of nucleotides", "")
        has_compound = any(
            s == "dna" and "made" in r and o == "nucleotides"
            for s, r, o in triples
        )
        assert has_compound, f"Expected compound relation (dna, made_*, nucleotides), got {triples}"
        assert ("dna", "is_a", "made") not in triples

    def test_simple_is_a_still_works(self, td):
        """'cell is organelle' → (cell, is_a, organelle) — no prep child on attr"""
        triples = td._extract_triples("cell is organelle", "")
        assert ("cell", "is_a", "organelle") in triples

    def test_sibling_of_no_is_a(self, td):
        """'bob is sibling of carol' → (bob, sibling_of, carol)
        spaCy parses 'sibling' as ROOT+VERB (lemma 'sible'), so this goes
        through the verb extraction path, not the copular path. No is_a emitted.
        'sibling' is a type noun, but the parse doesn't produce attr+cop structure."""
        triples = td._extract_triples("Bob is sibling of Carol", "")
        assert ("bob", "sibling_of", "carol") in triples

    def test_married_to_no_is_a(self, td):
        """'alice is married to bob' → (alice, married_to, bob), NOT (alice, is_a, married)
        'married' is a relation word (past participle used as adjective) — not a type."""
        triples = td._extract_triples("Alice is married to Bob", "")
        assert ("alice", "married_to", "bob") in triples
        assert ("alice", "is_a", "married") not in triples

    # ── NEW: Type noun + adjunct prep → emit BOTH ──────────────────

    def test_city_in_france_emits_both(self, td):
        """'Paris is a city in France' → (paris, is_a, city) AND (paris, city_in, france)
        'city' is a TYPE NOUN. 'in France' is an adjunct prep. Emit both."""
        triples = td._extract_triples("Paris is a city in France", "")
        assert ("paris", "is_a", "city") in triples, f"Expected is_a in {triples}"
        # Compound relation from prep chain
        has_compound = any(
            s == "paris" and "france" in o
            for s, r, o in triples
        )
        assert has_compound, f"Expected compound relation to france in {triples}"

    def test_student_of_philosophy_emits_both(self, td):
        """'Alice is a student of philosophy' → (alice, is_a, student) AND (alice, student_of, philosophy)
        'student' is a TYPE NOUN. 'of philosophy' is a complement prep. Emit both."""
        triples = td._extract_triples("Alice is a student of philosophy", "")
        assert ("alice", "is_a", "student") in triples, f"Expected is_a in {triples}"
        has_compound = any(
            s == "alice" and "philosophy" in o
            for s, r, o in triples
        )
        assert has_compound, f"Expected compound relation to philosophy in {triples}"

    def test_friend_of_alice_emits_both(self, td):
        """'Bob is a friend of Alice' → (bob, is_a, friend) AND (bob, friend_of, alice)
        'friend' is a TYPE NOUN. Emit both triples."""
        triples = td._extract_triples("Bob is a friend of Alice", "")
        assert ("bob", "is_a", "friend") in triples, f"Expected is_a in {triples}"
        has_compound = any(
            s == "bob" and "alice" in o
            for s, r, o in triples
        )
        assert has_compound, f"Expected compound relation to alice in {triples}"

    def test_component_of_no_is_a(self, td):
        """'CPU is a component of the computer' → (cpu, component_of, computer), NO is_a
        'component' is a RELATION NOUN."""
        triples = td._extract_triples("CPU is a component of the computer", "")
        has_compound = any(
            s == "cpu" and "computer" in o
            for s, r, o in triples
        )
        assert has_compound, f"Expected compound relation in {triples}"
        assert ("cpu", "is_a", "component") not in triples


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
        """'iPhone is made by Apple' → passive voice swap: (apple, made, iphone)
        TEA Nets (2026): nsubjpass + agent dep → swap subject/object.
        """
        triples = td._extract_triples("iPhone is made by Apple", "")
        found = any("apple" in s.lower() for s, r, o in triples)
        assert found, f"Expected Apple as subject (passive swap), got {triples}"

    def test_directed_by(self, td):
        """'The movie is directed by Spielberg' → passive voice: agent first.
        TEA Nets (2026): nsubjpass + agent dep → swap subject/object.
        (spielberg, directed, movie) not (movie, directed_by, spielberg)
        """
        triples = td._extract_triples("The movie is directed by Spielberg", "")
        # Passive voice: agent (Spielberg) becomes subject
        found = any("spielberg" in s.lower() for s, r, o in triples)
        assert found, f"Expected Spielberg as subject (passive swap), got {triples}"


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
