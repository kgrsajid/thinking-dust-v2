"""Tests for relation canonicalization and triple deduplication.

Tests the post-extraction canonicalization approach (Option B from
EDC framework) for solving the duplicate triple problem.

Reference: Zhang & Soh (2024), "Extract, Define, Canonicalize" — arXiv:2404.03868
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from td.perception.relation_canonicalizer import (
    canonicalize_relation,
    relation_specificity,
    deduplicate_triples,
    PREPOSITION_SUFFIXES,
    COMPOUND_RELATIONS,
)


# ─── Relation Canonicalization ──────────────────────────────────────

class TestCanonicalizeRelation:
    """Test the canonicalize_relation() function."""

    # Verb + preposition → lemmatized verb
    def test_went_to(self):
        assert canonicalize_relation("went_to") == "go"

    def test_invested_in(self):
        assert canonicalize_relation("invested_in") == "invest"

    def test_lived_in(self):
        assert canonicalize_relation("lived_in") == "live"

    def test_depended_on(self):
        """'depended_on' is NOT in COMPOUND_RELATIONS, so it gets canonicalized."""
        result = canonicalize_relation("depended_on")
        assert result == "depend"  # lemmatized

    def test_connected_to(self):
        """'connected_to' IS in COMPOUND_RELATIONS, kept as-is."""
        assert canonicalize_relation("connected_to") == "connected_to"

    # Bare verbs → lemmatized
    def test_went(self):
        assert canonicalize_relation("went") == "go"

    def test_invested(self):
        assert canonicalize_relation("invested") == "invest"

    def test_running(self):
        assert canonicalize_relation("running") == "run"

    def test_created(self):
        assert canonicalize_relation("created") == "create"

    # Compound relations → kept as-is
    def test_capital_of(self):
        assert canonicalize_relation("capital_of") == "capital_of"

    def test_part_of(self):
        assert canonicalize_relation("part_of") == "part_of"

    def test_depends_on(self):
        assert canonicalize_relation("depends_on") == "depends_on"

    def test_married_to(self):
        assert canonicalize_relation("married_to") == "married_to"

    # Non-verb relations → kept as-is
    def test_in(self):
        assert canonicalize_relation("in") == "in"

    def test_before(self):
        assert canonicalize_relation("before") == "before"

    def test_contains(self):
        assert canonicalize_relation("contains") == "contain"

    # Case insensitivity
    def test_uppercase(self):
        assert canonicalize_relation("WENT_TO") == "go"

    def test_mixed_case(self):
        assert canonicalize_relation("Invested_In") == "invest"

    # Edge cases
    def test_empty(self):
        assert canonicalize_relation("") == ""

    def test_underscore_only(self):
        """'_to' has no verb part (empty after stripping suffix) — return as-is."""
        # verb_part would be "" which is < 2 chars, so no canonicalization
        result = canonicalize_relation("_to")
        # spaCy lemmatizes "_to" → "_to" or "to" depending on model
        # The important thing is it doesn't crash
        assert result is not None

    def test_short_verb(self):
        """'go_to' — verb part 'go' is short but valid."""
        result = canonicalize_relation("go_to")
        assert result == "go"


# ─── Deduplication Pairs ────────────────────────────────────────────

class TestDeduplicationPairs:
    """Test that canonicalization produces matching keys for duplicates."""

    def test_went_to_vs_went(self):
        """The core duplicate case."""
        assert canonicalize_relation("went_to") == canonicalize_relation("went")

    def test_invested_in_vs_invested(self):
        assert canonicalize_relation("invested_in") == canonicalize_relation("invested")

    def test_lived_in_vs_lived(self):
        assert canonicalize_relation("lived_in") == canonicalize_relation("lived")

    def test_created_vs_created_by(self):
        """'created' vs 'created_by' — different because 'by' is in COMPOUND_RELATIONS."""
        c1 = canonicalize_relation("created")
        c2 = canonicalize_relation("created_by")
        # "created_by" is in COMPOUND_RELATIONS, kept as-is
        assert c1 != c2

    def test_capital_of_vs_capital(self):
        """'capital_of' is compound, 'capital' is bare. Different."""
        assert canonicalize_relation("capital_of") != canonicalize_relation("capital")

    def test_in_vs_in(self):
        """Same relation, same canonical."""
        assert canonicalize_relation("in") == canonicalize_relation("in")


# ─── Specificity Scoring ────────────────────────────────────────────

class TestSpecificity:
    """Test the relation_specificity() function."""

    def test_compound_is_most_specific(self):
        assert relation_specificity("capital_of") == 3

    def test_verb_prep_is_medium(self):
        assert relation_specificity("went_to") == 2

    def test_bare_verb_is_least(self):
        assert relation_specificity("went") == 1

    def test_preposition_is_least(self):
        assert relation_specificity("in") == 1

    def test_specificity_preference(self):
        """When duplicates found, more specific wins."""
        assert relation_specificity("went_to") > relation_specificity("went")
        assert relation_specificity("invested_in") > relation_specificity("invested")


# ─── Triple Deduplication ───────────────────────────────────────────

class TestDeduplicateTriples:
    """Test the deduplicate_triples() function."""

    def test_exact_duplicates(self):
        """Exact duplicates are removed."""
        triples = [
            ("alice", "went_to", "paris"),
            ("alice", "went_to", "paris"),
        ]
        result = deduplicate_triples(triples)
        assert len(result) == 1

    def test_canonical_duplicates_keep_specific(self):
        """went_to and went → keep went_to (more specific)."""
        triples = [
            ("alice", "went_to", "paris"),
            ("alice", "went", "paris"),
        ]
        result = deduplicate_triples(triples)
        assert len(result) == 1
        assert result[0] == ("alice", "went_to", "paris")

    def test_canonical_duplicates_preposition_wins(self):
        """invested_in and invested → keep invested_in."""
        triples = [
            ("company", "invested", "ai"),
            ("company", "invested_in", "ai"),
        ]
        result = deduplicate_triples(triples)
        assert len(result) == 1
        assert result[0] == ("company", "invested_in", "ai")

    def test_different_objects_not_deduplicated(self):
        """Same relation, different objects → keep both."""
        triples = [
            ("alice", "went_to", "paris"),
            ("alice", "went_to", "london"),
        ]
        result = deduplicate_triples(triples)
        assert len(result) == 2

    def test_different_subjects_not_deduplicated(self):
        """Same relation+object, different subjects → keep both."""
        triples = [
            ("alice", "went_to", "paris"),
            ("bob", "went", "paris"),
        ]
        result = deduplicate_triples(triples)
        assert len(result) == 2

    def test_no_duplicates(self):
        """All unique triples preserved."""
        triples = [
            ("alice", "went_to", "paris"),
            ("bob", "invested_in", "ai"),
            ("company", "capital_of", "country"),
        ]
        result = deduplicate_triples(triples)
        assert len(result) == 3

    def test_empty_input(self):
        assert deduplicate_triples([]) == []

    def test_single_triple(self):
        result = deduplicate_triples([("alice", "went", "paris")])
        assert len(result) == 1

    def test_complex_real_world(self):
        """Real-world scenario: Alice and Bob went to Paris."""
        triples = [
            # From dependency extraction
            ("alice", "went_to", "paris"),
            ("bob", "went_to", "paris"),
            # From clause segmenter
            ("alice", "went", "paris"),
            ("bob", "went", "paris"),
        ]
        result = deduplicate_triples(triples)
        assert len(result) == 2
        # Should keep the more specific "went_to"
        assert ("alice", "went_to", "paris") in result
        assert ("bob", "went_to", "paris") in result

    def test_mixed_compound_and_bare(self):
        """Compound relations preserved alongside bare verbs."""
        triples = [
            ("paris", "capital_of", "france"),  # compound, keep
            ("alice", "went_to", "paris"),       # verb+prep, canonicalize
            ("alice", "went", "paris"),          # bare verb, duplicate
        ]
        result = deduplicate_triples(triples)
        assert len(result) == 2
        assert ("paris", "capital_of", "france") in result
        assert ("alice", "went_to", "paris") in result


# ─── Integration with Parser ────────────────────────────────────────

try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

@pytest.mark.skipif(not HAS_SPACY, reason="spaCy not installed")
class TestParserIntegration:
    """Test canonicalization integrated into extract_triples_spacy()."""

    @pytest.fixture
    def parser(self):
        from td.perception.nl_parser import GenericNLParser
        from td.perception.hdc import build_default_vocabulary
        from td.memory.mhn import ModernHopfieldNetwork, MHNConfig
        p = GenericNLParser.__new__(GenericNLParser)
        from td.languages import get_language
        p._lang_config = get_language("en")
        p._fallback_stop_words = p._lang_config.stop_words
        p._nlp = spacy.load("en_core_web_sm")
        p._coref_enabled = False
        p.vocab = build_default_vocabulary(dim=10000)
        p.mhn = ModernHopfieldNetwork(MHNConfig(dim=10000, min_similarity=0.01))
        p.dim = 10000
        return p

    def test_alice_and_bob_went_to_paris(self, parser):
        """The original bug: 4 triples → 2 after deduplication."""
        triples = parser.extract_triples_spacy("Alice and Bob went to Paris")
        # Should have 2 triples, not 4
        alice_triples = [(s, r, o) for s, r, o in triples if s == "alice"]
        bob_triples = [(s, r, o) for s, r, o in triples if s == "bob"]
        assert len(alice_triples) == 1, f"Alice has {len(alice_triples)} triples: {alice_triples}"
        assert len(bob_triples) == 1, f"Bob has {len(bob_triples)} triples: {bob_triples}"

    def test_france_and_germany_in_europe(self, parser):
        """'France and Germany are in Europe' — deduplication handles this.

        Note: clause segmenter may extract 'are' as relation while dependency
        extracts 'in'. These are genuinely different relations (copula vs prep).
        The deduplication correctly keeps both when they canonicalize differently.
        """
        triples = parser.extract_triples_spacy("France and Germany are in Europe")
        france_triples = [(s, r, o) for s, r, o in triples if s == "france"]
        germany_triples = [(s, r, o) for s, r, o in triples if s == "germany"]
        # At minimum, we should have the dependency-extracted 'in' triples
        assert any(r == "in" for s, r, o in france_triples), f"France triples: {france_triples}"
        assert any(r == "in" for s, r, o in germany_triples), f"Germany triples: {germany_triples}"

    def test_simple_sentence_no_dedup(self, parser):
        """Simple sentence — no duplicates to remove."""
        triples = parser.extract_triples_spacy("Paris is the capital of France")
        assert len(triples) >= 1
        assert any("paris" in s for s, r, o in triples)

    def test_coordinated_objects_still_work(self, parser):
        """Coordinated objects still produce multiple triples."""
        triples = parser.extract_triples_spacy("Games have music, a story and visuals")
        objects = [o for s, r, o in triples]
        assert "music" in objects
        assert any("story" in o for o in objects)
        assert "visuals" in objects

    def test_cross_sentence_no_false_dedup(self, parser):
        """Different sentences with same relation → no false deduplication."""
        triples = parser.extract_triples_spacy(
            "Paris is in France. Berlin is in Germany."
        )
        paris_triples = [(s, r, o) for s, r, o in triples if s == "paris"]
        berlin_triples = [(s, r, o) for s, r, o in triples if s == "berlin"]
        assert len(paris_triples) >= 1
        assert len(berlin_triples) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
