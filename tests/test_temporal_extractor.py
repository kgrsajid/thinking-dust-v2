"""Tests for temporal ordering extraction from discourse connectives.

Tests the extraction of temporal ordering triples from text using
discourse connectives ("then", "after", "before", "subsequently").

References:
- Allen (1983), "Maintaining Knowledge about Temporal Intervals"
- TimeML (Pustejovsky et al., 2003)
- Consistent Discourse-level TRE (EMNLP 2025)
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import spacy
    nlp_test = spacy.load("en_core_web_sm")
    HAS_SPACY = True
except (ImportError, OSError):
    HAS_SPACY = False

pytestmark = pytest.mark.skipif(not HAS_SPACY, reason="spaCy not installed")

from td.perception.temporal_extractor import (
    extract_temporal_orderings,
    temporal_triples_from_text,
    TemporalOrdering,
)
from td.perception.temporal_connectives import (
    get_connectives,
    get_conditional_markers,
    AllenRelation,
    SemanticType,
)


@pytest.fixture
def nlp():
    return spacy.load("en_core_web_sm")


# ─── "Then" Connective ──────────────────────────────────────────────

class TestThenConnective:
    """Test temporal ordering with 'then'."""

    def test_then_coordinated_verbs(self, nlp):
        """'Alice went to Paris and then invested in stocks' → went BEFORE invested."""
        doc = nlp("Alice went to Paris and then invested in stocks")
        orderings = extract_temporal_orderings(doc)
        assert len(orderings) == 1
        assert orderings[0].relation == "before"
        assert "went" in orderings[0].event1_description
        assert "invested" in orderings[0].event2_description

    def test_then_cross_sentence(self, nlp):
        """'Alice went to Paris. Then she invested in stocks.' → went BEFORE invested."""
        doc = nlp("Alice went to Paris. Then she invested in stocks.")
        orderings = extract_temporal_orderings(doc)
        assert len(orderings) == 1
        assert orderings[0].relation == "before"

    def test_then_conditional_excluded(self, nlp):
        """'If you go to Paris then you should visit the Louvre' → no temporal ordering."""
        doc = nlp("If you go to Paris then you should visit the Louvre")
        orderings = extract_temporal_orderings(doc)
        assert len(orderings) == 0


# ─── "After" Connective ─────────────────────────────────────────────

class TestAfterConnective:
    """Test temporal ordering with 'after'."""

    def test_after_subordinating(self, nlp):
        """'After Alice went to Paris, she invested in stocks' → went BEFORE invested."""
        doc = nlp("After Alice went to Paris, she invested in stocks")
        orderings = extract_temporal_orderings(doc)
        assert len(orderings) == 1
        assert orderings[0].relation == "before"

    def test_after_preposition(self, nlp):
        """'Alice went to Paris after visiting London' → visiting BEFORE went."""
        doc = nlp("Alice went to Paris after visiting London")
        orderings = extract_temporal_orderings(doc)
        assert len(orderings) == 1
        assert orderings[0].relation == "before"


# ─── "Before" Connective ────────────────────────────────────────────

class TestBeforeConnective:
    """Test temporal ordering with 'before'."""

    def test_before_subordinating(self, nlp):
        """'Before leaving, Alice went to Paris' → went BEFORE leaving."""
        doc = nlp("Before leaving, Alice went to Paris")
        orderings = extract_temporal_orderings(doc)
        assert len(orderings) == 1
        assert orderings[0].relation == "before"

    def test_before_preposition(self, nlp):
        """'Alice went to Paris before investing in stocks' → went BEFORE investing."""
        doc = nlp("Alice went to Paris before investing in stocks")
        orderings = extract_temporal_orderings(doc)
        assert len(orderings) == 1
        assert orderings[0].relation == "before"


# ─── Other Connectives ──────────────────────────────────────────────

class TestOtherConnectives:
    """Test other temporal connectives."""

    def test_subsequently(self, nlp):
        """'Alice went to Paris. Subsequently, she invested in stocks.'"""
        doc = nlp("Alice went to Paris. Subsequently, she invested in stocks.")
        orderings = extract_temporal_orderings(doc)
        assert len(orderings) == 1
        assert orderings[0].relation == "before"

    def test_no_connective(self, nlp):
        """'Alice went to Paris and Bob went to London' → no temporal ordering."""
        doc = nlp("Alice went to Paris and Bob went to London")
        orderings = extract_temporal_orderings(doc)
        assert len(orderings) == 0

    def test_simple_sentence(self, nlp):
        """'Alice went to Paris' → no temporal ordering."""
        doc = nlp("Alice went to Paris")
        orderings = extract_temporal_orderings(doc)
        assert len(orderings) == 0


# ─── Edge Cases ─────────────────────────────────────────────────────

class TestTemporalEdgeCases:
    """Edge cases for temporal ordering extraction."""

    def test_multiple_connectives(self, nlp):
        """Multiple temporal connectives in one text."""
        doc = nlp("Alice went to Paris and then invested in stocks. Subsequently she moved to London.")
        orderings = extract_temporal_orderings(doc)
        assert len(orderings) >= 2

    def test_empty_text(self, nlp):
        """Empty text → no orderings."""
        doc = nlp("")
        orderings = extract_temporal_orderings(doc)
        assert len(orderings) == 0

    def test_convenience_function(self, nlp):
        """temporal_triples_from_text() returns tuples."""
        triples = temporal_triples_from_text(
            "Alice went to Paris and then invested in stocks", nlp
        )
        assert len(triples) == 1
        assert triples[0][1] == "before"  # relation


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
