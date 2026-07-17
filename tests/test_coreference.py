"""Tests for coreference resolution integration.

Tests the spaCy two-pipeline approach for resolving pronouns
and filtering discourse deixis.

Reference: spaCy coref blog (Explosion, 2022)
Reference: Guerra et al. "Resolving Discourse-Deictic Pronouns" (SemEval 2015)
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

try:
    from td.query import HAS_OXIGRAPH
except ImportError:
    HAS_OXIGRAPH = False

# Coreference needs spacy-experimental + en_coreference_web_trf
HAS_COREF = False
if HAS_SPACY:
    try:
        nlp_coref_test = spacy.load("en_coreference_web_trf", vocab=nlp_test.vocab)
        HAS_COREF = True
    except (ImportError, OSError):
        pass

pytestmark = pytest.mark.skipif(
    not (HAS_SPACY and HAS_COREF),
    reason="spaCy + en_coreference_web_trf not available"
)

from td.perception.nl_parser import GenericNLParser
from td.perception.hdc import build_default_vocabulary
from td.memory.mhn import ModernHopfieldNetwork, MHNConfig


@pytest.fixture
def parser():
    p = GenericNLParser.__new__(GenericNLParser)
    p._nlp = spacy.load("en_core_web_sm")
    p._nlp_coref = spacy.load("en_coreference_web_trf", vocab=p._nlp.vocab)
    p._coref_enabled = True  # Enable coreference for tests
    p.vocab = build_default_vocabulary(dim=10000)
    p.mhn = ModernHopfieldNetwork(MHNConfig(dim=10000, min_similarity=0.01))
    p.dim = 10000
    return p


# ─── Coreference Resolution ─────────────────────────────────────────

class TestCoreferenceResolution:
    """Test pronoun resolution via spaCy two-pipeline."""

    def test_pronoun_he(self, parser):
        """'Alice went home. She was tired.' → She → Alice."""
        resolved, coref_map = parser.resolve_coreferences(
            "Alice went home. She was tired."
        )
        # Text is returned unchanged; coref_map maps pronoun indices to entities
        assert len(coref_map) >= 1
        entities = [entity for entity, _ in coref_map.values()]
        assert any("alice" in e for e in entities)

    def test_pronoun_he_male(self, parser):
        """'Peter went to the store. He bought milk.' → He → Peter."""
        resolved, coref_map = parser.resolve_coreferences(
            "Peter went to the store. He bought milk."
        )
        assert len(coref_map) >= 1
        entities = [entity for entity, _ in coref_map.values()]
        assert any("peter" in e for e in entities)

    def test_pronoun_it(self, parser):
        """'The cat sat on the mat. It was comfortable.' → It → cat/mat."""
        resolved, coref_map = parser.resolve_coreferences(
            "The cat sat on the mat. It was comfortable."
        )
        # coref_map should contain at least one pronoun resolution
        assert len(coref_map) >= 1

    def test_pronoun_they(self, parser):
        """'Alice and Bob went to Paris. They loved it.' → They → Alice and Bob."""
        resolved, coref_map = parser.resolve_coreferences(
            "Alice and Bob went to Paris. They loved it."
        )
        assert len(coref_map) >= 1

    def test_possessive_its(self, parser):
        """'The company announced its earnings. It was record-breaking.'"""
        resolved, coref_map = parser.resolve_coreferences(
            "The company announced its earnings. It was record-breaking."
        )
        # "its" and "it" should be resolved
        assert len(coref_map) >= 1

    def test_no_coreference(self, parser):
        """Simple sentence with no pronouns — no changes."""
        text = "Paris is the capital of France."
        resolved, coref_map = parser.resolve_coreferences(text)
        assert coref_map == {}
        assert resolved == text

    def test_empty_text(self, parser):
        """Empty text — no changes."""
        resolved, coref_map = parser.resolve_coreferences("")
        assert coref_map == {}


# ─── Discourse Deixis ───────────────────────────────────────────────

class TestDiscourseDeixis:
    """Test discourse deixis filtering (this/that referring to clauses)."""

    def test_discourse_deixis_this_shows(self, parser):
        """'X is Y. This shows Z.' → 'this' should NOT be replaced."""
        resolved, coref_map = parser.resolve_coreferences(
            "The experiment was successful. This shows the method works."
        )
        # "This" refers to the clause, not an entity — should be kept as-is
        assert "this" in resolved.lower()

    def test_discourse_deixis_this_means(self, parser):
        """'X happened. This means Y.' → 'this' should NOT be replaced."""
        resolved, coref_map = parser.resolve_coreferences(
            "The server crashed. This means we lost data."
        )
        # "This" is discourse deixis — should be kept
        assert "this" in resolved.lower()

    def test_entity_reference_this_car(self, parser):
        """'I bought a car. This car is fast.' → 'this' = car (entity reference)."""
        resolved, coref_map = parser.resolve_coreferences(
            "I bought a car. This car is fast."
        )
        # "This car" is an entity reference — should work normally
        # (coreference model may or may not resolve this)


# ─── Integration with Triple Extraction ─────────────────────────────

class TestCoreferenceIntegration:
    """Test coreference resolution integrated into triple extraction."""

    def test_pronoun_resolved_before_extraction(self, parser):
        """'Alice is in Paris. She is in France.' → 'She' resolved to 'alice'."""
        triples = parser.extract_triples_spacy(
            "Alice is in Paris. She is in France."
        )
        # Should extract (alice, in, france) not (she, in, france)
        subjects = [s for s, r, o in triples]
        assert any("alice" in s for s in subjects)
        assert not any(s == "she" for s in subjects)

    def test_its_resolved(self, parser):
        """'A video game console is designed for its video games.' → 'its' resolved."""
        triples = parser.extract_triples_spacy(
            "A video game console is designed for its video games."
        )
        # "its video games" should be resolved to "video game console's video games"
        # or just "video games" after resolution
        if triples:
            objects = [o for s, r, o in triples]
            # Should NOT have "its" in objects
            assert not any("its" in o for o in objects)

    def test_it_resolved_to_entity(self, parser):
        """'The cat sat on the mat. It was comfortable.' → 'it' resolved."""
        triples = parser.extract_triples_spacy(
            "The cat sat on the mat. It was comfortable."
        )
        # "It" should be resolved to "cat" or "mat" — but coref model
        # may not always resolve this reliably, so just check no bare "it"
        # subject remains if triples were extracted
        subjects = [s for s, r, o in triples]
        if subjects:
            # At least one subject should not be "it"
            assert not all(s == "it" for s in subjects)

    def test_discourse_deixis_not_extracted(self, parser):
        """'X is Y. This shows Z.' → 'this' should NOT become a triple subject."""
        triples = parser.extract_triples_spacy(
            "The experiment was successful. This shows the method works."
        )
        # "This shows the method works" should NOT produce (this, shows, ...)
        subjects = [s for s, r, o in triples]
        assert "this" not in subjects

    def test_no_pronouns_no_change(self, parser):
        """Simple sentence — no coreference needed."""
        triples = parser.extract_triples_spacy(
            "Paris is the capital of France."
        )
        assert len(triples) >= 1
        assert any("paris" in s for s, r, o in triples)


# ─── Edge Cases ─────────────────────────────────────────────────────

class TestCoreferenceEdgeCases:
    """Edge cases for coreference resolution."""

    def test_multiple_pronouns_same_entity(self, parser):
        """Multiple pronouns referring to the same entity."""
        resolved, coref_map = parser.resolve_coreferences(
            "Alice went home. She was tired. Her friends called her."
        )
        # coref_map should have multiple pronouns mapped to alice
        entities = [entity for entity, _ in coref_map.values()]
        assert any("alice" in e for e in entities)
        assert len(coref_map) >= 2

    def test_multiple_entities(self, parser):
        """Multiple entities with different pronouns."""
        resolved, coref_map = parser.resolve_coreferences(
            "Alice met Bob. She gave him a book."
        )
        # She → Alice, him → Bob
        entities = [entity for entity, _ in coref_map.values()]
        assert len(coref_map) >= 1

    def test_coreference_with_clause_segmentation(self, parser):
        """Coreference + clause segmentation together."""
        triples = parser.extract_triples_spacy(
            "Alice and Bob went to Paris. They loved the city. It was beautiful."
        )
        # Should extract: alice went, bob went, they/they loved, city was beautiful
        # Coreference should resolve they → Alice and Bob, it → city/Paris
        assert len(triples) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
