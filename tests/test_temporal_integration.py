"""Integration tests: temporal ordering through the full TD v2 pipeline.

Tests that temporal extraction works end-to-end through:
- td.teach() → parser → temporal_extractor → KG → SPARQL
- td.think() → parser → KG/SPARQL query

These tests verify the full call chain, not just the extractor in isolation.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

try:
    from td.query import HAS_OXIGRAPH
except ImportError:
    HAS_OXIGRAPH = False

pytestmark = pytest.mark.skipif(
    not (HAS_SPACY and HAS_OXIGRAPH),
    reason="spaCy + pyoxigraph required"
)

from td.thinking import GenericThinkingDust
from td.perception.hdc import build_default_vocabulary
from td.memory.mhn import ModernHopfieldNetwork, MHNConfig


@pytest.fixture
def td():
    """Fresh thinking engine."""
    return GenericThinkingDust(
        vocab=build_default_vocabulary(dim=10000),
        mhn=ModernHopfieldNetwork(MHNConfig(dim=10000, min_similarity=0.01)),
        dim=10000, pure_mode=True
    )


# ─── End-to-End Temporal Extraction ────────────────────────────────

class TestTemporalEndToEnd:
    """Temporal ordering through td.teach() → KG → SPARQL."""

    def test_then_produces_temporal_triple(self, td):
        """'Alice went to Paris and then invested' → temporal triple in KG."""
        td.teach("Alice went to Paris and then invested in stocks", "Alice")

        # Should have: facts + temporal ordering
        triples = [(t.subject, t.relation, t.object) for t in td.kg.triples]
        temporal = [t for t in triples if t[1] == "before"]
        assert len(temporal) >= 1, f"No temporal triple found: {triples}"

    def test_temporal_triple_in_sparql(self, td):
        """Temporal triple should be synced to SPARQL store."""
        td.teach("Alice went to Paris and then invested in stocks", "Alice")

        # Check SPARQL store has the temporal triple
        raw = td.sparql_store.query_sparql_bindings(
            'SELECT ?s ?p ?o WHERE { ?s ?p ?o . '
            'FILTER(STRSTARTS(STR(?p), "http://thinking-dust.org/relation/before")) }'
        )
        assert len(raw) >= 1, "No temporal triple in SPARQL store"

    def test_after_subordinating_e2e(self, td):
        """'After Alice went to Paris, she invested' → temporal triple."""
        td.teach("After Alice went to Paris, she invested in stocks", "Alice")
        triples = [(t.subject, t.relation, t.object) for t in td.kg.triples]
        temporal = [t for t in triples if t[1] == "before"]
        assert len(temporal) >= 1, f"No temporal triple: {triples}"

    def test_before_preposition_e2e(self, td):
        """'Alice went to Paris before investing' → temporal triple."""
        td.teach("Alice went to Paris before investing in stocks", "Alice")
        triples = [(t.subject, t.relation, t.object) for t in td.kg.triples]
        temporal = [t for t in triples if t[1] == "before"]
        assert len(temporal) >= 1, f"No temporal triple: {triples}"

    def test_simultaneous_while_e2e(self, td):
        """'Alice sang while Bob played guitar' → overlaps triple."""
        td.teach("Alice sang while Bob played guitar", "Alice")
        triples = [(t.subject, t.relation, t.object) for t in td.kg.triples]
        temporal = [t for t in triples if t[1] in ("overlaps", "during")]
        assert len(temporal) >= 1, f"No simultaneous triple: {triples}"

    def test_no_temporal_when_simple(self, td):
        """Simple sentence — no temporal triple."""
        td.teach("Paris is the capital of France", "Paris")
        triples = [(t.subject, t.relation, t.object) for t in td.kg.triples]
        temporal = [t for t in triples if t[1] in ("before", "after", "overlaps", "during")]
        assert len(temporal) == 0, f"Unexpected temporal triple: {temporal}"


# ─── Temporal + Non-Temporal Coexistence ────────────────────────────

class TestTemporalCoexistence:
    """Temporal triples coexist with entity triples."""

    def test_facts_and_temporal_together(self, td):
        """Entity facts + temporal ordering stored together."""
        td.teach("Alice went to Paris and then invested in stocks", "Alice")

        triples = [(t.subject, t.relation, t.object) for t in td.kg.triples]

        # Entity facts
        entity_facts = [t for t in triples if t[0] == "alice"]
        assert len(entity_facts) >= 2, f"Missing entity facts: {entity_facts}"

        # Temporal ordering
        temporal = [t for t in triples if t[1] == "before"]
        assert len(temporal) >= 1, f"Missing temporal: {triples}"

    def test_temporal_with_teach_relation(self, td):
        """Temporal + explicit relation teaching."""
        td.teach_relation("depends_on", "transitive")
        td.teach("API depends on Database", "API")
        td.teach("Database depends on Server", "Database")
        td.teach("After deploying the API, we updated the Database", "We")

        # Both entity facts and temporal should coexist
        triples = [(t.subject, t.relation, t.object) for t in td.kg.triples]
        entity = [t for t in triples if t[1] == "depends_on"]
        temporal = [t for t in triples if t[1] == "before"]
        assert len(entity) >= 2
        assert len(temporal) >= 1


# ─── Temporal + Persistence ────────────────────────────────────────

class TestTemporalPersistence:
    """Temporal triples survive save/load."""

    def test_temporal_survives_save_load(self, td):
        """Save → load → temporal triple preserved."""
        td.teach("Alice went to Paris and then invested in stocks", "Alice")

        import tempfile, os
        tmp = tempfile.mktemp(suffix=".db")
        td.kg.save(tmp)

        # Load in fresh KG
        from td.kg import KnowledgeGraph
        kg2 = KnowledgeGraph()
        kg2.load(tmp)

        triples = [(t.subject, t.relation, t.object) for t in kg2.triples]
        temporal = [t for t in triples if t[1] == "before"]
        assert len(temporal) >= 1, f"Temporal lost after save/load: {triples}"

        os.unlink(tmp)
        import shutil
        shutil.rmtree(tmp.replace(".db", "_store"), ignore_errors=True)


# ─── Conditional Filtering ─────────────────────────────────────────

class TestConditionalFiltering:
    """Conditional 'then' should NOT produce temporal triples."""

    def test_conditional_then_filtered(self, td):
        """'If you go to Paris then visit the Louvre' → no temporal triple."""
        td.teach("If you go to Paris then you should visit the Louvre", "You")
        triples = [(t.subject, t.relation, t.object) for t in td.kg.triples]
        temporal = [t for t in triples if t[1] in ("before", "after")]
        assert len(temporal) == 0, f"Conditional 'then' produced temporal: {temporal}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
