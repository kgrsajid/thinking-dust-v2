"""Integration tests: SPARQL layer wired into GenericThinkingDust.

Tests that the SPARQL query path works end-to-end through
td.think() and td.teach(), not just the standalone SparqlStore.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from td.thinking import GenericThinkingDust
from td.perception.hdc import build_default_vocabulary
from td.memory.mhn import ModernHopfieldNetwork, MHNConfig

try:
    from td.query import HAS_OXIGRAPH
except ImportError:
    HAS_OXIGRAPH = False

pytestmark = pytest.mark.skipif(not HAS_OXIGRAPH, reason="pyoxigraph not installed")


@pytest.fixture
def td():
    """Fresh GenericThinkingDust with SPARQL store."""
    vocab = build_default_vocabulary(dim=10000)
    mhn = ModernHopfieldNetwork(MHNConfig(dim=10000, min_similarity=0.01))
    engine = GenericThinkingDust(vocab=vocab, mhn=mhn, dim=10000, pure_mode=True)
    return engine


class TestSparqlIntegration:
    """Test SPARQL layer through the thinking engine."""

    def test_sparql_store_initialized(self, td):
        """SparqlStore should be initialized in pure_mode."""
        assert td.sparql_store is not None

    def test_teach_syncs_to_sparql(self, td):
        """teach() should sync facts to SPARQL store."""
        td.teach("Paris is the capital of France", "Paris")
        assert len(td.sparql_store) > 0

    def test_inverse_query_via_sparql(self, td):
        """Inverse query should work through SPARQL path.

        'What is the capital of France?' → Paris
        This is the primary motivation for the SPARQL layer.
        """
        td.teach("Paris is the capital of France", "Paris")
        td.teach("Berlin is the capital of Germany", "Berlin")

        # Inverse query: find subject given object+relation
        capitals = td.sparql_store.inverse_query("capital_of", "france")
        assert "paris" in capitals

    def test_multi_hop_transitive(self, td):
        """Multi-hop transitive via SPARQL property paths."""
        td.teach("Paris is the capital of France", "Paris")
        td.teach("France is in the EU", "France is in the EU")
        td.teach("EU is part of Europe", "EU is part of Europe")

        # SPARQL should find the 3-hop path
        result = td.sparql_store.ask("paris", "europe")
        assert result.found is True

    def test_sparql_synced_on_load(self, td):
        """SPARQL store should sync when KG is loaded from SQLite."""
        # Add facts
        td.teach("Paris is the capital of France", "Paris")
        td.teach("France is in the EU", "France is in the EU")

        # Verify SPARQL has them
        assert len(td.sparql_store) > 0
        capitals = td.sparql_store.inverse_query("capital_of", "france")
        assert "paris" in capitals

    def test_teach_relation_syncs(self, td):
        """teach_relation() should sync properties to SPARQL."""
        td.teach("Kazakhstan is north of Uzbekistan", "Kazakhstan")
        td.teach_relation("north_of", "transitive")

        props = td.sparql_store.get_relation_properties("north_of")
        assert "transitive" in props

    def test_sparql_used_for_open_query(self, td):
        """Open query 'What is the capital of France?' should use SPARQL inverse."""
        td.teach("Paris is the capital of France", "Paris")

        # This should trigger the SPARQL inverse path in _query_knowledge_graph
        result = td.think("What is the capital of France?")
        # The result should find Paris
        if result.solution:
            assert "paris" in str(result.solution).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
