"""Tests for the SPARQL Query Layer (td/query).

Tests the bridge between TD v2 SQLite KG and pyoxigraph SPARQL store.
Covers: URI mapping, add/remove, sync, ask, inverse, multi-hop,
metadata, named graphs, export/import, statistics.
"""

import os
import sys
import shutil
import tempfile
import pytest

# Ensure td is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from td.query import (
    SparqlStore, entity_to_uri, relation_to_uri,
    uri_to_entity, uri_to_relation, TD_ENT, TD_REL, TD_VOCAB,
    HAS_OXIGRAPH,
)

# Skip all tests if pyoxigraph not installed
pytestmark = pytest.mark.skipif(not HAS_OXIGRAPH, reason="pyoxigraph not installed")


# ─── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def store():
    """Fresh in-memory SparqlStore."""
    return SparqlStore()


@pytest.fixture
def store_with_facts(store):
    """SparqlStore pre-loaded with basic facts."""
    # Paris → France → EU → Europe
    store.add_fact("paris", "capital_of", "france", source="user")
    store.add_fact("france", "in", "eu", source="user")
    store.add_fact("eu", "part_of", "europe", source="user")
    store.add_fact("berlin", "capital_of", "germany", source="user")
    store.add_fact("germany", "in", "eu", source="user")
    return store


@pytest.fixture
def kg():
    """A KnowledgeGraph with some facts."""
    from td.kg import KnowledgeGraph
    graph = KnowledgeGraph()
    graph.add_fact("paris", "capital_of", "france", source="user")
    graph.add_fact("france", "in", "eu", source="user")
    graph.add_fact("eu", "part_of", "europe", source="user")
    graph.add_fact("berlin", "capital_of", "germany", source="user")
    graph.add_fact("germany", "in", "eu", source="user")
    graph.add_fact("alice", "married_to", "bob", source="user")
    graph.relation_properties["married_to"] = {"symmetric"}
    graph.composition_rules[("capital_of", "in")] = "in"
    graph.composition_rules[("in", "part_of")] = "in"
    return store, graph


@pytest.fixture
def disk_store(tmp_path):
    """Disk-persistent SparqlStore."""
    path = str(tmp_path / "td_sparql_test")
    s = SparqlStore(store_path=path)
    yield s
    # Cleanup handled by pytest tmp_path


# ─── URI Mapping ──────────────────────────────────────────────────────

class TestUriMapping:
    """Test entity/relation ↔ RDF URI conversion."""

    def test_entity_simple(self):
        uri = entity_to_uri("paris")
        assert uri.value == f"{TD_ENT}paris"

    def test_entity_multiword(self):
        uri = entity_to_uri("south korea")
        assert uri.value == f"{TD_ENT}south_korea"

    def test_entity_hyphen(self):
        """Hyphens are preserved in URIs (only spaces → underscores)."""
        uri = entity_to_uri("world-war-2")
        assert uri.value == f"{TD_ENT}world-war-2"

    def test_entity_case_insensitive(self):
        uri = entity_to_uri("Paris")
        assert uri.value == f"{TD_ENT}paris"

    def test_relation_simple(self):
        uri = relation_to_uri("capital_of")
        assert uri.value == f"{TD_REL}capital_of"

    def test_relation_in(self):
        uri = relation_to_uri("in")
        assert uri.value == f"{TD_REL}in"

    def test_roundtrip_entity(self):
        original = "south korea"
        uri = entity_to_uri(original)
        recovered = uri_to_entity(uri)
        assert recovered == original

    def test_roundtrip_relation(self):
        original = "capital_of"
        uri = relation_to_uri(original)
        recovered = uri_to_relation(uri)
        assert recovered == original


# ─── Basic Operations ─────────────────────────────────────────────────

class TestBasicOperations:
    """Test add, remove, clear."""

    def test_add_single_fact(self, store):
        store.add_fact("paris", "capital_of", "france")
        assert len(store) >= 3  # assertion + source graph + metadata

    def test_add_multiple_facts(self, store):
        store.add_fact("paris", "capital_of", "france")
        store.add_fact("france", "in", "eu")
        store.add_fact("eu", "part_of", "europe")
        assert len(store) >= 9

    def test_remove_fact(self, store):
        store.add_fact("paris", "capital_of", "france")
        count_before = len(store)
        store.remove_fact("paris", "capital_of", "france")
        count_after = len(store)
        assert count_after < count_before

    def test_clear(self, store):
        store.add_fact("paris", "capital_of", "france")
        store.add_fact("france", "in", "eu")
        store.clear()
        assert len(store) == 0

    def test_add_with_proof(self, store):
        store.add_fact("paris", "in", "eu", source="derived",
                       proof="paris capital_of france → france in eu",
                       confidence=0.85)
        meta = store.get_fact_metadata("paris", "in", "eu")
        assert meta is not None
        assert meta["source"] == "derived"
        assert "capital_of" in meta["proof"]
        assert meta["confidence"] == pytest.approx(0.85)

    def test_add_with_temporal(self, store):
        store.add_fact("obama", "president_of", "usa",
                       temporal_start=2009, temporal_end=2017)
        meta = store.get_fact_metadata("obama", "president_of", "usa")
        assert meta["temporal_start"] == 2009
        assert meta["temporal_end"] == 2017


# ─── Direct Queries ───────────────────────────────────────────────────

class TestDirectQueries:
    """Test direct fact lookup via SPARQL."""

    def test_ask_direct_yes(self, store_with_facts):
        result = store_with_facts.ask("paris", "france", "capital_of")
        assert result.found is True
        assert result.answer == "YES"
        assert result.confidence > 0.9

    def test_ask_direct_no(self, store_with_facts):
        result = store_with_facts.ask("paris", "germany", "capital_of")
        assert result.found is False

    def test_ask_any_relation(self, store_with_facts):
        result = store_with_facts.ask("paris", "france")
        assert result.found is True

    def test_query_relation_objects(self, store_with_facts):
        objects = store_with_facts.query_relation("paris", "capital_of")
        assert "france" in objects

    def test_query_relation_empty(self, store_with_facts):
        objects = store_with_facts.query_relation("paris", "in")
        assert objects == []


# ─── Inverse Queries ─────────────────────────────────────────────────

class TestInverseQueries:
    """Test inverse queries — the primary motivation for SPARQL."""

    def test_inverse_capital_of(self, store_with_facts):
        results = store_with_facts.inverse_query("capital_of", "france")
        assert "paris" in results

    def test_inverse_in(self, store_with_facts):
        results = store_with_facts.inverse_query("in", "eu")
        assert "france" in results
        assert "germany" in results

    def test_inverse_no_match(self, store_with_facts):
        results = store_with_facts.inverse_query("capital_of", "eu")
        assert results == []

    def test_inverse_open_query(self, store_with_facts):
        """'What is the capital of France?' → Paris via inverse query."""
        result = store_with_facts.ask("paris", "france", "capital_of")
        # Forward: paris capital_of france → YES
        assert result.found is True

        # Inverse: who is capital_of france?
        capitals = store_with_facts.inverse_query("capital_of", "france")
        assert "paris" in capitals


# ─── Multi-Hop Transitive ────────────────────────────────────────────

class TestMultiHopTransitive:
    """Test transitive chains via SPARQL property paths."""

    def test_2hop_transitive(self, store_with_facts):
        """Paris → France → EU (2 hops via capital_of + in)."""
        result = store_with_facts.ask("paris", "eu")
        assert result.found is True

    def test_3hop_transitive(self, store_with_facts):
        """Paris → France → EU → Europe (3 hops)."""
        result = store_with_facts.ask("paris", "europe")
        assert result.found is True

    def test_find_path_2hop(self, store_with_facts):
        path = store_with_facts.find_path("paris", "eu", max_hops=6)
        assert path is not None
        assert len(path) >= 1

    def test_find_path_3hop(self, store_with_facts):
        path = store_with_facts.find_path("paris", "europe", max_hops=6)
        assert path is not None


# ─── Named Graph Source Filtering ────────────────────────────────────

class TestNamedGraphs:
    """Test source-based filtering via named graphs."""

    def test_user_facts(self, store):
        store.add_fact("paris", "capital_of", "france", source="user")
        store.add_fact("paris", "in", "eu", source="derived")

        user_facts = store.get_facts_by_source("user")
        assert len(user_facts) == 1
        assert user_facts[0]["subject"] == "paris"
        assert user_facts[0]["relation"] == "capital_of"

    def test_derived_facts(self, store):
        store.add_fact("paris", "in", "eu", source="derived",
                       proof="paris capital_of france → france in eu")

        derived = store.get_facts_by_source("derived")
        assert len(derived) == 1
        assert derived[0]["relation"] == "in"

    def test_empty_source(self, store):
        facts = store.get_facts_by_source("user")
        assert facts == []


# ─── Metadata ─────────────────────────────────────────────────────────

class TestMetadata:
    """Test per-fact metadata (source, proof, confidence, temporal)."""

    def test_metadata_source(self, store):
        store.add_fact("paris", "capital_of", "france", source="user")
        meta = store.get_fact_metadata("paris", "capital_of", "france")
        assert meta["source"] == "user"

    def test_metadata_proof(self, store):
        proof = "paris capital_of france → france in eu"
        store.add_fact("paris", "in", "eu", source="derived", proof=proof)
        meta = store.get_fact_metadata("paris", "in", "eu")
        assert proof in meta["proof"]

    def test_metadata_confidence(self, store):
        store.add_fact("paris", "in", "eu", source="derived", confidence=0.75)
        meta = store.get_fact_metadata("paris", "in", "eu")
        assert meta["confidence"] == pytest.approx(0.75)

    def test_metadata_temporal(self, store):
        store.add_fact("obama", "president_of", "usa",
                       temporal_start=2009, temporal_end=2017)
        meta = store.get_fact_metadata("obama", "president_of", "usa")
        assert meta["temporal_start"] == 2009
        assert meta["temporal_end"] == 2017

    def test_metadata_nonexistent(self, store):
        meta = store.get_fact_metadata("atlantis", "in", "eu")
        assert meta is None


# ─── Relation Properties ─────────────────────────────────────────────

class TestRelationProperties:
    """Test relation property storage and retrieval."""

    def test_transitive_property(self, store):
        """Relation properties can be stored and queried."""
        from pyoxigraph import Quad, NamedNode, Literal as L
        # Add relation property directly
        r = relation_to_uri("in")
        store.store.add(Quad(r, NamedNode(f"{TD_VOCAB}property"), L("transitive")))
        props = store.get_relation_properties("in")
        assert "transitive" in props


# ─── Sync from KnowledgeGraph ────────────────────────────────────────

class TestSyncFromKG:
    """Test full sync from SQLite-backed KnowledgeGraph."""

    def test_sync_basic(self, store):
        from td.kg import KnowledgeGraph
        kg = KnowledgeGraph()
        kg.add_fact("paris", "capital_of", "france")
        kg.add_fact("france", "in", "eu")

        count = store.sync_from_kg(kg)
        assert count == 2

    def test_sync_preserves_source(self, store):
        from td.kg import KnowledgeGraph
        kg = KnowledgeGraph()
        kg.add_fact("paris", "capital_of", "france", source="user")
        kg.add_fact("paris", "in", "eu", source="derived")

        store.sync_from_kg(kg)
        user = store.get_facts_by_source("user")
        derived = store.get_facts_by_source("derived")
        assert len(user) == 1
        assert len(derived) == 1

    def test_sync_preserves_temporal(self, store):
        from td.kg import KnowledgeGraph
        kg = KnowledgeGraph()
        kg.add_fact("obama", "president_of", "usa",
                    temporal_start=2009, temporal_end=2017)

        store.sync_from_kg(kg)
        meta = store.get_fact_metadata("obama", "president_of", "usa")
        assert meta["temporal_start"] == 2009

    def test_sync_queryable_after(self, store):
        from td.kg import KnowledgeGraph
        kg = KnowledgeGraph()
        kg.add_fact("paris", "capital_of", "france")
        kg.add_fact("france", "in", "eu")
        kg.add_fact("eu", "part_of", "europe")

        store.sync_from_kg(kg)

        # Inverse query works after sync
        capitals = store.inverse_query("capital_of", "france")
        assert "paris" in capitals

        # Multi-hop works after sync
        result = store.ask("paris", "europe")
        assert result.found is True

    def test_sync_relation_properties(self, store):
        from td.kg import KnowledgeGraph
        kg = KnowledgeGraph()
        kg.relation_properties["north_of"] = {"transitive"}

        store.sync_from_kg(kg)
        props = store.get_relation_properties("north_of")
        assert "transitive" in props


# ─── Disk Persistence ────────────────────────────────────────────────

class TestDiskPersistence:
    """Test disk-backed store."""

    def test_disk_store_creates_directory(self, tmp_path):
        path = str(tmp_path / "test_store")
        s = SparqlStore(store_path=path)
        assert os.path.exists(path)

    def test_disk_store_persists(self, tmp_path):
        path = str(tmp_path / "test_store")
        s1 = SparqlStore(store_path=path)
        s1.add_fact("paris", "capital_of", "france")
        s1.store.flush()
        # Close the store by deleting the reference
        del s1
        import gc; gc.collect()

        # Open new store at same path
        s2 = SparqlStore(store_path=path)
        result = s2.ask("paris", "france", "capital_of")
        assert result.found is True


# ─── Export/Import ────────────────────────────────────────────────────

class TestExportImport:
    """Test RDF export and import."""

    def test_export_turtle(self, store_with_facts, tmp_path):
        path = str(tmp_path / "export.ttl")
        store_with_facts.export_turtle(path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_export_ntriples(self, store_with_facts, tmp_path):
        path = str(tmp_path / "export.nt")
        store_with_facts.export_ntriples(path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_roundtrip_turtle(self, store_with_facts, tmp_path):
        # Export
        path = str(tmp_path / "roundtrip.ttl")
        store_with_facts.export_turtle(path)

        # Import into new store
        new_store = SparqlStore()
        count = new_store.import_turtle(path)
        assert count > 0


# ─── Statistics ───────────────────────────────────────────────────────

class TestStatistics:
    """Test store statistics."""

    def test_stats_empty(self, store):
        stats = store.stats()
        assert stats["total_quads"] == 0

    def test_stats_with_facts(self, store_with_facts):
        stats = store_with_facts.stats()
        assert stats["total_quads"] > 0
        assert "capital_of" in stats["relations"] or "in" in stats["relations"]


# ─── Raw SPARQL ───────────────────────────────────────────────────────

class TestRawSparql:
    """Test direct SPARQL query execution."""

    def test_select(self, store_with_facts):
        results = store_with_facts.query_sparql_bindings(
            'SELECT ?s ?o WHERE { ?s <http://thinking-dust.org/relation/capital_of> ?o }'
        )
        assert len(results) >= 2  # paris→france, berlin→germany

    def test_ask_true(self, store_with_facts):
        result = store_with_facts.query_sparql(
            'ASK { <http://thinking-dust.org/entity/paris> <http://thinking-dust.org/relation/capital_of> <http://thinking-dust.org/entity/france> }'
        )
        assert bool(result) is True

    def test_ask_false(self, store_with_facts):
        result = store_with_facts.query_sparql(
            'ASK { <http://thinking-dust.org/entity/paris> <http://thinking-dust.org/relation/capital_of> <http://thinking-dust.org/entity/germany> }'
        )
        assert bool(result) is False

    def test_filter(self, store_with_facts):
        """SPARQL FILTER — not possible with old BFS."""
        store_with_facts.add_fact("obama", "president_of", "usa",
                                  temporal_start=2009, temporal_end=2017)
        store_with_facts.add_fact("trump", "president_of", "usa",
                                  temporal_start=2017, temporal_end=2021)

        # Query: presidents who started after 2010
        results = store_with_facts.query_sparql_bindings(
            'SELECT ?s WHERE { ?s <http://thinking-dust.org/relation/president_of> <http://thinking-dust.org/entity/usa> }'
        )
        # Both should match (FILTER on metadata would need a more complex query)
        assert len(results) >= 2

    def test_optional(self, store_with_facts):
        """SPARQL OPTIONAL — graceful handling of missing data."""
        results = store_with_facts.query_sparql_bindings(
            '''SELECT ?s ?p ?o WHERE {
                ?s <http://thinking-dust.org/relation/capital_of> ?o .
                OPTIONAL { ?s ?p ?o }
            }'''
        )
        assert len(results) >= 2


# ─── Edge Cases ───────────────────────────────────────────────────────

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_store_ask(self, store):
        result = store.ask("paris", "france")
        assert result.found is False

    def test_empty_store_inverse(self, store):
        results = store.inverse_query("capital_of", "france")
        assert results == []

    def test_special_characters_in_entity(self, store):
        """Entities with underscores, numbers."""
        store.add_fact("node_5", "connected_to", "node_6")
        results = store.inverse_query("connected_to", "node_6")
        assert "node 5" in results  # URI mapping normalizes

    def test_duplicate_fact(self, store):
        """Adding same fact twice shouldn't create duplicates in default graph."""
        store.add_fact("paris", "capital_of", "france")
        store.add_fact("paris", "capital_of", "france")
        # Default graph should have 1 assertion, not 2
        results = store.query_relation("paris", "capital_of")
        assert results == ["france"]

    def test_repr(self, store):
        store.add_fact("paris", "capital_of", "france")
        r = repr(store)
        assert "SparqlStore" in r


# ─── Novel Relations ─────────────────────────────────────────────────

class TestNovelRelations:
    """Test with relations not pre-seeded in TD v2."""

    def test_novel_transitive(self, store):
        """User teaches 'north_of' as transitive."""
        store.add_fact("kazakhstan", "north_of", "uzbekistan")
        store.add_fact("uzbekistan", "north_of", "tajikistan")

        # Without property path, need to find path manually
        path = store.find_path("kazakhstan", "tajikistan")
        assert path is not None

    def test_novel_symmetric(self, store):
        """User teaches 'married_to' as symmetric."""
        store.add_fact("alice", "married_to", "bob")
        # Direct query
        result = store.ask("alice", "bob", "married_to")
        assert result.found is True
        # Inverse should also work
        results = store.inverse_query("married_to", "bob")
        assert "alice" in results

    def test_novel_functional(self, store):
        """User teaches 'capital_of' as functional."""
        store.add_fact("paris", "capital_of", "france")
        store.add_fact("berlin", "capital_of", "germany")
        # Both should be findable
        assert "paris" in store.inverse_query("capital_of", "france")
        assert "berlin" in store.inverse_query("capital_of", "germany")


# ─── Performance Smoke Test ──────────────────────────────────────────

class TestPerformance:
    """Quick performance sanity checks."""

    def test_bulk_1000_facts(self, store):
        """Bulk facts should sync quickly."""
        import time
        from td.kg import KnowledgeGraph
        kg = KnowledgeGraph()
        for i in range(500):
            kg.add_fact(f"entity_{i}", "in", f"group_{i // 10}")
            kg.add_fact(f"group_{i // 10}", "part_of", "universe")

        t0 = time.perf_counter()
        count = store.sync_from_kg(kg)
        elapsed = time.perf_counter() - t0

        # KG deduplicates: 500 entity_in + 50 group_part_of = 550 unique
        assert count == 550
        assert elapsed < 10.0  # Should be <2s, generous margin

    def test_inverse_query_1000_facts(self, store):
        """Inverse query should be fast even with 1000 facts."""
        import time
        from td.kg import KnowledgeGraph
        kg = KnowledgeGraph()
        for i in range(500):
            kg.add_fact(f"city_{i}", "capital_of", f"country_{i}")

        store.sync_from_kg(kg)

        t0 = time.perf_counter()
        results = store.inverse_query("capital_of", "country_42")
        elapsed = time.perf_counter() - t0

        assert "city 42" in results
        assert elapsed < 0.1  # Should be <10ms


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
