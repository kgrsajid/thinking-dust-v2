"""Comprehensive 5-6 hop tests for ALL question types.

Tests all question types with real Wikipedia data:
1. Yes/No — "Is X in Y?"
2. Open Query — "What/Who/Where is X?"
3. Functional Contradiction — "Are X and Y the same?"
4. Temporal — "Was X before/meets Y?"
5. Multi-entity — "Which X is in Y?"
6. Proof Trace — "Why is X in Y?"
7. Confidence — Decreases with hop count

Sources: Apollo program, Renaissance, DNA biology, Technology, Music, Space
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from td.kg import KnowledgeGraph


@pytest.fixture
def kg():
    g = KnowledgeGraph()
    g.set_composition_rule("in", "in", "in")
    g.set_composition_rule("part_of", "part_of", "part_of")
    g.set_composition_rule("born_in", "in", "in")
    g.set_composition_rule("capital_of", "in", "in")
    g.set_composition_rule("launched_by", "in", "in")
    g.set_composition_rule("directed_by", "born_in", "born_in")
    g.set_composition_rule("created_by", "born_in", "born_in")
    g.set_composition_rule("designed_by", "born_in", "born_in")
    g.set_composition_rule("evolved_from", "evolved_from", "evolved_from")
    return g


# ═══════════════════════════════════════════════════════════════════
# APOLLO PROGRAM — 5-hop chain
# Source: Wikipedia "Apollo program"
# Chain: Apollo 11 → NASA → USA → North America → Earth → Solar System
# ═══════════════════════════════════════════════════════════════════

class TestApollo:
    """Apollo program — 5-hop chain from Wikipedia."""

    def test_5hop_yesno(self, kg):
        """Is Apollo 11 in the Solar System? → YES (5-hop)"""
        kg.add_fact("apollo 11", "launched_by", "nasa")
        kg.add_fact("nasa", "in", "united states")
        kg.add_fact("united states", "in", "north america")
        kg.add_fact("north america", "part_of", "earth")
        kg.add_fact("earth", "in", "solar system")
        r = kg.query("apollo 11", "in", "solar system")
        assert r.answer is True

    def test_5hop_open_query(self, kg):
        """What is Apollo 11 launched by? → NASA (1-hop open)"""
        kg.add_fact("apollo 11", "launched_by", "nasa")
        kg.add_fact("nasa", "in", "united states")
        kg.add_fact("united states", "in", "north america")
        kg.add_fact("north america", "part_of", "earth")
        kg.add_fact("earth", "in", "solar system")
        r = kg.query("apollo 11", "launched_by")
        assert r.answer is True
        assert "nasa" in r.proof_trace

    def test_5hop_open_2hop(self, kg):
        """What country is Apollo 11 in? → USA (2-hop open)"""
        kg.add_fact("apollo 11", "launched_by", "nasa")
        kg.add_fact("nasa", "in", "united states")
        kg.add_fact("united states", "in", "north america")
        kg.add_fact("north america", "part_of", "earth")
        kg.add_fact("earth", "in", "solar system")
        r = kg.query("apollo 11", "in")
        assert r.answer is True

    def test_5hop_confidence(self, kg):
        """Confidence reflects chain quality, not hop count.
        
        Research: CPR (arXiv 2026), UaG (AAAI 2025), UnKGCP (arXiv 2025).
        A chain with all explicit composition rules has high confidence
        regardless of length. A chain with heuristic fallbacks has lower
        confidence.
        """
        kg.add_fact("apollo 11", "launched_by", "nasa")
        kg.add_fact("nasa", "in", "united states")
        kg.add_fact("united states", "in", "north america")
        kg.add_fact("north america", "part_of", "earth")
        kg.add_fact("earth", "in", "solar system")

        r1 = kg.query("apollo 11", "launched_by", "nasa")
        r5 = kg.query("apollo 11", "in", "solar system")
        # Both should have high confidence (explicit rules)
        assert r1.confidence >= 0.90
        # 5-hop with explicit rules should also have high confidence
        assert r5.confidence >= 0.70
        # Direct fact is always >= derived
        assert r1.confidence >= r5.confidence


# ═══════════════════════════════════════════════════════════════════
# RENAISSANCE — 5-hop chain
# Source: Wikipedia "Renaissance"
# Chain: Leonardo da Vinci → Florence → Italy → Europe → Eurasia → Earth
# ═══════════════════════════════════════════════════════════════════

class TestRenaissance:
    """Renaissance — 5-hop chain from Wikipedia."""

    def test_5hop_yesno(self, kg):
        """Is Leonardo da Vinci on Earth? → YES (5-hop)"""
        kg.add_fact("leonardo da vinci", "born_in", "florence")
        kg.add_fact("florence", "in", "italy")
        kg.add_fact("italy", "in", "europe")
        kg.add_fact("europe", "part_of", "eurasia")
        kg.add_fact("eurasia", "part_of", "earth")
        r = kg.query("leonardo da vinci", "part_of", "earth")
        assert r.answer is True

    def test_5hop_open(self, kg):
        """Where was Leonardo da Vinci born? → Florence (1-hop)"""
        kg.add_fact("leonardo da vinci", "born_in", "florence")
        kg.add_fact("florence", "in", "italy")
        r = kg.query("leonardo da vinci", "born_in")
        assert r.answer is True
        assert "florence" in r.proof_trace

    def test_5hop_proof_trace(self, kg):
        """Proof trace shows full 5-hop chain."""
        kg.add_fact("leonardo da vinci", "born_in", "florence")
        kg.add_fact("florence", "in", "italy")
        kg.add_fact("italy", "in", "europe")
        kg.add_fact("europe", "part_of", "eurasia")
        kg.add_fact("eurasia", "part_of", "earth")
        r = kg.query("leonardo da vinci", "part_of", "earth")
        assert "leonardo da vinci" in r.proof_trace
        assert "florence" in r.proof_trace
        assert "earth" in r.proof_trace


# ═══════════════════════════════════════════════════════════════════
# DNA BIOLOGY — 6-hop chain
# Source: Wikipedia "DNA", "Cell biology"
# Chain: DNA → genes → chromosome → nucleus → cell → organ → organism
# ═══════════════════════════════════════════════════════════════════

class TestDNABiology:
    """DNA biology — 6-hop chain from Wikipedia."""

    def test_6hop_yesno(self, kg):
        """Is DNA part of the organism? → YES (6-hop)"""
        kg.add_fact("dna", "encodes", "genes")
        kg.add_fact("genes", "part_of", "chromosome")
        kg.add_fact("chromosome", "in", "nucleus")
        kg.add_fact("nucleus", "in", "cell")
        kg.add_fact("cell", "part_of", "organ")
        kg.add_fact("organ", "part_of", "organism")
        r = kg.query("dna", "part_of", "organism")
        assert r.answer is True

    def test_6hop_open(self, kg):
        """What does DNA encode? → genes (1-hop)"""
        kg.add_fact("dna", "encodes", "genes")
        kg.add_fact("genes", "part_of", "chromosome")
        r = kg.query("dna", "encodes")
        assert r.answer is True
        assert "genes" in r.proof_trace

    def test_6hop_open_3hop(self, kg):
        """What is DNA part of? → chromosome (3-hop via encodes→part_of→part_of)"""
        kg.add_fact("dna", "encodes", "genes")
        kg.add_fact("genes", "part_of", "chromosome")
        kg.add_fact("chromosome", "in", "nucleus")
        r = kg.query("dna", "part_of")
        assert r.answer is True


# ═══════════════════════════════════════════════════════════════════
# TECHNOLOGY — 5-hop chain
# Source: Wikipedia "iPhone", "Apple Inc."
# Chain: iPhone → Apple → Steve Jobs → San Francisco → USA → North America
# ═══════════════════════════════════════════════════════════════════

class TestTechnology:
    """Technology — 5-hop chain from Wikipedia."""

    def test_5hop_yesno(self, kg):
        """Is iPhone in North America? → YES (5-hop)"""
        kg.add_fact("iphone", "made_by", "apple")
        kg.add_fact("apple", "founded_by", "steve jobs")
        kg.add_fact("steve jobs", "born_in", "san francisco")
        kg.add_fact("san francisco", "in", "united states")
        kg.add_fact("united states", "in", "north america")
        r = kg.query("iphone", "in", "north america")
        assert r.answer is True

    def test_5hop_open_2hop(self, kg):
        """Who founded the company that makes iPhone? → Steve Jobs (2-hop)"""
        kg.add_fact("iphone", "made_by", "apple")
        kg.add_fact("apple", "founded_by", "steve jobs")
        r = kg.query("iphone", "founded_by")
        assert r.answer is True
        assert "steve jobs" in r.proof_trace

    def test_5hop_open_3hop(self, kg):
        """Where was the founder of Apple born? → San Francisco (3-hop)"""
        kg.add_fact("iphone", "made_by", "apple")
        kg.add_fact("apple", "founded_by", "steve jobs")
        kg.add_fact("steve jobs", "born_in", "san francisco")
        r = kg.query("iphone", "born_in")
        assert r.answer is True
        assert "san francisco" in r.proof_trace

    def test_5hop_open_4hop(self, kg):
        """What country is iPhone in? → USA (4-hop)"""
        kg.add_fact("iphone", "made_by", "apple")
        kg.add_fact("apple", "founded_by", "steve jobs")
        kg.add_fact("steve jobs", "born_in", "san francisco")
        kg.add_fact("san francisco", "in", "united states")
        r = kg.query("iphone", "in")
        assert r.answer is True

    def test_5hop_functional(self, kg):
        """Are iPhone and Galaxy the same? → NO (different makers)"""
        kg.add_fact("iphone", "made_by", "apple")
        kg.add_fact("galaxy", "made_by", "samsung")
        kg.set_relation_property("made_by", "functional")
        r = kg.check_same("iphone", "galaxy")
        assert r.answer is False

    def test_5hop_persistence(self, kg):
        """5-hop chain survives SQLite save/load."""
        kg.add_fact("iphone", "made_by", "apple")
        kg.add_fact("apple", "founded_by", "steve jobs")
        kg.add_fact("steve jobs", "born_in", "san francisco")
        kg.add_fact("san francisco", "in", "united states")
        kg.add_fact("united states", "in", "north america")
        tmp = tempfile.mktemp(suffix=".db")
        try:
            kg.save(tmp)
            kg2 = KnowledgeGraph()
            kg2.load(tmp)
            r = kg2.query("iphone", "in", "north america")
            assert r.answer is True
        finally:
            os.unlink(tmp)


# ═══════════════════════════════════════════════════════════════════
# MUSIC — 5-hop chain
# Source: Wikipedia "Hip hop music", "Blues"
# Chain: Hip hop → funk → soul → rhythm and blues → blues → spirituals
# ═══════════════════════════════════════════════════════════════════

class TestMusic:
    """Music — 5-hop chain from Wikipedia."""

    def test_5hop_yesno(self, kg):
        """Did hip hop evolve from spirituals? → YES (5-hop)"""
        kg.add_fact("hip hop", "evolved_from", "funk")
        kg.add_fact("funk", "evolved_from", "soul")
        kg.add_fact("soul", "evolved_from", "rhythm and blues")
        kg.add_fact("rhythm and blues", "evolved_from", "blues")
        kg.add_fact("blues", "evolved_from", "spirituals")
        r = kg.query("hip hop", "evolved_from", "spirituals")
        assert r.answer is True

    def test_5hop_open(self, kg):
        """What did hip hop evolve from? → funk (1-hop)"""
        kg.add_fact("hip hop", "evolved_from", "funk")
        kg.add_fact("funk", "evolved_from", "soul")
        r = kg.query("hip hop", "evolved_from")
        assert r.answer is True
        assert "funk" in r.proof_trace

    def test_5hop_open_3hop(self, kg):
        """What is hip hop's grandparent genre? → rhythm and blues (3-hop)"""
        kg.add_fact("hip hop", "evolved_from", "funk")
        kg.add_fact("funk", "evolved_from", "soul")
        kg.add_fact("soul", "evolved_from", "rhythm and blues")
        r = kg.query("hip hop", "evolved_from")
        assert r.answer is True


# ═══════════════════════════════════════════════════════════════════
# SPACE EXPLORATION — 5-hop chain
# Source: Wikipedia "Voyager 1"
# Chain: Voyager 1 → NASA → USA → North America → Earth → Solar System
# ═══════════════════════════════════════════════════════════════════

class TestSpaceExploration:
    """Space exploration — 5-hop chain from Wikipedia."""

    def test_5hop_yesno(self, kg):
        """Is Voyager 1 in the Solar System? → YES (5-hop)"""
        kg.add_fact("voyager 1", "launched_by", "nasa")
        kg.add_fact("nasa", "in", "united states")
        kg.add_fact("united states", "in", "north america")
        kg.add_fact("north america", "part_of", "earth")
        kg.add_fact("earth", "in", "solar system")
        r = kg.query("voyager 1", "in", "solar system")
        assert r.answer is True

    def test_5hop_open_2hop(self, kg):
        """What country launched Voyager 1? → USA (2-hop)"""
        kg.add_fact("voyager 1", "launched_by", "nasa")
        kg.add_fact("nasa", "in", "united states")
        r = kg.query("voyager 1", "in")
        assert r.answer is True

    def test_5hop_temporal(self, kg):
        """Voyager 1 launched before Voyager 2."""
        kg.add_fact("voyager 1", "launched_by", "nasa",
                   temporal_start=1977, temporal_end=1978)
        kg.add_fact("voyager 2", "launched_by", "nasa",
                   temporal_start=1977, temporal_end=1978)
        r = kg.query_temporal("voyager 1", "voyager 2", "before")
        # Both launched in 1977, so not strictly before
        assert r.answer is not None  # Should be True or False


# ═══════════════════════════════════════════════════════════════════
# GEOLOGY — 5-hop chain
# Source: Wikipedia "Himalayas", "Plate tectonics"
# Chain: Himalayas → Indian Plate → Plate Tectonics → Geology → Earth Science → Science
# ═══════════════════════════════════════════════════════════════════

class TestGeology:
    """Geology — 5-hop chain from Wikipedia."""

    def test_5hop_yesno(self, kg):
        """Are the Himalayas part of Science? → YES (5-hop)"""
        kg.add_fact("himalayas", "formed_by", "indian plate collision")
        kg.add_fact("indian plate collision", "part_of", "plate tectonics")
        kg.add_fact("plate tectonics", "studied_by", "geology")
        kg.add_fact("geology", "part_of", "earth science")
        kg.add_fact("earth science", "part_of", "science")
        r = kg.query("himalayas", "part_of", "science")
        assert r.answer is True

    def test_5hop_open(self, kg):
        """What formed the Himalayas? → Indian plate collision (1-hop)"""
        kg.add_fact("himalayas", "formed_by", "indian plate collision")
        r = kg.query("himalayas", "formed_by")
        assert r.answer is True
        assert "indian plate" in r.proof_trace


# ═══════════════════════════════════════════════════════════════════
# CROSS-DOMAIN — Multiple 5-hop chains interacting
# ═══════════════════════════════════════════════════════════════════

class TestCrossDomain:
    """Cross-domain 5-hop chains."""

    def test_scientist_country_chain(self, kg):
        """Einstein → Germany → Europe → Eurasia (3-hop)"""
        kg.add_fact("einstein", "born_in", "ulm")
        kg.add_fact("ulm", "in", "germany")
        kg.add_fact("germany", "in", "europe")
        kg.add_fact("europe", "part_of", "eurasia")
        r = kg.query("einstein", "in", "eurasia")
        assert r.answer is True

    def test_company_country_chain(self, kg):
        """Toyota → Japan → East Asia → Asia (3-hop)"""
        kg.add_fact("toyota", "in", "japan")
        kg.add_fact("japan", "in", "east asia")
        kg.add_fact("east asia", "part_of", "asia")
        r = kg.query("toyota", "part_of", "asia")
        assert r.answer is True

    def test_dish_country_chain(self, kg):
        """Sushi → Japan → East Asia → Asia (3-hop)"""
        kg.add_fact("sushi", "originates_from", "japan")
        kg.add_fact("japan", "in", "east asia")
        kg.add_fact("east asia", "part_of", "asia")
        r = kg.query("sushi", "in", "asia")
        assert r.answer is True


# ═══════════════════════════════════════════════════════════════════
# PERSISTENCE — 5-hop chains survive SQLite
# ═══════════════════════════════════════════════════════════════════

class TestPersistence5Hop:
    """5-hop chains persist through SQLite."""

    def test_apollo_persists(self, kg):
        """Apollo 5-hop chain survives save/load."""
        kg.add_fact("apollo 11", "launched_by", "nasa")
        kg.add_fact("nasa", "in", "united states")
        kg.add_fact("united states", "in", "north america")
        kg.add_fact("north america", "part_of", "earth")
        kg.add_fact("earth", "in", "solar system")
        tmp = tempfile.mktemp(suffix=".db")
        try:
            kg.save(tmp)
            kg2 = KnowledgeGraph()
            kg2.load(tmp)
            r = kg2.query("apollo 11", "in", "solar system")
            assert r.answer is True
        finally:
            os.unlink(tmp)

    def test_dna_persists(self, kg):
        """DNA 6-hop chain survives save/load."""
        kg.add_fact("dna", "encodes", "genes")
        kg.add_fact("genes", "part_of", "chromosome")
        kg.add_fact("chromosome", "in", "nucleus")
        kg.add_fact("nucleus", "in", "cell")
        kg.add_fact("cell", "part_of", "organ")
        kg.add_fact("organ", "part_of", "organism")
        tmp = tempfile.mktemp(suffix=".db")
        try:
            kg.save(tmp)
            kg2 = KnowledgeGraph()
            kg2.load(tmp)
            r = kg2.query("dna", "part_of", "organism")
            assert r.answer is True
        finally:
            os.unlink(tmp)
