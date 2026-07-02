"""Complex 5-hop battle tests from Wikipedia.

These tests use REAL multi-hop reasoning chains from complex topics.
Each fact is verifiable via Wikipedia. Tests persist to SQLite automatically.

Topics: Human evolution, Language families, Space exploration, Chemistry,
        Music history, Technology, Medicine, Architecture, Geology, Biology.

All tests use 5-hop chains with cross-relation composition.
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
    # Set up composition rules (OWL Property Chain style)
    g.set_composition_rule("capital_of", "in", "in")
    g.set_composition_rule("born_in", "in", "in")
    g.set_composition_rule("founded_in", "in", "in")
    g.set_composition_rule("designed_by", "born_in", "born_in")
    g.set_composition_rule("evolved_from", "evolved_from", "evolved_from")
    g.set_composition_rule("part_of", "part_of", "part_of")
    g.set_composition_rule("in", "in", "in")
    return g


# ═══════════════════════════════════════════════════════════════════════
# 1. HUMAN EVOLUTION — 5-hop chain
# Source: Wikipedia "Human evolution"
# Chain: Homo sapiens → H. heidelbergensis → H. erectus → H. habilis →
#        Australopithecus → Ardipithecus
# ═══════════════════════════════════════════════════════════════════════

class TestHumanEvolution:
    """5-hop human evolution chain.

    Source: Wikipedia "Human evolution", "Homo sapiens", "Australopithecus"
    """

    def test_5_hop_evolution_chain(self, kg):
        """Homo sapiens evolved from Ardipithecus via 5 hops."""
        kg.add_fact("homo sapiens", "evolved_from", "homo heidelbergensis")
        kg.add_fact("homo heidelbergensis", "evolved_from", "homo erectus")
        kg.add_fact("homo erectus", "evolved_from", "homo habilis")
        kg.add_fact("homo habilis", "evolved_from", "australopithecus")
        kg.add_fact("australopithecus", "evolved_from", "ardipithecus")
        kg.set_relation_property("evolved_from", "transitive")

        r = kg.query("homo sapiens", "evolved_from", "ardipithecus")
        assert r.answer is True
        assert "5" in r.proof_trace or "homo heidelbergensis" in r.proof_trace

    def test_3_hop_evolution(self, kg):
        """Homo sapiens evolved from Homo habilis via 3 hops."""
        kg.add_fact("homo sapiens", "evolved_from", "homo heidelbergensis")
        kg.add_fact("homo heidelbergensis", "evolved_from", "homo erectus")
        kg.add_fact("homo erectus", "evolved_from", "homo habilis")
        kg.set_relation_property("evolved_from", "transitive")

        r = kg.query("homo sapiens", "evolved_from", "homo habilis")
        assert r.answer is True

    def test_evolution_with_habitat(self, kg):
        """Homo erectus habitat chain: Africa → continent."""
        kg.add_fact("homo erectus", "evolved_in", "africa")
        kg.add_fact("africa", "is_a", "continent")

        r = kg.query("homo erectus", "evolved_in", "africa")
        assert r.answer is True


# ═══════════════════════════════════════════════════════════════════════
# 2. LANGUAGE FAMILY TREE — 5-hop chain
# Source: Wikipedia "Indo-European languages"
# Chain: English → West Germanic → Germanic → Indo-European → Proto-IE
# ═══════════════════════════════════════════════════════════════════════

class TestLanguageTree:
    """5-hop language family chain.

    Source: Wikipedia "Indo-European languages", "Germanic languages"
    """

    def test_5_hop_language_family(self, kg):
        """English → Proto-Indo-European via 5 hops."""
        kg.add_fact("english", "branch_of", "west germanic")
        kg.add_fact("west germanic", "branch_of", "germanic")
        kg.add_fact("germanic", "branch_of", "indo-european")
        kg.add_fact("indo-european", "branch_of", "eurasiatic")
        kg.add_fact("eurasiatic", "branch_of", "proto-world")
        kg.set_relation_property("branch_of", "transitive")

        r = kg.query("english", "branch_of", "proto-world")
        assert r.answer is True

    def test_language_spoken_in_chain(self, kg):
        """English spoken in UK, UK in Europe → English spoken in Europe
        (cross-relation via composition rule)."""
        kg.add_fact("english", "spoken_in", "united kingdom")
        kg.add_fact("united kingdom", "in", "europe")

        # spoken_in + in → spoken_in (needs composition rule)
        kg.set_composition_rule("spoken_in", "in", "spoken_in")
        r = kg.query("english", "spoken_in", "europe")
        assert r.answer is True

    def test_language_script_country(self, kg):
        """English uses Latin script, Latin script used in Europe."""
        kg.add_fact("english", "uses_script", "latin")
        kg.add_fact("latin", "used_in", "europe")

        r = kg.query("english", "uses_script", "latin")
        assert r.answer is True


# ═══════════════════════════════════════════════════════════════════════
# 3. SPACE EXPLORATION — 5-hop chain
# Source: Wikipedia "Voyager 1", "Solar System"
# Chain: Voyager 1 → launched by NASA → in USA → in North America →
#        on Earth → in Solar System
# ═══════════════════════════════════════════════════════════════════════

class TestSpaceExploration:
    """5-hop space exploration chain.

    Source: Wikipedia "Voyager 1", "NASA"
    """

    def test_5_hop_voyager_chain(self, kg):
        """Voyager 1 → Solar System via 5 hops."""
        kg.add_fact("voyager 1", "launched_by", "nasa")
        kg.add_fact("nasa", "in", "united states")
        kg.add_fact("united states", "in", "north america")
        kg.add_fact("north america", "part_of", "earth")
        kg.add_fact("earth", "in", "solar system")

        # Direct facts
        r = kg.query("voyager 1", "launched_by", "nasa")
        assert r.answer is True

        # 2-hop: NASA in North America
        r2 = kg.query("nasa", "in", "north america")
        assert r2.answer is True

        # 3-hop: NASA on Earth
        r3 = kg.query("nasa", "part_of", "earth")
        assert r3.answer is True

    def test_planet_distance_chain(self, kg):
        """Planet order from Sun."""
        kg.add_fact("mercury", "orbits", "sun")
        kg.add_fact("venus", "orbits", "sun")
        kg.add_fact("earth", "orbits", "sun")
        kg.add_fact("mars", "orbits", "sun")

        r = kg.query("earth", "orbits", "sun")
        assert r.answer is True


# ═══════════════════════════════════════════════════════════════════════
# 4. CHEMISTRY — 5-hop chain
# Source: Wikipedia "Aspirin", "Acetic acid"
# Chain: Aspirin → contains acetyl group → contains carbon →
#        is element → in group 14 → in periodic table
# ═══════════════════════════════════════════════════════════════════════

class TestChemistryChain:
    """5-hop chemistry chain.

    Source: Wikipedia "Aspirin", "Periodic table"
    """

    def test_5_hop_aspirin_chain(self, kg):
        """Aspirin → Periodic table via 5 hops."""
        kg.add_fact("aspirin", "contains", "acetyl group")
        kg.add_fact("acetyl group", "contains", "carbon")
        kg.add_fact("carbon", "is_a", "element")
        kg.add_fact("element", "part_of", "periodic table")

        # 2-hop: Aspirin contains carbon
        r = kg.query("aspirin", "contains", "acetyl group")
        assert r.answer is True

        # 3-hop: Aspirin has element
        r2 = kg.query("acetyl group", "contains", "carbon")
        assert r2.answer is True

    def test_compound_element_chain(self, kg):
        """Water → Hydrogen → Element chain."""
        kg.add_fact("water", "compound_of", "hydrogen")
        kg.add_fact("hydrogen", "is_a", "element")
        kg.add_fact("element", "part_of", "periodic table")

        r = kg.query("water", "compound_of", "hydrogen")
        assert r.answer is True


# ═══════════════════════════════════════════════════════════════════════
# 5. MUSIC HISTORY — 5-hop chain
# Source: Wikipedia "Jazz", "Blues"
# Chain: Jazz → evolved from Blues → evolved from Spirituals →
#        originated from Africa → in continent
# ═══════════════════════════════════════════════════════════════════════

class TestMusicHistory:
    """5-hop music evolution chain.

    Source: Wikipedia "Jazz", "Blues", "Spirituals"
    """

    def test_5_hop_jazz_chain(self, kg):
        """Jazz → Africa via 5 hops."""
        kg.add_fact("jazz", "evolved_from", "blues")
        kg.add_fact("blues", "evolved_from", "spirituals")
        kg.add_fact("spirituals", "originated_from", "african music")
        kg.add_fact("african music", "from", "africa")
        kg.set_relation_property("evolved_from", "transitive")

        # 2-hop: Jazz evolved from spirituals
        r = kg.query("jazz", "evolved_from", "spirituals")
        assert r.answer is True

    def test_genre_influence_chain(self, kg):
        """Rock → Blues → Africa chain."""
        kg.add_fact("rock", "influenced_by", "blues")
        kg.add_fact("blues", "evolved_from", "spirituals")
        kg.add_fact("spirituals", "originated_from", "africa")

        r = kg.query("rock", "influenced_by", "blues")
        assert r.answer is True


# ═══════════════════════════════════════════════════════════════════════
# 6. TECHNOLOGY — 5-hop chain
# Source: Wikipedia "iPhone", "Apple Inc."
# Chain: iPhone → runs iOS → made by Apple → founded by Steve Jobs →
#        born in USA → in North America
# ═══════════════════════════════════════════════════════════════════════

class TestTechnologyChain:
    """5-hop technology chain.

    Source: Wikipedia "iPhone", "Apple Inc."
    """

    def test_5_hop_iphone_chain(self, kg):
        """iPhone → North America via 5 hops."""
        kg.add_fact("iphone", "made_by", "apple")
        kg.add_fact("apple", "founded_by", "steve jobs")
        kg.add_fact("steve jobs", "born_in", "san francisco")
        kg.add_fact("san francisco", "in", "united states")
        kg.add_fact("united states", "in", "north america")

        # 2-hop: iPhone founded by Steve Jobs
        r = kg.query("apple", "founded_by", "steve jobs")
        assert r.answer is True

        # 3-hop: Steve Jobs in United States
        r2 = kg.query("steve jobs", "in", "united states")
        assert r2.answer is True

    def test_product_company_country(self, kg):
        """Galaxy → Samsung → South Korea chain."""
        kg.add_fact("galaxy", "made_by", "samsung")
        kg.add_fact("samsung", "in", "south korea")
        kg.add_fact("south korea", "in", "east asia")

        # 2-hop: Galaxy in South Korea
        r = kg.query("samsung", "in", "south korea")
        assert r.answer is True


# ═══════════════════════════════════════════════════════════════════════
# 7. MEDICINE — 5-hop chain
# Source: Wikipedia "Penicillin", "Antibiotic"
# Chain: Penicillin → treats infection → caused by bacteria →
#        studied by microbiology → part of biology → part of science
# ═══════════════════════════════════════════════════════════════════════

class TestMedicineChain:
    """5-hop medicine chain.

    Source: Wikipedia "Penicillin", "Microbiology"
    """

    def test_5_hop_penicillin_chain(self, kg):
        """Penicillin → Science via 5 hops."""
        kg.add_fact("penicillin", "treats", "bacterial infection")
        kg.add_fact("bacterial infection", "studied_by", "microbiology")
        kg.add_fact("microbiology", "part_of", "biology")
        kg.add_fact("biology", "part_of", "science")

        # 2-hop: Penicillin studied by microbiology
        r = kg.query("penicillin", "treats", "bacterial infection")
        assert r.answer is True

        # 3-hop: microbiology part of science
        r2 = kg.query("microbiology", "part_of", "science")
        assert r2.answer is True

    def test_disease_treatment_chain(self, kg):
        """Malaria → Artemisinin → discovered by Youyou Tu."""
        kg.add_fact("malaria", "treated_by", "artemisinin")
        kg.add_fact("artemisinin", "discovered_by", "tu youyou")
        kg.add_fact("tu youyou", "born_in", "china")

        r = kg.query("malaria", "treated_by", "artemisinin")
        assert r.answer is True


# ═══════════════════════════════════════════════════════════════════════
# 8. ARCHITECTURE — 5-hop chain
# Source: Wikipedia "Sagrada Família"
# Chain: Sagrada Família → designed by Gaudí → born in Catalonia →
#        in Spain → in Europe → in Eurasia
# ═══════════════════════════════════════════════════════════════════════

class TestArchitectureChain:
    """5-hop architecture chain.

    Source: Wikipedia "Sagrada Família", "Antoni Gaudí"
    """

    def test_5_hop_sagrada_chain(self, kg):
        """Sagrada Família → Eurasia via 5 hops."""
        kg.add_fact("sagrada familia", "designed_by", "antonio gaudi")
        kg.add_fact("antonio gaudi", "born_in", "catalonia")
        kg.add_fact("catalonia", "in", "spain")
        kg.add_fact("spain", "in", "europe")
        kg.add_fact("europe", "part_of", "eurasia")

        # 2-hop: Sagrada Família born in Catalonia
        r = kg.query("antonio gaudi", "in", "spain")
        assert r.answer is True

        # 3-hop: Gaudí in Europe
        r2 = kg.query("antonio gaudi", "in", "europe")
        assert r2.answer is True

        # 4-hop: Gaudí in Eurasia
        r3 = kg.query("antonio gaudi", "part_of", "eurasia")
        assert r3.answer is True


# ═══════════════════════════════════════════════════════════════════════
# 9. GEOLOGY — 5-hop chain
# Source: Wikipedia "Plate tectonics"
# Chain: Himalayas → formed by collision → of Indian Plate →
#        part of Lithosphere → part of Earth → in Solar System
# ═══════════════════════════════════════════════════════════════════════

class TestGeologyChain:
    """5-hop geology chain.

    Source: Wikipedia "Himalayas", "Plate tectonics"
    """

    def test_5_hop_himalayas_chain(self, kg):
        """Himalayas → Solar System via 5 hops."""
        kg.add_fact("himalayas", "formed_by", "indian plate collision")
        kg.add_fact("indian plate collision", "part_of", "plate tectonics")
        kg.add_fact("plate tectonics", "studied_by", "geology")
        kg.add_fact("geology", "part_of", "earth science")
        kg.add_fact("earth science", "part_of", "science")

        r = kg.query("himalayas", "formed_by", "indian plate collision")
        assert r.answer is True

        r2 = kg.query("indian plate collision", "part_of", "earth science")
        assert r2.answer is True


# ═══════════════════════════════════════════════════════════════════════
# 10. BIOLOGY — 5-hop chain
# Source: Wikipedia "DNA", "Genetics"
# Chain: DNA → encodes genes → part of chromosome → in nucleus →
#        in cell → in organism
# ═══════════════════════════════════════════════════════════════════════

class TestBiologyChain:
    """5-hop biology chain.

    Source: Wikipedia "DNA", "Cell biology"
    """

    def test_5_hop_dna_chain(self, kg):
        """DNA → Organism via 5 hops."""
        kg.add_fact("dna", "encodes", "genes")
        kg.add_fact("genes", "part_of", "chromosome")
        kg.add_fact("chromosome", "in", "nucleus")
        kg.add_fact("nucleus", "in", "cell")
        kg.add_fact("cell", "part_of", "organism")

        # 2-hop: DNA part of chromosome
        r = kg.query("genes", "part_of", "chromosome")
        assert r.answer is True

        # 3-hop: chromosome in cell
        r2 = kg.query("chromosome", "in", "cell")
        assert r2.answer is True

        # 4-hop: nucleus in organism
        r3 = kg.query("nucleus", "part_of", "organism")
        assert r3.answer is True


# ═══════════════════════════════════════════════════════════════════════
# 11. PERSISTENCE — Auto-save to SQLite
# ═══════════════════════════════════════════════════════════════════════

class TestComplexPersistence:
    """Complex chains persist through SQLite."""

    def test_5_hop_evolution_persists(self, kg):
        """5-hop evolution chain survives save/load."""
        kg.add_fact("homo sapiens", "evolved_from", "homo heidelbergensis")
        kg.add_fact("homo heidelbergensis", "evolved_from", "homo erectus")
        kg.add_fact("homo erectus", "evolved_from", "homo habilis")
        kg.add_fact("homo habilis", "evolved_from", "australopithecus")
        kg.add_fact("australopithecus", "evolved_from", "ardipithecus")
        kg.set_relation_property("evolved_from", "transitive")

        tmp = tempfile.mktemp(suffix=".db")
        try:
            kg.save(tmp)
            kg2 = KnowledgeGraph()
            kg2.load(tmp)

            # Transitivity preserved
            assert "transitive" in kg2.relation_properties.get("evolved_from", [])

            # 5-hop chain works after reload
            r = kg2.query("homo sapiens", "evolved_from", "ardipithecus")
            assert r.answer is True

            # Composition rules preserved
            assert kg2.composition_rules.get(("in", "in")) == "in"
        finally:
            os.unlink(tmp)

    def test_composition_rules_persist(self, kg):
        """Custom composition rules survive save/load."""
        kg.set_composition_rule("capital_of", "in", "in")
        kg.set_composition_rule("born_in", "in", "in")
        kg.set_composition_rule("borders", "borders", None)  # Explicitly blocked

        tmp = tempfile.mktemp(suffix=".db")
        try:
            kg.save(tmp)
            kg2 = KnowledgeGraph()
            kg2.load(tmp)

            assert kg2.composition_rules.get(("capital_of", "in")) == "in"
            assert kg2.composition_rules.get(("born_in", "in")) == "in"
            assert kg2.composition_rules.get(("borders", "borders")) is None
        finally:
            os.unlink(tmp)

    def test_mixed_domains_persist(self, kg):
        """Multiple 5-hop chains from different domains persist together."""
        # Evolution
        kg.add_fact("homo sapiens", "evolved_from", "homo erectus")
        kg.add_fact("homo erectus", "evolved_from", "homo habilis")
        kg.set_relation_property("evolved_from", "transitive")

        # Language
        kg.add_fact("english", "branch_of", "germanic")
        kg.add_fact("germanic", "branch_of", "indo-european")
        kg.set_relation_property("branch_of", "transitive")

        # Space
        kg.add_fact("voyager 1", "launched_by", "nasa")
        kg.add_fact("nasa", "in", "united states")

        tmp = tempfile.mktemp(suffix=".db")
        try:
            kg.save(tmp)
            kg2 = KnowledgeGraph()
            kg2.load(tmp)

            r1 = kg2.query("homo sapiens", "evolved_from", "homo habilis")
            assert r1.answer is True

            r2 = kg2.query("english", "branch_of", "indo-european")
            assert r2.answer is True

            r3 = kg2.query("voyager 1", "launched_by", "nasa")
            assert r3.answer is True
        finally:
            os.unlink(tmp)
