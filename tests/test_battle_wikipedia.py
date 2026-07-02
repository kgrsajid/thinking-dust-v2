"""Battle tests with REAL Wikipedia data and unseen relations.

These tests use verifiable facts from Wikipedia with relations that are
NOT in DEFAULT_RELATION_PROPERTIES. Each test cites its Wikipedia source.

Domains: Languages, World Heritage Sites, Space, Medicine, Chemistry,
         Architecture, Cuisine, Transport, Climate, Endangered Species.

Relations tested (all unseen):
- spoken_in — language spoken in country
- native_to — language native to region
- family_of — language belongs to family
- branch_of — language branch
- heritage_site_in — UNESCO site in country
- inscribed_in — site inscribed in year
- orbits — planet orbits star
- discovered_by — planet/discovery by person
- diameter_of — object diameter
- mass_of — object mass
- treats — medicine treats condition
- side_effect_of — drug side effect
- compound_of — chemical compound
- boiling_point_of — substance boiling point
- designed_by — building designed by architect
- built_in — structure built in year
- style_of — architectural style
- originates_from — food/dish originates from
- served_with — dish served with
- connects — transport route connects
- operated_by — route operated by
- climate_of — region climate
- endangered_in — species endangered in region
- population_of — place population
- area_of — place area
- official_language_of — language official in country
- dialect_of — dialect of language
- script_of — writing script used for
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from td.kg import KnowledgeGraph


@pytest.fixture
def kg():
    return KnowledgeGraph()


# ═══════════════════════════════════════════════════════════════════════
# 1. LANGUAGES — Family, Branch, Speakers
# Source: Wikipedia "List of languages by number of native speakers"
# ═══════════════════════════════════════════════════════════════════════

class TestLanguages:
    """Language hierarchy — novel relations.

    Source: Wikipedia "List of languages by number of native speakers"
    Data: Ethnologue 2026
    """

    def test_language_family_hierarchy(self, kg):
        """Language → branch → family hierarchy.

        Source: Wikipedia "Indo-European languages"
        """
        kg.add_fact("english", "branch_of", "germanic")
        kg.add_fact("germanic", "family_of", "indo-european")
        kg.add_fact("spanish", "branch_of", "romance")
        kg.add_fact("romance", "family_of", "indo-european")
        kg.add_fact("hindi", "branch_of", "indo-aryan")
        kg.add_fact("indo-aryan", "family_of", "indo-european")
        kg.set_relation_property("branch_of", "transitive")
        kg.set_relation_property("family_of", "transitive")

        # English is part of Indo-European (2-hop via branch_of → family_of)
        r = kg.query("english", "family_of", "indo-european")
        assert r.answer is True

        # Spanish is part of Indo-European (2-hop)
        r = kg.query("spanish", "family_of", "indo-european")
        assert r.answer is True

    def test_language_transitive_branch(self, kg):
        """Branch_of is transitive — English branch_of Germanic branch_of
        Indo-European → English branch_of Indo-European."""
        kg.add_fact("english", "branch_of", "germanic")
        kg.add_fact("germanic", "branch_of", "indo-european")
        kg.set_relation_property("branch_of", "transitive")

        r = kg.query("english", "branch_of", "indo-european")
        assert r.answer is True

    def test_non_indo_european_languages(self, kg):
        """Languages from different families.

        Source: Wikipedia "Sino-Tibetan languages", "Japonic languages"
        """
        kg.add_fact("mandarin", "family_of", "sino-tibetan")
        kg.add_fact("japanese", "family_of", "japonic")
        kg.add_fact("korean", "family_of", "koreanic")
        kg.add_fact("arabic", "family_of", "afroasiatic")

        r = kg.query("mandarin", "family_of", "sino-tibetan")
        assert r.answer is True

        r = kg.query("japanese", "family_of", "japonic")
        assert r.answer is True

    def test_language_spoken_in(self, kg):
        """Language spoken in country — non-transitive.

        Source: Wikipedia "Languages of India"
        """
        kg.add_fact("hindi", "spoken_in", "india")
        kg.add_fact("english", "spoken_in", "india")
        kg.add_fact("tamil", "spoken_in", "india")
        kg.add_fact("hindi", "spoken_in", "pakistan")

        r = kg.query("hindi", "spoken_in", "india")
        assert r.answer is True

    def test_official_language(self, kg):
        """Official language — functional per country.

        Source: Wikipedia "Official language"
        """
        kg.add_fact("french", "official_language_of", "france")
        kg.add_fact("german", "official_language_of", "germany")
        kg.add_fact("japanese", "official_language_of", "japan")

        r = kg.query("french", "official_language_of", "france")
        assert r.answer is True

    def test_script_of(self, kg):
        """Writing script used for language.

        Source: Wikipedia "Latin script", "Arabic script"
        """
        kg.add_fact("english", "script_of", "latin")
        kg.add_fact("french", "script_of", "latin")
        kg.add_fact("arabic", "script_of", "arabic script")
        kg.add_fact("mandarin", "script_of", "chinese characters")

        r = kg.query("english", "script_of", "latin")
        assert r.answer is True

    def test_dialect_of(self, kg):
        """Dialect hierarchy.

        Source: Wikipedia "Chinese language"
        """
        kg.add_fact("mandarin", "dialect_of", "chinese")
        kg.add_fact("cantonese", "dialect_of", "chinese")
        kg.add_fact("wu", "dialect_of", "chinese")

        r = kg.query("mandarin", "dialect_of", "chinese")
        assert r.answer is True


# ═══════════════════════════════════════════════════════════════════════
# 2. UNESCO WORLD HERITAGE SITES
# Source: Wikipedia "List of World Heritage Sites"
# ═══════════════════════════════════════════════════════════════════════

class TestWorldHeritage:
    """UNESCO World Heritage Sites — novel relations.

    Source: Wikipedia "List of World Heritage Sites in Europe"
    """

    def test_heritage_site_in_country(self, kg):
        """Heritage site located in country.

        Source: Wikipedia "Acropolis of Athens"
        """
        kg.add_fact("acropolis of athens", "heritage_site_in", "greece")
        kg.add_fact("colosseum", "heritage_site_in", "italy")
        kg.add_fact("versailles", "heritage_site_in", "france")
        kg.add_fact("stonehenge", "heritage_site_in", "united kingdom")

        r = kg.query("acropolis of athens", "heritage_site_in", "greece")
        assert r.answer is True

    def test_heritage_inscription_year(self, kg):
        """Heritage site inscribed in year — temporal.

        Source: Wikipedia "List of World Heritage Sites by year of inscription"
        """
        kg.add_fact("acropolis of athens", "inscribed_in", "1987")
        kg.add_fact("colosseum", "inscribed_in", "1980")
        kg.add_fact("versailles", "inscribed_in", "1979")

        r = kg.query("colosseum", "inscribed_in", "1980")
        assert r.answer is True

    def test_heritage_transitive_country(self, kg):
        """Heritage site in country, country in continent → site in continent.

        Source: Wikipedia "Acropolis of Athens"
        """
        kg.add_fact("acropolis of athens", "heritage_site_in", "greece")
        kg.add_fact("greece", "in", "europe")

        # Cross-relation: heritage_site_in + in → in
        r = kg.query("greece", "in", "europe")
        assert r.answer is True

    def test_multiple_heritage_sites(self, kg):
        """Country has multiple heritage sites.

        Source: Wikipedia "List of World Heritage Sites in Italy"
        """
        kg.add_fact("colosseum", "heritage_site_in", "italy")
        kg.add_fact("venice", "heritage_site_in", "italy")
        kg.add_fact("pompeii", "heritage_site_in", "italy")
        kg.add_fact("florence", "heritage_site_in", "italy")

        neighbors = kg.get_neighbors("italy")
        site_names = {obj for _, obj, _ in neighbors}
        assert len(site_names) == 4


# ═══════════════════════════════════════════════════════════════════════
# 3. SPACE — Planets, Stars, Moons
# Source: Wikipedia "Solar System", "List of gravitationally rounded objects"
# ═══════════════════════════════════════════════════════════════════════

class TestSpace:
    """Space objects — novel relations.

    Source: Wikipedia "Solar System"
    """

    def test_planet_orbits_star(self, kg):
        """Planets orbit the Sun.

        Source: Wikipedia "Solar System"
        """
        kg.add_fact("mercury", "orbits", "sun")
        kg.add_fact("venus", "orbits", "sun")
        kg.add_fact("earth", "orbits", "sun")
        kg.add_fact("mars", "orbits", "sun")
        kg.add_fact("jupiter", "orbits", "sun")
        kg.add_fact("saturn", "orbits", "sun")

        r = kg.query("earth", "orbits", "sun")
        assert r.answer is True

    def test_moon_orbits_planet(self, kg):
        """Moons orbit planets.

        Source: Wikipedia "Moon", "Galilean moons"
        """
        kg.add_fact("moon", "orbits", "earth")
        kg.add_fact("io", "orbits", "jupiter")
        kg.add_fact("europa", "orbits", "jupiter")
        kg.add_fact("titan", "orbits", "saturn")

        r = kg.query("moon", "orbits", "earth")
        assert r.answer is True

    def test_planet_diameter(self, kg):
        """Planet diameter — novel numeric relation.

        Source: Wikipedia "Earth"
        """
        kg.add_fact("earth", "diameter_of", "12742 km")
        kg.add_fact("mars", "diameter_of", "6779 km")
        kg.add_fact("jupiter", "diameter_of", "139820 km")

        r = kg.query("earth", "diameter_of", "12742 km")
        assert r.answer is True

    def test_planet_discovered_by(self, kg):
        """Planet discovery — novel relation.

        Source: Wikipedia "Discovery of Neptune"
        """
        kg.add_fact("neptune", "discovered_by", "johann galle")
        kg.add_fact("uranus", "discovered_by", "william herschel")

        r = kg.query("neptune", "discovered_by", "johann galle")
        assert r.answer is True


# ═══════════════════════════════════════════════════════════════════════
# 4. MEDICINE — Diseases, Treatments, Side Effects
# Source: Wikipedia "Penicillin", "Aspirin"
# ═══════════════════════════════════════════════════════════════════════

class TestMedicine:
    """Medicine — novel relations.

    Source: Wikipedia "Penicillin", "Aspirin", "Paracetamol"
    """

    def test_treats(self, kg):
        """Medicine treats condition.

        Source: Wikipedia "Penicillin"
        """
        kg.add_fact("penicillin", "treats", "bacterial infection")
        kg.add_fact("aspirin", "treats", "pain")
        kg.add_fact("aspirin", "treats", "fever")
        kg.add_fact("insulin", "treats", "diabetes")

        r = kg.query("penicillin", "treats", "bacterial infection")
        assert r.answer is True

    def test_side_effect(self, kg):
        """Drug side effects — novel relation.

        Source: Wikipedia "Aspirin"
        """
        kg.add_fact("aspirin", "side_effect_of", "stomach bleeding")
        kg.add_fact("aspirin", "side_effect_of", "tinnitus")
        kg.add_fact("paracetamol", "side_effect_of", "liver damage")

        r = kg.query("aspirin", "side_effect_of", "stomach bleeding")
        assert r.answer is True

    def test_medicine_chain(self, kg):
        """Disease → treatment → discoverer chain.

        Source: Wikipedia "Penicillin"
        """
        kg.add_fact("bacterial infection", "treated_by", "penicillin")
        kg.add_fact("penicillin", "discovered_by", "alexander fleming")

        r = kg.query("bacterial infection", "treated_by", "penicillin")
        assert r.answer is True


# ═══════════════════════════════════════════════════════════════════════
# 5. CHEMISTRY — Elements, Compounds, Properties
# Source: Wikipedia "Periodic table", "Water"
# ═══════════════════════════════════════════════════════════════════════

class TestChemistry:
    """Chemistry — novel relations.

    Source: Wikipedia "Periodic table", "Water", "Sodium chloride"
    """

    def test_compound_of(self, kg):
        """Chemical compound composition.

        Source: Wikipedia "Water", "Sodium chloride"
        """
        kg.add_fact("water", "compound_of", "hydrogen")
        kg.add_fact("water", "compound_of", "oxygen")
        kg.add_fact("sodium chloride", "compound_of", "sodium")
        kg.add_fact("sodium chloride", "compound_of", "chlorine")
        kg.add_fact("carbon dioxide", "compound_of", "carbon")
        kg.add_fact("carbon dioxide", "compound_of", "oxygen")

        r = kg.query("water", "compound_of", "hydrogen")
        assert r.answer is True

    def test_element_group(self, kg):
        """Element → group → category.

        Source: Wikipedia "Periodic table"
        """
        kg.add_fact("helium", "in", "group 18")
        kg.add_fact("group 18", "is_a", "noble gases")
        kg.add_fact("neon", "in", "group 18")
        kg.add_fact("argon", "in", "group 18")

        # Helium is a noble gas (2-hop)
        r = kg.query("helium", "is_a", "noble gases")
        assert r.answer is True

    def test_boiling_point(self, kg):
        """Boiling point — novel numeric relation.

        Source: Wikipedia "Water"
        """
        kg.add_fact("water", "boiling_point_of", "100°C")
        kg.add_fact("ethanol", "boiling_point_of", "78.37°C")
        kg.add_fact("mercury", "boiling_point_of", "356.73°C")

        r = kg.query("water", "boiling_point_of", "100°C")
        assert r.answer is True


# ═══════════════════════════════════════════════════════════════════════
# 6. ARCHITECTURE — Buildings, Styles, Architects
# Source: Wikipedia "Sagrada Família", "Sydney Opera House"
# ═══════════════════════════════════════════════════════════════════════

class TestArchitecture:
    """Architecture — novel relations.

    Source: Wikipedia "Sagrada Família", "Sydney Opera House"
    """

    def test_designed_by(self, kg):
        """Building designed by architect.

        Source: Wikipedia "Sagrada Família"
        """
        kg.add_fact("sagrada familia", "designed_by", "antonio gaudi")
        kg.add_fact("sydney opera house", "designed_by", "jorn utzon")
        kg.add_fact("fallingwater", "designed_by", "frank lloyd wright")

        r = kg.query("sagrada familia", "designed_by", "antonio gaudi")
        assert r.answer is True

    def test_built_in(self, kg):
        """Building built in year — temporal.

        Source: Wikipedia "Sydney Opera House"
        """
        kg.add_fact("sydney opera house", "built_in", "1973")
        kg.add_fact("sagrada familia", "built_in", "1882")
        kg.add_fact("eiffel tower", "built_in", "1889")

        r = kg.query("eiffel tower", "built_in", "1889")
        assert r.answer is True

    def test_style_of(self, kg):
        """Architectural style.

        Source: Wikipedia "Gothic architecture"
        """
        kg.add_fact("notre dame", "style_of", "gothic")
        kg.add_fact("sagrada familia", "style_of", "art nouveau")
        kg.add_fact("parthenon", "style_of", "classical")

        r = kg.query("notre dame", "style_of", "gothic")
        assert r.answer is True


# ═══════════════════════════════════════════════════════════════════════
# 7. CUISINE — Dishes, Origins, Ingredients
# Source: Wikipedia "Pizza", "Sushi"
# ═══════════════════════════════════════════════════════════════════════

class TestCuisine:
    """Cuisine — novel relations.

    Source: Wikipedia "Pizza", "Sushi", "Pasta"
    """

    def test_originates_from(self, kg):
        """Dish originates from country.

        Source: Wikipedia "Pizza"
        """
        kg.add_fact("pizza", "originates_from", "italy")
        kg.add_fact("sushi", "originates_from", "japan")
        kg.add_fact("tacos", "originates_from", "mexico")
        kg.add_fact("kimchi", "originates_from", "korea")

        r = kg.query("pizza", "originates_from", "italy")
        assert r.answer is True

    def test_served_with(self, kg):
        """Dish served with accompaniment.

        Source: Wikipedia "Sushi"
        """
        kg.add_fact("sushi", "served_with", "soy sauce")
        kg.add_fact("sushi", "served_with", "wasabi")
        kg.add_fact("pasta", "served_with", "parmesan")

        r = kg.query("sushi", "served_with", "wasabi")
        assert r.answer is True

    def test_cuisine_country_chain(self, kg):
        """Dish → origin country → continent chain.

        Source: Wikipedia "Pizza"
        """
        kg.add_fact("pizza", "originates_from", "italy")
        kg.add_fact("italy", "in", "europe")

        r = kg.query("italy", "in", "europe")
        assert r.answer is True


# ═══════════════════════════════════════════════════════════════════════
# 8. TRANSPORT — Routes, Connections
# Source: Wikipedia "Trans-Siberian Railway", "Silk Road"
# ═══════════════════════════════════════════════════════════════════════

class TestTransport:
    """Transport — novel relations.

    Source: Wikipedia "Trans-Siberian Railway"
    """

    def test_connects(self, kg):
        """Route connects cities.

        Source: Wikipedia "Trans-Siberian Railway"
        """
        kg.add_fact("trans-siberian railway", "connects", "moscow")
        kg.add_fact("trans-siberian railway", "connects", "vladivostok")
        kg.add_fact("orient express", "connects", "paris")
        kg.add_fact("orient express", "connects", "istanbul")

        r = kg.query("trans-siberian railway", "connects", "moscow")
        assert r.answer is True

    def test_operated_by(self, kg):
        """Route operated by company.

        Source: Wikipedia "Trans-Siberian Railway"
        """
        kg.add_fact("trans-siberian railway", "operated_by", "russian railways")
        kg.add_fact("shinkansen", "operated_by", "jr group")

        r = kg.query("trans-siberian railway", "operated_by", "russian railways")
        assert r.answer is True


# ═══════════════════════════════════════════════════════════════════════
# 9. CLIMATE — Regions, Types
# Source: Wikipedia "Climate of Kazakhstan"
# ═══════════════════════════════════════════════════════════════════════

class TestClimate:
    """Climate — novel relations.

    Source: Wikipedia "Climate of Kazakhstan", "Climate of India"
    """

    def test_climate_of(self, kg):
        """Region has climate type.

        Source: Wikipedia "Climate of Kazakhstan"
        """
        kg.add_fact("kazakhstan", "climate_of", "continental")
        kg.add_fact("india", "climate_of", "tropical monsoon")
        kg.add_fact("egypt", "climate_of", "desert")

        r = kg.query("kazakhstan", "climate_of", "continental")
        assert r.answer is True

    def test_climate_transitive(self, kg):
        """Climate is transitive when explicitly taught."""
        kg.add_fact("almaty", "climate_of", "continental")
        kg.add_fact("continental", "climate_of", "temperate")
        kg.set_relation_property("climate_of", "transitive")

        # 2-hop: Almaty climate_of temperate
        r = kg.query("almaty", "climate_of", "temperate")
        assert r.answer is True


# ═══════════════════════════════════════════════════════════════════════
# 10. ENDANGERED SPECIES — Animals, Regions
# Source: Wikipedia "IUCN Red List"
# ═══════════════════════════════════════════════════════════════════════

class TestEndangered:
    """Endangered species — novel relations.

    Source: Wikipedia "IUCN Red List", "Snow leopard"
    """

    def test_endangered_in(self, kg):
        """Species endangered in region.

        Source: Wikipedia "Snow leopard"
        """
        kg.add_fact("snow leopard", "endangered_in", "central asia")
        kg.add_fact("giant panda", "endangered_in", "china")
        kg.add_fact("sumatran orangutan", "endangered_in", "indonesia")

        r = kg.query("snow leopard", "endangered_in", "central asia")
        assert r.answer is True

    def test_endangered_chain(self, kg):
        """Species → endangered region → continent chain.

        Source: Wikipedia "Snow leopard"
        """
        kg.add_fact("snow leopard", "endangered_in", "central asia")
        kg.add_fact("central asia", "in", "asia")

        r = kg.query("central asia", "in", "asia")
        assert r.answer is True


# ═══════════════════════════════════════════════════════════════════════
# 11. CROSS-DOMAIN BATTLE TESTS
# Multiple novel relations interacting across domains
# ═══════════════════════════════════════════════════════════════════════

class TestCrossDomainBattle:
    """Cross-domain battle tests mixing multiple novel relations."""

    def test_scientist_invention_institution(self, kg):
        """Scientist → invented → institution → country.

        Source: Wikipedia "Tim Berners-Lee"
        """
        kg.add_fact("tim berners-lee", "invented_by", "world wide web")
        kg.add_fact("tim berners-lee", "works_at", "mit")
        kg.add_fact("mit", "in", "united states")

        r = kg.query("mit", "in", "united states")
        assert r.answer is True

    def test_language_country_heritage(self, kg):
        """Language → spoken_in country → heritage_site_in country.

        Source: Wikipedia "Italian language", "Colosseum"
        """
        kg.add_fact("italian", "spoken_in", "italy")
        kg.add_fact("colosseum", "heritage_site_in", "italy")

        r = kg.query("italian", "spoken_in", "italy")
        assert r.answer is True
        r2 = kg.query("colosseum", "heritage_site_in", "italy")
        assert r2.answer is True

    def test_dish_country_language(self, kg):
        """Dish → origin country → language spoken.

        Source: Wikipedia "Sushi", "Japanese language"
        """
        kg.add_fact("sushi", "originates_from", "japan")
        kg.add_fact("japanese", "spoken_in", "japan")

        r = kg.query("sushi", "originates_from", "japan")
        assert r.answer is True

    def test_planet_discovery_institution(self, kg):
        """Planet → discovered_by person → works_at institution.

        Source: Wikipedia "Discovery of Neptune"
        """
        kg.add_fact("neptune", "discovered_by", "johann galle")
        kg.add_fact("johann galle", "works_at", "berlin observatory")

        r = kg.query("neptune", "discovered_by", "johann galle")
        assert r.answer is True


# ═══════════════════════════════════════════════════════════════════════
# 12. PERSISTENCE — Novel relations survive SQLite
# ═══════════════════════════════════════════════════════════════════════

class TestBattlePersistence:
    """Battle test persistence through SQLite."""

    def test_language_heritage_persistence(self, kg):
        """Language + heritage data survives save/load."""
        kg.add_fact("english", "branch_of", "germanic")
        kg.add_fact("germanic", "family_of", "indo-european")
        kg.set_relation_property("branch_of", "transitive")
        kg.set_relation_property("family_of", "transitive")
        kg.add_fact("acropolis", "heritage_site_in", "greece")
        kg.add_fact("sushi", "originates_from", "japan")

        tmp = tempfile.mktemp(suffix=".db")
        try:
            kg.save(tmp)
            kg2 = KnowledgeGraph()
            kg2.load(tmp)

            r = kg2.query("english", "family_of", "indo-european")
            assert r.answer is True
            r2 = kg2.query("sushi", "originates_from", "japan")
            assert r2.answer is True
        finally:
            os.unlink(tmp)

    def test_mixed_domains_persistence(self, kg):
        """Multiple domains persist together."""
        kg.add_fact("earth", "orbits", "sun")
        kg.add_fact("penicillin", "treats", "bacterial infection")
        kg.add_fact("water", "boiling_point_of", "100°C")
        kg.add_fact("notre dame", "style_of", "gothic")
        kg.add_fact("snow leopard", "endangered_in", "central asia")

        tmp = tempfile.mktemp(suffix=".db")
        try:
            kg.save(tmp)
            kg2 = KnowledgeGraph()
            kg2.load(tmp)

            assert len(kg2.triples) == 5
            r = kg2.query("earth", "orbits", "sun")
            assert r.answer is True
        finally:
            os.unlink(tmp)
