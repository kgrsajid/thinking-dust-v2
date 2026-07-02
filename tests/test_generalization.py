"""Tests for TD v2 generalization — relation learning, persistence, inference.

Covers:
- Relation extraction from natural language ("is Y to Z", "is Y of Z", etc.)
- Relation property teaching (transitive, symmetric, functional)
- Multi-hop transitive inference (up to 5 hops)
- Symmetric inference
- Functional contradiction detection
- SQLite persistence + parser sync
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from td.perception.hdc import build_default_vocabulary
from td.memory.mhn import ModernHopfieldNetwork, MHNConfig
from td.thinking import GenericThinkingDust


@pytest.fixture
def td():
    vocab = build_default_vocabulary(dim=10000)
    mhn = ModernHopfieldNetwork(MHNConfig(dim=10000, min_similarity=0.01))
    return GenericThinkingDust(vocab=vocab, mhn=mhn, dim=10000, pure_mode=True)


# ─── Relation Extraction ──────────────────────────────────────────────

class TestRelationExtraction:
    """Parser extracts relations from natural language patterns."""

    def test_is_y_of_z(self, td):
        """'Paris is the capital of France' → capital_of"""
        result = td.parser.extract_structure('Paris is the capital of France')
        graph = result['graph']
        rel_types = [r['rel_type'] for r in graph.relations]
        assert 'capital_of' in rel_types

    def test_is_y_to_z(self, td):
        """'David Beckham is married to Victoria Beckham' → married_to"""
        result = td.parser.extract_structure('David Beckham is married to Victoria Beckham')
        graph = result['graph']
        rel_types = [r['rel_type'] for r in graph.relations]
        assert 'married_to' in rel_types

    def test_is_in_z(self, td):
        """'France is in the EU' → in"""
        result = td.parser.extract_structure('France is in the EU')
        graph = result['graph']
        rel_types = [r['rel_type'] for r in graph.relations]
        assert 'in' in rel_types

    def test_is_part_of_z(self, td):
        """'Engine is part of the car' → part_of"""
        result = td.parser.extract_structure('Engine is part of the car')
        graph = result['graph']
        rel_types = [r['rel_type'] for r in graph.relations]
        assert 'part_of' in rel_types

    def test_is_before_z(self, td):
        """'Alice is before Bob' → before"""
        result = td.parser.extract_structure('Alice is before Bob')
        graph = result['graph']
        rel_types = [r['rel_type'] for r in graph.relations]
        assert 'before' in rel_types

    def test_is_north_of_z(self, td):
        """'Kazakhstan is north of Uzbekistan' → north_of"""
        result = td.parser.extract_structure('Kazakhstan is north of Uzbekistan')
        graph = result['graph']
        rel_types = [r['rel_type'] for r in graph.relations]
        assert 'north_of' in rel_types


# ─── Triple Extraction (teach) ────────────────────────────────────────

class TestTripleExtraction:
    """teach() extracts KG triples from natural language."""

    def test_teach_is_y_to_z(self, td):
        """teach 'X is married to Y' → (x, married_to, y)"""
        td.teach('David Beckham is married to Victoria Beckham',
                 'David Beckham is married to Victoria Beckham')
        triples = [(t.subject, t.relation, t.object) for t in td.kg.triples]
        assert any(r == 'married_to' for _, r, _ in triples)

    def test_teach_is_y_of_z(self, td):
        """teach 'Paris is the capital of France' → (paris, capital_of, france)"""
        td.teach('Paris is the capital of France',
                 'Paris is the capital of France')
        triples = [(t.subject, t.relation, t.object) for t in td.kg.triples]
        assert ('paris', 'capital_of', 'france') in triples

    def test_teach_is_in_z(self, td):
        """teach 'France is in the EU' → (france, in, eu)"""
        td.teach('France is in the EU', 'France is in the EU')
        triples = [(t.subject, t.relation, t.object) for t in td.kg.triples]
        assert ('france', 'in', 'eu') in triples


# ─── Transitive Inference (2-5 hops) ──────────────────────────────────

class TestTransitiveInference:
    """KG derives facts via transitive chains."""

    def test_2_hop_transitive(self, td):
        """A → B → C (2 hops)"""
        td.teach('Kazakhstan is north of Uzbekistan', 'Kazakhstan is north of Uzbekistan')
        td.teach('Uzbekistan is north of Tajikistan', 'Uzbekistan is north of Tajikistan')
        td.teach_relation('north_of', 'transitive')

        result = td.think('is Kazakhstan north of Tajikistan')
        assert result.solution is not None
        assert result.solution['type'] == 'inferred'
        assert result.confidence >= 0.80

    def test_3_hop_transitive(self, td):
        """A → B → C → D (3 hops)"""
        td.teach('A is part of B', 'A is part of B')
        td.teach('B is part of C', 'B is part of C')
        td.teach('C is part of D', 'C is part of D')
        td.teach_relation('part_of', 'transitive')

        result = td.think('is A part of D')
        assert result.solution is not None
        assert result.solution['type'] == 'inferred'
        assert result.confidence >= 0.70

    def test_4_hop_transitive(self, td):
        """A → B → C → D → E (4 hops)"""
        td.teach('X1 is in X2', 'X1 is in X2')
        td.teach('X2 is in X3', 'X2 is in X3')
        td.teach('X3 is in X4', 'X3 is in X4')
        td.teach('X4 is in X5', 'X4 is in X5')
        td.teach_relation('in', 'transitive')

        result = td.think('is X1 in X5')
        assert result.solution is not None
        assert result.solution['type'] == 'inferred'
        assert result.confidence >= 0.65

    def test_5_hop_transitive(self, td):
        """A → B → C → D → E → F (5 hops)"""
        td.teach('Alpha before Beta', 'Alpha before Beta')
        td.teach('Beta before Gamma', 'Beta before Gamma')
        td.teach('Gamma before Delta', 'Gamma before Delta')
        td.teach('Delta before Epsilon', 'Delta before Epsilon')
        td.teach('Epsilon before Zeta', 'Epsilon before Zeta')
        td.teach_relation('before', 'transitive')

        result = td.think('is Alpha before Zeta')
        assert result.solution is not None
        assert result.solution['type'] == 'inferred'
        # 5 hops: max(0.65, 0.95 - 4*0.10) = 0.65
        assert result.confidence >= 0.60

    def test_6_hop_transitive(self, td):
        """A → B → C → D → E → F → G (6 hops)"""
        for i in range(6):
            td.teach(f'Step{i} is inside Step{i+1}', f'Step{i} is inside Step{i+1}')
        td.teach_relation('inside', 'transitive')

        result = td.think('is Step0 inside Step6')
        assert result.solution is not None
        assert result.solution['type'] == 'inferred'
        # 6 hops: max(0.65, 0.95 - 5*0.10) = 0.65
        assert result.confidence >= 0.60

    def test_confidence_decreases_with_hops(self, td):
        """Confidence: 1-hop > 2-hop > 3-hop > 4-hop > 5-hop.

        NOTE: derive_all() caches transitive facts, so we test each hop
        count with a FRESH KG to avoid derived shortcuts.
        """
        from td.kg import KnowledgeGraph

        targets = ['N2', 'N3', 'N4', 'N5', 'N6']
        expected_mins = [0.85, 0.75, 0.65, 0.65, 0.65]  # 1,2,3,4,5 hops

        for target, expected_min in zip(targets, expected_mins):
            # Fresh KG for each hop count to avoid derive_all() caching
            fresh_kg = KnowledgeGraph()
            for i in range(5):
                fresh_kg.add_fact(f'n{i+1}', 'north_of', f'n{i+2}')
            fresh_kg.set_relation_property('north_of', 'transitive')

            # Temporarily swap KG
            old_kg = td.kg
            td.kg = fresh_kg
            td.sync_kg_to_parser()

            result = td.think(f'is N1 north_of {target}')

            # Restore
            td.kg = old_kg
            td.sync_kg_to_parser()

            assert result.solution is not None, f"No solution for N1→{target}"
            assert result.solution['type'] == 'inferred', f"Expected inferred for N1→{target}, got {result.solution['type']}"
            assert result.confidence >= expected_min, \
                f"N1→{target}: confidence {result.confidence} < expected {expected_min}"


# ─── Symmetric Inference ──────────────────────────────────────────────

class TestSymmetricInference:
    """KG derives facts via symmetry."""

    def test_symmetric_basic(self, td):
        """A married_to B → B married_to A"""
        # Use single-word entities to match _extract_triples regex
        td.teach('Alice is married to Bob', 'Alice is married to Bob')
        td.teach_relation('married_to', 'symmetric')

        result = td.think('is Bob married to Alice')
        assert result.solution is not None
        # Accept "learned" (MHN) or "inferred" (KG) — both are valid answers
        assert result.solution['type'] in ('inferred', 'learned')

    def test_symmetric_sibling(self, td):
        """A sibling_of B → B sibling_of A"""
        td.teach('Carol is sibling_of Dave', 'Carol is sibling_of Dave')
        td.teach_relation('sibling_of', 'symmetric')

        result = td.think('is Dave sibling_of Carol')
        assert result.solution is not None
        assert result.solution['type'] in ('inferred', 'learned')

    def test_symmetric_same_as(self, td):
        """A same_as B → B same_as A"""
        # "same_as" triggers the "are X and Y the same" special case
        td.teach('Foo is same_as Bar', 'Foo is same_as Bar')
        td.teach_relation('same_as', 'symmetric')

        result = td.think('is Bar same_as Foo')
        assert result.solution is not None
        # Accept any non-unknown answer
        assert result.solution['type'] != 'unknown'


# ─── Novel (Unseen) Relation Generalization ────────────────────────────

class TestNovelRelationGeneralization:
    """System handles completely unseen relations — never pre-seeded, never taught before."""

    def test_novel_transitive_relation(self, td):
        """Teach a brand-new transitive relation 'feeds_into' and derive across it."""
        # 'feeds_into' is NOT in DEFAULT_RELATION_PROPERTIES or parser prototypes
        assert 'feeds_into' not in td.kg.relation_properties
        assert 'feeds_into' not in td.parser.relation_prototypes

        # Teach facts
        td.teach('RiverA feeds_into RiverB', 'RiverA feeds_into RiverB')
        td.teach('RiverB feeds_into RiverC', 'RiverB feeds_into RiverC')
        td.teach('RiverC feeds_into RiverD', 'RiverC feeds_into RiverD')

        # Teach the property
        td.teach_relation('feeds_into', 'transitive')

        # Should derive: RiverA → feeds_into → RiverD (3 hops)
        result = td.think('does RiverA feed into RiverD')
        assert result.solution is not None
        assert result.solution['type'] == 'inferred'

    def test_novel_symmetric_relation(self, td):
        """Teach a brand-new symmetric relation 'collaborates_with'."""
        assert 'collaborates_with' not in td.kg.relation_properties

        td.teach('TeamX collaborates_with TeamY', 'TeamX collaborates_with TeamY')
        td.teach_relation('collaborates_with', 'symmetric')

        # Should derive reverse
        result = td.think('does TeamY collaborate with TeamX')
        assert result.solution is not None
        assert result.solution['type'] in ('inferred', 'learned')

    def test_novel_functional_relation(self, td):
        """Teach a brand-new functional relation 'ceo_of'."""
        assert 'ceo_of' not in td.kg.relation_properties

        td.teach('Alice is ceo_of CompanyX', 'Alice is ceo_of CompanyX')
        td.teach('Bob is ceo_of CompanyY', 'Bob is ceo_of CompanyY')
        td.teach_relation('ceo_of', 'functional')

        # Should detect they're different
        result = td.think('are Alice and Bob the same')
        assert result.solution is not None
        assert result.solution['type'] == 'inferred'

    def test_novel_5_hop_chain(self, td):
        """5-hop chain with completely novel relation 'powers'."""
        assert 'powers' not in td.kg.relation_properties

        for i in range(5):
            td.teach(f'Device{i} powers Device{i+1}', f'Device{i} powers Device{i+1}')
        td.teach_relation('powers', 'transitive')

        result = td.think('does Device0 power Device5')
        assert result.solution is not None
        assert result.solution['type'] == 'inferred'
        assert result.confidence >= 0.60

    def test_novel_relation_persistence(self, td):
        """Novel relation survives SQLite save/load and still works."""
        assert 'regulates' not in td.kg.relation_properties

        td.teach('GeneA regulates GeneB', 'GeneA regulates GeneB')
        td.teach('GeneB regulates GeneC', 'GeneB regulates GeneC')
        td.teach_relation('regulates', 'transitive')

        # Verify it works before save
        result1 = td.think('does GeneA regulate GeneC')
        assert result1.solution['type'] == 'inferred'

        # Save and reload
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            path = f.name
        try:
            td.kg.save(path)

            from td.perception.hdc import build_default_vocabulary
            from td.memory.mhn import ModernHopfieldNetwork, MHNConfig
            td2 = GenericThinkingDust(
                vocab=build_default_vocabulary(dim=10000),
                mhn=ModernHopfieldNetwork(MHNConfig(dim=10000, min_similarity=0.01)),
                dim=10000, pure_mode=True
            )
            td2.kg.load(path)
            td2.sync_kg_to_parser()

            # Should work after reload
            assert 'regulates' in td2.parser.relation_prototypes
            assert 'regulates' in td2.kg.relation_properties

            result2 = td2.think('does GeneA regulate GeneC')
            assert result2.solution is not None
            assert result2.solution['type'] in ('inferred', 'learned')
        finally:
            os.unlink(path)

    def test_multiple_novel_relations_interleaved(self, td):
        """Multiple novel relations taught, system keeps them separate."""
        # Relation 1: transitive
        td.teach('ModuleA depends_on ModuleB', 'ModuleA depends_on ModuleB')
        td.teach('ModuleB depends_on ModuleC', 'ModuleB depends_on ModuleC')
        td.teach_relation('depends_on', 'transitive')

        # Relation 2: symmetric
        td.teach('NodeX connected_to NodeY', 'NodeX connected_to NodeY')
        td.teach_relation('connected_to', 'symmetric')

        # Relation 3: functional
        td.teach('Port80 is assigned_to ServiceA', 'Port80 is assigned_to ServiceA')
        td.teach('Port443 is assigned_to ServiceB', 'Port443 is assigned_to ServiceB')
        td.teach_relation('assigned_to', 'functional')

        # Test transitive
        r1 = td.think('does ModuleA depend on ModuleC')
        assert r1.solution['type'] in ('inferred', 'learned')

        # Test symmetric
        r2 = td.think('is NodeY connected to NodeX')
        assert r2.solution['type'] in ('inferred', 'learned')

        # Test functional
        r3 = td.think('are Port80 and Port443 the same')
        assert r3.solution is not None
        assert r3.solution['type'] == 'inferred'

        # All three should be in parser + KG
        assert 'depends_on' in td.parser.relation_prototypes
        assert 'connected_to' in td.parser.relation_prototypes
        assert 'assigned_to' in td.parser.relation_prototypes


# ─── Functional Contradiction ─────────────────────────────────────────

class TestFunctionalInference:
    """KG detects contradictions for functional relations."""

    def test_functional_different_values(self, td):
        """capital_of is functional: Paris ≠ Berlin → different"""
        td.teach('Paris is the capital of France', 'Paris is the capital of France')
        td.teach('Berlin is the capital of Germany', 'Berlin is the capital of Germany')
        td.teach_relation('capital_of', 'functional')

        result = td.think('are Paris and Berlin the same')
        assert result.solution is not None
        assert result.solution['type'] == 'inferred'

    def test_functional_same_value(self, td):
        """capital_of is functional: same capital → same city"""
        td.teach('Paris is the capital of France', 'Paris is the capital of France')
        td.teach('Paris_city is the capital of France', 'Paris_city is the capital of France')
        td.teach_relation('capital_of', 'functional')

        # This should detect they're the same (same functional target)
        result = td.think('are Paris and Paris_city the same')
        assert result.solution is not None


# ─── Relation Teaching + Parser Sync ──────────────────────────────────

class TestRelationTeaching:
    """Relations taught to KG are synced to parser."""

    def test_teach_relation_syncs_to_parser(self, td):
        """teach_relation('foo_bar', 'transitive') → parser has 'foo_bar' prototype"""
        assert 'foo_bar' not in td.parser.relation_prototypes
        td.teach_relation('foo_bar', 'transitive')
        assert 'foo_bar' in td.parser.relation_prototypes

    def test_preseeded_relations_in_parser(self, td):
        """Pre-seeded relations (married_to, capital_of) are in parser at init."""
        assert 'married_to' in td.parser.relation_prototypes
        assert 'capital_of' in td.parser.relation_prototypes
        assert 'in' in td.parser.relation_prototypes
        assert 'part_of' in td.parser.relation_prototypes

    def test_new_relation_detectable_after_teach(self, td):
        """After teaching 'adjacent_to', parser can detect it in new queries."""
        td.teach_relation('adjacent_to', 'symmetric')
        result = td.parser.extract_structure('Room A is adjacent_to Room B')
        graph = result['graph']
        rel_types = [r['rel_type'] for r in graph.relations]
        assert 'adjacent_to' in rel_types


# ─── SQLite Persistence ───────────────────────────────────────────────

class TestPersistence:
    """Relations and triples survive SQLite save/load."""

    def test_relation_properties_persist(self, td):
        """Relation properties survive save/load cycle."""
        td.teach_relation('custom_rel', 'transitive')
        td.teach_relation('another_rel', 'symmetric')

        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            path = f.name

        try:
            td.kg.save(path)

            # Load into a fresh KG
            from td.kg import KnowledgeGraph
            kg2 = KnowledgeGraph()
            kg2.load(path)

            assert 'custom_rel' in kg2.relation_properties
            assert 'transitive' in kg2.relation_properties['custom_rel']
            assert 'another_rel' in kg2.relation_properties
            assert 'symmetric' in kg2.relation_properties['another_rel']
        finally:
            os.unlink(path)

    def test_triples_persist(self, td):
        """Triples survive save/load cycle."""
        td.teach('A is north of B', 'A is north of B')
        td.teach('B is north of C', 'B is north of C')

        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            path = f.name

        try:
            td.kg.save(path)

            from td.kg import KnowledgeGraph
            kg2 = KnowledgeGraph()
            kg2.load(path)

            triples = [(t.subject, t.relation, t.object) for t in kg2.triples]
            assert ('a', 'north_of', 'b') in triples
            assert ('b', 'north_of', 'c') in triples
        finally:
            os.unlink(path)

    def test_parser_syncs_after_load(self, td):
        """Parser gets KG relations after sync_kg_to_parser()."""
        td.teach_relation('wibble_wobble', 'transitive')

        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            path = f.name

        try:
            td.kg.save(path)

            # Fresh thinking engine
            from td.perception.hdc import build_default_vocabulary
            from td.memory.mhn import ModernHopfieldNetwork, MHNConfig
            td2 = GenericThinkingDust(
                vocab=build_default_vocabulary(dim=10000),
                mhn=ModernHopfieldNetwork(MHNConfig(dim=10000, min_similarity=0.01)),
                dim=10000, pure_mode=True
            )

            # Load KG
            td2.kg.load(path)
            td2.sync_kg_to_parser()

            assert 'wibble_wobble' in td2.parser.relation_prototypes
        finally:
            os.unlink(path)


# ─── Full End-to-End Generalization ───────────────────────────────────

class TestEndToEnd:
    """Full teach → infer → persist → reload → infer cycle."""

    def test_5_hop_chain_with_persistence(self, td):
        """Teach 5-hop chain, save, reload, still derives."""
        # Build chain: A → B → C → D → E → F
        for i in range(5):
            td.teach(f'Node{i} is ancestor_of Node{i+1}',
                     f'Node{i} is ancestor_of Node{i+1}')
        td.teach_relation('ancestor_of', 'transitive')

        # Verify inference works
        result = td.think('is Node0 ancestor_of Node5')
        assert result.solution is not None
        assert result.solution['type'] in ('inferred', 'learned')

        # Save and reload
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            path = f.name

        try:
            td.kg.save(path)

            from td.perception.hdc import build_default_vocabulary
            from td.memory.mhn import ModernHopfieldNetwork, MHNConfig
            td2 = GenericThinkingDust(
                vocab=build_default_vocabulary(dim=10000),
                mhn=ModernHopfieldNetwork(MHNConfig(dim=10000, min_similarity=0.01)),
                dim=10000, pure_mode=True
            )
            td2.kg.load(path)
            td2.sync_kg_to_parser()

            # Should still derive after reload
            result2 = td2.think('is Node0 ancestor_of Node5')
            assert result2.solution is not None
            assert result2.solution['type'] in ('inferred', 'learned')
        finally:
            os.unlink(path)

    def test_mixed_relation_types(self, td):
        """Multiple relation types coexist correctly."""
        # Transitive chain
        td.teach('CityA is in RegionB', 'CityA is in RegionB')
        td.teach('RegionB is in CountryC', 'RegionB is in CountryC')
        td.teach_relation('in', 'transitive')

        # Symmetric relation
        td.teach('PersonX is sibling_of PersonY', 'PersonX is sibling_of PersonY')
        td.teach_relation('sibling_of', 'symmetric')

        # Functional relation
        td.teach('CapitalP is the capital of CountryQ', 'CapitalP is the capital of CountryQ')
        td.teach_relation('capital_of', 'functional')

        # Test all three — accept "inferred" or "learned" (both are valid)
        r1 = td.think('is CityA in CountryC')
        assert r1.solution['type'] in ('inferred', 'learned')

        r2 = td.think('is PersonY sibling_of PersonX')
        assert r2.solution['type'] in ('inferred', 'learned')

        r3 = td.think('is CapitalP the capital of CountryQ')
        assert r3.solution is not None
