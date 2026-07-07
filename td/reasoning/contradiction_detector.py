"""Lightweight Ontological Type Guard (LOTG) — Contradiction Detection for TD v2.

A pre-commit hook that catches type contradictions before facts are stored.
Uses domain/range constraints on relations, entity type inference, and a
disjointness table. Warns but never rejects — the user is the authority.

Performance target: <1ms per check (dict lookups + set intersections).

Research backing:
    - OWL 2 rdfs:domain/rdfs:range (W3C, 2009) — property constraints
    - OWL 2 owl:disjointWith (W3C, 2009) — mutually exclusive types
    - RDFS rdfs:subClassOf (W3C, 2004) — type hierarchy / subsumption
    - Wikidata constraint violations — soft constraints, not hard blocks
    - Open World Assumption (OWL) — absence of fact ≠ negation of fact
    - NELL (CMU, 2010+) — entity type inference from relation patterns
    - Description Logic (Baader et al., 2003) — disjointness as DL axiom
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


# ─── Relation Schema: Domain/Range Constraints ─────────────────────
# Each relation can declare what types its subject (domain) and
# object (range) should have. None = unconstrained.
#
# Reference: OWL 2 rdfs:domain, rdfs:range (W3C Recommendation, 2009)
# Reference: Wikidata property constraints (https://www.wikidata.org/wiki/Wikidata:Database_reports/Constraint_violations)

RELATION_SCHEMA: dict[str, dict[str, str | None]] = {
    # Geography
    "capital_of":   {"domain": "city",          "range": "country"},
    "located_in":   {"domain": "place",         "range": "place"},
    "borders":      {"domain": "country",       "range": "country"},
    # People
    "born_in":      {"domain": "person",        "range": "place"},
    "lives_in":     {"domain": "person",        "range": "place"},
    "married_to":   {"domain": "person",        "range": "person"},
    "sibling_of":   {"domain": "person",        "range": "person"},
    "parent_of":    {"domain": "person",        "range": "person"},
    "child_of":     {"domain": "person",        "range": "person"},
    # Organizations
    "founded_by":   {"domain": "organization",  "range": "person"},
    "employs":      {"domain": "organization",  "range": "person"},
    "subsidiary_of":{"domain": "organization",  "range": "organization"},
    # Products / creation
    "made_by":      {"domain": "product",       "range": "organization"},
    "invented_by":  {"domain": "invention",     "range": "person"},
    "discovered_by":{"domain": "discovery",     "range": "person"},
    "directed_by":  {"domain": "film",          "range": "person"},
    "composed_by":  {"domain": "composition",   "range": "person"},
    "painted_by":   {"domain": "artwork",       "range": "person"},
    # Generic (unconstrained — too broad to type-check)
    "is_a":         {"domain": None,            "range": None},
    "in":           {"domain": None,            "range": None},
    "part_of":      {"domain": None,            "range": None},
    "before":       {"domain": None,            "range": None},
    "after":        {"domain": None,            "range": None},
    "contains":     {"domain": None,            "range": None},
    "has_property": {"domain": None,            "range": None},
}


# ─── Disjoint Types: Mutually Exclusive Categories ─────────────────
# Entities cannot belong to two types in the same frozenset.
# Only checked when both types are present in the same set.
#
# Reference: OWL 2 owl:disjointWith axiom (W3C Recommendation, 2009)
# Reference: Description Logic — disjointness is a fundamental DL axiom
#            (Baader et al., "The Description Logic Handbook", Cambridge, 2003)

DISJOINT_TYPES: set[frozenset[str]] = {
    frozenset({"city", "country", "continent", "state"}),
    frozenset({"person", "organization"}),
    frozenset({"animal", "plant"}),
    frozenset({"living", "non_living"}),
    frozenset({"product", "invention", "discovery", "artwork", "film", "composition"}),
}


# ─── Type Hierarchy: Subsumption (Subtype → Supertypes) ────────────
# Enables: "Paris is a city" AND "Paris is a place" = no conflict,
# because city ⊑ place (city is a subtype of place).
#
# Reference: OWL 2 rdfs:subClassOf (W3C Recommendation, 2009)
# Reference: Wikidata instance_of / subclass_of hierarchy
# Reference: RDFS Semantics (W3C, 2004)

TYPE_HIERARCHY: dict[str, set[str]] = {
    # Geographic
    "city":          {"settlement", "place"},
    "country":       {"place", "geopolitical_entity"},
    "continent":     {"place"},
    "state":         {"place", "geopolitical_entity"},
    "river":         {"place", "geographical_feature"},
    "ocean":         {"place", "geographical_feature"},
    "mountain":      {"place", "geographical_feature"},
    "island":        {"place", "geographical_feature"},
    # Entities
    "person":        {"living", "entity"},
    "organization":  {"entity"},
    "product":       {"entity"},
    "invention":     {"entity"},
    "discovery":     {"entity"},
    "artwork":       {"entity"},
    "film":          {"entity", "artwork"},
    "composition":   {"entity", "artwork"},
    # Abstract
    "settlement":    {"place"},
    "place":         {"entity"},
    "living":        {"entity"},
    "entity":        set(),  # root
}


# ─── Warning Data Structure ────────────────────────────────────────

@dataclass
class ContradictionWarning:
    """A type contradiction detected during fact ingestion."""
    entity: str
    existing_type: str
    new_type: str
    existing_proof: str
    new_proof: str
    relation: str = ""

    def __str__(self) -> str:
        return (
            f"⚠️  Contradiction for '{self.entity}': "
            f"previously inferred as '{self.existing_type}' ({self.existing_proof}), "
            f"now identified as '{self.new_type}' ({self.new_proof}). "
            f"These types are mutually exclusive."
        )


# ─── The Detector ──────────────────────────────────────────────────

class ContradictionDetector:
    """Lightweight ontological type guard for TD v2.

    Infers entity types from relations, tracks them, and checks for
    type contradictions against a disjointness table. Warns but never
    rejects — the user is the authority.

    Usage:
        detector = ContradictionDetector()
        warnings = detector.check("paris", "capital_of", "france")
        # paris → city (from domain), france → country (from range)
        warnings = detector.check("paris", "is_a", "country")
        # ⚠️ Contradiction: paris was 'city', now 'country'
    """

    def __init__(self):
        # entity → set of inferred types
        self.entity_types: dict[str, set[str]] = defaultdict(set)
        # entity → type → proof string
        self.type_proofs: dict[str, dict[str, str]] = defaultdict(dict)

    def check(self, subject: str, relation: str, obj: str) -> list[ContradictionWarning]:
        """Check a proposed triple for type contradictions.

        Returns a list of warnings (empty = no contradictions found).
        The triple should still be stored regardless — warnings are informational.

        Args:
            subject: Triple subject (already normalized to lowercase)
            relation: Triple relation (already normalized to lowercase)
            obj: Triple object (already normalized to lowercase)

        Returns:
            List of ContradictionWarning objects. Empty if no conflicts.
        """
        warnings: list[ContradictionWarning] = []

        # ── Case 1: Explicit type declaration (is_a) ──────────────
        if relation == "is_a":
            new_type = obj
            warnings.extend(self._check_and_record(subject, new_type,
                f"explicitly taught: '{subject} is_a {obj}'"))
            return warnings

        # ── Case 2: Relation has domain/range constraints ─────────
        schema = RELATION_SCHEMA.get(relation)
        if schema is None:
            return warnings  # Unknown relation, no constraints

        domain = schema.get("domain")
        range_ = schema.get("range")

        # Infer subject type from domain
        if domain:
            proof = f"inferred from: {subject} {relation} {obj} (domain of '{relation}' is {domain})"
            warnings.extend(self._check_and_record(subject, domain, proof))

        # Infer object type from range
        if range_:
            proof = f"inferred from: {subject} {relation} {obj} (range of '{relation}' is {range_})"
            warnings.extend(self._check_and_record(obj, range_, proof))

        return warnings

    def _check_and_record(self, entity: str, new_type: str, proof: str) -> list[ContradictionWarning]:
        """Check a new type against existing types, then record it.

        Uses the disjointness table AND type hierarchy:
        - If new_type and existing_type are in the same disjoint set
          AND neither subsumes the other → contradiction
        - If one subsumes the other → compatible, no warning
        """
        warnings: list[ContradictionWarning] = []
        existing = self.entity_types.get(entity, set())

        for existing_type in existing:
            if existing_type == new_type:
                continue  # Same type, no conflict

            # Check if one subsumes the other (compatible)
            if self._is_subsumed(new_type, existing_type):
                continue  # new_type ⊑ existing_type, OK
            if self._is_subsumed(existing_type, new_type):
                continue  # existing_type ⊑ new_type, OK

            # Check if they're in the same disjoint set
            if self._are_disjoint(new_type, existing_type):
                old_proof = self.type_proofs.get(entity, {}).get(existing_type, "unknown")
                warnings.append(ContradictionWarning(
                    entity=entity,
                    existing_type=existing_type,
                    new_type=new_type,
                    existing_proof=old_proof,
                    new_proof=proof,
                ))

        # Always record the new type (user is authority)
        self.entity_types[entity].add(new_type)
        if new_type not in self.type_proofs[entity]:
            self.type_proofs[entity][new_type] = proof

        return warnings

    def _are_disjoint(self, type1: str, type2: str) -> bool:
        """Check if two types are in the same disjoint set."""
        for disjoint_set in DISJOINT_TYPES:
            if type1 in disjoint_set and type2 in disjoint_set:
                return True
        return False

    def _is_subsumed(self, subtype: str, supertype: str) -> bool:
        """Check if subtype is a sub-type of supertype (transitively).

        Uses BFS over TYPE_HIERARCHY to handle multi-level hierarchies.
        """
        if subtype == supertype:
            return True
        visited = {subtype}
        queue = [subtype]
        while queue:
            current = queue.pop(0)
            parents = TYPE_HIERARCHY.get(current, set())
            for parent in parents:
                if parent == supertype:
                    return True
                if parent not in visited:
                    visited.add(parent)
                    queue.append(parent)
        return False

    def get_entity_types(self, entity: str) -> set[str]:
        """Get all inferred types for an entity."""
        return self.entity_types.get(entity, set()).copy()

    def get_type_proof(self, entity: str, type_name: str) -> str | None:
        """Get the proof for why an entity was inferred to have a type."""
        return self.type_proofs.get(entity, {}).get(type_name)

    def reset(self):
        """Clear all inferred types. Used for testing."""
        self.entity_types.clear()
        self.type_proofs.clear()
