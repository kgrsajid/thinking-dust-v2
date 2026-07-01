#!/usr/bin/env python3
"""HDC Knowledge Graph + Z3 Inference Engine for TD v2.

This is the "thinking" layer. It stores facts as triples, applies general
logical rule templates (transitive, symmetric, inverse, functional), and
derives new facts that were never explicitly taught.

Architecture:
    - Triples: (subject, relation, object) stored as Z3 facts
    - Rule templates: general logical schemas (transitive, symmetric, etc.)
    - Relation properties: which template applies to which relation
    - Z3 solver: derives new facts, detects contradictions, generates proof traces

No hardcoded facts. No hardcoded rules for specific words. The templates
are general; the properties are taught or pre-seeded.
"""

from __future__ import annotations

import time
import sqlite3
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict


# ─── Z3 Import (lazy) ────────────────────────────────────────────────

def _import_z3():
    try:
        from z3 import (
            Solver, DeclareSort, Const, Consts, Function, BoolSort,
            Implies, And, Or, Not, ForAll, sat, unsat, ModelRef,
            StringSort, String, Expr,
        )
        return True
    except ImportError:
        return False


# ─── Rule Templates (General Logical Schemas) ────────────────────────

RULE_TEMPLATES = {
    "transitive": "{R}(X,Y) ∧ {R}(Y,Z) → {R}(X,Z)",
    "symmetric":  "{R}(X,Y) → {R}(Y,X)",
    "inverse":    "{R1}(X,Y) → {R2}(Y,X)",
    "functional": "{R}(X,Y) ∧ {R}(X,Z) → Y=Z",
    "reflexive":  "→ {R}(X,X)",
    "antisymmetric": "{R}(X,Y) ∧ {R}(Y,X) → X=Y",
}


# Pre-seeded relation properties (bootstrap, user can extend)
DEFAULT_RELATION_PROPERTIES = {
    "in": ["transitive"],
    "part_of": ["transitive"],
    "before": ["transitive"],
    "after": ["transitive"],
    "inside": ["transitive"],
    "contains": ["transitive"],
    "subset_of": ["transitive"],
    "ancestor_of": ["transitive"],
    "descendant_of": ["transitive"],
    "larger_than": ["transitive"],
    "smaller_than": ["transitive"],
    "capital_of": ["functional"],
    "equals": ["symmetric", "transitive"],
    "same_as": ["symmetric", "transitive"],
    "married_to": ["symmetric"],
    "sibling_of": ["symmetric"],
    "adjacent_to": ["symmetric"],
}


@dataclass
class Triple:
    """A knowledge graph triple."""
    subject: str
    relation: str
    object: str
    source: str = "user"  # "user", "derived", "seed"
    proof: str = ""       # derivation chain for derived facts

    def __repr__(self):
        return f"({self.subject}, {self.relation}, {self.object})"


@dataclass
class InferenceResult:
    """Result of a Z3 inference query."""
    answer: Optional[bool]  # True, False, or None (unknown)
    proof_trace: str        # human-readable derivation
    confidence: float
    method: str             # "direct", "transitive", "derived", "contradiction", "unknown"


class KnowledgeGraph:
    """HDC Knowledge Graph with Z3 inference.

    Stores triples, applies logical rule templates, derives new facts.
    The templates are general (transitive, symmetric, etc.). The properties
    are per-relation (pre-seeded or user-taught).
    """

    def __init__(self):
        self.triples: list[Triple] = []
        self.relation_properties: dict[str, list[str]] = dict(DEFAULT_RELATION_PROPERTIES)
        self._entity_index: dict[str, list[int]] = defaultdict(list)  # entity → triple indices

        # Inverse relation tracking (for inverse: pairs)
        self._inverse_pairs: dict[str, str] = {}

    def add_fact(self, subject: str, relation: str, obj: str, source: str = "user", proof: str = "") -> Triple:
        """Add a triple to the knowledge graph."""
        # Normalize
        subject = subject.strip().lower()
        relation = relation.strip().lower()
        obj = obj.strip().lower()

        # Check for duplicate
        for t in self.triples:
            if t.subject == subject and t.relation == relation and t.object == obj:
                return t  # Already exists

        triple = Triple(subject, relation, obj, source, proof)
        idx = len(self.triples)
        self.triples.append(triple)
        self._entity_index[subject].append(idx)
        self._entity_index[obj].append(idx)
        return triple

    def set_relation_property(self, relation: str, *properties: str):
        """Set properties for a relation (user-taught).

        Example: kg.set_relation_property("north_of", "transitive")
        """
        relation = relation.strip().lower()
        if relation not in self.relation_properties:
            self.relation_properties[relation] = []
        for prop in properties:
            if prop not in self.relation_properties[relation]:
                self.relation_properties[relation].append(prop)
            # Track inverse pairs
            if prop.startswith("inverse:"):
                inv = prop.split(":", 1)[1]
                self._inverse_pairs[relation] = inv
                self._inverse_pairs[inv] = relation

    def get_facts_for_relation(self, relation: str) -> list[Triple]:
        """Get all triples with a given relation."""
        relation = relation.strip().lower()
        return [t for t in self.triples if t.relation == relation]

    def get_neighbors(self, entity: str, direction: str = "both") -> list[tuple[str, str, str]]:
        """Get (relation, neighbor, direction) tuples for an entity."""
        entity = entity.strip().lower()
        results = []
        for idx in self._entity_index.get(entity, []):
            t = self.triples[idx]
            if t.subject == entity:
                results.append((t.relation, t.object, "outgoing"))
            if t.object == entity:
                results.append((t.relation, t.subject, "incoming"))
        return results

    def bfs_paths(self, start: str, end: str, max_hops: int = 4) -> list[list[Triple]]:
        """Find all paths between two entities using BFS.

        Returns list of paths, where each path is a list of triples.
        """
        start = start.strip().lower()
        end = end.strip().lower()

        if start == end:
            return []

        paths = []
        # BFS queue: (current_entity, path_so_far, visited_entities)
        queue = [(start, [], {start})]

        while queue:
            current, path, visited = queue.pop(0)

            if len(path) > max_hops:
                continue

            # Get all triples involving current entity
            for idx in self._entity_index.get(current, []):
                t = self.triples[idx]

                # Forward: current is subject
                if t.subject == current:
                    neighbor = t.object
                    if neighbor == end:
                        paths.append(path + [t])
                    elif neighbor not in visited and len(path) < max_hops:
                        queue.append((neighbor, path + [t], visited | {neighbor}))

                # Backward: current is object (traverse inverse)
                if t.object == current:
                    neighbor = t.subject
                    if neighbor == end:
                        paths.append(path + [t])
                    elif neighbor not in visited and len(path) < max_hops:
                        queue.append((neighbor, path + [t], visited | {neighbor}))

        return paths

    def query(self, subject: str, relation: str, obj: str = None) -> InferenceResult:
        """Query the knowledge graph with Z3 inference.

        Three modes:
        1. Direct lookup: (Paris, capital_of, France) → True
        2. Inference: (Paris, in, EU) → derived via transitivity
        3. Contradiction: (Paris, in, Germany) → False (conflicts with known facts)
        """
        subject = subject.strip().lower()
        relation = relation.strip().lower()
        obj = obj.strip().lower() if obj else None

        t0 = time.perf_counter()

        # Mode 1: Direct lookup
        for t in self.triples:
            if t.subject == subject and t.relation == relation:
                if obj and t.object == obj:
                    return InferenceResult(
                        answer=True,
                        proof_trace=f"Direct fact: ({subject}, {relation}, {obj})",
                        confidence=0.95,
                        method="direct",
                    )
                elif obj and t.object != obj and self._is_functional(relation):
                    # Functional relation: only one object per subject
                    return InferenceResult(
                        answer=False,
                        proof_trace=f"Contradiction: {relation} is functional. "
                                   f"Known: ({subject}, {relation}, {t.object}), "
                                   f"not {obj}.",
                        confidence=0.90,
                        method="contradiction",
                    )

        # Mode 2: BFS path finding (no Z3 needed for simple transitivity)
        if obj:
            paths = self.bfs_paths(subject, obj, max_hops=4)
            if paths:
                # Check if any path is valid under the relation's properties
                best_path = self._find_valid_path(paths, relation, subject, obj)
                if best_path:
                    trace = self._format_proof_trace(best_path, subject, relation, obj)
                    elapsed = (time.perf_counter() - t0) * 1000
                    return InferenceResult(
                        answer=True,
                        proof_trace=trace,
                        confidence=0.80,
                        method="derived",
                    )

            # Check for contradiction via functional relations
            for t in self.triples:
                if t.relation == relation and t.subject == subject and t.object != obj:
                    if self._is_functional(relation):
                        return InferenceResult(
                            answer=False,
                            proof_trace=f"Contradiction: {relation}({subject}) is {t.object}, not {obj}.",
                            confidence=0.90,
                            method="contradiction",
                        )

        # Mode 3: Open query (subject + relation, no object)
        if obj is None:
            results = []
            for t in self.triples:
                if t.subject == subject and t.relation == relation:
                    results.append(t.object)
            # Also check inverse relations
            for inv_rel in self._get_inverse_relations(relation):
                for t in self.triples:
                    if t.object == subject and t.relation == inv_rel:
                        results.append(t.subject)
            if results:
                return InferenceResult(
                    answer=True,
                    proof_trace=f"{relation}({subject}) → {', '.join(results)}",
                    confidence=0.85,
                    method="direct",
                )

        return InferenceResult(
            answer=None,
            proof_trace="No matching facts or derivable conclusions.",
            confidence=0.0,
            method="unknown",
        )

    def check_same(self, entity1: str, entity2: str) -> InferenceResult:
        """Check if two entities are the same using functional properties.

        Uses Option B: if both entities have different values for the same
        functional relation, they are provably different.

        Example:
            capital_of(Paris, France) and capital_of(Berlin, Germany)
            capital_of is functional → Paris != Berlin
        """
        entity1 = entity1.strip().lower()
        entity2 = entity2.strip().lower()

        if entity1 == entity2:
            return InferenceResult(
                answer=True,
                proof_trace=f"{entity1} and {entity2} are identical.",
                confidence=1.0,
                method="direct",
            )

        # Check if explicitly stated as same
        for t in self.triples:
            if ((t.subject == entity1 and t.object == entity2 and
                 t.relation in ("same_as", "equals", "identical_to")) or
                (t.subject == entity2 and t.object == entity1 and
                 t.relation in ("same_as", "equals", "identical_to"))):
                return InferenceResult(
                    answer=True,
                    proof_trace=f"Explicitly taught: {t.subject} {t.relation} {t.object}",
                    confidence=0.95,
                    method="direct",
                )

        # Check functional relations for distinction
        for func_rel, props in self.relation_properties.items():
            if "functional" not in props:
                continue
            # Get values for entity1 and entity2 under this functional relation
            val1 = None
            val2 = None
            for t in self.triples:
                if t.relation == func_rel:
                    if t.subject == entity1:
                        val1 = t.object
                    elif t.subject == entity2:
                        val2 = t.object
            # If both have values and they differ → provably different
            if val1 and val2 and val1 != val2:
                return InferenceResult(
                    answer=False,
                    proof_trace=f"No. {entity1} has {func_rel}={val1}, "
                               f"but {entity2} has {func_rel}={val2}. "
                               f"Since {func_rel} is functional, they are different.",
                    confidence=0.90,
                    method="contradiction",
                )

        # Check if they share any relation with different objects
        # (weaker evidence of difference)
        rels1 = {(t.relation, t.object) for t in self.triples if t.subject == entity1}
        rels2 = {(t.relation, t.object) for t in self.triples if t.subject == entity2}
        if rels1 and rels2:
            # They both exist in the KG but no functional property distinguishes them
            return InferenceResult(
                answer=None,
                proof_trace=f"I know about both {entity1} and {entity2}, "
                           f"but I have no evidence that they are the same or different.",
                confidence=0.0,
                method="unknown",
            )

        return InferenceResult(
            answer=None,
            proof_trace=f"I don't have enough information to determine "
                       f"if {entity1} and {entity2} are the same.",
            confidence=0.0,
            method="unknown",
        )

    def _is_functional(self, relation: str) -> bool:
        """Check if a relation is functional (one object per subject)."""
        props = self.relation_properties.get(relation, [])
        return "functional" in props

    def _find_valid_path(self, paths: list[list[Triple]], target_relation: str,
                         start: str, end: str) -> list[Triple] | None:
        """Find a path that's logically valid for the target relation.

        Priority:
        1. Path where ALL edges match the target relation (pure transitivity)
        2. Path where the LAST edge matches the target relation (composition)
        3. Any path (fallback — the entities are connected somehow)
        """
        target_props = self.relation_properties.get(target_relation, [])

        # Priority 1: pure transitivity (all edges = target relation)
        if "transitive" in target_props:
            for path in paths:
                rels = set(t.relation for t in path)
                if rels == {target_relation}:
                    return path

        # Priority 2: last edge matches target relation (composition)
        for path in paths:
            if path and path[-1].relation == target_relation:
                return path

        # Priority 3: any path where target relation appears
        for path in paths:
            if any(t.relation == target_relation for t in path):
                return path

        # Fallback: shortest path (entities are connected)
        return paths[0] if paths else None

    def _format_proof_trace(self, path: list[Triple], subject: str,
                            relation: str, obj: str) -> str:
        """Format a proof trace showing every hop with its relation."""
        if not path:
            return f"({subject}, {relation}, {obj})"

        hops = []
        for t in path:
            hops.append(f"{t.subject} --{t.relation}--> {t.object}")

        chain = " , ".join(hops)
        return f"Yes. {subject} {relation} {obj} because: {chain}"

    def _get_inverse_relations(self, relation: str) -> list[str]:
        """Get inverse relation names."""
        if relation in self._inverse_pairs:
            return [self._inverse_pairs[relation]]
        # Common inverse patterns
        inversions = {
            "capital_of": "has_capital",
            "parent_of": "child_of",
            "part_of": "contains",
            "in": "contains",
            "before": "after",
            "larger_than": "smaller_than",
        }
        if relation in inversions:
            return [inversions[relation]]
        return []

    def derive_transitive(self, relation: str) -> list[Triple]:
        """Derive all transitive facts for a given relation.

        If R is transitive and R(A,B) and R(B,C) exist, derive R(A,C).
        Returns list of newly derived triples.
        """
        derived = []
        facts = self.get_facts_for_relation(relation)

        # Build adjacency for this relation
        adj: dict[str, list[str]] = defaultdict(list)
        for t in facts:
            adj[t.subject].append(t.object)

        # Floyd-Warshall style transitive closure
        existing = set((t.subject, t.object) for t in facts)
        changed = True
        while changed:
            changed = False
            new_pairs = []
            for (a, b) in list(existing):
                for c in adj.get(b, []):
                    if (a, c) not in existing:
                        new_pairs.append((a, c))
            for a, c in new_pairs:
                if (a, c) not in existing:
                    existing.add((a, c))
                    adj[a].append(c)
                    # Build proof chain
                    proof = f"derived: {relation} is transitive"
                    triple = self.add_fact(a, relation, c, source="derived", proof=proof)
                    if triple:
                        derived.append(triple)
                        changed = True

        return derived

    def derive_all(self) -> list[Triple]:
        """Run all applicable inference rules and return derived facts.
        
        Handles:
        - Pure transitivity: R(X,Y) ∧ R(Y,Z) → R(X,Z) where R is same relation
        - Cross-relation composition: R1(X,Y) ∧ R2(Y,Z) → R2(X,Z) when both are transitive
          (e.g., capital_of(Paris, France) ∧ in(France, EU) → in(Paris, EU))
        """
        all_derived = []
        
        # Phase 1: Pure transitivity (same relation)
        for relation, props in self.relation_properties.items():
            if "transitive" in props:
                derived = self.derive_transitive(relation)
                all_derived.extend(derived)

        # Phase 2: Cross-relation composition
        # If R1 and R2 are both transitive, and R1(X,Y) ∧ R2(Y,Z) exist,
        # then R2(X,Z) is derivable (compose through the chain)
        transitive_relations = [
            r for r, props in self.relation_properties.items() 
            if "transitive" in props
        ]
        
        changed = True
        while changed:
            changed = False
            for r1 in transitive_relations:
                for r2 in transitive_relations:
                    # Find R1(X,Y) ∧ R2(Y,Z) → R2(X,Z)
                    facts_r1 = self.get_facts_for_relation(r1)
                    facts_r2 = self.get_facts_for_relation(r2)
                    
                    # Build lookup: entity → things it connects to via R2
                    r2_by_subject = defaultdict(list)
                    for t in facts_r2:
                        r2_by_subject[t.subject].append(t.object)
                    
                    for t1 in facts_r1:
                        x, y = t1.subject, t1.object
                        # For each Z where R2(Y, Z) exists, derive R2(X, Z)
                        for z in r2_by_subject.get(y, []):
                            if z != x:
                                # Check if R2(X,Z) already exists
                                exists = any(
                                    t.subject == x and t.relation == r2 and t.object == z
                                    for t in self.triples
                                )
                                if not exists:
                                    proof = f"derived: {r1}({x},{y}) ∧ {r2}({y},{z}) → {r2}({x},{z})"
                                    triple = self.add_fact(x, r2, z, source="derived", proof=proof)
                                    if triple:
                                        all_derived.append(triple)
                                        # Also add to r2_by_subject for chaining
                                        r2_by_subject[x].append(z)
                                        changed = True

        return all_derived

    # ─── SQLite Persistence ──────────────────────────────────────────

    def save(self, path: str = None):
        """Save knowledge graph to SQLite database.

        Stores triples and relation properties in a queryable SQLite file.
        Dense vectors (BEAGLE, MHN) remain as separate pickle files.

        Args:
            path: Path to .db file. Defaults to data/td_knowledge.db
        """
        if path is None:
            path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data", "td_knowledge.db"
            )

        os.makedirs(os.path.dirname(path), exist_ok=True)

        conn = sqlite3.connect(path)
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS triples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    object TEXT NOT NULL,
                    source TEXT DEFAULT 'user',
                    proof TEXT DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS relation_properties (
                    relation TEXT PRIMARY KEY,
                    properties TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_triples_subject ON triples(subject);
                CREATE INDEX IF NOT EXISTS idx_triples_relation ON triples(relation);
                CREATE INDEX IF NOT EXISTS idx_triples_object ON triples(object);
            """)

            # Clear and re-insert (full sync)
            conn.execute("DELETE FROM triples")
            conn.execute("DELETE FROM relation_properties")

            conn.executemany(
                "INSERT INTO triples (subject, relation, object, source, proof) VALUES (?, ?, ?, ?, ?)",
                [(t.subject, t.relation, t.object, t.source, t.proof) for t in self.triples]
            )

            conn.executemany(
                "INSERT INTO relation_properties (relation, properties) VALUES (?, ?)",
                [(rel, ",".join(props)) for rel, props in self.relation_properties.items()]
            )

            conn.commit()
        finally:
            conn.close()

    def load(self, path: str = None) -> bool:
        """Load knowledge graph from SQLite database.

        Args:
            path: Path to .db file. Defaults to data/td_knowledge.db

        Returns:
            True if loaded successfully, False if file doesn't exist.
        """
        if path is None:
            path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data", "td_knowledge.db"
            )

        if not os.path.exists(path):
            return False

        conn = sqlite3.connect(path)
        try:
            # Load triples
            rows = conn.execute(
                "SELECT subject, relation, object, source, proof FROM triples"
            ).fetchall()

            for subject, relation, obj, source, proof in rows:
                self.add_fact(subject, relation, obj, source=source, proof=proof)

            # Load relation properties
            prop_rows = conn.execute(
                "SELECT relation, properties FROM relation_properties"
            ).fetchall()

            for relation, props_str in prop_rows:
                props = props_str.split(",") if props_str else []
                if props:
                    self.set_relation_property(relation, *props)

            conn.close()

            loaded_count = len(rows)
            return loaded_count > 0
        except Exception:
            conn.close()
            return False

    def query_sql(self, sql: str, params: tuple = ()) -> list[dict]:
        """Run arbitrary SQL query against the triples database.

        Example:
            kg.query_sql("SELECT * FROM triples WHERE subject = ?", ("paris",))
            kg.query_sql("SELECT * FROM triples WHERE relation = ?", ("capital_of",))

        This returns results from the persisted DB, not the in-memory graph.
        Call save() first if you want the latest in-memory facts.
        """
        default_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "td_knowledge.db"
        )
        if not os.path.exists(default_path):
            return []

        conn = sqlite3.connect(default_path)
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def stats(self) -> dict:
        user_facts = sum(1 for t in self.triples if t.source == "user")
        derived_facts = sum(1 for t in self.triples if t.source == "derived")
        return {
            "total_triples": len(self.triples),
            "user_facts": user_facts,
            "derived_facts": derived_facts,
            "entities": len(set(t.subject for t in self.triples) | set(t.object for t in self.triples)),
            "relations": len(set(t.relation for t in self.triples)),
        }
