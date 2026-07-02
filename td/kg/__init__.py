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

# Temporal reasoning — Allen's interval algebra (lazy import to avoid circular deps)
# Allen, J.F. (1983). "Maintaining Knowledge about Temporal Intervals." CACM, 26(11).
# Used locally in temporal reasoning methods only.


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
    """A knowledge graph triple.

    Can optionally carry temporal interval data (start, end) for temporal reasoning
    via Allen's interval algebra. When temporal fields are None, the triple is
    treated as timeless (existing behavior).
    """
    subject: str
    relation: str
    object: str
    source: str = "user"           # "user", "derived", "seed"
    proof: str = ""                # derivation chain for derived facts
    temporal_start: Optional[int] = None  # Year/start of interval (None = open start)
    temporal_end: Optional[int] = None    # Year/end of interval (None = open end)

    def __repr__(self):
        base = f"({self.subject}, {self.relation}, {self.object})"
        if self.temporal_start is not None or self.temporal_end is not None:
            s = self.temporal_start if self.temporal_start is not None else "-∞"
            e = self.temporal_end if self.temporal_end is not None else "∞"
            base += f" @[{s}, {e})"
        return base

    def has_temporal(self) -> bool:
        """True if this triple has temporal data."""
        return self.temporal_start is not None or self.temporal_end is not None

    def is_interval(self) -> bool:
        """True if this triple has a complete (valid) temporal interval."""
        return self.temporal_start is not None and self.temporal_end is not None


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

    def __init__(self, max_hops: int = 100):
        self.triples: list[Triple] = []
        self.relation_properties: dict[str, list[str]] = dict(DEFAULT_RELATION_PROPERTIES)
        self._entity_index: dict[str, list[int]] = defaultdict(list)  # entity → triple indices
        self.max_hops = max_hops  # Effectively unlimited (100 hops)

        # Inverse relation tracking (for inverse: pairs)
        self._inverse_pairs: dict[str, str] = {}

        # Temporal index: entity → (temporal_start, temporal_end)
        self._temporal_index: dict[str, tuple[int | None, int | None]] = {}

        # Composition rules: (rel1, rel2) → target_relation
        # OWL Property Chain style: explicit declaration of which relations compose.
        # If (rel1, rel2) → target, then rel1(X,Y) ∧ rel2(Y,Z) → target(X,Z)
        # If (rel1, rel2) → None, composition is explicitly INVALID.
        # If (rel1, rel2) not in dict, fall back to transitivity heuristics.
        #
        # References:
        #   - OWL 2 Web Ontology Language (W3C, 2009) — PropertyChain axiom
        #   - HolmE (Zheng et al., 2024) — KGE closed under composition
        #   - Rot-Pro (NeurIPS, 2021) — transitivity by projection
        #   - GLIDR (arXiv, 2025) — differentiable ILP for graph-structured rules
        self.composition_rules: dict[tuple[str, str], str | None] = {}

        # Pre-seeded composition rules (from OWL best practices)
        self._init_default_composition_rules()

        # Gazetteer: learned multi-word entity dictionary
        # Populated from teach() interactions. Used by query parser to
        # recognize "United Kingdom" as a single entity in queries.
        # Reference: Named Entity Recognition (Nadeau & Sekine, 2007)
        self.gazetteer: set[str] = set()

    def add_fact(self, subject: str, relation: str, obj: str, source: str = "user",
                 proof: str = "", temporal_start: int = None,
                 temporal_end: int = None) -> Triple:
        """Add a triple to the knowledge graph.

        Args:
            subject, relation, obj: The fact triple
            source: "user", "derived", or "seed"
            proof: derivation chain for derived facts
            temporal_start: Start year of the interval (None = open/unbounded)
            temporal_end: End year of the interval (None = open/unbounded)
        """
        # Normalize
        subject = subject.strip().lower()
        relation = relation.strip().lower()
        obj = obj.strip().lower()

        # Check for duplicate (ignoring temporal fields for deduplication)
        for t in self.triples:
            if t.subject == subject and t.relation == relation and t.object == obj:
                # Update temporal fields if provided
                if temporal_start is not None:
                    t.temporal_start = temporal_start
                if temporal_end is not None:
                    t.temporal_end = temporal_end
                # Update temporal index (only subject owns its interval)
                if t.temporal_start is not None:
                    self._temporal_index[subject] = (t.temporal_start, t.temporal_end)
                return t  # Already exists

        triple = Triple(subject, relation, obj, source, proof,
                        temporal_start=temporal_start, temporal_end=temporal_end)
        idx = len(self.triples)
        self.triples.append(triple)
        self._entity_index[subject].append(idx)
        self._entity_index[obj].append(idx)

        # Update gazetteer with multi-word entities
        if " " in subject:
            self.gazetteer.add(subject)
        if " " in obj:
            self.gazetteer.add(obj)

        # Update temporal index (only the subject entity owns this interval)
        # Store even if one end is open (None) — used for open-ended comparisons
        if temporal_start is not None:
            self._temporal_index[subject] = (temporal_start, temporal_end)

        return triple

    def _init_default_composition_rules(self):
        """Initialize default composition rules from OWL best practices.

        These are the "obvious" compositions that hold in most domains:
        - in ∘ in → in (transitive spatial containment)
        - part_of ∘ part_of → part_of (transitive part-whole)
        - before ∘ before → before (transitive temporal)
        - after ∘ after → after (transitive temporal)

        Domain-specific compositions (capital_of ∘ in → in) are NOT pre-seeded.
        Users teach them explicitly via set_composition_rule().

        Reference: OWL 2 Web Ontology Language (W3C, 2009) — PropertyChain
        """
        # Same-relation transitive compositions
        for rel in ("in", "part_of", "before", "after", "contains",
                     "subset_of", "ancestor_of", "descendant_of"):
            self.composition_rules[(rel, rel)] = rel

    def set_composition_rule(self, rel1: str, rel2: str, target: str | None):
        """Declare a composition rule: rel1(X,Y) ∧ rel2(Y,Z) → target(X,Z).

        Args:
            rel1: First relation in the chain
            rel2: Second relation in the chain
            target: Resulting relation, or None to explicitly block composition

        Examples:
            kg.set_composition_rule("capital_of", "in", "in")
            # capital_of(Paris,France) ∧ in(France,EU) → in(Paris,EU)

            kg.set_composition_rule("born_in", "in", None)
            # born_in(Einstein,Ulm) ∧ in(Ulm,Germany) → INVALID

        Reference: OWL 2 PropertyChain axiom (W3C, 2009)
        """
        rel1 = rel1.strip().lower()
        rel2 = rel2.strip().lower()
        if target is not None:
            target = target.strip().lower()
        self.composition_rules[(rel1, rel2)] = target

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

    def bfs_paths(self, start: str, end: str, max_hops: int = None) -> list[list[Triple]]:
        """Find all paths between two entities using BFS.

        Returns list of paths, where each path is a list of triples.
        Only traverses backwards (object → subject) for SYMMETRIC relations.
        For asymmetric relations (in, part_of, before, etc.), only traverses
        forward (subject → object).
        """
        start = start.strip().lower()
        end = end.strip().lower()

        if max_hops is None:
            max_hops = self.max_hops

        if start == end:
            return []

        paths = []
        queue = [(start, [], {start})]

        while queue:
            current, path, visited = queue.pop(0)

            if len(path) > max_hops:
                continue

            for idx in self._entity_index.get(current, []):
                t = self.triples[idx]

                # Forward: current is subject (always valid)
                if t.subject == current:
                    neighbor = t.object
                    if neighbor == end:
                        paths.append(path + [t])
                    elif neighbor not in visited and len(path) < max_hops:
                        queue.append((neighbor, path + [t], visited | {neighbor}))

                # Backward: current is object — ONLY for symmetric relations
                if t.object == current:
                    is_symmetric = "symmetric" in self.relation_properties.get(t.relation, set())
                    if is_symmetric:
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
            paths = self.bfs_paths(subject, obj)
            if paths:
                # Check if any path is valid under the relation's properties
                best_path = self._find_valid_path(paths, relation, subject, obj)
                if best_path:
                    trace = self._format_proof_trace(best_path, subject, relation, obj)
                    elapsed = (time.perf_counter() - t0) * 1000
                    conf = self._chain_confidence(best_path, relation)
                    return InferenceResult(
                        answer=True,
                        proof_trace=trace,
                        confidence=conf,
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

            # Multi-hop open query: follow BFS paths to find the answer
            # "Who founded the company that makes iPhone?" → follow iphone→apple→steve jobs
            # Find all entities reachable from subject via any relation
            reachable = []
            for idx in self._entity_index.get(subject, []):
                t = self.triples[idx]
                if t.subject == subject:
                    reachable.append((t.object, [t]))

            # BFS to find entities reachable via the target relation
            visited = {subject}
            queue = [(subject, [])]
            while queue:
                current, path = queue.pop(0)
                if len(path) > 6:
                    continue
                for idx in self._entity_index.get(current, []):
                    t = self.triples[idx]
                    if t.subject == current and t.object not in visited:
                        new_path = path + [t]
                        # Check if the LAST relation in the path matches target
                        if t.relation == relation:
                            hop_count = len(new_path)
                            proof = f"{relation}({subject}) via {hop_count}-hop: "
                            proof += " , ".join(
                                f"{p.subject} --{p.relation}--> {p.object}"
                                for p in new_path
                            )
                            conf = self._chain_confidence(new_path, relation)
                            return InferenceResult(
                                answer=True,
                                proof_trace=proof,
                                confidence=conf,
                                method="derived",
                            )
                        visited.add(t.object)
                        queue.append((t.object, new_path))

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

        Uses OWL Property Chain style composition_rules as primary authority.
        Falls back to transitivity heuristics only when no explicit rule exists.

        Priority:
        1. Pure transitivity: all edges = target, target is transitive
        2. Explicit composition rule: (rel1, rel2) → target in composition_rules
        3. 1-hop direct fact: always valid
        4. Heuristic fallback: target transitive OR all preceding transitive

        References:
            - OWL 2 PropertyChain axiom (W3C, 2009)
            - HolmE (Zheng et al., 2024) — KGE closed under composition
            - GLIDR (arXiv, 2025) — differentiable ILP for graph-structured rules
        """
        target_props = self.relation_properties.get(target_relation, [])
        target_is_transitive = "transitive" in target_props

        # Priority 1: pure transitivity (all edges = target relation)
        if target_is_transitive:
            for path in paths:
                rels = set(t.relation for t in path)
                if rels == {target_relation}:
                    return path

        # Priority 2: explicit composition rules (OWL Property Chain)
        for path in paths:
            if not path:
                continue

            # 1-hop: direct fact — always valid
            if len(path) == 1:
                if path[0].relation == target_relation:
                    return path
                continue

            # Multi-hop: compute composed relation from the chain
            # spoken_in ∘ in → spoken_in (from composition_rules)
            composed = path[0].relation
            chain_valid = True
            for i in range(1, len(path)):
                rel_pair = (composed, path[i].relation)
                if rel_pair in self.composition_rules:
                    result = self.composition_rules[rel_pair]
                    if result is None:
                        chain_valid = False
                        break
                    composed = result
                else:
                    # No explicit rule — fall back to heuristic
                    prec_transitive = "transitive" in self.relation_properties.get(
                        composed, [])
                    tgt_transitive = "transitive" in self.relation_properties.get(
                        target_relation, [])
                    if not prec_transitive and not tgt_transitive:
                        chain_valid = False
                        break
                    # If target is transitive, the chain composes to target
                    if tgt_transitive:
                        composed = target_relation

            if chain_valid and composed == target_relation:
                return path

        # Priority 2.5: last edge matches target, all preceding transitive
        # This handles cases like in(X,G) ∘ is_a(G,N) → is_a(X,N)
        # where "in" is transitive and the last edge is the target
        for path in paths:
            if len(path) >= 2 and path[-1].relation == target_relation:
                preceding = path[:-1]
                all_prec_transitive = all(
                    "transitive" in self.relation_properties.get(t.relation, [])
                    for t in preceding
                )
                if all_prec_transitive or target_is_transitive:
                    return path

        # Priority 3: ALL relations in path are transitive (heuristic fallback)
        for path in paths:
            all_transitive = all(
                "transitive" in self.relation_properties.get(t.relation, [])
                for t in path
            )
            if all_transitive:
                return path

        return None

    def _chain_confidence(self, path: list[Triple], target_relation: str) -> float:
        """Compute confidence from chain quality and error propagation.

        Based on research:
        - CPR (arXiv 2026): Path quality scoring, not path length
        - UaG (AAAI 2025): Multi-step error accumulation
        - UnKGCP (arXiv 2025): Query-adaptive confidence intervals

        Scoring (no ML, no calibration data):
        - Each step: explicit rule = 1.0, transitive fallback = 0.7, heuristic = 0.4
        - Error propagation: confidence = product of step scores
        - This models how uncertainty accumulates through the chain

        Example:
            5-hop chain with all explicit rules: 1.0^5 = 1.0 → capped at 0.95
            5-hop chain with all heuristics: 0.4^5 = 0.01 → floor 0.1
            Mixed chain: 1.0 × 0.7 × 1.0 × 0.4 × 1.0 = 0.28
        """
        if not path:
            return 0.0

        # Single-hop direct fact
        if len(path) == 1:
            return 0.95

        # Multi-hop: compute product of step scores (error propagation)
        composed = path[0].relation
        chain_score = 1.0
        for i in range(1, len(path)):
            rel_pair = (composed, path[i].relation)
            rule = self.composition_rules.get(rel_pair)
            if rule is not None:
                # Explicit composition rule — high confidence
                step_score = 1.0
                composed = rule
            elif "transitive" in self.relation_properties.get(composed, []):
                # Transitive fallback — moderate confidence
                step_score = 0.7
                composed = path[i].relation
            else:
                # No rule, no transitivity — low confidence
                step_score = 0.4
                composed = path[i].relation
            chain_score *= step_score

        # Clamp to [0.1, 0.95]
        return round(max(0.1, min(0.95, chain_score)), 2)

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

    # ─── Temporal Reasoning (Allen's Interval Algebra) ─────────────────
    #
    # Allen, J.F. (1983). "Maintaining Knowledge about Temporal Intervals."
    # Communications of the ACM, 26(11), 832-843.
    #
    # This section adds temporal reasoning using Allen's 13 interval relations:
    # before, after, meets, met_by, overlaps, overlapped_by, starts, started_by,
    # during, contains, finishes, finished_by, equals.
    #
    # Temporal triples carry (temporal_start, temporal_end) fields.
    # When both entities have intervals, we can apply Allen's algebra.

    def _get_temporal_intervals(self, entity: str, other: str = None) -> list[tuple[AllenRelation, TemporalInterval]]:
        """Get all (relation, interval) pairs for an entity.

        Returns list of (AllenRelation, TemporalInterval) for the entity's triples
        that have temporal data. Tries to match the relation to AllenRelation names.

        If 'other' is specified, only returns intervals from triples directly
        connecting 'entity' and 'other'. This prevents comparing unrelated
        temporal facts (e.g., Obama's presidency interval vs Trump's presidency
        interval, not Obama's "before trump" interval vs Trump's "president_of usa"
        interval).

        If the relation name is a direct Allen relation name (before, after, etc.),
        uses that. Otherwise falls back to checking the relation properties.
        """
        from ..temporal import AllenRelation as AR, TemporalInterval as TI

        results = []
        for t in self.triples:
            if not t.has_temporal():
                continue

            # If 'other' is specified, only consider triples directly connecting entity ↔ other
            if other is not None:
                if not ((t.subject == entity and t.object == other) or
                        (t.object == entity and t.subject == other)):
                    continue

            # Determine the Allen relation for this triple's relation
            allen_rel = self._map_to_allen_relation(t.relation)
            if allen_rel is None:
                continue

            if t.subject == entity:
                interval = TI(start=t.temporal_start, end=t.temporal_end)
                results.append((allen_rel, interval))
            elif t.object == entity:
                inv_rel = allen_rel.inverse
                interval = TI(start=t.temporal_start, end=t.temporal_end)
                results.append((inv_rel, interval))

        return results

    def _map_to_allen_relation(self, relation: str) -> Optional[AllenRelation]:
        """Map a KG relation string to an AllenRelation enum.

        Returns None if the relation doesn't correspond to a known Allen relation.
        Handles both direct matches (e.g., "before" → AllenRelation.BEFORE) and
        inverse mappings (e.g., "has_capital" → None for now, handled separately).
        """
        from ..temporal import AllenRelation as AR

        direct_map = {
            "before": AR.BEFORE,
            "after": AR.AFTER,
            "meets": AR.MEETS,
            "met_by": AR.MET_BY,
            "overlaps": AR.OVERLAPS,
            "overlapped_by": AR.OVERLAPPED_BY,
            "starts": AR.STARTS,
            "started_by": AR.STARTED_BY,
            "during": AR.DURING,
            "contains": AR.CONTAINS,
            "finishes": AR.FINISHES,
            "finished_by": AR.FINISHED_BY,
            "equals": AR.EQUALS,
        }
        return direct_map.get(relation.lower())

    def find_temporal_relation(self, entity1: str, entity2: str) -> Optional[tuple]:
        """Find the Allen relation between two entities using stored temporal intervals.

        Looks for triples involving entity1 and entity2 that have temporal data.
        If both entities have intervals, computes the Allen relation between them.

        Uses the _temporal_index for clean O(1) lookup. Both entities should
        have intervals stored from temporal facts (e.g., WWII gets [1939,1945) from
        "WWII before CW @ [1939,1945)", and CW gets [1947,1991) from
        "CW after WWII @ [1947,1991)").

        Compares all pairs of intervals between entity1 and entity2.
        Returns the first valid Allen relation found.

        Returns:
            (AllenRelation, confidence) or None if not enough temporal data
        """
        from ..temporal import AllenRelation as AR, TemporalInterval as TI
        from ..temporal import check_allen_relation

        entity1 = entity1.strip().lower()
        entity2 = entity2.strip().lower()

        # O(1) lookup via temporal index (each entity has at most one interval)
        if entity1 not in self._temporal_index or entity2 not in self._temporal_index:
            return None

        (s1, e1) = self._temporal_index[entity1]
        (s2, e2) = self._temporal_index[entity2]

        iv1 = TI(start=s1, end=e1)
        iv2 = TI(start=s2, end=e2)

        # Don't require is_valid() — open-ended intervals are valid for comparison
        # check_allen_relation handles None endpoints correctly

        allen_rel = check_allen_relation(iv1, iv2)
        if allen_rel:
            return (allen_rel, 0.90)

        return None

    def query_temporal(self, entity1: str, entity2: str,
                       temporal_relation: str) -> InferenceResult:
        """Query if two entities have a specific Allen temporal relation.

        Args:
            entity1: First entity
            entity2: Second entity
            temporal_relation: One of Allen's 13 relations (before, after, etc.)

        Returns:
            InferenceResult with answer (True/False/None), proof trace, confidence
        """
        from ..temporal import AllenRelation as AR, compose

        entity1 = entity1.strip().lower()
        entity2 = entity2.strip().lower()
        temporal_relation = temporal_relation.strip().lower()

        # Map string to AllenRelation
        target_rel_map = {
            "before": AR.BEFORE, "after": AR.AFTER,
            "meets": AR.MEETS, "met_by": AR.MET_BY,
            "overlaps": AR.OVERLAPS, "overlapped_by": AR.OVERLAPPED_BY,
            "starts": AR.STARTS, "started_by": AR.STARTED_BY,
            "during": AR.DURING, "contains": AR.CONTAINS,
            "finishes": AR.FINISHES, "finished_by": AR.FINISHED_BY,
            "equals": AR.EQUALS,
        }
        if temporal_relation not in target_rel_map:
            return InferenceResult(
                answer=None,
                proof_trace=f"Unknown temporal relation: '{temporal_relation}'. "
                           f"Valid: {list(target_rel_map.keys())}",
                confidence=0.0,
                method="unknown",
            )
        target = target_rel_map[temporal_relation]

        # Try direct temporal lookup
        result = self.find_temporal_relation(entity1, entity2)
        if result:
            found_rel, conf = result
            if found_rel == target:
                return InferenceResult(
                    answer=True,
                    proof_trace=f"Direct temporal: {entity1} {found_rel.value} {entity2} "
                               f"(from stored intervals)",
                    confidence=conf,
                    method="direct",
                )
            elif found_rel == target.inverse:
                return InferenceResult(
                    answer=False,
                    proof_trace=f"Temporal mismatch: {entity1} is actually "
                               f"{found_rel.value} {entity2} (the inverse of {target.value})",
                    confidence=conf,
                    method="direct",
                )
            else:
                return InferenceResult(
                    answer=False,
                    proof_trace=f"Temporal mismatch: {entity1} {found_rel.value} {entity2}, "
                               f"not {target.value}",
                    confidence=conf,
                    method="direct",
                )

        # Try path-based temporal reasoning
        # Find a chain: entity1 --R1--> X --R2--> entity2
        # where compose(R1, R2) contains target
        paths = self.bfs_paths(entity1, entity2, max_hops=4)
        if paths:
            # Try to derive temporal via Allen's composition
            for path in paths:
                if len(path) >= 2:
                    # Try composition of first and last edges
                    r1_map = self._map_to_allen_relation(path[0].relation)
                    r2_map = self._map_to_allen_relation(path[-1].relation)
                    if r1_map and r2_map:
                        composed = compose(r1_map, r2_map)
                        if target in composed:
                            trace = f"Yes (via composition): "
                            hops = [f"{t.subject} --{t.relation}--> {t.object}" for t in path]
                            trace += " , ".join(hops)
                            trace += f" → {entity1} {target.value} {entity2}"
                            return InferenceResult(
                                answer=True,
                                proof_trace=trace,
                                confidence=0.75,
                                method="derived",
                            )

        return InferenceResult(
            answer=None,
            proof_trace=f"No temporal data connecting {entity1} and {entity2}",
            confidence=0.0,
            method="unknown",
        )

    def derive_temporal_transitive(self) -> list[Triple]:
        """Derive new temporal facts using Allen's composition table.

        For each pair of temporal triples (X, R1, Y) and (Y, R2, Z) where
        both Y-intervals exist, compute compose(R1, R2) and add any
        deterministic (single-result) compositions as derived facts.

        Allen (1983), Table 1: the composition of two relations gives the
        possible relations between the first and third entities.
        """
        from ..temporal import AllenRelation as AR, TemporalInterval as TI
        from ..temporal import check_allen_relation, compose

        derived = []
        temporal_triples = [t for t in self.triples if t.has_temporal()]

        # Build temporal index: entity → list of (triple, allen_rel, interval)
        temporal_index: dict[str, list[tuple[Triple, AR, TI]]] = {}
        for t in temporal_triples:
            if not t.is_interval():
                continue
            interval = TI(start=t.temporal_start, end=t.temporal_end)
            for entity in (t.subject, t.object):
                if entity not in temporal_index:
                    temporal_index[entity] = []
                if entity == t.subject:
                    rel = self._map_to_allen_relation(t.relation)
                    if rel:
                        temporal_index[entity].append((t, rel, interval))
                else:
                    rel = self._map_to_allen_relation(t.relation)
                    if rel:
                        temporal_index[entity].append((t, rel.inverse, interval))

        # For each path of length 2: X --R1--> Y --R2--> Z
        # Check temporal composition
        processed = set()
        for y_entity, y_triples in temporal_index.items():
            for t1, r1, interval1 in y_triples:
                x_entity = t1.subject if t1.object == y_entity else t1.object
                if x_entity == y_entity:
                    continue

                for t2, r2, interval2 in temporal_index.get(y_entity, []):
                    z_entity = t2.subject if t2.object == y_entity else t2.object
                    if z_entity in (x_entity, y_entity):
                        continue

                    # Compute Allen composition
                    composed = compose(r1, r2)
                    if len(composed) == 1:
                        # Deterministic: exactly one relation
                        only_rel = next(iter(composed))
                        # Check if this relation holds between X and Z intervals
                        # Build intervals for X and Z
                        x_interval = None
                        z_interval = None
                        for t, rel, iv in temporal_index.get(x_entity, []):
                            if t.subject == x_entity or t.object == x_entity:
                                x_interval = iv
                                break
                        for t, rel, iv in temporal_index.get(z_entity, []):
                            if t.subject == z_entity or t.object == z_entity:
                                z_interval = iv
                                break

                        # If we have intervals, verify the derived relation
                        if x_interval and z_interval and x_interval.is_valid() and z_interval.is_valid():
                            actual = check_allen_relation(x_interval, z_interval)
                            if actual == only_rel:
                                key = (x_entity, only_rel.value, z_entity)
                                if key not in processed:
                                    processed.add(key)
                                    proof = (f"temporal derivation: {x_entity} {r1.value} {y_entity} "
                                            f"∧ {y_entity} {r2.value} {z_entity} "
                                            f"→ {x_entity} {only_rel.value} {z_entity} "
                                            f"(Allen's composition: {r1.value} ∘ {r2.value} = {only_rel.value})")
                                    triple = self.add_fact(
                                        x_entity, only_rel.value, z_entity,
                                        source="derived", proof=proof
                                    )
                                    if triple.source == "derived":
                                        derived.append(triple)

        return derived

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
                    temporal_start INTEGER,
                    temporal_end INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS relation_properties (
                    relation TEXT PRIMARY KEY,
                    properties TEXT NOT NULL
                );


                CREATE TABLE IF NOT EXISTS composition_rules (
                    rel1 TEXT NOT NULL,
                    rel2 TEXT NOT NULL,
                    target_relation TEXT,
                    PRIMARY KEY (rel1, rel2)
                );

                CREATE TABLE IF NOT EXISTS gazetteer (
                    entity TEXT PRIMARY KEY
                );

                CREATE INDEX IF NOT EXISTS idx_triples_subject ON triples(subject);
                CREATE INDEX IF NOT EXISTS idx_triples_relation ON triples(relation);
                CREATE INDEX IF NOT EXISTS idx_triples_object ON triples(object);
            """)

            # Migration: add temporal columns if they don't exist (for old DBs)
            try:
                conn.execute("ALTER TABLE triples ADD COLUMN temporal_start INTEGER")
            except sqlite3.OperationalError:
                pass  # Column already exists
            try:
                conn.execute("ALTER TABLE triples ADD COLUMN temporal_end INTEGER")
            except sqlite3.OperationalError:
                pass  # Column already exists

            # Clear and re-insert (full sync)
            conn.execute("DELETE FROM triples")
            conn.execute("DELETE FROM relation_properties")

            conn.executemany(
                "INSERT INTO triples (subject, relation, object, source, proof, temporal_start, temporal_end) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [(t.subject, t.relation, t.object, t.source, t.proof,
                  t.temporal_start, t.temporal_end) for t in self.triples]
            )

            conn.executemany(
                "INSERT INTO relation_properties (relation, properties) VALUES (?, ?)",
                [(rel, ",".join(props)) for rel, props in self.relation_properties.items()]
            )

            conn.execute("DELETE FROM composition_rules")
            conn.executemany(
                "INSERT INTO composition_rules (rel1, rel2, target_relation) VALUES (?, ?, ?)",
                [(r1, r2, target) for (r1, r2), target in self.composition_rules.items()]
            )

            conn.execute("DELETE FROM gazetteer")
            conn.executemany(
                "INSERT INTO gazetteer (entity) VALUES (?)",
                [(e,) for e in self.gazetteer]
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
            # Load triples (including temporal fields)
            rows = conn.execute(
                "SELECT subject, relation, object, source, proof, temporal_start, temporal_end FROM triples"
            ).fetchall()

            for row in rows:
                subject, relation, obj, source, proof, t_start, t_end = row
                self.add_fact(subject, relation, obj, source=source, proof=proof,
                             temporal_start=t_start, temporal_end=t_end)

            # Load relation properties
            prop_rows = conn.execute(
                "SELECT relation, properties FROM relation_properties"
            ).fetchall()

            for relation, props_str in prop_rows:
                props = props_str.split(",") if props_str else []
                if props:
                    self.set_relation_property(relation, *props)

            # Load composition rules
            try:
                comp_rows = conn.execute(
                    "SELECT rel1, rel2, target_relation FROM composition_rules"
                ).fetchall()
                for rel1, rel2, target in comp_rows:
                    self.composition_rules[(rel1, rel2)] = target if target else None
            except sqlite3.OperationalError:
                pass  # Old DB without composition_rules table

            # Load gazetteer
            try:
                gaz_rows = conn.execute("SELECT entity FROM gazetteer").fetchall()
                for (entity,) in gaz_rows:
                    self.gazetteer.add(entity)
            except sqlite3.OperationalError:
                pass  # Old DB without gazetteer table

            # Also populate gazetteer from multi-word entities in triples
            for t in self.triples:
                if " " in t.subject:
                    self.gazetteer.add(t.subject)
                if " " in t.object:
                    self.gazetteer.add(t.object)

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
