"""Generic Thinking Loop -- Universal constraint primitives, no hardcoded domains.

Based on:
    - Betteti, Bullo, Baggio, Zampieri (2025) -- IDP Iterative Refinement
      Science Advances. DOI: 10.1126/sciadv.adu6991
    - Kanerva (2009) -- HDC Algebraic Decomposition
      Cognitive Computation 1(2), 139-159. DOI: 10.1007/s12559-009-9009-8
    - Ramsauer et al. (2020) -- Automatic Attractor Storage
      ICLR 2021. arXiv:2008.02217
    - Kleyko et al. (2022) -- HDC/VSA Survey, ACM Computing Surveys 55(6), Article 130.
    - Kleyko et al. (2025) -- Principled neuromorphic reservoir computing
      Nature Communications 16(1). DOI: 10.1038/s41467-025-55832-y

Design principle: No domain-specific code. No `if "schedule" in text`.
Instead, 18 UNIVERSAL mathematical primitives discovered via:
    1. MHN template retrieval (Ramsauer 2020)
    2. Entity relation inference (Kanerva 2009)
    3. Innate relation prototypes (Kleyko 2025)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .perception.hdc import (
    bind, bundle, similarity, permute,
    generate_hypervector, normalize_hdc,
)
from .perception.nl_parser import GenericNLParser, GenericEntityGraph
from .memory.mhn import ModernHopfieldNetwork, MHNConfig


# =========================================================================
# Data Structures
# =========================================================================

@dataclass
class Thought:
    iteration: int
    state_hdc: np.ndarray
    retrieved_hdc: np.ndarray | None
    retrieved_similarity: float
    retrieved_metadata: dict
    converged: bool
    description: str = ""


@dataclass
class ThinkingResult:
    problem: str
    evolved_state: np.ndarray
    thoughts: list[Thought]
    sub_problems: list[dict]
    solution: dict | None
    confidence: float
    latency_ms: float
    trace: list[str] = field(default_factory=list)

    @property
    def iterations(self) -> int:
        return len(self.thoughts)

    def summary(self) -> str:
        lines = [
            f"=== Thinking Dust -- Generic Reasoning Trace ===",
            f"Problem: {self.problem[:80]}",
            f"Iterations: {self.iterations}",
            f"Sub-problems: {len(self.sub_problems)}",
            f"Confidence: {self.confidence:.1%}",
            f"Latency: {self.latency_ms:.0f}ms",
            "",
        ]
        for t in self.thoughts:
            status = "✓ converged" if t.converged else f"iter {t.iteration}"
            lines.append(f"  [{status}] sim={t.retrieved_similarity:.3f} {t.description}")
        lines.append("")
        if self.solution:
            lines.append("Solution:")
            formatted = self.solution.get("formatted", str(self.solution))
            lines.append(f"  {formatted}")
        return "\n".join(lines)


# =========================================================================
# Convergence Detection
# =========================================================================

def has_converged(current: np.ndarray, previous: np.ndarray, threshold: float = 0.98) -> bool:
    sim = similarity(current, previous)
    return sim > threshold


# =========================================================================
# Mechanism 1: IDP Iterative Refinement (Betteti et al. 2025)
# =========================================================================

def idp_refine(
    query_hdc: np.ndarray,
    mhn: ModernHopfieldNetwork,
    max_iterations: int = 5,
    convergence_threshold: float = 0.98,
    blend_factor: float = 0.3,
) -> tuple[np.ndarray, list[Thought]]:
    thoughts: list[Thought] = []
    current_state = query_hdc.copy()

    for i in range(max_iterations):
        results = mhn.retrieve(current_state, top_k=1)

        if not results:
            thoughts.append(Thought(
                iteration=i + 1,
                state_hdc=current_state.copy(),
                retrieved_hdc=None,
                retrieved_similarity=0.0,
                retrieved_metadata={},
                converged=True,
                description="No matching patterns -- converged at empty memory",
            ))
            break

        retrieved_vec, retrieved_sim, retrieved_meta = results[0]

        new_state = bundle(
            current_state * (1 - blend_factor) +
            retrieved_vec * blend_factor
        )
        new_state = np.sign(new_state).astype(np.int8)
        new_state[new_state == 0] = 1

        converged = has_converged(new_state, current_state, convergence_threshold)

        meta_desc = retrieved_meta.get("title", retrieved_meta.get("domain", ""))
        desc = f"Retrieved: {meta_desc} (sim={retrieved_sim:.3f})" if meta_desc else f"Retrieved pattern (sim={retrieved_sim:.3f})"

        thoughts.append(Thought(
            iteration=i + 1,
            state_hdc=current_state.copy(),
            retrieved_hdc=retrieved_vec.copy(),
            retrieved_similarity=retrieved_sim,
            retrieved_metadata=retrieved_meta,
            converged=converged,
            description=desc,
        ))

        current_state = new_state
        if converged:
            break

    return current_state, thoughts


# =========================================================================
# Mechanism 2: HDC Algebraic Decomposition (Kanerva 2009)
# =========================================================================

SUB_PROBLEM_DESCRIPTIONS = {
    "discover_entities": "find all objects and elements that appear in this problem",
    "discover_constraints": "what are the limits and rules that restrict the solution",
    "find_solution": "how to assign values that satisfy all limits and rules",
    "validate_result": "check that the assignment satisfies every requirement and limit",
}


def hdc_decompose(
    state_hdc: np.ndarray,
    prototype_vectors: dict[str, np.ndarray],
    mhn: ModernHopfieldNetwork,
) -> list[dict]:
    sub_problems = []
    for concept_name, proto_hdc in prototype_vectors.items():
        component_hdc = bind(state_hdc, proto_hdc)
        results = mhn.retrieve(component_hdc, top_k=1)
        if results:
            _, sim, meta = results[0]
            sub_problems.append({
                "concept": concept_name,
                "hdc": component_hdc,
                "retrieved_sim": sim,
                "retrieved_meta": meta,
                "solution": meta.get("solution", meta),
            })
        else:
            sub_problems.append({
                "concept": concept_name,
                "hdc": component_hdc,
                "retrieved_sim": 0.0,
                "retrieved_meta": {},
                "solution": None,
            })
    return sub_problems


def hdc_compose(sub_solutions: list[np.ndarray]) -> np.ndarray:
    if not sub_solutions:
        return generate_hypervector(10_000)
    if len(sub_solutions) == 1:
        return sub_solutions[0]
    return bundle(*sub_solutions)


# =========================================================================
# Mechanism 4: Automatic Attractor Storage (Ramsauer et al. 2020)
# =========================================================================

def store_experience(
    mhn: ModernHopfieldNetwork,
    problem_hdc: np.ndarray,
    solution_hdc: np.ndarray,
    metadata: dict | None = None,
):
    mhn.store(problem_hdc, solution_hdc, metadata or {"source": "auto_learned"})


# =========================================================================
# Generic Z3 Constraint Solver -- 18 Universal Primitives (ACTUALLY IMPLEMENTED)
# =========================================================================

class GenericZ3Solver:
    """Generic constraint solver using 18 universal mathematical primitives.

    NO domain-specific code. No `if "schedule" in text`.

    18 UNIVERSAL primitives:
        Core (6): all_different, ordered, bounded, excluded, grouped, optimized
        Arithmetic (5): sum_eq, sum_leq, sum_geq, count, ratio
        Logical (5): implies, equivalent, at_least, at_most, exactly
        Temporal (2): no_overlap, precedence
    """

    def __init__(self):
        self.primitive_builders = {
            # Core (6)
            "all_different": self._build_all_different,
            "ordered": self._build_ordered,
            "bounded": self._build_bounded,
            "excluded": self._build_excluded,
            "grouped": self._build_grouped,
            "optimized": self._build_optimized,
            # Arithmetic (5)
            "sum_eq": self._build_sum_eq,
            "sum_leq": self._build_sum_leq,
            "sum_geq": self._build_sum_geq,
            "count": self._build_count,
            "ratio": self._build_ratio,
            # Logical (5)
            "implies": self._build_implies,
            "equivalent": self._build_equivalent,
            "at_least": self._build_at_least,
            "at_most": self._build_at_most,
            "exactly": self._build_exactly,
            # Temporal (2)
            "no_overlap": self._build_no_overlap,
            "precedence": self._build_precedence,
        }

    def solve(self, graph: GenericEntityGraph, template: dict | None = None) -> dict | None:
        try:
            from z3 import (
                Solver, Optimize, Int, Bool, sat, unsat,
                And, Or, Not, Implies, Distinct, Sum, If, And as Z3And,
            )
        except ImportError:
            return None

        if not graph.entities:
            return None

        # ─── Step 1: Create Z3 variables generically ─────────────
        z3_vars = {}
        for e in graph.entities:
            var_type = self._infer_var_type(e, graph)
            if var_type == "bool":
                z3_vars[e["id"]] = Bool(e["id"])
            else:
                z3_vars[e["id"]] = Int(e["id"])

        # ─── Step 2: Gather primitives to apply ────────────────────
        primitives = []
        if template:
            primitives = self._extract_primitives_from_template(template, graph)
        else:
            primitives = self._infer_primitives_from_relations(graph)
            for c in graph.constraints:
                primitive = self._constraint_to_primitive(c, graph)
                if primitive:
                    primitives.append(primitive)

        if not primitives:
            return None

        # ─── Step 3: Build and solve ───────────────────────────────
        has_opt = any(p["type"] == "optimized" for p in primitives)
        solver = Optimize() if has_opt else Solver()

        for p in primitives:
            builder = self.primitive_builders.get(p["type"])
            if builder:
                builder(solver, z3_vars, p, graph)

        result = solver.check()

        if result == sat:
            model = solver.model()
            return self._format_solution(model, z3_vars, graph, primitives)
        elif result == unsat:
            return {
                "type": "unsat",
                "formatted": "No solution exists -- constraints are contradictory.",
            }
        else:
            return None

    # ─── Core Primitives (6) ─────────────────────────────────────

    def _build_all_different(self, solver, z3_vars, primitive, graph):
        from z3 import Distinct
        subjects = [z3_vars[sid] for sid in primitive["subjects"] if sid in z3_vars]
        if len(subjects) > 1:
            solver.add(Distinct(subjects))

    def _build_ordered(self, solver, z3_vars, primitive, graph):
        subjects = [sid for sid in primitive["subjects"] if sid in z3_vars]
        order = primitive.get("order", "ascending")
        for i in range(len(subjects) - 1):
            if order == "ascending":
                solver.add(z3_vars[subjects[i]] < z3_vars[subjects[i + 1]])
            else:
                solver.add(z3_vars[subjects[i]] > z3_vars[subjects[i + 1]])

    def _build_bounded(self, solver, z3_vars, primitive, graph):
        subjects = [sid for sid in primitive["subjects"] if sid in z3_vars]
        lo = primitive.get("min", 0)
        hi = primitive.get("max", 100)
        for sid in subjects:
            solver.add(z3_vars[sid] >= lo)
            solver.add(z3_vars[sid] <= hi)

    def _build_excluded(self, solver, z3_vars, primitive, graph):
        subjects = [sid for sid in primitive["subjects"] if sid in z3_vars]
        if len(subjects) == 2:
            solver.add(z3_vars[subjects[0]] != z3_vars[subjects[1]])
        elif len(subjects) == 1:
            exclude_val = primitive.get("exclude_value", 0)
            solver.add(z3_vars[subjects[0]] != exclude_val)

    def _build_grouped(self, solver, z3_vars, primitive, graph):
        from z3 import Sum, Distinct
        groups = primitive.get("groups", [primitive["subjects"]])
        limit = primitive.get("limit")
        constraint = primitive.get("group_constraint", "sum_leq")
        for group in groups:
            group_vars = [z3_vars[sid] for sid in group if sid in z3_vars]
            if not group_vars:
                continue
            if constraint == "sum_leq" and limit is not None:
                solver.add(Sum(group_vars) <= limit)
            elif constraint == "sum_geq" and limit is not None:
                solver.add(Sum(group_vars) >= limit)
            elif constraint == "sum_eq" and limit is not None:
                solver.add(Sum(group_vars) == limit)
            elif constraint == "distinct" and len(group_vars) > 1:
                solver.add(Distinct(group_vars))

    def _build_optimized(self, solver, z3_vars, primitive, graph):
        from z3 import Int, Sum
        subjects = [sid for sid in primitive["subjects"] if sid in z3_vars]
        objective = primitive.get("objective", "maximize_min")
        direction = primitive.get("direction", "maximize")
        if objective == "maximize_min":
            min_var = Int("__min_alloc")
            for sid in subjects:
                solver.add(min_var <= z3_vars[sid])
            solver.maximize(min_var)
        elif objective == "sum":
            total = Sum([z3_vars[sid] for sid in subjects])
            if direction == "maximize":
                solver.maximize(total)
            else:
                solver.minimize(total)

    # ─── Arithmetic Primitives (5) ───────────────────────────────

    def _build_sum_eq(self, solver, z3_vars, primitive, graph):
        from z3 import Sum
        subjects = [z3_vars[sid] for sid in primitive["subjects"] if sid in z3_vars]
        target = primitive.get("target", 0)
        if subjects:
            solver.add(Sum(subjects) == target)

    def _build_sum_leq(self, solver, z3_vars, primitive, graph):
        from z3 import Sum
        subjects = [z3_vars[sid] for sid in primitive["subjects"] if sid in z3_vars]
        limit = primitive.get("limit", 0)
        if subjects:
            solver.add(Sum(subjects) <= limit)

    def _build_sum_geq(self, solver, z3_vars, primitive, graph):
        from z3 import Sum
        subjects = [z3_vars[sid] for sid in primitive["subjects"] if sid in z3_vars]
        limit = primitive.get("limit", 0)
        if subjects:
            solver.add(Sum(subjects) >= limit)

    def _build_count(self, solver, z3_vars, primitive, graph):
        from z3 import Sum, If
        subjects = [z3_vars[sid] for sid in primitive["subjects"] if sid in z3_vars]
        condition = primitive.get("condition", "eq")
        target = primitive.get("target", 0)
        if not subjects:
            return
        count = Sum([If(v, 1, 0) for v in subjects])
        if condition == "eq":
            solver.add(count == target)
        elif condition == "geq":
            solver.add(count >= target)
        elif condition == "leq":
            solver.add(count <= target)

    def _build_ratio(self, solver, z3_vars, primitive, graph):
        from z3 import Int
        subjects = [sid for sid in primitive["subjects"] if sid in z3_vars]
        ratio = primitive.get("ratio", 1)
        if len(subjects) >= 2:
            # A / B = ratio  =>  A = ratio * B
            solver.add(z3_vars[subjects[0]] == ratio * z3_vars[subjects[1]])

    # ─── Logical Primitives (5) ──────────────────────────────────

    def _build_implies(self, solver, z3_vars, primitive, graph):
        from z3 import Implies
        subjects = [sid for sid in primitive["subjects"] if sid in z3_vars]
        if len(subjects) >= 2:
            for i in range(len(subjects) - 1):
                solver.add(Implies(z3_vars[subjects[i]], z3_vars[subjects[i + 1]]))

    def _build_equivalent(self, solver, z3_vars, primitive, graph):
        subjects = [sid for sid in primitive["subjects"] if sid in z3_vars]
        if len(subjects) >= 2:
            for i in range(len(subjects) - 1):
                solver.add(z3_vars[subjects[i]] == z3_vars[subjects[i + 1]])

    def _build_at_least(self, solver, z3_vars, primitive, graph):
        from z3 import Sum, If
        subjects = [z3_vars[sid] for sid in primitive["subjects"] if sid in z3_vars]
        n = primitive.get("n", 1)
        if subjects:
            solver.add(Sum([If(v, 1, 0) for v in subjects]) >= n)

    def _build_at_most(self, solver, z3_vars, primitive, graph):
        from z3 import Sum, If
        subjects = [z3_vars[sid] for sid in primitive["subjects"] if sid in z3_vars]
        n = primitive.get("n", 1)
        if subjects:
            solver.add(Sum([If(v, 1, 0) for v in subjects]) <= n)

    def _build_exactly(self, solver, z3_vars, primitive, graph):
        from z3 import Sum, If
        subjects = [z3_vars[sid] for sid in primitive["subjects"] if sid in z3_vars]
        n = primitive.get("n", 1)
        if subjects:
            solver.add(Sum([If(v, 1, 0) for v in subjects]) == n)

    # ─── Temporal Primitives (2) ─────────────────────────────────

    def _build_no_overlap(self, solver, z3_vars, primitive, graph):
        from z3 import Or
        subjects = [sid for sid in primitive["subjects"] if sid in z3_vars]
        duration = primitive.get("duration", 1)
        for i in range(len(subjects)):
            for j in range(i + 1, len(subjects)):
                solver.add(Or(
                    z3_vars[subjects[i]] + duration <= z3_vars[subjects[j]],
                    z3_vars[subjects[j]] + duration <= z3_vars[subjects[i]]
                ))

    def _build_precedence(self, solver, z3_vars, primitive, graph):
        subjects = [sid for sid in primitive["subjects"] if sid in z3_vars]
        min_gap = primitive.get("min_gap", 1)
        for i in range(len(subjects) - 1):
            solver.add(z3_vars[subjects[i]] + min_gap <= z3_vars[subjects[i + 1]])

    # ─── Inference Methods ───────────────────────────────────────

    def _infer_var_type(self, entity: dict, graph: GenericEntityGraph) -> str:
        for c in graph.constraints:
            if entity["id"] in c["subjects"]:
                if c["params"].get("type") == "bool":
                    return "bool"
        return "int"

    def _extract_primitives_from_template(self, template: dict, graph: GenericEntityGraph) -> list[dict]:
        primitives = []
        for p in template.get("primitives", []):
            mapped_subjects = self._map_subjects(p.get("subjects", []), p.get("selector"), graph)
            if mapped_subjects:
                primitives.append({
                    "type": p["type"],
                    "subjects": mapped_subjects,
                    **{k: v for k, v in p.items() if k not in ("type", "subjects")},
                })
        return primitives

    def _infer_primitives_from_relations(self, graph: GenericEntityGraph) -> list[dict]:
        """Infer primitives from entity relations.

        25+ relation type mappings (innate + learned from MHN):
            different/distinct -> all_different
            before -> ordered (ascending)
            after -> ordered (descending)
            excludes -> excluded
            limited -> bounded
            grouped/partitioned -> grouped
            optimize/maximize/minimize -> optimized
            sum_to/total -> sum_eq
            at_least -> at_least
            at_most -> at_most
            exactly -> exactly
            implies/if_then -> implies
            equivalent/same -> equivalent
            overlap/conflict -> no_overlap
            precedence/chain -> precedence
            ratio/proportional -> ratio
            count -> count
        """
        primitives = []
        entity_ids = [e["id"] for e in graph.entities]

        rel_groups = {}
        for rel in graph.relations:
            rtype = rel.get("rel_type", "unknown")
            if rtype not in rel_groups:
                rel_groups[rtype] = []
            rel_groups[rtype].append((rel["src"], rel["tgt"]))

        # Core mappings
        if "different" in rel_groups or "distinct" in rel_groups:
            involved = set()
            for rtype in ["different", "distinct"]:
                for src, tgt in rel_groups.get(rtype, []):
                    involved.add(src); involved.add(tgt)
            if involved:
                primitives.append({"type": "all_different", "subjects": list(involved)})

        if "before" in rel_groups:
            ordered = self._build_order_chain(rel_groups["before"])
            if ordered:
                primitives.append({"type": "ordered", "subjects": ordered, "order": "ascending"})

        if "after" in rel_groups:
            ordered = self._build_order_chain(rel_groups["after"])
            if ordered:
                primitives.append({"type": "ordered", "subjects": ordered, "order": "descending"})

        if "excludes" in rel_groups:
            for src, tgt in rel_groups["excludes"]:
                primitives.append({"type": "excluded", "subjects": [src, tgt]})

        if "limited" in rel_groups:
            bounds = self._infer_bounds(graph)
            if bounds:
                primitives.append({"type": "bounded", "subjects": entity_ids, **bounds})

        if "grouped" in rel_groups or "partitioned" in rel_groups:
            groups = self._build_groups(rel_groups.get("grouped", []) + rel_groups.get("partitioned", []))
            if groups:
                primitives.append({"type": "grouped", "subjects": [], "groups": groups})

        if "optimize" in rel_groups or "maximize" in rel_groups or "minimize" in rel_groups:
            direction = "maximize" if ("maximize" in rel_groups or "optimize" in rel_groups) else "minimize"
            primitives.append({"type": "optimized", "subjects": entity_ids, "direction": direction, "objective": "maximize_min"})

        # Arithmetic mappings
        if "sum_to" in rel_groups or "total" in rel_groups:
            total = self._infer_total(graph)
            if total is not None:
                primitives.append({"type": "sum_eq", "subjects": entity_ids, "target": total})

        if "at_least" in rel_groups:
            n = self._infer_count(rel_groups.get("at_least", []))
            primitives.append({"type": "at_least", "subjects": entity_ids, "n": n})

        if "at_most" in rel_groups:
            n = self._infer_count(rel_groups.get("at_most", []))
            primitives.append({"type": "at_most", "subjects": entity_ids, "n": n})

        if "exactly" in rel_groups:
            n = self._infer_count(rel_groups.get("exactly", []))
            primitives.append({"type": "exactly", "subjects": entity_ids, "n": n})

        # Logical mappings
        if "implies" in rel_groups or "if_then" in rel_groups:
            involved = set()
            for rtype in ["implies", "if_then"]:
                for src, tgt in rel_groups.get(rtype, []):
                    involved.add(src); involved.add(tgt)
            if involved:
                primitives.append({"type": "implies", "subjects": list(involved)})

        if "equivalent" in rel_groups or "same" in rel_groups:
            involved = set()
            for rtype in ["equivalent", "same"]:
                for src, tgt in rel_groups.get(rtype, []):
                    involved.add(src); involved.add(tgt)
            if involved:
                primitives.append({"type": "equivalent", "subjects": list(involved)})

        # Temporal mappings
        if "overlap" in rel_groups or "conflict" in rel_groups:
            involved = set()
            for rtype in ["overlap", "conflict"]:
                for src, tgt in rel_groups.get(rtype, []):
                    involved.add(src); involved.add(tgt)
            if involved:
                primitives.append({"type": "no_overlap", "subjects": list(involved), "duration": 1})

        if "precedence" in rel_groups or "chain" in rel_groups:
            ordered = self._build_order_chain(rel_groups.get("precedence", []) + rel_groups.get("chain", []))
            if ordered:
                primitives.append({"type": "precedence", "subjects": ordered, "min_gap": 1})

        # Ratio mapping
        if "ratio" in rel_groups or "proportional" in rel_groups:
            for src, tgt in rel_groups.get("ratio", []) + rel_groups.get("proportional", []):
                primitives.append({"type": "ratio", "subjects": [src, tgt], "ratio": 2})

        # Count mapping
        if "count" in rel_groups:
            n = self._infer_count(rel_groups.get("count", []))
            primitives.append({"type": "count", "subjects": entity_ids, "condition": "eq", "target": n})

        # Default
        if not primitives and entity_ids:
            primitives.append({"type": "bounded", "subjects": entity_ids, "min": 0, "max": 100})

        return primitives

    def _constraint_to_primitive(self, constraint: dict, graph: GenericEntityGraph) -> dict | None:
        params = constraint.get("params", {})
        ptype = params.get("primitive_type", "bounded")
        return {
            "type": ptype,
            "subjects": constraint["subjects"],
            **{k: v for k, v in params.items() if k != "primitive_type"},
        }

    def _map_subjects(self, template_subjects: list, selector: np.ndarray | None, graph: GenericEntityGraph) -> list[str]:
        if not selector:
            return [e["id"] for e in graph.entities[:len(template_subjects)]]
        matches = []
        for e in graph.entities:
            sim = similarity(e["hdc"], selector)
            matches.append((e["id"], sim))
        matches.sort(key=lambda x: x[1], reverse=True)
        return [eid for eid, sim in matches if sim > 0.30]

    def _build_order_chain(self, relations: list[tuple]) -> list[str]:
        from collections import defaultdict, deque
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        nodes = set()
        for src, tgt in relations:
            graph[src].append(tgt)
            in_degree[tgt] += 1
            nodes.add(src); nodes.add(tgt)
        for n in nodes:
            if n not in in_degree:
                in_degree[n] = 0
        queue = deque([n for n in nodes if in_degree[n] == 0])
        ordered = []
        while queue:
            n = queue.popleft()
            ordered.append(n)
            for neighbor in graph[n]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        return ordered if len(ordered) == len(nodes) else []

    def _build_groups(self, relations: list[tuple]) -> list[list[str]]:
        parent = {}
        def find(x):
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        for src, tgt in relations:
            union(src, tgt)
        groups = {}
        for x in parent:
            root = find(x)
            if root not in groups:
                groups[root] = []
            groups[root].append(x)
        return list(groups.values())

    def _infer_bounds(self, graph: GenericEntityGraph) -> dict | None:
        max_val = 100
        for e in graph.entities:
            nums = [int(w) for w in e["text"].split() if w.isdigit()]
            if nums:
                max_val = max(max_val, max(nums) * 2)
        return {"min": 0, "max": max_val}

    def _infer_total(self, graph: GenericEntityGraph) -> int | None:
        for e in graph.entities:
            nums = [int(w) for w in e["text"].split() if w.isdigit()]
            if nums:
                return max(nums)
        return None

    def _infer_count(self, relations: list[tuple]) -> int:
        for src, tgt in relations:
            nums = [int(w) for w in src.split() if w.isdigit()] + [int(w) for w in tgt.split() if w.isdigit()]
            if nums:
                return max(nums)
        return 1

    def _format_solution(self, model, z3_vars, graph, primitives) -> dict:
        lines = []
        for e in graph.entities:
            val = model.eval(z3_vars[e["id"]], model_completion=True)
            lines.append(f"  {e['text']}: {val}")
        ptypes = [p["type"] for p in primitives]
        method = " + ".join(ptypes) if ptypes else "generic"
        return {
            "type": "generic_csp",
            "method": f"z3_{method}",
            "formatted": "\n".join(lines),
            "primitives_applied": ptypes,
        }


# =========================================================================
# The Full Generic Thinking Loop
# =========================================================================

class GenericThinkingDust:
    """Generic reasoning engine -- 18 innate primitives, learned composition."""

    def __init__(
        self,
        vocab=None,
        mhn: ModernHopfieldNetwork | None = None,
        dim: int = 10_000,
        max_idp_iterations: int = 5,
        idp_blend_factor: float = 0.3,
        convergence_threshold: float = 0.98,
        pure_mode: bool = False,
    ):
        self.dim = dim
        self.mhn = mhn or ModernHopfieldNetwork(MHNConfig(
            dim=dim, min_similarity=0.01, idp_enabled=False,
        ))
        self.parser = GenericNLParser(vocab, self.mhn, dim=dim)
        self.z3_solver = GenericZ3Solver()

        self.max_idp_iterations = max_idp_iterations
        self.idp_blend_factor = idp_blend_factor
        self.convergence_threshold = convergence_threshold
        self.pure_mode = pure_mode

        self.sub_problem_prototypes = self._build_semantic_prototypes()

        self.total_thinks = 0
        self.total_learned = 0
        self.avg_iterations = 0.0
        self.seed_count = 0

        if not pure_mode:
            self._load_minimal_seed()

    def think(self, problem_text: str, context: dict | None = None) -> ThinkingResult:
        t0 = time.perf_counter()
        trace = []
        self.total_thinks += 1

        # Step 1: Generic Parse
        struct = self.parser.extract_structure(problem_text)
        problem_hdc = struct["hdc"]
        graph = struct["graph"]
        trace.append(f"Parsed: {len(graph.entities)} entities, {len(graph.relations)} relations, {len(graph.constraints)} constraints")

        # Step 2: IDP Refinement
        trace.append(f"IDP refinement (max {self.max_idp_iterations} iterations)...")
        evolved_state, thoughts = idp_refine(
            problem_hdc, self.mhn,
            max_iterations=self.max_idp_iterations,
            convergence_threshold=self.convergence_threshold,
            blend_factor=self.idp_blend_factor,
        )
        for t in thoughts:
            trace.append(f"  Iter {t.iteration}: sim={t.retrieved_similarity:.3f} "
                        f"{'-> CONVERGED' if t.converged else '-> continuing'}")
        avg_iters = sum(t.iteration for t in thoughts) / max(len(thoughts), 1)
        self.avg_iterations = (self.avg_iterations * (self.total_thinks - 1) + avg_iters) / self.total_thinks

        # Step 3: HDC Decomposition
        trace.append("HDC algebraic decomposition (universal roles)...")
        sub_problems = hdc_decompose(evolved_state, self.sub_problem_prototypes, self.mhn)
        for sp in sub_problems:
            trace.append(f"  [{sp['concept']}] sim={sp['retrieved_sim']:.3f}")

        # Step 4: Retrieve Constraint Template
        template = None
        for sp in sub_problems:
            if sp["concept"] == "discover_constraints":
                meta = sp.get("retrieved_meta", {})
                template = meta.get("constraint_template")
                if template:
                    trace.append(f"  Retrieved template: {template.get('primitives', [])}")
                break

        # Step 5: Generic Z3 Solving
        trace.append("Generic Z3 solving (18 primitives)...")
        solution = self.z3_solver.solve(graph, template)
        if solution:
            if solution.get("type") == "unsat":
                trace.append("  Z3: UNSAT -- constraints contradictory")
            else:
                trace.append(f"  Z3: SAT -- primitives: {solution.get('primitives_applied', [])}")
        else:
            trace.append("  Z3: No solution (no matching primitives)")

        # Step 5b: If only default bounded, try MHN fallback
        if solution and solution.get("type") == "generic_csp":
            prims = solution.get("primitives_applied", [])
            if prims == ["bounded"]:
                fallback = self._fallback_mhn_solve(thoughts, graph)
                if fallback and fallback.get("similarity", 0) > 0.85:
                    solution = fallback
                    trace.append("  MHN override: better retrieval than default Z3")

        # Step 6: Fallback if Z3 fails
        if not solution or solution.get("type") == "unsat":
            fallback = self._fallback_mhn_solve(thoughts, graph)
            if fallback:
                solution = fallback
                trace.append("  Fallback: MHN retrieved similar solution")

        # Step 7: Store
        sub_solution_vecs = [sp["hdc"] for sp in sub_problems if sp.get("hdc") is not None]
        composed_hdc = hdc_compose(sub_solution_vecs) if sub_solution_vecs else evolved_state
        metadata = {
            "source": "auto_learned",
            "problem": problem_text[:200],
            "entity_count": len(graph.entities),
            "relation_count": len(graph.relations),
            "constraint_count": len(graph.constraints),
            "primitives_applied": solution.get("primitives_applied", []) if solution else [],
            "timestamp": time.time(),
        }
        if template:
            metadata["constraint_template"] = template
        store_experience(self.mhn, problem_hdc, composed_hdc, metadata)
        self.total_learned += 1
        trace.append(f"Stored as new attractor (memory: {len(self.mhn.patterns)} patterns)")

        confidence = self._compute_confidence(solution, thoughts, sub_problems)
        latency = (time.perf_counter() - t0) * 1000

        return ThinkingResult(
            problem=problem_text, evolved_state=evolved_state,
            thoughts=thoughts, sub_problems=sub_problems,
            solution=solution, confidence=confidence,
            latency_ms=latency, trace=trace,
        )

    def teach(self, problem_text: str, solution_text: str,
              constraint_template: dict | None = None, metadata: dict | None = None):
        problem_hdc = self.parser.parse(problem_text)
        solution_hdc = self.parser.parse(solution_text)
        title = (metadata or {}).get("title", " ".join(solution_text.split()[:6]))
        meta = {
            "source": "human_taught", "title": title,
            "effectiveness": (metadata or {}).get("effectiveness", 0.75),
            "description": solution_text, "problem": problem_text[:200],
            "solution_text": solution_text[:200], "timestamp": time.time(),
        }
        if constraint_template:
            meta["constraint_template"] = constraint_template
        if metadata:
            meta.update(metadata)
        self.mhn.store(problem_hdc, solution_hdc, meta)
        self.total_learned += 1
        return {
            "status": "learned", "problem": problem_text[:80],
            "solution": solution_text[:80],
            "memory_size": len(self.mhn.patterns),
            "message": "Got it. I'll remember this for next time.",
        }

    def needs_teaching(self, result: ThinkingResult) -> bool:
        if self.total_thinks < 3 and len(self.mhn.patterns) < 10:
            return True
        if result.confidence < 0.25:
            return True
        return False

    def _build_semantic_prototypes(self) -> dict[str, np.ndarray]:
        descriptions = {
            "discover_entities": "find all objects and elements that appear in this problem",
            "discover_constraints": "what are the limits and rules that restrict the solution",
            "find_solution": "how to assign values that satisfy all limits and rules",
            "validate_result": "check that the assignment satisfies every requirement and limit",
        }
        return {name: self.parser.parse(text) for name, text in descriptions.items()}

    def _fallback_mhn_solve(self, thoughts: list[Thought], graph: GenericEntityGraph) -> dict | None:
        best_sim = 0
        best_meta = {}
        for t in thoughts:
            if t.retrieved_hdc is not None and t.retrieved_similarity > best_sim:
                best_sim = t.retrieved_similarity
                best_meta = t.retrieved_metadata
        if best_sim > 0.3 and best_meta:
            text = best_meta.get("description", best_meta.get("solution_text", ""))
            if text:
                return {"type": "learned", "formatted": text, "similarity": best_sim}
        return None

    def _compute_confidence(self, solution: dict | None, thoughts: list[Thought], sub_problems: list[dict]) -> float:
        if solution:
            sol_type = solution.get("type", "unknown")
            if sol_type == "generic_csp":
                prims = solution.get("primitives_applied", [])
                if not prims or prims == ["bounded"]:
                    return 0.30
                return min(0.60 + len(prims) * 0.10, 0.90)
            elif sol_type == "unsat":
                return 0.95
            elif sol_type == "learned":
                sim = solution.get("similarity", 0.5)
                if sim < 0.40:       return 0.10
                elif sim < 0.60:    return 0.25
                elif sim < 0.75:    return 0.40
                elif sim < 0.90:    return 0.55
                else:                return min(sim * 0.85, 0.85)
        best_sim = max((t.retrieved_similarity for t in thoughts if t.retrieved_hdc is not None), default=0)
        if best_sim > 0.5:
            return min(best_sim * 0.7, 0.70)
        elif best_sim > 0.3:
            return min(best_sim * 0.5, 0.40)
        return 0.20

    def _load_minimal_seed(self):
        try:
            from .minimal_seed import ALL_SEED_PATTERNS
            for pattern in ALL_SEED_PATTERNS:
                query_hdc = self.parser.parse(pattern.text)
                solution_hdc = self.parser.parse(pattern.solution)
                meta = {"label": "seed", **pattern.metadata}
                if hasattr(pattern, "constraint_template") and pattern.constraint_template:
                    meta["constraint_template"] = pattern.constraint_template
                self.mhn.store(query_hdc, solution_hdc, meta)
                self.seed_count += 1
        except ImportError:
            pass

    def stats(self) -> dict:
        total = len(self.mhn.patterns)
        seed_pct = (self.seed_count / total * 100) if total > 0 else 0
        return {
            "total_thinks": self.total_thinks, "total_learned": self.total_learned,
            "memory_size": total, "seed_patterns": self.seed_count,
            "learned_patterns": total - self.seed_count,
            "seed_ratio_pct": round(seed_pct, 1),
            "avg_iterations": round(self.avg_iterations, 2),
            "pure_mode": self.pure_mode,
        }
