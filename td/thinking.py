"""Generic Thinking Loop with Structure-Driven Routing.

Based on:
    - Betteti et al. (2025) -- IDP Iterative Refinement, Science Advances
    - Kanerva (2009) -- HDC Algebraic Decomposition, Cognitive Computation
    - Ramsauer et al. (2020) -- Automatic Attractor Storage, ICLR 2021
    - Kleyko et al. (2022) -- HDC/VSA Survey, ACM Computing Surveys 55(6), Article 130.
    - Kleyko et al. (2025) -- Principled neuromorphic reservoir computing, Nature Communications

Structure-Driven Routing:
    The parser discovers structure (entities, relations, constraints).
    The structure determines routing:

    1. KG can answer (via inference) → KG result
    2. Relations found → Z3 constraint solving
    3. No relations but MHN match → retrieve answer
    4. Nothing → honest unknown

    No keyword lists. No intent classifier. Pure structural routing.
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
    intent: str = "unknown"
    trace: list[str] = field(default_factory=list)

    @property
    def iterations(self) -> int:
        return len(self.thoughts)

    def summary(self) -> str:
        lines = [
            f"=== Thinking Dust -- {self.intent.upper()} ===",
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


def has_converged(current, previous, threshold=0.98):
    return similarity(current, previous) > threshold


def idp_refine(query_hdc, mhn, max_iterations=5, convergence_threshold=0.98, blend_factor=0.3):
    thoughts = []
    current_state = query_hdc.copy()
    for i in range(max_iterations):
        results = mhn.retrieve(current_state, top_k=1)
        if not results:
            thoughts.append(Thought(
                iteration=i+1, state_hdc=current_state.copy(),
                retrieved_hdc=None, retrieved_similarity=0.0,
                retrieved_metadata={}, converged=True,
                description="No matching patterns -- converged at empty memory",
            ))
            break
        retrieved_vec, retrieved_sim, retrieved_meta = results[0]
        # Proper weighted blend: sign(α·current + β·retrieved)
        weighted = (current_state.astype(np.float32) * (1 - blend_factor) +
                    retrieved_vec.astype(np.float32) * blend_factor)
        new_state = np.sign(weighted).astype(np.int8)
        new_state[new_state == 0] = 1
        converged = has_converged(new_state, current_state, convergence_threshold)
        meta_desc = retrieved_meta.get("title", retrieved_meta.get("domain", ""))
        desc = f"Retrieved: {meta_desc} (sim={retrieved_sim:.3f})" if meta_desc else f"Retrieved pattern (sim={retrieved_sim:.3f})"
        thoughts.append(Thought(
            iteration=i+1, state_hdc=current_state.copy(),
            retrieved_hdc=retrieved_vec.copy(), retrieved_similarity=retrieved_sim,
            retrieved_metadata=retrieved_meta, converged=converged, description=desc,
        ))
        current_state = new_state
        if converged:
            break
    return current_state, thoughts


# =========================================================================
# HDC Decomposition
# =========================================================================

SUB_PROBLEM_DESCRIPTIONS = {
    "discover_entities": "find all objects and elements that appear in this problem",
    "discover_constraints": "what are the limits and rules that restrict the solution",
    "find_solution": "how to assign values that satisfy all limits and rules",
    "validate_result": "check that the assignment satisfies every requirement and limit",
}


def hdc_decompose(state_hdc, prototype_vectors, mhn):
    sub_problems = []
    for concept_name, proto_hdc in prototype_vectors.items():
        component_hdc = bind(state_hdc, proto_hdc)
        results = mhn.retrieve(component_hdc, top_k=1)
        if results:
            _, sim, meta = results[0]
            sub_problems.append({
                "concept": concept_name, "hdc": component_hdc,
                "retrieved_sim": sim, "retrieved_meta": meta,
                "solution": meta.get("solution", meta),
            })
        else:
            sub_problems.append({
                "concept": concept_name, "hdc": component_hdc,
                "retrieved_sim": 0.0, "retrieved_meta": {}, "solution": None,
            })
    return sub_problems


def hdc_compose(sub_solutions):
    if not sub_solutions:
        return generate_hypervector(10_000)
    if len(sub_solutions) == 1:
        return sub_solutions[0]
    return bundle(*sub_solutions)


def store_experience(mhn, problem_hdc, solution_hdc, metadata=None):
    mhn.store(problem_hdc, solution_hdc, metadata or {"source": "auto_learned"})


# =========================================================================
# Generic Z3 Solver -- 18 Primitives
# =========================================================================

class GenericZ3Solver:
    def __init__(self):
        self.primitive_builders = {
            "all_different": self._build_all_different,
            "ordered": self._build_ordered,
            "bounded": self._build_bounded,
            "excluded": self._build_excluded,
            "grouped": self._build_grouped,
            "optimized": self._build_optimized,
            "sum_eq": self._build_sum_eq,
            "sum_leq": self._build_sum_leq,
            "sum_geq": self._build_sum_geq,
            "count": self._build_count,
            "ratio": self._build_ratio,
            "implies": self._build_implies,
            "equivalent": self._build_equivalent,
            "at_least": self._build_at_least,
            "at_most": self._build_at_most,
            "exactly": self._build_exactly,
            "no_overlap": self._build_no_overlap,
            "precedence": self._build_precedence,
        }

    def solve(self, graph, template=None):
        try:
            from z3 import Solver, Optimize, Int, Bool, sat, unsat, And, Or, Not, Implies, Distinct, Sum, If
        except ImportError:
            return None
        if not graph.entities:
            return None

        has_template = template is not None and template.get("primitives")
        has_relations = len(graph.relations) > 0
        has_constraints = len(graph.constraints) > 0
        if not has_template and not has_relations and not has_constraints:
            return None

        z3_vars = {}
        for e in graph.entities:
            var_type = self._infer_var_type(e, graph)
            z3_vars[e["id"]] = Bool(e["id"]) if var_type == "bool" else Int(e["id"])

        primitives = []
        if template:
            primitives = self._extract_primitives_from_template(template, graph)
        else:
            primitives = self._infer_primitives_from_relations(graph)
            for c in graph.constraints:
                primitive = self._constraint_to_primitive(c, graph)
                if primitive:
                    primitives.append(primitive)

        if not primitives or primitives == [{"type": "bounded", "subjects": list(z3_vars.keys()), "min": 0, "max": 100}]:
            return None

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
            return {"type": "unsat", "formatted": "No solution exists -- constraints are contradictory."}
        return None

    # Builders (all 18)
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
            solver.add(z3_vars[subjects[0]] != primitive.get("exclude_value", 0))

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

    def _build_sum_eq(self, solver, z3_vars, primitive, graph):
        from z3 import Sum
        subjects = [z3_vars[sid] for sid in primitive["subjects"] if sid in z3_vars]
        if subjects:
            solver.add(Sum(subjects) == primitive.get("target", 0))

    def _build_sum_leq(self, solver, z3_vars, primitive, graph):
        from z3 import Sum
        subjects = [z3_vars[sid] for sid in primitive["subjects"] if sid in z3_vars]
        if subjects:
            solver.add(Sum(subjects) <= primitive.get("limit", 0))

    def _build_sum_geq(self, solver, z3_vars, primitive, graph):
        from z3 import Sum
        subjects = [z3_vars[sid] for sid in primitive["subjects"] if sid in z3_vars]
        if subjects:
            solver.add(Sum(subjects) >= primitive.get("limit", 0))

    def _build_count(self, solver, z3_vars, primitive, graph):
        from z3 import Sum, If
        subjects = [z3_vars[sid] for sid in primitive["subjects"] if sid in z3_vars]
        if not subjects:
            return
        count = Sum([If(v, 1, 0) for v in subjects])
        condition = primitive.get("condition", "eq")
        target = primitive.get("target", 0)
        if condition == "eq":
            solver.add(count == target)
        elif condition == "geq":
            solver.add(count >= target)
        elif condition == "leq":
            solver.add(count <= target)

    def _build_ratio(self, solver, z3_vars, primitive, graph):
        subjects = [sid for sid in primitive["subjects"] if sid in z3_vars]
        ratio = primitive.get("ratio", 1)
        if len(subjects) >= 2:
            solver.add(z3_vars[subjects[0]] == ratio * z3_vars[subjects[1]])

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
        if subjects:
            solver.add(Sum([If(v, 1, 0) for v in subjects]) >= primitive.get("n", 1))

    def _build_at_most(self, solver, z3_vars, primitive, graph):
        from z3 import Sum, If
        subjects = [z3_vars[sid] for sid in primitive["subjects"] if sid in z3_vars]
        if subjects:
            solver.add(Sum([If(v, 1, 0) for v in subjects]) <= primitive.get("n", 1))

    def _build_exactly(self, solver, z3_vars, primitive, graph):
        from z3 import Sum, If
        subjects = [z3_vars[sid] for sid in primitive["subjects"] if sid in z3_vars]
        if subjects:
            solver.add(Sum([If(v, 1, 0) for v in subjects]) == primitive.get("n", 1))

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

    # Inference methods
    def _infer_var_type(self, entity, graph):
        for c in graph.constraints:
            if entity["id"] in c["subjects"]:
                if c["params"].get("type") == "bool":
                    return "bool"
        return "int"

    def _extract_primitives_from_template(self, template, graph):
        primitives = []
        for p in template.get("primitives", []):
            mapped = self._map_subjects(p.get("subjects", []), p.get("selector"), graph)
            if mapped:
                primitives.append({"type": p["type"], "subjects": mapped,
                    **{k: v for k, v in p.items() if k not in ("type", "subjects")}})
        return primitives

    def _infer_primitives_from_relations(self, graph):
        primitives = []
        entity_ids = [e["id"] for e in graph.entities]
        rel_groups = {}
        for rel in graph.relations:
            rtype = rel.get("rel_type", "unknown")
            if rtype not in rel_groups:
                rel_groups[rtype] = []
            rel_groups[rtype].append((rel["src"], rel["tgt"]))

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

        if "ratio" in rel_groups or "proportional" in rel_groups:
            for src, tgt in rel_groups.get("ratio", []) + rel_groups.get("proportional", []):
                primitives.append({"type": "ratio", "subjects": [src, tgt], "ratio": 2})

        if "count" in rel_groups:
            n = self._infer_count(rel_groups.get("count", []))
            primitives.append({"type": "count", "subjects": entity_ids, "condition": "eq", "target": n})

        return primitives

    def _constraint_to_primitive(self, constraint, graph):
        params = constraint.get("params", {})
        ptype = params.get("primitive_type", "bounded")
        return {"type": ptype, "subjects": constraint["subjects"],
                **{k: v for k, v in params.items() if k != "primitive_type"}}

    def _map_subjects(self, template_subjects, selector, graph):
        if not selector:
            return [e["id"] for e in graph.entities[:len(template_subjects)]]
        matches = []
        for e in graph.entities:
            sim = similarity(e["hdc"], selector)
            matches.append((e["id"], sim))
        matches.sort(key=lambda x: x[1], reverse=True)
        return [eid for eid, sim in matches if sim > 0.30]

    def _build_order_chain(self, relations):
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

    def _build_groups(self, relations):
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

    def _infer_bounds(self, graph):
        max_val = 100
        for e in graph.entities:
            nums = [int(w) for w in e["text"].split() if w.isdigit()]
            if nums:
                max_val = max(max_val, max(nums) * 2)
        return {"min": 0, "max": max_val}

    def _infer_total(self, graph):
        for e in graph.entities:
            nums = [int(w) for w in e["text"].split() if w.isdigit()]
            if nums:
                return max(nums)
        return None

    def _infer_count(self, relations):
        for src, tgt in relations:
            nums = [int(w) for w in src.split() if w.isdigit()] + [int(w) for w in tgt.split() if w.isdigit()]
            if nums:
                return max(nums)
        return 1

    def _format_solution(self, model, z3_vars, graph, primitives):
        lines = []
        for e in graph.entities:
            val = model.eval(z3_vars[e["id"]], model_completion=True)
            lines.append(f"  {e['text']}: {val}")
        ptypes = [p["type"] for p in primitives]
        method = " + ".join(ptypes) if ptypes else "generic"
        return {
            "type": "generic_csp", "method": f"z3_{method}",
            "formatted": "\n".join(lines), "primitives_applied": ptypes,
        }


# =========================================================================
# Intent-Based Thinking Loop
# =========================================================================

class GenericThinkingDust:
    """Generic reasoning engine with structure-driven routing.

    Routing is determined by parser-discovered structure:
        1. KG inference (if KG can answer)
        2. Z3 constraint solving (if relations/constraints found)
        3. MHN retrieval (if semantic match exists)
        4. Honest unknown (if nothing matches)
    """

    def __init__(self, vocab=None, mhn=None, dim=10_000, max_idp_iterations=5,
                 idp_blend_factor=0.3, convergence_threshold=0.98, pure_mode=False,
                 word_vectors=None):
        self.dim = dim
        self.mhn = mhn or ModernHopfieldNetwork(MHNConfig(dim=dim, min_similarity=0.01, idp_enabled=False))
        self.parser = GenericNLParser(vocab, self.mhn, dim=dim)
        # Coreference resolution is opt-in via td.parser.enable_coreference()
        # Not auto-enabled — loads 490MB model, slows tests

        self.z3_solver = GenericZ3Solver()
        self.max_idp_iterations = max_idp_iterations
        self.idp_blend_factor = idp_blend_factor
        self.convergence_threshold = convergence_threshold
        self.pure_mode = pure_mode

        # Semantic word vectors (BEAGLE-style, trained on corpus)
        # If None, fall back to parser.parse() for MHN keys
        # If loaded, use encode_query() for position-independent semantic keys
        self.wvm = word_vectors

        # Knowledge Graph for multi-hop inference (the "thinking" layer)
        from .kg import KnowledgeGraph
        self.kg = KnowledgeGraph()

        # SPARQL query layer (pyoxigraph bridge)
        # Provides: inverse queries, property paths, named graphs, FILTER
        # Falls back gracefully if pyoxigraph not installed
        self.sparql_store = None
        try:
            from .query import SparqlStore
            self.sparql_store = SparqlStore()
        except ImportError:
            pass  # pyoxigraph not installed; BFS-only mode

        # Sync pre-seeded relation properties to parser prototypes
        # (married_to, capital_of, in, part_of, etc.)
        self.sync_kg_to_parser()

        # Sub-problem prototypes for reasoning decomposition
        self.sub_problem_prototypes = self._build_semantic_prototypes()



        self.total_thinks = 0
        self.total_learned = 0
        self.avg_iterations = 0.0
        self.seed_count = 0
        if not pure_mode:
            self._load_minimal_seed()

    def think(self, problem_text, context=None):
        """Process input through structure-driven routing.

        The parser discovers structure (entities, relations, constraints).
        The structure determines routing:

        1. KG can answer (via inference) → KG result with proof trace
        2. Relations found → Z3 constraint solving
        3. No relations but MHN match → retrieve answer
        4. Nothing → honest unknown

        This is NOT hardcoded routing. The parser uses HDC algebra + CA
        reservoir + stop-word filtering — all algebraic, no keyword lists.
        """
        t0 = time.perf_counter()
        trace = []
        self.total_thinks += 1

        # ─── Parse ──────────────────────────────────────────────────
        struct = self.parser.extract_structure(problem_text)
        problem_hdc = struct["hdc"]
        graph = struct["graph"]
        trace.append(f"Parsed: {len(graph.entities)} entities, "
                     f"{len(graph.relations)} relations, "
                     f"{len(graph.constraints)} constraints")

        # ─── Check Knowledge Graph first (inference > constraint) ────
        # If the KG can answer this via derivation, use it
        if self.kg and self.kg.triples:
            kg_result = self._query_knowledge_graph(problem_text)
            if kg_result:
                trace.append(f"Route: KG inference ({kg_result['method']})")
                latency = (time.perf_counter() - t0) * 1000
                # Auto-derive new facts periodically
                self.kg.derive_all()
                # Sync newly derived facts to SPARQL store
                if self.sparql_store is not None:
                    self.sparql_store.sync_from_kg(self.kg)
                return ThinkingResult(
                    problem=problem_text, evolved_state=problem_hdc,
                    thoughts=[], sub_problems=[],
                    solution=kg_result,
                    confidence=kg_result["confidence"],
                    latency_ms=latency, intent="inference", trace=trace,
                )

        # ─── Route by discovered structure ──────────────────────────
        # Check for constraint templates (MHN patterns with Z3 primitives)
        constraint_template = self._find_constraint_template(graph, problem_hdc)

        # Route to constraint solving if:
        # 1. MHN has a template (learned pattern), OR
        # 2. Parser found a relation that matches a Z3 constraint signal
        #    Only the innate 14 constraint types route to Z3.
        #    BUT: if the KG has actual triples with this relation AND it's been
        #    taught as transitive/symmetric/functional, prefer KG inference.
        kg_relations_with_facts = set(t.relation for t in self.kg.triples)
        has_constraint_relations = any(
            r.get("rel_type") in self.parser.constraint_signals
            and r.get("rel_type") not in kg_relations_with_facts
            for r in graph.relations
        )

        if constraint_template or has_constraint_relations:
            trace.append("Route: constraint template → Z3 solving")
            return self._handle_constraint(problem_text, problem_hdc, graph, trace, t0, constraint_template)

        # No relations. Try MHN retrieval using semantic key.
        query_hdc = self._encode_key(problem_text)
        evolved_state, thoughts = idp_refine(
            query_hdc, self.mhn,
            max_iterations=self.max_idp_iterations,
            convergence_threshold=self.convergence_threshold,
            blend_factor=self.idp_blend_factor,
        )
        for t in thoughts:
            trace.append(f"  Iter {t.iteration}: sim={t.retrieved_similarity:.3f} "
                        f"{'-> CONVERGED' if t.converged else ''}")

        fallback = self._fallback_mhn_solve(thoughts, graph)

        if fallback:
            trace.append(f"Route: MHN match (sim={fallback['similarity']:.3f})")
            latency = (time.perf_counter() - t0) * 1000
            return ThinkingResult(
                problem=problem_text, evolved_state=evolved_state,
                thoughts=thoughts, sub_problems=[],
                solution=fallback,
                confidence=self._compute_confidence(fallback, thoughts, []),
                latency_ms=latency, intent="retrieval", trace=trace,
            )

        # No relations, no MHN match. Honest unknown.
        trace.append("Route: no match → unknown")
        latency = (time.perf_counter() - t0) * 1000
        return ThinkingResult(
            problem=problem_text, evolved_state=evolved_state,
            thoughts=thoughts, sub_problems=[],
            solution={"type": "unknown",
                      "formatted": "I don't know this one yet."},
            confidence=0.15,
            latency_ms=latency, intent="unknown", trace=trace,
        )

    # ─── Intent Classification (Kleyko 2022: HDC prototype similarity) ──

    def _find_constraint_template(self, graph, problem_hdc):
        """Check if MHN has a constraint template for this problem.

        Returns template dict if found, None otherwise.
        This replaces the need for parser-discovered relations to trigger
        constraint solving — the MHN templates are the source of truth.
        """
        results = self.mhn.retrieve(problem_hdc, top_k=3)
        for vec, sim, meta in results:
            if sim < 0.30:
                continue
            template = meta.get("constraint_template")
            if template and template.get("primitives"):
                return template
        return None

    def _handle_constraint(self, problem_text, problem_hdc, graph, trace, t0, template=None):
        """Handle constraint problems — Z3 solving.

        Uses MHN templates (learned from corpus or taught) for Z3 primitives.
        No parser prototypes needed — the MHN is the source of truth.
        """
        # IDP Refinement
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

        # If no template passed in, try to find one via MHN retrieval
        if not template:
            for sp in thoughts:
                if hasattr(sp, 'retrieved_metadata'):
                    meta = sp.retrieved_metadata or {}
                    template = meta.get("constraint_template")
                    if template:
                        break

        # HDC Decomposition
        trace.append("HDC algebraic decomposition...")
        sub_problems = hdc_decompose(evolved_state, self.sub_problem_prototypes, self.mhn)
        for sp in sub_problems:
            trace.append(f"  [{sp['concept']}] sim={sp['retrieved_sim']:.3f}")

        if template:
            trace.append(f"  Template: {template.get('primitives', [])}")

        # Z3 Solve
        trace.append("Generic Z3 solving...")
        solution = self.z3_solver.solve(graph, template)
        if solution:
            if solution.get("type") == "unsat":
                trace.append("  Z3: UNSAT")
            else:
                trace.append(f"  Z3: SAT -- primitives: {solution.get('primitives_applied', [])}")
        else:
            trace.append("  Z3: Not a constraint problem (no relations/template)")

        # Fallback
        if not solution:
            fallback = self._fallback_mhn_solve(thoughts, graph)
            if fallback:
                solution = fallback
                trace.append(f"  MHN: Retrieved similar answer (sim={fallback['similarity']:.2f})")

        # Store
        sub_solution_vecs = [sp["hdc"] for sp in sub_problems if sp.get("hdc") is not None]
        composed_hdc = hdc_compose(sub_solution_vecs) if sub_solution_vecs else evolved_state
        metadata = {
            "source": "auto_learned", "problem": problem_text[:200],
            "entity_count": len(graph.entities), "relation_count": len(graph.relations),
            "constraint_count": len(graph.constraints),
            "primitives_applied": solution.get("primitives_applied", []) if solution else [],
            "intent": "constraint", "timestamp": time.time(),
        }
        if template:
            metadata["constraint_template"] = template
        # Store solution text if available
        if solution and solution.get("formatted") and solution.get("type") not in ("unknown", "unsat"):
            metadata["description"] = solution["formatted"][:500]
            metadata["solution_text"] = solution["formatted"][:500]
        store_experience(self.mhn, problem_hdc, composed_hdc, metadata)
        self.total_learned += 1
        trace.append(f"Stored as new attractor (memory: {len(self.mhn.patterns)} patterns)")

        confidence = self._compute_confidence(solution, thoughts, sub_problems)
        latency = (time.perf_counter() - t0) * 1000
        return ThinkingResult(
            problem=problem_text, evolved_state=evolved_state,
            thoughts=thoughts, sub_problems=sub_problems,
            solution=solution, confidence=confidence,
            latency_ms=latency, intent="constraint", trace=trace,
        )

    # ─── Teach ────────────────────────────────────────────────────────

    def teach(self, problem_text, solution_text, constraint_template=None, metadata=None):
        # Use semantic key for storage (enables paraphrase matching)
        problem_hdc = self._encode_key(problem_text)
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

        # Online BEAGLE update: word vectors learn from this interaction
        if self.wvm is not None:
            self.wvm.train_incremental(problem_text)
            self.wvm.train_incremental(solution_text)

        # Try to extract knowledge graph triples
        triples = self._extract_triples(problem_text, solution_text)
        for (s, r, o) in triples:
            self.kg.add_fact(s, r, o)
            # Sync to SPARQL store (if available)
            if self.sparql_store is not None:
                self.sparql_store.add_fact(s, r, o, source="user")

        return {
            "status": "learned", "problem": problem_text[:80],
            "solution": solution_text[:80], "memory_size": len(self.mhn.patterns),
            "message": "Got it. I'll remember this for next time.",
        }

    def teach_relation(self, relation: str, *properties: str) -> dict:
        """Teach logical properties of a relation.

        This is the "teaching dust how to think" interface for relation
        properties. The user tells TD how a relation behaves, and TD
        applies general logical templates.

        Examples:
            td.teach_relation("north_of", "transitive")
            td.teach_relation("married_to", "symmetric")
            td.teach_relation("capital_of", "functional", "inverse:has_capital")

        Properties:
            transitive  — R(X,Y) ∧ R(Y,Z) → R(X,Z)
            symmetric   — R(X,Y) → R(Y,X)
            inverse:R2  — R1(X,Y) → R2(Y,X)
            functional  — R(X,Y) ∧ R(X,Z) → Y=Z
        """
        self.kg.set_relation_property(relation, *properties)
        # Also register lemmatized form (spaCy lemmatizes "powers" → "power")
        if self.parser.nlp:
            doc = self.parser.nlp(relation)
            if doc and doc[0].lemma_ != relation:
                self.kg.set_relation_property(doc[0].lemma_, *properties)
        # Sync to parser so it can detect this relation in future queries
        self.parser.register_relation(relation)
        # Sync to SPARQL store (if available)
        if self.sparql_store is not None:
            self.sparql_store.sync_from_kg(self.kg)
        return {
            "status": "learned",
            "relation": relation,
            "properties": list(properties),
            "message": f"Got it. '{relation}' is now {', '.join(properties)}.",
        }

    def sync_kg_to_parser(self):
        """Sync all KG relation properties to the parser's prototypes.

        Call this after loading from SQLite so the parser can detect
        previously learned relations in new queries.
        """
        for relation in self.kg.relation_properties:
            self.parser.register_relation(relation)

        # Sync full KG to SPARQL store (if available)
        if self.sparql_store is not None:
            count = self.sparql_store.sync_from_kg(self.kg)
            if count > 0:
                import sys
                print(f"  ✓ SPARQL store synced: {count} facts", file=sys.stderr)

    def needs_teaching(self, result):
        if self.total_thinks < 3 and len(self.mhn.patterns) < 10:
            return True
        if result.confidence < 0.25:
            return True
        return False

    # ─── Internal ─────────────────────────────────────────────────────

    def _build_semantic_prototypes(self):
        descriptions = {
            "discover_entities": "find all objects and elements that appear in this problem",
            "discover_constraints": "what are the limits and rules that restrict the solution",
            "find_solution": "how to assign values that satisfy all limits and rules",
            "validate_result": "check that the assignment satisfies every requirement and limit",
        }
        return {name: self.parser.parse(text) for name, text in descriptions.items()}

    def _encode_key(self, text: str) -> np.ndarray:
        """Encode text as MHN storage/retrieval key.

        If semantic word vectors (BEAGLE) are loaded, use position-independent
        bag-of-content-words encoding. This enables paraphrase matching:
        "what is the capital of france" and "capital of france?" produce the
        same key.

        If no word vectors, fall back to parser.parse() (position-dependent,
        no paraphrase matching).
        """
        if self.wvm is not None:
            return self.wvm.encode_query(text)
        return self.parser.parse(text)

    # ─── Knowledge Graph Triple Extraction ──────────────────────────

    def _extract_triples(self, problem: str, solution: str) -> list[tuple[str, str, str]]:
        """Extract knowledge graph triples from teach() input.

        Uses spaCy dependency parsing when available (language-agnostic).
        Falls back to regex patterns when spaCy is not installed.

        spaCy provides:
        - Dependency parsing → proper PP attachment
        - Noun chunks → compound noun detection ("United Kingdom")
        - POS tagging → preposition detection (no hardcoded _pp_words)
        - Lemmatization → "collaborates" → "collaborate"

        Reference: Honnibal & Montani (2017), "spaCy 2"
        """
        text = problem.lower().strip()

        # Try spaCy first (language-agnostic, no hardcoded rules)
        # Preserve original case for spaCy (lowercase confuses NER/POS)
        spacy_triples = self.parser.extract_triples_spacy(problem)
        if spacy_triples:
            return spacy_triples

        # Fallback: regex patterns (English-only, hardcoded)
        return self._extract_triples_regex(text)

    def _extract_triples_regex(self, text: str) -> list[tuple[str, str, str]]:
        """Regex-based triple extraction (fallback when spaCy unavailable).

        Hardcoded English patterns. Replaced by spaCy when available.
        Lemmatizes relations via spaCy when available for consistency.
        """
        import re

        triples = []
        # text is already lowercased by caller

        # Lemmatize relation via spaCy (consistency with spaCy extraction)
        def _lemmatize(rel: str) -> str:
            if self.parser.nlp:
                doc = self.parser.nlp(rel)
                if doc:
                    return doc[0].lemma_
            return rel

        # Helper: strip trailing prepositional phrases from entities
        # "austria on danube" → "austria"
        # "united kingdom in europe" → "united kingdom"
        # Reference: Prepositional phrase attachment (Manning & Schütze, 1999)
        _pp_words = {"on", "in", "at", "from", "to", "by", "for", "with", "into",
                     "through", "during", "before", "after", "about", "near", "over"}
        def _strip_pp(entity: str) -> str:
            words = entity.split()
            # Strip trailing preposition + its object
            # "austria on danube" → ["austria", "on", "danube"] → "austria"
            while len(words) >= 3 and words[-2] in _pp_words:
                words = words[:-2]
            return " ".join(words)

        # Pattern: X is the Y of Z → (X, Y, Z)
        # "Paris is the capital of France" → (paris, capital, france)
        # Multi-word: "Seoul is the capital of South Korea" → (seoul, capital, south korea)
        m = re.search(r'(\w+(?:\s+\w+)*)\s+is\s+(?:the\s+)?(\w+)\s+of\s+(?:the\s+)?(\w+(?:\s+\w+)*)', text)
        if m:
            s, r, o = m.group(1), m.group(2), m.group(3)
            triples.append((s, f"{r}_of", o))

        # Pattern: X is in Y → (X, in, Y)
        m = re.search(r'(\w+(?:\s+\w+)*)\s+is\s+in\s+(?:the\s+)?(\w+(?:\s+\w+)*)', text)
        if m:
            triples.append((_strip_pp(m.group(1)), "in", _strip_pp(m.group(2))))

        # Pattern: X is inside Y → (X, inside, Y)
        m = re.search(r'(\w+(?:\s+\w+)*)\s+is\s+inside\s+(?:the\s+)?(\w+)', text)
        if m:
            triples.append((m.group(1), "inside", m.group(2)))

        # Pattern: X is within Y → (X, inside, Y) — alias
        m = re.search(r'(\w+(?:\s+\w+)*)\s+is\s+within\s+(?:the\s+)?(\w+)', text)
        if m and not any(r == "inside" for _, r, _ in triples):
            triples.append((m.group(1), "inside", m.group(2)))

        # Pattern: X contains Y → (X, contains, Y)
        m = re.search(r'(\w+(?:\s+\w+)*)\s+contains\s+(?:the\s+)?(\w+)', text)
        if m:
            triples.append((m.group(1), "contains", m.group(2)))

        # Pattern: X is part of Y → (X, part_of, Y)
        m = re.search(r'(\w+(?:\s+\w+)*)\s+is\s+part\s+of\s+(?:the\s+)?(\w+(?:\s+\w+)*)', text)
        if m:
            triples.append((m.group(1), "part_of", m.group(2)))

        # Pattern: X is before Y → (X, before, Y)
        m = re.search(r'(\w+(?:\s+\w+)*)\s+is\s+before\s+(\w+(?:\s+\w+)*)', text)
        if m:
            triples.append((_strip_pp(m.group(1)), "before", _strip_pp(m.group(2))))

        # Pattern: X before Y → (X, before, Y) — without "is"
        m = re.search(r'(\w+(?:\s+\w+)*)\s+before\s+(\w+(?:\s+\w+)*)', text)
        if m and not any(r == "before" for _, r, _ in triples):
            triples.append((_strip_pp(m.group(1)), "before", _strip_pp(m.group(2))))

        # Pattern: X is after Y → (X, after, Y)
        m = re.search(r'(\w+)\s+is\s+after\s+(\w+)', text)
        if m:
            triples.append((m.group(1), "after", m.group(2)))

        # Pattern: X after Y → (X, after, Y) — without "is"
        m = re.search(r'(\w+)\s+after\s+(\w+)', text)
        if m and not any(r == "after" for _, r, _ in triples):
            triples.append((m.group(1), "after", m.group(2)))

        # Pattern: X means Y → (X, means, Y)
        m = re.search(r'(\w+)\s+means\s+(\w+)', text)
        if m:
            triples.append((m.group(1), "means", m.group(2)))

        # Pattern: X is Y to Z → (X, Y_to, Z)
        # "David Beckham is married to Victoria Beckham" → (david, married_to, victoria)
        m = re.search(r'(\w+)\s+is\s+(\w+)\s+to\s+(\w+)', text)
        if m:
            s, r, o = m.group(1), m.group(2), m.group(3)
            triples.append((s, f"{r}_to", o))

        # Pattern: X is Y Z → (X, Y_Z, Z) for compound relations
        # "X is north of Y" → (x, north_of, y) — already covered by "is Y of Z"
        # But also: "X is larger than Y" → (x, larger_than, y)
        m = re.search(r'(\w+)\s+is\s+(\w+)\s+than\s+(\w+)', text)
        if m:
            s, r, o = m.group(1), m.group(2), m.group(3)
            triples.append((s, f"{r}_than", o))

        # Pattern: X is R Y → (X, R, Y) where R is a compound relation (e.g., sibling_of)
        # "Carol is sibling_of Dave" → (carol, sibling_of, dave)
        # This is a general fallback for teach() input with explicit relation names.
        if not triples:
            m = re.search(r'(\w+)\s+is\s+([a-z]+_[a-z]+)\s+(\w+)', text)
            if m:
                s, r, o = m.group(1), m.group(2), m.group(3)
                triples.append((s, r, o))

        # Fallback: X relation Y (no "is" — e.g., "RiverA feeds_into RiverB")
        # Only match if relation is not a common stop word.
        if not triples:
            stop_words = {
                'the', 'a', 'an', 'and', 'or', 'is', 'are', 'was', 'were',
                'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
                'will', 'would', 'could', 'should', 'may', 'might', 'must',
                'can', 'shall', 'it', 'its', 'this', 'that', 'these', 'those',
                'there', 'their', 'they', 'them', 'of', 'to', 'in', 'for',
                'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
                'during', 'before', 'after', 'above', 'below', 'between',
            }
            m = re.search(r'(\w+)\s+([a-z_]+)\s+(\w+)', text)
            if m:
                s, r, o = m.group(1), m.group(2), m.group(3)
                if r not in stop_words and len(r) > 1:
                    triples.append((s, _lemmatize(r), o))

        return triples

    def _query_knowledge_graph(self, text: str) -> dict | None:
        """Query the knowledge graph for inferred answers.

        GENERIC approach — no relation-specific regex patterns.
        Works with ANY relation the KG knows about.

        1. Find entities in the query that exist in the KG
        2. Find relation words in the query that match KG relations
        3. For each entity pair, check if KG has a path
        4. If relation word found, filter paths; else return any path
        """
        import re
        text_lower = text.lower().strip()
        tokens = re.findall(r'\w+', text_lower)  # \w includes underscores

        # Collect entities that exist in the KG
        kg_entities = set()
        for t in self.kg.triples:
            kg_entities.add(t.subject)
            kg_entities.add(t.object)
        entities_in_query = [t for t in tokens if t in kg_entities]

        if len(entities_in_query) < 2:
            # Try multi-word entities from gazetteer (learned from teach())
            # Reference: Nadeau & Sekine (2007) — Named Entity Recognition
            query_text = " ".join(tokens)
            for entity in self.kg.gazetteer:
                if entity in query_text:
                    entities_in_query.append(entity)

            # Also check KG entities (for entities not yet in gazetteer)
            for entity in kg_entities:
                if " " in entity and entity in query_text:
                    if entity not in entities_in_query:
                        entities_in_query.append(entity)
                    for part in entity.split():
                        if part in entities_in_query:
                            entities_in_query.remove(part)

            if len(entities_in_query) < 2:
                # Try inverse/open query: "What is the capital of France?"
                # Only for "what/who/where" questions (not "is X Y?" yes/no questions)
                if len(entities_in_query) == 1:
                    question_words = {"what", "who", "where", "which"}
                    is_question = any(t in question_words for t in tokens)
                    if not is_question:
                        return None

                    kg_relations = set(t.relation for t in self.kg.triples)
                    kg_relations.update(self.kg.relation_properties.keys())
                    entity = entities_in_query[0]

                    def _try_inverse_query(rel):
                        """Try entity as subject first, then as object."""
                        # Forward: capital_of(france, ?) → what does France have?
                        result = self.kg.query(entity, rel)
                        if result.answer is not None:
                            return result
                        # Inverse: capital_of(?, france) → what is the capital of France?
                        for t in self.kg.triples:
                            if t.relation == rel and t.object == entity:
                                return self.kg.query(t.subject, rel, entity)
                        return None

                    for token in tokens:
                        if token in kg_relations:
                            result = _try_inverse_query(token)
                            if result and result.answer is not None:
                                return {
                                    "type": "inferred",
                                    "formatted": result.proof_trace,
                                    "confidence": result.confidence,
                                    "method": result.method,
                                }
                    # Also check compound relations
                    for i, token in enumerate(tokens):
                        for rel in kg_relations:
                            parts = rel.split("_")
                            if len(parts) == 2 and i + 1 < len(tokens):
                                if token == parts[0] and tokens[i + 1] == parts[1]:
                                    result = _try_inverse_query(rel)
                                    if result and result.answer is not None:
                                        return {
                                            "type": "inferred",
                                            "formatted": result.proof_trace,
                                            "confidence": result.confidence,
                                            "method": result.method,
                                        }
                return None

        # Collect relation words in the query that match KG relations
        kg_relations = set(t.relation for t in self.kg.triples)
        # Also check relation_properties (for relations taught but no triples yet)
        kg_relations.update(self.kg.relation_properties.keys())

        relation_in_query = None
        for token in tokens:
            if token in kg_relations:
                relation_in_query = token
                break
        # Also check compound relations like "part_of" from "part of"
        # Must match TWO parts to avoid false positives (e.g., "to" in "married_to")
        for i, token in enumerate(tokens):
            if relation_in_query:
                break
            for rel in kg_relations:
                parts = rel.split("_")
                if len(parts) == 2 and i + 1 < len(tokens):
                    if token == parts[0] and tokens[i + 1] == parts[1]:
                        relation_in_query = rel
                        break

        # Fuzzy fallback: SequenceMatcher (stdlib) for morphology
        # "collaborate" matches "collaborates_with" (0.96 similarity)
        # Language-agnostic, no external deps, battle-tested since Python 2.1
        if not relation_in_query:
            import difflib
            for rel in kg_relations:
                parts = rel.split("_")
                if len(parts) == 2:
                    best0 = max(difflib.SequenceMatcher(None, t, parts[0]).ratio() for t in tokens)
                    best1 = max(difflib.SequenceMatcher(None, t, parts[1]).ratio() for t in tokens)
                    if best0 >= 0.75 and best1 >= 0.75:
                        relation_in_query = rel
                        break
                elif len(parts) == 1:
                    best = max(difflib.SequenceMatcher(None, t, parts[0]).ratio() for t in tokens)
                    if best >= 0.75:
                        relation_in_query = rel
                        break

        # Check "are X and Y the same?" — special case for functional comparison
        if "same" in tokens or "equal" in tokens or "identical" in tokens:
            if len(entities_in_query) >= 2:
                result = self.kg.check_same(entities_in_query[0], entities_in_query[1])
                if result.answer is not None:
                    return {
                        "type": "inferred",
                        "formatted": result.proof_trace,
                        "confidence": result.confidence,
                        "method": result.method,
                    }

        # For each entity pair, check for paths in the KG
        # ─── SPARQL-first path (pyoxigraph) ─────────────────────────
        # Try SPARQL before BFS: handles inverse queries, property paths,
        # multi-hop transitive chains, and FILTER natively.
        if self.sparql_store is not None and self.sparql_store._synced:
            for i, e1 in enumerate(entities_in_query):
                for e2 in entities_in_query[i + 1:]:
                    # SPARQL ask with relation (if found)
                    if relation_in_query:
                        result = self.sparql_store.ask(e1, e2, relation_in_query)
                        if result.found:
                            return {
                                "type": "inferred",
                                "formatted": result.proof_trace,
                                "confidence": result.confidence,
                                "method": result.method,
                            }
                    # SPARQL ask without relation (finds any path)
                    result = self.sparql_store.ask(e1, e2)
                    if result.found:
                        return {
                            "type": "inferred",
                            "formatted": result.proof_trace,
                            "confidence": result.confidence,
                            "method": result.method,
                        }

            # SPARQL inverse query: "What is the capital of France?"
            # (single entity + relation, open question)
            if len(entities_in_query) == 1 and relation_in_query:
                question_words = {"what", "who", "where", "which"}
                is_question = any(t in question_words for t in tokens)
                if is_question:
                    inv_results = self.sparql_store.inverse_query(
                        relation_in_query, entities_in_query[0]
                    )
                    if inv_results:
                        return {
                            "type": "inferred",
                            "formatted": f"{relation_in_query}({', '.join(inv_results)}) → {entities_in_query[0]}",
                            "confidence": 0.90,
                            "method": "sparql_inverse",
                        }

        # ─── BFS fallback (existing logic) ─────────────────────────
        for i, e1 in enumerate(entities_in_query):
            for e2 in entities_in_query[i + 1:]:
                # For non-symmetric relations, only try query direction (e1 → e2)
                # For symmetric relations, try both directions
                is_symmetric = "symmetric" in self.kg.relation_properties.get(
                    relation_in_query or "", set()
                )
                directions = [(e1, e2)] if not is_symmetric else [(e1, e2), (e2, e1)]

                for subj, obj in directions:
                    paths = self.kg.bfs_paths(subj, obj)
                    if paths:
                        best_path = self.kg._find_valid_path(
                            paths, relation_in_query or "", subj, obj
                        )
                        if best_path:
                            # Format: use relation word if found, else generic
                            rel_word = relation_in_query or "connects to"
                            trace = self.kg._format_proof_trace(
                                best_path, subj, rel_word, obj
                            )
                            # Confidence from chain quality (research-backed)
                            # CPR (arXiv 2026), UaG (AAAI 2025), UnKGCP (arXiv 2025)
                            conf = self.kg._chain_confidence(
                                best_path, relation_in_query or ""
                            )
                            return {
                                "type": "inferred",
                                "formatted": trace,
                                "confidence": conf,
                                "method": "derived",
                            }

        return None

    def _fallback_mhn_solve(self, thoughts, graph):
        best_sim = 0
        best_meta = {}
        for t in thoughts:
            if t.retrieved_hdc is not None and t.retrieved_similarity > best_sim:
                best_sim = t.retrieved_similarity
                best_meta = t.retrieved_metadata
        if best_sim > 0.3 and best_meta:
            text = (best_meta.get("description") or
                    best_meta.get("solution_text") or
                    best_meta.get("problem", ""))
            if text and text != "I don't know this one yet.":
                # Entity validation for medium-similarity cases to prevent hallucinations.
                # High similarity (>=0.7) = near-perfect match, trust it.
                # Low similarity (<=0.3) = already filtered by sim threshold.
                # Medium similarity (0.3-0.7) = validate entity overlap.
                if best_sim >= 0.7:
                    return {"type": "learned", "formatted": text, "similarity": best_sim}

                # Validate: ALL query entities must appear in the answer.
                # Prevents "is Norway part of Europe" → "EU is part of Europe" hallucination.
                # Norway is in the query but not in the answer → reject.
                query_entities = {e['text'].lower() for e in graph.entities}
                retrieved_words = set(text.lower().split())
                matched = query_entities & retrieved_words
                if matched == query_entities:
                    return {"type": "learned", "formatted": text, "similarity": best_sim}
        return None

    def _compute_confidence(self, solution, thoughts, sub_problems):
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
                if sim >= 0.99:     return 0.95
                elif sim >= 0.90: return 0.85
                elif sim >= 0.80: return 0.75
                elif sim >= 0.70: return 0.65
                elif sim >= 0.60: return 0.55
                elif sim >= 0.50: return 0.45
                elif sim >= 0.40: return 0.35
                elif sim >= 0.30: return 0.30
                else:               return 0.20
            elif sol_type == "conversation":
                return 0.95
            elif sol_type == "suggestion":
                return 0.90
            elif sol_type == "command":
                return 0.90
            elif sol_type == "meta":
                return 0.70
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

    def _auto_store(self, problem_hdc, graph, solution, intent_label, trace):
        """Store an experience automatically (Ramsauer 2020: online learning)."""
        metadata = {
            "source": "auto_learned",
            "problem": "",  # filled by caller if available
            "entity_count": len(graph.entities),
            "relation_count": len(graph.relations),
            "constraint_count": len(graph.constraints),
            "intent": intent_label,
            "timestamp": time.time(),
        }
        sol_text = solution.get("formatted", "")
        if sol_text and solution.get("type") not in ("unknown", "unsat"):
            metadata["description"] = sol_text[:500]
            metadata["solution_text"] = sol_text[:500]
        composed = hdc_compose([e["hdc"] for e in graph.entities]) if graph.entities else problem_hdc
        store_experience(self.mhn, problem_hdc, composed, metadata)
        self.total_learned += 1
        trace.append(f"Stored as new attractor (memory: {len(self.mhn.patterns)} patterns)")

    def stats(self):
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
