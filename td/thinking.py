"""Thinking Dust — Honest Architecture.

Based on:
    - Betteti et al. (2025) — IDP Iterative Refinement, Science Advances
    - Kanerva (2009) — HDC Algebraic Decomposition, Cognitive Computation
    - Ramsauer et al. (2020) — MHN Attractor Storage, ICLR 2021
    - Kleyko et al. (2022) — HDC/VSA Survey, ACM Computing Surveys 55(6), Article 130.

Pure Mode Architecture (0 domain seed):
    2 innate intents:
        1. CONVERSATION: MHN retrieval from 5 innate social patterns (Ramsauer 2020)
        2. REASONING: Parser → Entity Graph → Z3 (if relations) or MHN (if no relations)

    Intent classification: Structural (token count), not semantic.
    - ≤ 2 tokens → conversation (too short for meaningful parsing)
    - > 2 tokens → reasoning (parse into entity graph)

    Conversation path: Retrieve closest innate pattern from MHN by HDC similarity.
    Reasoning path: Parser discovers entities and relations. If relations exist,
    Z3 solves. If no relations, MHN retrieves past answers.

    This is honest. The 2-intent split is structural (token count), not semantic.
    Semantic classification emerges from MHN learning, not hardcoded rules.
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
            f"=== Thinking Dust — {self.intent.upper()} ===",
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
                description="No matching patterns — converged at empty memory",
            ))
            break
        retrieved_vec, retrieved_sim, retrieved_meta = results[0]
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
# Z3 Solver — 18 Primitives (unchanged, research-backed)
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
            return {"type": "unsat", "formatted": "No solution exists — constraints are contradictory."}
        return None

    # All 18 builders (unchanged from v3)
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
# The Thinking Loop — Honest 2-Intent Architecture
# =========================================================================

class GenericThinkingDust:
    """Honest architecture: 2 intents in pure mode, research-backed throughout.

    Intent split: STRUCTURAL (token count), not semantic.
        - ≤ 2 tokens → conversation (too short to parse meaningfully)
        - > 2 tokens → reasoning (parse into entity graph)

    Conversation: MHN retrieval from innate patterns (Ramsauer 2020).
    Reasoning: Parser → Entity Graph → Z3 (if relations) or MHN (if no relations).

    NO keyword matching. NO hardcoded semantic rules. Pure HDC + MHN + Z3.
    """

    def __init__(self, vocab=None, mhn=None, dim=10_000, max_idp_iterations=5,
                 idp_blend_factor=0.3, convergence_threshold=0.98, pure_mode=False):
        self.dim = dim
        self.mhn = mhn or ModernHopfieldNetwork(MHNConfig(dim=dim, min_similarity=0.01, idp_enabled=False))
        self.parser = GenericNLParser(vocab, self.mhn, dim=dim)
        self.z3_solver = GenericZ3Solver()
        self.max_idp_iterations = max_idp_iterations
        self.idp_blend_factor = idp_blend_factor
        self.convergence_threshold = convergence_threshold
        self.pure_mode = pure_mode

        self.sub_problem_prototypes = self._build_semantic_prototypes()

        # Innate conversation patterns: stored in MHN at init.
        # These are NOT domain seed data. They are the social substrate —
        # like a baby's innate reflexes (crying, sucking, smiling).
        # The baby doesn't learn these; they're wired in.
        # TD doesn't learn these; they're the conversational substrate.
        self._load_innate_conversation_patterns()

        self.total_thinks = 0
        self.total_learned = 0
        self.avg_iterations = 0.0
        self.seed_count = 0
        if not pure_mode:
            self._load_minimal_seed()

    def think(self, problem_text, context=None):
        t0 = time.perf_counter()
        trace = []
        self.total_thinks += 1

        # ─── Intent: STRUCTURAL split (token count) ─────────────────
        tokens = problem_text.lower().strip().split()
        if len(tokens) <= 2:
            intent = "conversation"
        else:
            intent = "reasoning"
        trace.append(f"Intent: {intent} (structural: {len(tokens)} tokens)")

        # ─── Route ──────────────────────────────────────────────────
        if intent == "conversation":
            return self._handle_conversation(problem_text, trace, t0)
        else:
            return self._handle_reasoning(problem_text, trace, t0)

    # ─── Conversation: MHN Retrieval from Innate Patterns ───────────────

    def _handle_conversation(self, problem_text, trace, t0):
        """Handle conversation via MHN retrieval from innate patterns.

        Based on Ramsauer et al. (2020): MHN retrieves by similarity.
        Innate patterns are stored at init. Input retrieves closest match.
        If no match above threshold → generic acknowledgment.
        """
        problem_hdc = self.parser.parse(problem_text)

        # Retrieve from MHN (innate patterns + learned patterns)
        results = self.mhn.retrieve(problem_hdc, top_k=1)

        if results and results[0][1] > 0.30:
            _, sim, meta = results[0]
            response = meta.get("description", "Hello!")
            trace.append(f"MHN retrieved conversation pattern (sim={sim:.3f})")
            confidence = min(sim * 1.2, 0.95)
        else:
            # No match: generic acknowledgment (honest uncertainty)
            response = "I see. Tell me more about that."
            trace.append("No matching conversation pattern — generic acknowledgment")
            confidence = 0.40

        # Store this interaction (learns conversational style)
        store_experience(self.mhn, problem_hdc, self.parser.parse(response), {
            "source": "auto_learned", "intent": "conversation",
            "problem": problem_text[:200], "timestamp": time.time(),
        })
        self.total_learned += 1
        trace.append(f"Stored as new attractor (memory: {len(self.mhn.patterns)} patterns)")

        latency = (time.perf_counter() - t0) * 1000
        return ThinkingResult(
            problem=problem_text, evolved_state=problem_hdc,
            thoughts=[], sub_problems=[],
            solution={"type": "conversation", "formatted": response},
            confidence=confidence, latency_ms=latency, intent="conversation", trace=trace,
        )

    def _load_innate_conversation_patterns(self):
        """Load 5 innate conversation patterns into MHN.

        These are the social substrate — not learned, not domain knowledge.
        Like a baby's innate reflexes. They enable basic social interaction
        before any learning occurs. Stored as MHN attractors (Ramsauer 2020).
        """
        innate = [
            ("hello hi hey", "Hello! What are we working on today?"),
            ("thanks thank you", "You're welcome! Let me know if you need anything else."),
            ("ok okay", "👍"),
            ("bye goodbye", "Goodbye! Come back when you have more problems to solve."),
            ("how are you", "I'm a reasoning engine — always ready to learn!"),
        ]
        for trigger_text, response_text in innate:
            trigger_hdc = self.parser.parse(trigger_text)
            response_hdc = self.parser.parse(response_text)
            self.mhn.store(trigger_hdc, response_hdc, {
                "source": "innate",
                "title": trigger_text[:30],
                "description": response_text,
                "intent": "conversation",
            })

    # ─── Reasoning: Parser → Z3 or MHN ────────────────────────────────

    def _handle_reasoning(self, problem_text, trace, t0):
        """Handle reasoning: parse → entity graph → Z3 (if relations) or MHN (if no relations).

        The parser discovers structure (entities, relations, constraints).
        If relations exist → constraint problem → Z3 solve.
        If no relations → question → MHN retrieve.
        This is structural, not keyword-based.
        """
        # Step 1: Parse into entity graph
        struct = self.parser.extract_structure(problem_text)
        problem_hdc = struct["hdc"]
        graph = struct["graph"]
        trace.append(f"Parsed: {len(graph.entities)} entities, {len(graph.relations)} relations, {len(graph.constraints)} constraints")

        # Step 2: IDP Refinement (Betteti 2025)
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

        # Step 3: HDC Decomposition (Kanerva 2009)
        trace.append("HDC algebraic decomposition...")
        sub_problems = hdc_decompose(evolved_state, self.sub_problem_prototypes, self.mhn)
        for sp in sub_problems:
            trace.append(f"  [{sp['concept']}] sim={sp['retrieved_sim']:.3f}")

        # Step 4: Retrieve constraint template (if any)
        template = None
        for sp in sub_problems:
            if sp["concept"] == "discover_constraints":
                meta = sp.get("retrieved_meta", {})
                template = meta.get("constraint_template")
                if template:
                    trace.append(f"  Retrieved template: {template.get('primitives', [])}")
                break

        # Step 5: Route by structure (relations exist → constraint; else → question)
        if graph.relations or (template and template.get("primitives")):
            # Constraint path
            trace.append("Structural: relations detected → constraint solving")
            solution = self.z3_solver.solve(graph, template)
            if solution:
                if solution.get("type") == "unsat":
                    trace.append("  Z3: UNSAT")
                else:
                    trace.append(f"  Z3: SAT — primitives: {solution.get('primitives_applied', [])}")
            else:
                trace.append("  Z3: No solution (no matching primitives)")
            intent_label = "constraint"
        else:
            # Question path (no relations in entity graph)
            trace.append("Structural: no relations → question retrieval")
            solution = self._fallback_mhn_solve(thoughts, graph)
            if solution:
                trace.append(f"  MHN: Retrieved answer (sim={solution['similarity']:.2f})")
            else:
                trace.append("  MHN: No matching patterns")
            intent_label = "question"

        # Step 6: Store experience ONLY if we have a real answer
        # (Don't pollute MHN with "I don't know" patterns that have no solution_text)
        sub_solution_vecs = [sp["hdc"] for sp in sub_problems if sp.get("hdc") is not None]
        composed_hdc = hdc_compose(sub_solution_vecs) if sub_solution_vecs else evolved_state
        metadata = {
            "source": "auto_learned", "problem": problem_text[:200],
            "entity_count": len(graph.entities), "relation_count": len(graph.relations),
            "constraint_count": len(graph.constraints),
            "primitives_applied": solution.get("primitives_applied", []) if solution else [],
            "intent": intent_label, "timestamp": time.time(),
        }
        if template:
            metadata["constraint_template"] = template
        if solution and solution.get("formatted") and solution.get("type") not in ("unknown", "unsat"):
            metadata["description"] = solution["formatted"][:500]
            metadata["solution_text"] = solution["formatted"][:500]
            store_experience(self.mhn, problem_hdc, composed_hdc, metadata)
            self.total_learned += 1
            trace.append(f"Stored as new attractor (memory: {len(self.mhn.patterns)} patterns)")
        else:
            trace.append("Not stored — no valid solution to remember")

        confidence = self._compute_confidence(solution, thoughts, sub_problems)
        latency = (time.perf_counter() - t0) * 1000
        return ThinkingResult(
            problem=problem_text, evolved_state=evolved_state,
            thoughts=thoughts, sub_problems=sub_problems,
            solution=solution, confidence=confidence,
            latency_ms=latency, intent=intent_label, trace=trace,
        )

    # ─── Teach ────────────────────────────────────────────────────────

    def teach(self, problem_text, solution_text, constraint_template=None, metadata=None):
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
            "solution": solution_text[:80], "memory_size": len(self.mhn.patterns),
            "message": "Got it. I'll remember this for next time.",
        }

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

    def _fallback_mhn_solve(self, thoughts, graph):
        best_sim = 0
        best_meta = {}
        for t in thoughts:
            if t.retrieved_hdc is None:
                continue
            meta = t.retrieved_metadata
            # Filter out conversation/innate patterns in reasoning path
            if meta.get("intent") == "conversation" or meta.get("source") == "innate":
                continue
            if t.retrieved_similarity > best_sim:
                best_sim = t.retrieved_similarity
                best_meta = meta
        if best_sim > 0.3 and best_meta:
            # ONLY use actual answer text — NEVER fall back to the original problem text
            text = best_meta.get("description") or best_meta.get("solution_text") or ""
            if text and text != "I don't know this one yet.":
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
                else:               return 0.25
            elif sol_type == "conversation":
                return 0.95
        # Filter out conversation/innate patterns when computing fallback confidence
        reasoning_thoughts = [t for t in thoughts if t.retrieved_hdc is not None
                              and t.retrieved_metadata.get("intent") != "conversation"
                              and t.retrieved_metadata.get("source") != "innate"]
        best_sim = max((t.retrieved_similarity for t in reasoning_thoughts), default=0)
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
