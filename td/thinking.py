"""Generic Thinking Loop -- Universal constraint primitives, no hardcoded domains.

Based on:
    - Betteti, Bullo, Baggio, Zampieri (2025) -- IDP Iterative Refinement
      "Input-driven dynamics for robust memory retrieval in Hopfield networks"
      Science Advances. DOI: 10.1126/sciadv.adu6991
      Key insight: The query EVOLVES during retrieval. Each retrieved pattern
      reshapes the energy landscape, guiding convergence regardless of initial state.

    - Kanerva (2009) -- HDC Algebraic Decomposition
      "Hyperdimensional Computing: An Introduction", Cognitive Computation 1(2), 139-159.
      Key insight: bind(A, B) extracts the B-relevant component of A.
      Decomposition is projection, not classification.

    - Ramsauer et al. (2020) -- Automatic Attractor Storage
      "Hopfield Networks is All You Need", ICLR 2021.
      Key insight: MHN stores by appending vectors. No retraining.
      Zero catastrophic forgetting. Memory grows monotonically.

    - Kleyko et al. (2022) -- HDC/VSA Survey, ACM Computing Surveys 55(6), Article 130.
      Key insight: HDC can represent trees, records, and constraint structures
      using bind and bundle. No fixed schema required.

    - Kleyko et al. (2025) -- Principled neuromorphic reservoir computing
      Nature Communications 16(1). DOI: 10.1038/s41467-025-55832-y
      Key insight: CA reservoir extracts features with zero parameters.
      Combined with HDC, enables generic pattern discovery.

Design principle: No domain-specific code. No `if "schedule" in text`.
Instead, the system uses UNIVERSAL constraint primitives:
    - all_different: variables must take distinct values
    - ordered: variables must satisfy ordering (before/after)
    - bounded: variables have domain bounds
    - excluded: certain value combinations are forbidden
    - grouped: variables partitioned with per-group constraints
    - optimized: maximize/minimize an objective function

These primitives are mathematical universals, not domain-specific.
Scheduling uses all_different + ordered + bounded.
Budget uses bounded + grouped + optimized.
Logic uses all_different (for booleans) + excluded.
CSP uses all_different + bounded + grouped.

The system discovers WHICH primitives to apply by:
    1. Retrieving past solved problems from MHN (Ramsauer 2020)
    2. Extracting their constraint templates via HDC binding (Kanerva 2009)
    3. Mapping those templates to the current problem's entity graph
    4. If no template matches, inferring primitives from entity relations
       (e.g., "different" relation -> all_different primitive)
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
    """A single iteration in the thinking process."""
    iteration: int
    state_hdc: np.ndarray
    retrieved_hdc: np.ndarray | None
    retrieved_similarity: float
    retrieved_metadata: dict
    converged: bool
    description: str = ""


@dataclass
class ThinkingResult:
    """Full result of the thinking loop."""
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
    """Check if the IDP state has converged (Betteti et al. 2025)."""
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
    """IDP Iterative Refinement -- the thinking loop.

    Based on Betteti et al. (2025): "The stimulus from the external world
    dynamically reshapes the energy landscape, guiding retrieval regardless
    of initial position."

    The query EVOLVES during retrieval. Each iteration:
        1. Retrieve nearest pattern from MHN given current state
        2. Blend retrieved pattern into current state (energy landscape reshapes)
        3. Check convergence (state barely changed -> done)
    """
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

        # Blend: update state by bundling with retrieved pattern (IDP mechanism)
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

# Generic sub-problem prototypes -- semantic descriptions encoded as HDC vectors.
# These are NOT domain-specific. They describe UNIVERSAL reasoning steps:
#   1. What exists in the problem? (entities)
#   2. What limits apply? (constraints)
#   3. What satisfies the limits? (solution)
#   4. Is the satisfaction correct? (validation)
#
# The prototypes are encoded from natural language sentences describing
# these universal reasoning roles, not from domain vocabulary.

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
    """HDC Algebraic Decomposition -- extract sub-problems via binding.

    Based on Kanerva (2009): bind(A, B) creates association.
    bind(problem, prototype) extracts the prototype-relevant component.

    Since HDC binding is self-inverse, bind(state, prototype) projects
    the problem onto the prototype's "axis" in high-dimensional space.

    Args:
        state_hdc: Evolved HDC state from IDP refinement.
        prototype_vectors: Dict of {name: prototype_hdc} for each reasoning role.
        mhn: MHN for retrieving sub-problem solutions.

    Returns:
        List of sub-problem dicts with HDC vectors and retrieved solutions.
    """
    sub_problems = []

    for concept_name, proto_hdc in prototype_vectors.items():
        # Algebraic extraction: bind the state with the prototype
        # This projects the problem onto the prototype's "axis" (Kanerva 2009)
        component_hdc = bind(state_hdc, proto_hdc)

        # Retrieve from MHN using the extracted component
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
    """Compose sub-solutions into a single HDC vector via bundling.

    Based on Kanerva (2009): bundle (+) creates superposition.
    bundle(sub1, sub2, sub3) ≈ the combination of all sub-solutions.
    """
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
    """Store a problem-solution pair as an MHN attractor.

    Based on Ramsauer et al. (2020): MHN stores by appending vectors.
    No retraining. No gradient descent. Zero catastrophic forgetting.

    After every solve(), this is called to grow the memory.
    """
    mhn.store(problem_hdc, solution_hdc, metadata or {"source": "auto_learned"})


# =========================================================================
# Generic Z3 Constraint Solver -- Universal Primitives
# =========================================================================

class GenericZ3Solver:
    """Generic constraint solver using universal mathematical primitives.

    NO domain-specific code. No `if "schedule" in text`. No hardcoded
    department names or knapsack weights.

    Instead, six UNIVERSAL primitives:
        1. all_different -- variables must take distinct values
        2. ordered -- variables must satisfy ordering (before/after)
        3. bounded -- variables have domain bounds [min, max]
        4. excluded -- certain value combinations are forbidden
        5. grouped -- variables partitioned into groups with constraints
        6. optimized -- maximize/minimize an objective

    These primitives are discovered from:
        a) Retrieved constraint templates from MHN (Ramsauer 2020)
        b) Entity relations in the generic graph (Kanerva 2009)

    Example mappings:
        Scheduling  -> all_different + ordered + bounded
        Budget      -> bounded + grouped + optimized
        Logic       -> all_different (booleans) + excluded
        CSP         -> all_different + bounded + grouped
        Graph color -> all_different + bounded
        Sudoku      -> all_different + grouped
    """

    def __init__(self):
        self.primitive_builders = {
            "all_different": self._build_all_different,
            "ordered": self._build_ordered,
            "bounded": self._build_bounded,
            "excluded": self._build_excluded,
            "grouped": self._build_grouped,
            "optimized": self._build_optimized,
        }

    def solve(self, graph: GenericEntityGraph, template: dict | None = None) -> dict | None:
        """Solve a generic entity graph using Z3.

        Args:
            graph: GenericEntityGraph with entities, relations, constraints.
            template: Optional constraint template from MHN retrieval.
                      If provided, uses template-driven solving.
                      If None, uses relation-driven primitive inference.

        Returns:
            Solution dict with formatted output, or None if unsat/unknown.
        """
        try:
            from z3 import (
                Solver, Optimize, Int, Bool, sat, unsat,
                And, Or, Not, Implies, Distinct, Sum, If,
            )
        except ImportError:
            return None

        if not graph.entities:
            return None

        # ─── Step 1: Create Z3 variables generically ─────────────
        # Variable type inferred from entity properties and constraints
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
            # Template-driven: extract primitives from template
            primitives = self._extract_primitives_from_template(template, graph)
        else:
            # Relation-driven: infer primitives from entity relations
            primitives = self._infer_primitives_from_relations(graph)

            # Also add explicit constraints from graph
            for c in graph.constraints:
                primitive = self._constraint_to_primitive(c, graph)
                if primitive:
                    primitives.append(primitive)

        if not primitives:
            return None

        # ─── Step 3: Build and solve ───────────────────────────────
        # Determine if optimization is needed
        has_opt = any(p["type"] == "optimized" for p in primitives)

        if has_opt:
            solver = Optimize()
        else:
            solver = Solver()

        # Apply each primitive
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

    # ─── Primitive Builders (universal, not domain-specific) ─────

    def _build_all_different(self, solver, z3_vars, primitive, graph):
        """all_different: variables must take distinct values."""
        from z3 import Distinct
        subjects = [z3_vars[sid] for sid in primitive["subjects"] if sid in z3_vars]
        if len(subjects) > 1:
            solver.add(Distinct(subjects))

    def _build_ordered(self, solver, z3_vars, primitive, graph):
        """ordered: variables must satisfy ordering relation."""
        subjects = [sid for sid in primitive["subjects"] if sid in z3_vars]
        order = primitive.get("order", "ascending")  # ascending or descending
        for i in range(len(subjects) - 1):
            if order == "ascending":
                solver.add(z3_vars[subjects[i]] < z3_vars[subjects[i + 1]])
            else:
                solver.add(z3_vars[subjects[i]] > z3_vars[subjects[i + 1]])

    def _build_bounded(self, solver, z3_vars, primitive, graph):
        """bounded: variables have domain bounds [min, max]."""
        subjects = [sid for sid in primitive["subjects"] if sid in z3_vars]
        lo = primitive.get("min", 0)
        hi = primitive.get("max", 100)
        for sid in subjects:
            solver.add(z3_vars[sid] >= lo)
            solver.add(z3_vars[sid] <= hi)

    def _build_excluded(self, solver, z3_vars, primitive, graph):
        """excluded: certain value combinations are forbidden.

        For boolean variables: if A then not B (A excludes B).
        For integer variables: A != B (mutual exclusion).
        """
        subjects = [sid for sid in primitive["subjects"] if sid in z3_vars]
        if len(subjects) == 2:
            # Mutual exclusion
            solver.add(z3_vars[subjects[0]] != z3_vars[subjects[1]])
        elif len(subjects) == 1:
            # Exclude specific value
            exclude_val = primitive.get("exclude_value", 0)
            solver.add(z3_vars[subjects[0]] != exclude_val)

    def _build_grouped(self, solver, z3_vars, primitive, graph):
        """grouped: variables partitioned into groups with constraints.

        Example: sum of group <= limit (budget allocation).
        Example: all_different within each group (Sudoku).
        """
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
            elif constraint == "distinct":
                if len(group_vars) > 1:
                    solver.add(Distinct(group_vars))

    def _build_optimized(self, solver, z3_vars, primitive, graph):
        """optimized: maximize/minimize an objective function.

        Currently supports: maximize minimum allocation (fairness).
        """
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

    # ─── Inference Methods (no hardcoding) ───────────────────────

    def _infer_var_type(self, entity: dict, graph: GenericEntityGraph) -> str:
        """Infer Z3 variable type from entity properties.

        bool: entity has boolean-like type vector or is in a logic constraint.
        int: default.
        """
        # Check if entity is involved in a boolean constraint
        for c in graph.constraints:
            if entity["id"] in c["subjects"]:
                if c["params"].get("type") == "bool":
                    return "bool"
        # Check type vector similarity to boolean prototype
        # (This would be retrieved from MHN in a full implementation)
        return "int"

    def _extract_primitives_from_template(self, template: dict, graph: GenericEntityGraph) -> list[dict]:
        """Extract primitive constraints from an MHN-retrieved template.

        Templates are stored metadata from past solved problems.
        They describe which primitives to apply and to which entities.
        """
        primitives = []
        for p in template.get("primitives", []):
            # Map template subjects to current graph entities by HDC similarity
            mapped_subjects = self._map_subjects(p.get("subjects", []), p.get("selector"), graph)
            if mapped_subjects:
                primitives.append({
                    "type": p["type"],
                    "subjects": mapped_subjects,
                    **{k: v for k, v in p.items() if k not in ("type", "subjects")},
                })
        return primitives

    def _infer_primitives_from_relations(self, graph: GenericEntityGraph) -> list[dict]:
        """Infer constraint primitives from entity relations.

        This is the KEY generic mechanism. Relations are discovered by
        the parser (not hardcoded). Each relation type maps to a primitive:

            "different"  -> all_different
            "before"     -> ordered (ascending)
            "after"      -> ordered (descending)
            "same"       -> excluded (A != B is wrong, actually equality)
            "excludes"   -> excluded
            "limited"    -> bounded
            "grouped"    -> grouped
            "optimize"   -> optimized
        """
        primitives = []
        entity_ids = [e["id"] for e in graph.entities]

        # Collect relation types
        rel_groups = {}
        for rel in graph.relations:
            rtype = rel.get("rel_type", "unknown")
            if rtype not in rel_groups:
                rel_groups[rtype] = []
            rel_groups[rtype].append((rel["src"], rel["tgt"]))

        # Map relation types to primitives
        if "different" in rel_groups or "distinct" in rel_groups:
            # all_different on all entities involved in "different" relations
            involved = set()
            for rtype in ["different", "distinct"]:
                for src, tgt in rel_groups.get(rtype, []):
                    involved.add(src)
                    involved.add(tgt)
            if involved:
                primitives.append({"type": "all_different", "subjects": list(involved)})

        if "before" in rel_groups:
            # Build ordered chain from before relations
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
            # bounded: infer bounds from entity text or constraint params
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

        # Default: if no relations but entities exist, add bounded
        if not primitives and entity_ids:
            primitives.append({"type": "bounded", "subjects": entity_ids, "min": 0, "max": 100})

        return primitives

    def _constraint_to_primitive(self, constraint: dict, graph: GenericEntityGraph) -> dict | None:
        """Convert an explicit graph constraint to a primitive.

        Explicit constraints come from MHN-retrieved constraint patterns.
        """
        params = constraint.get("params", {})
        ptype = params.get("primitive_type", "bounded")
        return {
            "type": ptype,
            "subjects": constraint["subjects"],
            **{k: v for k, v in params.items() if k != "primitive_type"},
        }

    def _map_subjects(self, template_subjects: list, selector: np.ndarray | None, graph: GenericEntityGraph) -> list[str]:
        """Map template subjects to current graph entities by HDC similarity."""
        if not selector:
            # No selector: try to match by count
            return [e["id"] for e in graph.entities[:len(template_subjects)]]

        matches = []
        for e in graph.entities:
            sim = similarity(e["hdc"], selector)
            matches.append((e["id"], sim))

        matches.sort(key=lambda x: x[1], reverse=True)
        return [eid for eid, sim in matches if sim > 0.30]

    def _build_order_chain(self, relations: list[tuple]) -> list[str]:
        """Build an ordered chain from before/after relations."""
        # Simple topological sort
        from collections import defaultdict, deque
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        nodes = set()

        for src, tgt in relations:
            graph[src].append(tgt)
            in_degree[tgt] += 1
            nodes.add(src)
            nodes.add(tgt)

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

    def _infer_bounds(self, graph: GenericEntityGraph) -> dict | None:
        """Infer domain bounds from entity texts (numbers in text)."""
        max_val = 100
        for e in graph.entities:
            text = e["text"]
            # Extract numbers from entity text
            nums = [int(w) for w in text.split() if w.isdigit()]
            if nums:
                max_val = max(max_val, max(nums) * 2)
        return {"min": 0, "max": max_val}

    def _build_groups(self, relations: list[tuple]) -> list[list[str]]:
        """Build groups from grouped relations."""
        # Union-find for grouping
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

    def _format_solution(self, model, z3_vars, graph, primitives) -> dict:
        """Format Z3 solution into human-readable output."""
        lines = []
        for e in graph.entities:
            val = model.eval(z3_vars[e["id"]], model_completion=True)
            lines.append(f"  {e['text']}: {val}")

        # Add primitive summary
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
    """Generic reasoning engine -- universal primitives, no hardcoded domains.

    1. IDP: Query evolves through iterative MHN retrieval (Betteti 2025)
    2. HDC: Sub-problems extracted via algebraic binding (Kanerva 2009)
    3. Z3: Universal constraint primitives, not domain models
    4. Auto-storage: Every solve grows the memory (Ramsauer 2020)
    """

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
            dim=dim,
            min_similarity=0.01,
            idp_enabled=False,
        ))
        self.parser = GenericNLParser(vocab, self.mhn, dim=dim)
        self.z3_solver = GenericZ3Solver()

        self.max_idp_iterations = max_idp_iterations
        self.idp_blend_factor = idp_blend_factor
        self.convergence_threshold = convergence_threshold
        self.pure_mode = pure_mode

        # Semantic prototypes for universal reasoning roles (Kanerva 2009)
        self.sub_problem_prototypes = self._build_semantic_prototypes()

        # Stats
        self.total_thinks = 0
        self.total_learned = 0
        self.avg_iterations = 0.0
        self.seed_count = 0

        if not pure_mode:
            self._load_minimal_seed()

    def think(self, problem_text: str, context: dict | None = None) -> ThinkingResult:
        """Run the full generic thinking loop on a problem.

        1. Parse into generic entity graph (no hardcoded types)
        2. IDP: Iteratively refine the query state
        3. HDC: Decompose evolved state into universal reasoning roles
        4. Retrieve constraint template from MHN (if any)
        5. Generic Z3 solving with universal primitives
        6. Store (problem, solution) as new attractor
        """
        t0 = time.perf_counter()
        trace = []
        self.total_thinks += 1

        # ─── Step 1: Generic Parse ────────────────────────────────
        struct = self.parser.extract_structure(problem_text)
        problem_hdc = struct["hdc"]
        graph = struct["graph"]
        trace.append(f"Parsed: {len(graph.entities)} entities, {len(graph.relations)} relations, {len(graph.constraints)} constraints")

        # ─── Step 2: IDP Iterative Refinement ─────────────────────
        trace.append(f"IDP refinement (max {self.max_idp_iterations} iterations)...")
        evolved_state, thoughts = idp_refine(
            problem_hdc,
            self.mhn,
            max_iterations=self.max_idp_iterations,
            convergence_threshold=self.convergence_threshold,
            blend_factor=self.idp_blend_factor,
        )

        for t in thoughts:
            trace.append(f"  Iter {t.iteration}: sim={t.retrieved_similarity:.3f} "
                        f"{'-> CONVERGED' if t.converged else '-> continuing'}")

        avg_iters = sum(t.iteration for t in thoughts) / max(len(thoughts), 1)
        self.avg_iterations = (self.avg_iterations * (self.total_thinks - 1) + avg_iters) / self.total_thinks

        # ─── Step 3: HDC Algebraic Decomposition ──────────────────
        trace.append("HDC algebraic decomposition (universal roles)...")
        sub_problems = hdc_decompose(evolved_state, self.sub_problem_prototypes, self.mhn)

        for sp in sub_problems:
            trace.append(f"  [{sp['concept']}] sim={sp['retrieved_sim']:.3f}")

        # ─── Step 4: Retrieve Constraint Template ──────────────────
        # Look for constraint templates in the "discover_constraints" sub-problem
        template = None
        for sp in sub_problems:
            if sp["concept"] == "discover_constraints":
                meta = sp.get("retrieved_meta", {})
                template = meta.get("constraint_template")
                if template:
                    trace.append(f"  Retrieved constraint template: {template.get('primitives', [])}")
                break

        # ─── Step 5: Generic Z3 Solving ───────────────────────────
        trace.append("Generic Z3 solving (universal primitives)...")
        solution = self.z3_solver.solve(graph, template)

        if solution:
            if solution.get("type") == "unsat":
                trace.append("  Z3: UNSAT -- constraints contradictory")
            else:
                trace.append(f"  Z3: SAT -- primitives: {solution.get('primitives_applied', [])}")
        else:
            trace.append("  Z3: No solution (no matching primitives or unknown)")

        # ─── Step 5b: If Z3 only used default 'bounded', try MHN fallback ─
        if solution and solution.get("type") == "generic_csp":
            prims = solution.get("primitives_applied", [])
            if prims == ["bounded"]:
                # Default bounded is not meaningful — check MHN first
                fallback = self._fallback_mhn_solve(thoughts, graph)
                if fallback and fallback.get("similarity", 0) > 0.85:
                    solution = fallback
                    trace.append("  MHN override: better retrieval than default Z3")

        # ─── Step 6: Fallback to MHN retrieval if Z3 fails ────────
        if not solution or solution.get("type") == "unsat":
            fallback = self._fallback_mhn_solve(thoughts, graph)
            if fallback:
                solution = fallback
                trace.append("  Fallback: MHN retrieved similar solution")

        # ─── Step 7: Automatic Attractor Storage ──────────────────
        # Compose solution HDC from sub-problem vectors
        sub_solution_vecs = [sp["hdc"] for sp in sub_problems if sp.get("hdc") is not None]
        composed_hdc = hdc_compose(sub_solution_vecs) if sub_solution_vecs else evolved_state

        # Store with constraint template if Z3 found one
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

        # ─── Confidence (honest, based on actual evidence) ─────────
        confidence = self._compute_confidence(solution, thoughts, sub_problems)
        latency = (time.perf_counter() - t0) * 1000

        return ThinkingResult(
            problem=problem_text,
            evolved_state=evolved_state,
            thoughts=thoughts,
            sub_problems=sub_problems,
            solution=solution,
            confidence=confidence,
            latency_ms=latency,
            trace=trace,
        )

    def teach(self, problem_text: str, solution_text: str, constraint_template: dict | None = None, metadata: dict | None = None):
        """Learn from explicit human teaching.

        The human provides a problem, its solution, and optionally a
        constraint template describing the universal primitives used.
        TD encodes both as HDC vectors and stores them as a new attractor.

        Args:
            problem_text: The problem or question.
            solution_text: The solution or answer.
            constraint_template: Optional dict describing primitives used,
                e.g., {"primitives": [{"type": "all_different", ...}]}
            metadata: Optional metadata.

        Returns:
            dict with storage confirmation.
        """
        problem_hdc = self.parser.parse(problem_text)
        solution_hdc = self.parser.parse(solution_text)

        title = (metadata or {}).get("title", " ".join(solution_text.split()[:6]))

        meta = {
            "source": "human_taught",
            "title": title,
            "effectiveness": (metadata or {}).get("effectiveness", 0.75),
            "description": solution_text,
            "problem": problem_text[:200],
            "solution_text": solution_text[:200],
            "timestamp": time.time(),
        }
        if constraint_template:
            meta["constraint_template"] = constraint_template
        if metadata:
            meta.update(metadata)

        self.mhn.store(problem_hdc, solution_hdc, meta)
        self.total_learned += 1

        return {
            "status": "learned",
            "problem": problem_text[:80],
            "solution": solution_text[:80],
            "memory_size": len(self.mhn.patterns),
            "message": "Got it. I'll remember this for next time.",
        }

    def needs_teaching(self, result: ThinkingResult) -> bool:
        """Check if the system should ask to be taught."""
        if self.total_thinks < 3 and len(self.mhn.patterns) < 10:
            return True
        if result.confidence < 0.25:
            return True
        return False

    # ─── Internal Methods ───────────────────────────────────────────────

    def _build_semantic_prototypes(self) -> dict[str, np.ndarray]:
        """Build sub-problem prototypes from universal reasoning role sentences.

        These are NOT domain-specific. They describe universal reasoning steps
        that apply to ANY problem: scheduling, budget, logic, CSP, etc.
        """
        descriptions = {
            "discover_entities": "find all objects and elements that appear in this problem",
            "discover_constraints": "what are the limits and rules that restrict the solution",
            "find_solution": "how to assign values that satisfy all limits and rules",
            "validate_result": "check that the assignment satisfies every requirement and limit",
        }
        return {name: self.parser.parse(text) for name, text in descriptions.items()}

    def _fallback_mhn_solve(self, thoughts: list[Thought], graph: GenericEntityGraph) -> dict | None:
        """Fallback: retrieve a similar past solution from MHN."""
        best_sim = 0
        best_meta = {}
        for t in thoughts:
            if t.retrieved_hdc is not None and t.retrieved_similarity > best_sim:
                best_sim = t.retrieved_similarity
                best_meta = t.retrieved_metadata

        if best_sim > 0.3 and best_meta:
            text = best_meta.get("description", best_meta.get("solution_text", ""))
            if text:
                return {
                    "type": "learned",
                    "formatted": text,
                    "similarity": best_sim,
                }
        return None

    def _compute_confidence(self, solution: dict | None, thoughts: list[Thought], sub_problems: list[dict]) -> float:
        """Compute honest confidence based on actual evidence."""
        if solution:
            sol_type = solution.get("type", "unknown")
            if sol_type == "generic_csp":
                # Confidence scales with number of primitives applied
                prims = solution.get("primitives_applied", [])
                if len(prims) <= 1 and prims == ["bounded"]:
                    # Default bounded with no real constraints = low confidence
                    return 0.30
                # Multiple primitives = real constraint solving
                return min(0.60 + len(prims) * 0.10, 0.90)
            elif sol_type == "unsat":
                return 0.95
            elif sol_type == "learned":
                sim = solution.get("similarity", 0.5)
                # Honest scaling: only exact matches = high confidence
                if sim < 0.40:       return 0.10
                elif sim < 0.60:    return 0.25
                elif sim < 0.75:    return 0.40
                elif sim < 0.90:    return 0.55
                else:                return min(sim * 0.85, 0.85)

        # No Z3 solution: confidence based on IDP retrieval quality
        best_sim = max((t.retrieved_similarity for t in thoughts if t.retrieved_hdc is not None), default=0)
        if best_sim > 0.5:
            return min(best_sim * 0.7, 0.70)
        elif best_sim > 0.3:
            return min(best_sim * 0.5, 0.40)
        else:
            return 0.20

    def _load_minimal_seed(self):
        """Load minimal seed patterns -- innate reflexes, not pretraining."""
        try:
            from .minimal_seed import ALL_SEED_PATTERNS
            for pattern in ALL_SEED_PATTERNS:
                query_hdc = self.parser.parse(pattern.text)
                solution_hdc = self.parser.parse(pattern.solution)
                meta = {"label": "seed", **pattern.metadata}
                # If seed has constraint template, store it
                if hasattr(pattern, "constraint_template"):
                    meta["constraint_template"] = pattern.constraint_template
                self.mhn.store(query_hdc, solution_hdc, meta)
                self.seed_count += 1
        except ImportError:
            pass

    def stats(self) -> dict:
        """Return engine statistics."""
        total = len(self.mhn.patterns)
        seed_pct = (self.seed_count / total * 100) if total > 0 else 0
        return {
            "total_thinks": self.total_thinks,
            "total_learned": self.total_learned,
            "memory_size": total,
            "seed_patterns": self.seed_count,
            "learned_patterns": total - self.seed_count,
            "seed_ratio_pct": round(seed_pct, 1),
            "avg_iterations": round(self.avg_iterations, 2),
            "pure_mode": self.pure_mode,
        }
