"""TD Decomposer — Recursive problem decomposition via HDC prototype matching.

This is the PRIMARY reasoning path for Thinking Dust. Instead of classifying
input into hardcoded domains, the decomposer:

1. Encodes the problem as HDC vector
2. Compares against sub-problem prototypes (what TYPE of sub-problem is this?)
3. Recursively breaks the problem into smaller pieces
4. For each leaf sub-problem: retrieve from MHN or solve with Z3
5. Composes the full solution by binding sub-solutions

The router (hierarchical_router.py) becomes a FALLBACK for when the decomposer
has low confidence. Over time, as the decomposer learns, router usage should
drop below 10%.

Design based on Kimi K2.6's incremental migration plan:
- Keep router as safety net (don't delete working code)
- Build decomposer as new primary path
- Measure fallback rate over 4 weeks
- Target: <10% router usage, then remove router

Prototype Categories (50 hand-written):
    CONSTRAINT_SATISFACTION — "find X such that constraints hold"
    OPTIMIZATION — "find the best X according to objective"
    SCHEDULING — "assign time slots to tasks"
    ALLOCATION — "assign resources to tasks"
    TRANSFORMATION — "convert X to format Y"
    VALIDATION — "check if X meets criteria"
    COMPARISON — "compare X and Y"
    SEQUENCING — "order these items"
    GROUPING — "cluster these items"
    FILTERING — "select subset matching criteria"
    COUNTING — "how many X satisfy Y"
    SEARCH — "find X in dataset"
    SORTING — "arrange by some key"
    PARSING — "extract structure from text"
    GENERATION — "create new X from spec"
    PROOF — "show that X implies Y"
    DEBUGGING — "find bug in code"
    EXPLANATION — "explain why X happens"
    PREDICTION — "what happens if X"
    PLANNING — "sequence of actions to reach goal"
    Each has ~3 sub-prototypes for specialization.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .perception.hdc import (
    ConceptVocabulary, build_default_vocabulary,
    generate_hypervector, bind, bundle, similarity, permute,
)
from .perception.nl_parser import NLParser
from .memory.mhn import ModernHopfieldNetwork, MHNConfig
from .reasoning.z3_bridge import Z3Bridge, Z3Result
from .reasoning.confidence import compute_confidence


# =========================================================================
# Sub-Problem Prototypes (50 hand-written)
# =========================================================================

PROTOTYPE_CATEGORIES = [
    # --- Constraint Satisfaction (find X such that constraints hold) ---
    "constraint_satisfaction",
    "constraint_scheduling",       # time-based constraints
    "constraint_assignment",       # assign values to variables

    # --- Optimization ---
    "optimization",
    "optimization_linear",         # linear objective
    "optimization_multi_objective", # multiple competing objectives

    # --- Scheduling ---
    "scheduling",
    "scheduling_meetings",         # calendar/time-slot allocation
    "scheduling_tasks",            # task ordering with deadlines

    # --- Allocation ---
    "allocation",
    "allocation_resources",        # assign people/equipment to tasks
    "allocation_budget",           # distribute money/budget

    # --- Transformation ---
    "transformation",
    "transformation_format",       # convert format (CSV→JSON, etc)
    "transformation_data",         # transform data values

    # --- Validation ---
    "validation",
    "validation_schema",           # check against schema
    "validation_logic",            # check logical consistency

    # --- Comparison ---
    "comparison",
    "comparison_ranking",          # rank items by criteria
    "comparison_equality",         # check if items match

    # --- Sequencing ---
    "sequencing",
    "sequencing_temporal",         # order by time
    "sequencing_dependency",       # topological sort

    # --- Grouping ---
    "grouping",
    "grouping_cluster",            # cluster by similarity
    "grouping_partition",          # split into groups

    # --- Filtering ---
    "filtering",
    "filtering_query",             # SQL-like query
    "filtering_search",            # search for items

    # --- Counting ---
    "counting",
    "counting_conditional",        # count items meeting condition
    "counting_set",                # set cardinality

    # --- Search ---
    "search",
    "search_exact",                # find exact match
    "search_approximate",          # find nearest match

    # --- Sorting ---
    "sorting",
    "sorting_single_key",          # sort by one criterion
    "sorting_multi_key",           # sort by multiple criteria

    # --- Parsing ---
    "parsing",
    "parsing_structured",          # parse JSON/CSV/YAML
    "parsing_natural",             # parse natural language

    # --- Generation ---
    "generation",
    "generation_template",         # fill in template
    "generation_creative",         # create from scratch

    # --- Proof ---
    "proof",
    "proof_logical",               # formal logic proof
    "proof_correctness",           # prove code is correct

    # --- Debugging ---
    "debugging",
    "debugging_logic_error",       # find logic bug
    "debugging_runtime_error",     # find crash cause

    # --- Explanation ---
    "explanation",
    "explanation_causal",          # why did X happen
    "explanation_summary",         # summarize what happened

    # --- Prediction ---
    "prediction",
    "prediction_conditional",      # what if X happens
    "prediction_trend",            # what will happen over time

    # --- Planning ---
    "planning",
    "planning_sequential",         # step-by-step plan
    "planning_conditional",        # plan with branches
]


# Keywords that trigger each prototype category
PROTOTYPE_KEYWORDS: dict[str, list[str]] = {
    "constraint_satisfaction": ["satisfy", "meet", "fulfill", "subject to", "such that", "constrained"],
    "constraint_scheduling": ["schedule", "calendar", "time slot", "availability", "book"],
    "constraint_assignment": ["assign", "allocate", "match", "pair"],
    "optimization": ["optimize", "minimize", "maximize", "best", "optimal", "efficient"],
    "optimization_linear": ["cheapest", "fastest", "shortest", "minimum cost", "maximum profit"],
    "optimization_multi_objective": ["balance", "trade-off", "tradeoff", "compromise", "pareto"],
    "scheduling": ["schedule", "timetable", "agenda", "plan time", "arrange"],
    "scheduling_meetings": ["meeting", "appointment", "booking", "reservation", "calendar"],
    "scheduling_tasks": ["task", "deadline", "sprint", "milestone", "deliverable"],
    "allocation": ["allocate", "distribute", "assign", "budget", "divide"],
    "allocation_resources": ["resource", "team", "staff", "equipment", "workload"],
    "allocation_budget": ["budget", "cost", "spend", "expense", "financial"],
    "transformation": ["convert", "transform", "change", "translate", "map"],
    "transformation_format": ["csv", "json", "yaml", "xml", "format", "export"],
    "transformation_data": ["normalize", "standardize", "clean", "process"],
    "validation": ["validate", "check", "verify", "ensure", "confirm"],
    "validation_schema": ["schema", "columns", "fields", "structure", "format check"],
    "validation_logic": ["consistent", "contradiction", "integrity", "correct"],
    "comparison": ["compare", "versus", "vs", "difference", "better"],
    "comparison_ranking": ["rank", "sort by quality", "best to worst", "priority"],
    "comparison_equality": ["match", "identical", "same", "duplicate", "equal"],
    "sequencing": ["order", "sequence", "arrange", "sort", "prioritize"],
    "sequencing_temporal": ["chronological", "timeline", "date order", "sequence"],
    "sequencing_dependency": ["depend", "before", "after", "prerequisite", "topological"],
    "grouping": ["group", "cluster", "categorize", "classify", "segment"],
    "grouping_cluster": ["similar", "cluster", "k-means", "nearest"],
    "grouping_partition": ["split", "partition", "divide into groups", "bin"],
    "filtering": ["filter", "select", "find", "where", "query"],
    "filtering_query": ["sql", "select where", "condition", "criteria"],
    "filtering_search": ["search", "find", "locate", "look up"],
    "counting": ["count", "how many", "number of", "total"],
    "counting_conditional": ["how many meet", "count where", "count if"],
    "counting_set": ["unique", "distinct", "cardinality", "set size"],
    "search": ["find", "search", "locate", "lookup", "get"],
    "search_exact": ["exact match", "find by id", "lookup", "retrieve"],
    "search_approximate": ["fuzzy", "similar", "closest", "nearest", "approximate"],
    "sorting": ["sort", "order", "arrange", "organize"],
    "sorting_single_key": ["sort by", "order by", "ascending", "descending"],
    "sorting_multi_key": ["sort by multiple", "primary secondary", "multi-key"],
    "parsing": ["parse", "extract", "read", "interpret", "analyze structure"],
    "parsing_structured": ["json", "csv", "xml", "yaml", "parse file"],
    "parsing_natural": ["extract from text", "parse sentence", "ner", "entity"],
    "generation": ["generate", "create", "produce", "make", "build"],
    "generation_template": ["template", "fill in", "populate", "from pattern"],
    "generation_creative": ["design", "compose", "invent", "brainstorm"],
    "proof": ["prove", "show that", "demonstrate", "verify formally"],
    "proof_logical": ["implies", "therefore", "if then", "modus", "syllogism"],
    "proof_correctness": ["correct", "sound", "secure", "no bugs", "verified"],
    "debugging": ["debug", "bug", "error", "wrong", "broken", "fix"],
    "debugging_logic_error": ["wrong result", "incorrect", "logic error", "off by"],
    "debugging_runtime_error": ["crash", "exception", "null pointer", "stack trace"],
    "explanation": ["explain", "why", "reason", "cause", "because"],
    "explanation_causal": ["why did", "cause of", "reason for", "led to"],
    "explanation_summary": ["summarize", "overview", "recap", "digest"],
    "prediction": ["predict", "forecast", "estimate", "project"],
    "prediction_conditional": ["what if", "suppose", "assume", "hypothetical"],
    "prediction_trend": ["trend", "future", "projection", "growth"],
    "planning": ["plan", "strategy", "roadmap", "steps", "approach"],
    "planning_sequential": ["step by step", "procedure", "workflow", "process"],
    "planning_conditional": ["if then else", "contingency", "fallback", "backup plan"],
}


# =========================================================================
# Data Structures
# =========================================================================

@dataclass
class SubProblem:
    """A single sub-problem in a decomposition tree."""
    description: str
    prototype: str               # prototype category name
    hdc_vector: np.ndarray       # HDC encoding of this sub-problem
    similarity: float            # similarity to matched prototype
    children: list["SubProblem"] = field(default_factory=list)
    solution: dict | None = None  # solution from MHN/Z3
    source: str = ""              # "mhn", "z3", "router_fallback", "unsolved"

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def is_solved(self) -> bool:
        return self.solution is not None


@dataclass
class DecompositionResult:
    """Full result of problem decomposition."""
    root: SubProblem
    prototype_match: str
    prototype_similarity: float
    sub_problems: list[SubProblem]
    solution: dict | None
    confidence: float
    latency_ms: float
    trace: list[str] = field(default_factory=list)
    used_router_fallback: bool = False

    @property
    def is_solved(self) -> bool:
        return self.solution is not None

    def summary(self) -> str:
        lines = [
            f"=== TD Decomposition ===",
            f"Prototype: {self.prototype_match} (sim={self.prototype_similarity:.3f})",
            f"Sub-problems: {len(self.sub_problems)}",
            f"Confidence: {self.confidence:.3f}",
            f"Latency: {self.latency_ms:.1f}ms",
            f"Router fallback: {'YES' if self.used_router_fallback else 'no'}",
            "",
        ]
        for i, sp in enumerate(self.sub_problems):
            status = "✅" if sp.is_solved else "❌"
            lines.append(f"  {status} [{sp.prototype}] {sp.description[:60]}")
        lines.append("")
        lines.append("Trace:")
        for t in self.trace:
            lines.append(f"  → {t}")
        return "\n".join(lines)


# =========================================================================
# Decomposer
# =========================================================================

class PrototypeBank:
    """Bank of sub-problem prototypes encoded as HDC vectors.

    Prototypes are HDC vectors representing canonical sub-problem types.
    Input is matched against prototypes via cosine similarity to determine
    what type of sub-problem it is.
    """

    def __init__(self, vocab: ConceptVocabulary, dim: int = 10_000):
        self.vocab = vocab
        self.dim = dim
        self.prototypes: dict[str, np.ndarray] = {}
        self._build()

    def _build(self):
        """Build prototype HDC vectors from keyword sets.

        Each prototype is encoded using the SAME pipeline as input:
        NLParser.parse(keyword_sentence) → HDC vector.
        This ensures prototypes and inputs are in the same HDC "space".

        We construct a synthetic sentence from keywords and parse it,
        so the role-filler binding structure matches.
        """
        parser = NLParser(self.vocab)
        for proto_name, keywords in PROTOTYPE_KEYWORDS.items():
            # Build a synthetic sentence that the parser can understand
            # Pick the 3 most distinctive keywords
            key_phrases = keywords[:3]
            sentence = " ".join(key_phrases)
            proto_vec = parser.parse(sentence)
            self.prototypes[proto_name] = proto_vec

    def match(self, hdc_vector: np.ndarray, top_k: int = 3) -> list[tuple[str, float]]:
        """Find best matching prototypes for an HDC vector.

        Args:
            hdc_vector: Problem HDC vector.
            top_k: Number of top matches to return.

        Returns:
            List of (prototype_name, similarity) tuples, sorted by similarity.
        """
        scores = []
        for name, proto_vec in self.prototypes.items():
            sim = similarity(hdc_vector, proto_vec)
            scores.append((name, sim))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]


class Decomposer:
    """Recursive problem decomposer — the PRIMARY reasoning path for TD.

    Pipeline:
        1. Encode problem → HDC vector
        2. Match against prototypes (what type of sub-problem is this?)
        3. Check MHN for similar past solutions
        4. If novel: decompose into sub-problems recursively
        5. For each leaf: try MHN → Z3 → router fallback
        6. Compose full solution

    The router (hierarchical_router) is only used as FALLBACK when the
    decomposer has low confidence. Over time, router usage should drop <10%.
    """

    def __init__(
        self,
        vocab: ConceptVocabulary | None = None,
        mhn: ModernHopfieldNetwork | None = None,
        z3_bridge: Z3Bridge | None = None,
        router=None,  # HierarchicalRouter (fallback only)
        dim: int = 10_000,
        similarity_threshold: float = 0.15,  # min prototype similarity
        mhn_threshold: float = 0.25,        # min MHN similarity
        max_depth: int = 3,                 # max decomposition depth
    ):
        self.vocab = vocab or build_default_vocabulary(dim=dim)
        self.mhn = mhn or ModernHopfieldNetwork(MHNConfig(dim=dim, min_similarity=0.15))
        self.z3_bridge = z3_bridge or Z3Bridge()
        self.router = router  # Fallback, may be None
        self.dim = dim
        self.similarity_threshold = similarity_threshold
        self.mhn_threshold = mhn_threshold
        self.max_depth = max_depth

        # Build prototype bank
        self.prototypes = PrototypeBank(self.vocab, dim)

        # Stats for measuring router fallback rate
        self.stats = {
            "total_decompositions": 0,
            "router_fallbacks": 0,
            "mhn_hits": 0,
            "z3_solves": 0,
            "unsolved": 0,
        }

    def decompose(
        self,
        problem_text: str,
        context: dict | None = None,
        depth: int = 0,
    ) -> DecompositionResult:
        """Decompose a problem into sub-problems and solve.

        Args:
            problem_text: Natural language problem description.
            context: Optional context (constraints, prior knowledge).
            depth: Current recursion depth (internal).

        Returns:
            DecompositionResult with solution tree.
        """
        t0 = time.perf_counter()
        trace = []
        context = context or {}
        self.stats["total_decompositions"] += 1

        # 1. Encode problem as HDC
        parser = NLParser(self.vocab)
        hdc_vector = parser.parse(problem_text)
        trace.append(f"Encoded problem ({len(problem_text)} chars)")

        # 2. Match against prototypes
        proto_matches = self.prototypes.match(hdc_vector, top_k=3)
        best_proto, best_sim = proto_matches[0]
        trace.append(f"Best prototype: {best_proto} (sim={best_sim:.3f})")

        # 3. Check MHN first (have we solved something like this before?)
        mhn_results = self.mhn.retrieve(hdc_vector, top_k=3)

        if mhn_results and mhn_results[0][1] >= self.mhn_threshold:
            # High MHN similarity — retrieve and return
            _, sim, meta = mhn_results[0]
            trace.append(f"MHN hit (sim={sim:.3f}) — retrieving stored solution")
            self.stats["mhn_hits"] += 1

            latency = (time.perf_counter() - t0) * 1000
            return DecompositionResult(
                root=SubProblem(
                    description=problem_text,
                    prototype=best_proto,
                    hdc_vector=hdc_vector,
                    similarity=best_sim,
                    solution=meta.get("solution"),
                    source="mhn",
                ),
                prototype_match=best_proto,
                prototype_similarity=best_sim,
                sub_problems=[],
                solution=meta.get("solution"),
                confidence=sim,
                latency_ms=latency,
                trace=trace,
            )

        # 4. Novel problem — try to decompose
        if best_sim < self.similarity_threshold or depth >= self.max_depth:
            # Can't decompose further — try Z3, then router fallback
            trace.append(f"Can't decompose (sim={best_sim:.3f}, depth={depth})")
            solution, source, confidence = self._solve_leaf(problem_text, hdc_vector, context, trace)

            latency = (time.perf_counter() - t0) * 1000
            used_router = source == "router_fallback"
            if used_router:
                self.stats["router_fallbacks"] += 1

            return DecompositionResult(
                root=SubProblem(
                    description=problem_text,
                    prototype=best_proto,
                    hdc_vector=hdc_vector,
                    similarity=best_sim,
                    solution=solution,
                    source=source,
                ),
                prototype_match=best_proto,
                prototype_similarity=best_sim,
                sub_problems=[],
                solution=solution,
                confidence=confidence,
                latency_ms=latency,
                trace=trace,
                used_router_fallback=used_router,
            )

        # 5. Decompose into sub-problems
        trace.append(f"Decomposing into sub-problems (depth={depth})")
        sub_descriptions = self._generate_sub_problems(problem_text, best_proto, context)
        sub_problems: list[SubProblem] = []

        for sub_desc in sub_descriptions:
            sub_hdc = parser.parse(sub_desc)
            sub_proto_matches = self.prototypes.match(sub_hdc, top_k=1)
            sub_proto, sub_sim = sub_proto_matches[0]

            # Check MHN for sub-problem
            sub_mhn = self.mhn.retrieve(sub_hdc, top_k=1)
            if sub_mhn and sub_mhn[0][1] >= self.mhn_threshold:
                _, sim, meta = sub_mhn[0]
                trace.append(f"  Sub-problem [{sub_proto}] solved via MHN (sim={sim:.3f})")
                self.stats["mhn_hits"] += 1
                sub_problems.append(SubProblem(
                    description=sub_desc,
                    prototype=sub_proto,
                    hdc_vector=sub_hdc,
                    similarity=sub_sim,
                    solution=meta.get("solution"),
                    source="mhn",
                ))
            else:
                # Try Z3 for this leaf
                solution, source, conf = self._solve_leaf(sub_desc, sub_hdc, context, trace)
                sub_problems.append(SubProblem(
                    description=sub_desc,
                    prototype=sub_proto,
                    hdc_vector=sub_hdc,
                    similarity=sub_sim,
                    solution=solution,
                    source=source,
                ))

        # 6. Compose solution
        all_solved = all(sp.is_solved for sp in sub_problems)
        if all_solved:
            composed_solution = self._compose(sub_problems)
            trace.append(f"All {len(sub_problems)} sub-problems solved — composing")
            confidence = 0.85  # High confidence when all sub-problems solved
        else:
            unsolved = [sp for sp in sub_problems if not sp.is_solved]
            trace.append(f"{len(unsolved)}/{len(sub_problems)} sub-problems unsolved")
            composed_solution = None
            confidence = 0.3  # Low confidence when some unsolved
            self.stats["unsolved"] += len(unsolved)

        latency = (time.perf_counter() - t0) * 1000

        return DecompositionResult(
            root=SubProblem(
                description=problem_text,
                prototype=best_proto,
                hdc_vector=hdc_vector,
                similarity=best_sim,
                children=sub_problems,
            ),
            prototype_match=best_proto,
            prototype_similarity=best_sim,
            sub_problems=sub_problems,
            solution=composed_solution,
            confidence=confidence,
            latency_ms=latency,
            trace=trace,
        )

    def _generate_sub_problems(
        self,
        problem_text: str,
        prototype: str,
        context: dict,
    ) -> list[str]:
        """Generate sub-problem descriptions by pattern-matching the prototype.

        This is a lightweight heuristic decomposer. For TD Pro, this would
        use the Liquid-KAN controller + hypernetworks for learned decomposition.

        For now, we use simple patterns:
        - scheduling_meetings → [find availability, resolve conflicts, optimize preferences]
        - optimization → [identify variables, define constraints, define objective, solve]
        - validation → [parse input, check constraints, report errors]
        - etc.
        """
        decomposition_patterns: dict[str, list[str]] = {
            "scheduling_meetings": [
                "find available time slots for all participants",
                "resolve scheduling conflicts between participants",
                "optimize meeting times for preferences and priorities",
            ],
            "scheduling": [
                "identify all tasks and their durations",
                "determine constraints and dependencies between tasks",
                "find optimal task ordering satisfying constraints",
            ],
            "scheduling_tasks": [
                "list all tasks with deadlines and priorities",
                "identify dependencies between tasks",
                "sequence tasks to meet all deadlines",
            ],
            "optimization": [
                "identify decision variables and their domains",
                "define hard constraints that must be satisfied",
                "define the objective function to optimize",
                "solve the optimization problem",
            ],
            "optimization_linear": [
                "identify variables and their cost coefficients",
                "define linear constraints",
                "minimize or maximize the linear objective",
            ],
            "optimization_multi_objective": [
                "identify competing objectives and their priorities",
                "define constraints for each objective",
                "find Pareto-optimal solution balancing objectives",
            ],
            "allocation_budget": [
                "list all expense categories and their minimum amounts",
                "determine total available budget",
                "allocate budget to maximize utility within constraints",
            ],
            "allocation_resources": [
                "list resources and their capabilities",
                "list tasks and their requirements",
                "assign resources to tasks optimally",
            ],
            "validation_schema": [
                "parse the input data structure",
                "check each field against the expected schema",
                "report all validation errors found",
            ],
            "validation_logic": [
                "extract all logical assertions from the input",
                "check for contradictions between assertions",
                "report any inconsistencies found",
            ],
            "transformation_format": [
                "parse the source format and extract data",
                "map source fields to target format fields",
                "generate output in the target format",
            ],
            "transformation_data": [
                "identify the current data format and issues",
                "apply normalization or cleaning rules",
                "verify the transformed data is correct",
            ],
            "comparison_ranking": [
                "extract evaluation criteria from the problem",
                "evaluate each item against all criteria",
                "rank items by total score",
            ],
            "planning_sequential": [
                "identify the goal and starting state",
                "list all available actions and their preconditions",
                "find a sequence of actions from start to goal",
            ],
            "proof_logical": [
                "identify premises and the conclusion",
                "apply inference rules to derive intermediate steps",
                "connect premises to conclusion via valid inferences",
            ],
            "debugging_logic_error": [
                "trace through the code execution path",
                "identify where the output diverges from expected",
                "determine the root cause of the incorrect logic",
            ],
            "filtering_query": [
                "parse the query conditions",
                "apply conditions to each item in the dataset",
                "return all items that satisfy all conditions",
            ],
            "parsing_structured": [
                "read the input and identify its structure",
                "extract fields and values from the structured data",
                "validate the extracted data types",
            ],
            "counting_conditional": [
                "identify the counting condition",
                "scan all items and test the condition",
                "return the total count of matching items",
            ],
            "search_approximate": [
                "compute similarity between query and all items",
                "rank items by similarity score",
                "return the top matches above threshold",
            ],
            "sorting_multi_key": [
                "identify primary and secondary sort keys",
                "sort by primary key first",
                "within each primary group, sort by secondary key",
            ],
            "generation_template": [
                "identify the template and its slots",
                "gather values for each slot from input data",
                "fill the template with the gathered values",
            ],
        }

        # Default decomposition: break into 2 generic sub-problems
        default = [
            f"identify the key components of: {problem_text[:80]}",
            f"determine the solution approach for: {problem_text[:80]}",
        ]

        # Find the most specific matching pattern
        patterns = []
        for key, subs in decomposition_patterns.items():
            if key == prototype:
                return subs
            if prototype.startswith(key):
                patterns = subs

        return patterns or default

    def _solve_leaf(
        self,
        problem_text: str,
        hdc_vector: np.ndarray,
        context: dict,
        trace: list[str],
    ) -> tuple[dict | None, str, float]:
        """Try to solve a leaf sub-problem.

        Priority: Z3 → router fallback → unsolved.

        Returns:
            Tuple of (solution, source, confidence).
        """
        # Try Z3 (if constraints available)
        constraints = context.get("constraints", {})
        if constraints:
            z3_result = self.z3_bridge.validate_action([], constraints)
            if z3_result.is_valid:
                trace.append(f"  Z3 solved leaf")
                self.stats["z3_solves"] += 1
                return {"status": "sat", "proof": z3_result.proof}, "z3", 0.9

        # Try router fallback
        if self.router is not None:
            try:
                routing = self.router.route(hdc_vector)
                trace.append(f"  Router fallback: {routing.domain}/{routing.task_type}")
                return (
                    {"domain": routing.domain, "task_type": routing.task_type,
                     "strategy": routing.strategy},
                    "router_fallback",
                    routing.combined_confidence,
                )
            except Exception:
                pass

        # Unsolved
        return None, "unsolved", 0.0

    def _compose(self, sub_problems: list[SubProblem]) -> dict:
        """Compose solutions from sub-problems into a unified solution.

        Simple composition: merge all sub-solution dicts, tag with source.
        """
        composed = {"steps": [], "sub_solutions": []}
        for i, sp in enumerate(sub_problems):
            if sp.solution:
                composed["steps"].append({
                    "step": i + 1,
                    "description": sp.description,
                    "prototype": sp.prototype,
                    "solution": sp.solution,
                    "source": sp.source,
                })
                composed["sub_solutions"].append(sp.solution)
        composed["total_steps"] = len(composed["steps"])
        return composed

    def learn(
        self,
        problem_text: str,
        solution: dict,
        outcome: str = "success",
        metadata: dict | None = None,
    ):
        """Store a problem-solution pair in MHN for future retrieval.

        Args:
            problem_text: Problem description.
            solution: Solution that worked.
            outcome: "success", "failure", or "partial".
            metadata: Additional context.
        """
        parser = NLParser(self.vocab)
        problem_hdc = parser.parse(problem_text)
        solution_hdc = parser.parse(str(solution)[:500])  # Truncate for encoding

        meta = {**(metadata or {}), "solution": solution, "outcome": outcome}
        self.mhn.store(problem_hdc, solution_hdc, meta)

    def get_stats(self) -> dict:
        """Return decomposer statistics including router fallback rate."""
        total = self.stats["total_decompositions"]
        fallback_rate = self.stats["router_fallbacks"] / max(total, 1)
        return {
            **self.stats,
            "router_fallback_rate": f"{fallback_rate:.1%}",
            "mhn_hit_rate": f"{self.stats['mhn_hits'] / max(total, 1):.1%}",
            "z3_solve_rate": f"{self.stats['z3_solves'] / max(total, 1):.1%}",
        }
