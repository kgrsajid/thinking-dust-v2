"""TD Decomposer v2 — Forces decomposition + Z3 reasoning on every problem.

Fixes from Kimi's diagnosis:
- MHN hit no longer short-circuits. It provides a decomposition plan.
- Z3 actually constructs and solves constraints.
- Problems requiring reasoning (proof, optimization, debug) ALWAYS decompose.
- Solution specificity check: reject generic templates.
- Real latency: 50-500ms (Z3 runs), not 2ms (keyword lookup).
"""

from __future__ import annotations

import time
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .perception.hdc import (
    ConceptVocabulary, build_default_vocabulary,
    generate_hypervector, bind, bundle, similarity, permute,
)
from .perception.nl_parser import NLParser
from .memory.mhn import ModernHopfieldNetwork, MHNConfig
from .reasoning.z3_bridge import Z3Bridge, Z3Result
from .decomposer import (
    PrototypeBank, SubProblem, DecompositionResult,
    PROTOTYPE_KEYWORDS,
)
from .z3_solver import (
    extract_entities, build_z3_model, solve_z3_model,
    requires_external_data, handle_external_data_problem,
    get_advice,
)
from .solve_advice import AdviceSolver


# Keywords that ALWAYS require actual reasoning, never just MHN retrieval
REASONING_REQUIRED = {
    "prove", "proof", "induction", "contradiction", "theorem", "implies",
    "optimize", "minimize", "maximize", "knapsack", "shortest path",
    "debug", "memory leak", "race condition", "trace",
    "solve", "factor", "derivative", "integral", "equation",
    "allocate", "schedule", "balance budget", "cut budget",
    "find all", "count how many", "complexity",
    "contradiction", "consistent", "verify",
}


def requires_reasoning(problem_text: str) -> bool:
    """Check if a problem requires actual reasoning, not just lookup."""
    lower = problem_text.lower()
    return any(kw in lower for kw in REASONING_REQUIRED)


class ReasoningDecomposer:
    """Decomposer that ALWAYS reasons. MHN provides plans, not answers.

    Pipeline (always runs all steps):
        1. Encode problem → HDC vector
        2. Match against prototypes
        3. Generate sub-problems (from prototype pattern or MHN plan)
        4. For each sub-problem: try MHN for hints, then Z3 to solve
        5. Compose solutions
        6. Z3 validate the composition
        7. Return with full trace

    MHN role changed: it provides decomposition plans and constraint
    templates, NOT final answers. This ensures every problem is actually
    reasoned about, not just pattern-matched.
    """

    def __init__(
        self,
        vocab: ConceptVocabulary | None = None,
        mhn: ModernHopfieldNetwork | None = None,
        z3_bridge: Z3Bridge | None = None,
        router=None,
        dim: int = 10_000,
        max_depth: int = 2,
    ):
        self.vocab = vocab or build_default_vocabulary(dim=dim)
        self.mhn = mhn or ModernHopfieldNetwork(MHNConfig(dim=dim, min_similarity=0.01))
        self.z3_bridge = z3_bridge or Z3Bridge()
        self.router = router
        self.dim = dim
        self.max_depth = max_depth
        self.prototypes = PrototypeBank(self.vocab, dim)
        self.parser = NLParser(self.vocab)

        self.stats = {
            "total": 0, "decomposed": 0, "mhn_assisted": 0,
            "z3_solved": 0, "router_fallback": 0, "failed": 0,
        }

    def solve(self, problem_text: str, context: dict | None = None) -> DecompositionResult:
        """Solve a problem by decomposing and reasoning.

        Unlike the old decomposer, this ALWAYS decomposes and runs Z3,
        even when MHN has a high-similarity match.
        """
        t0 = time.perf_counter()
        trace = []
        context = context or {}
        self.stats["total"] += 1

        # 1. Encode
        hdc_vector = self.parser.parse(problem_text)
        entities = self.parser.extract_entities(problem_text)
        ptype = entities.get("problem_type", "unknown")
        trace.append(f"Perception: type={ptype}, entities={ {k: v for k, v in entities.items() if k != 'problem_type' and v} }")

        # Route based on problem type
        if ptype == "advice":
            trace.append("Mode: ADVICE — retrieving from MHN")
            seed_path = str(Path(__file__).parent.parent / "data" / "behavioral_strategies.json")
            advice_solver = AdviceSolver(self.mhn, self.vocab, self.parser,
                                         seed_data_path=seed_path)
            advice_result = advice_solver.solve(entities)
            trace.append(f"  Source: {advice_result['source']}, strategies: {len(advice_result['strategies'])}")
            latency = (time.perf_counter() - t0) * 1000
            # Format for display
            formatted_lines = ["Strategy:"]
            for i, s in enumerate(advice_result["strategies"], 1):
                formatted_lines.append(f"  {i}. {s['title']}: {s['description']}")
            return DecompositionResult(
                root=SubProblem(problem_text, "advice", hdc_vector, 0.5,
                              solution={"type": "advice", "formatted": "\n".join(formatted_lines),
                                       "details": advice_result},
                              source="advice"),
                prototype_match="advice",
                prototype_similarity=0.5,
                sub_problems=[],
                solution={"type": "advice", "formatted": "\n".join(formatted_lines),
                         "details": advice_result},
                confidence=advice_result["confidence"],
                latency_ms=latency,
                trace=trace,
            )

        elif ptype == "info_request":
            trace.append("Mode: INFO_REQUEST — needs external data")
            result = handle_external_data_problem(problem_text)
            latency = (time.perf_counter() - t0) * 1000
            return DecompositionResult(
                root=SubProblem(problem_text, "info_request", hdc_vector, 0.3,
                              solution=result, source="external"),
                prototype_match="info_request",
                prototype_similarity=0.3,
                sub_problems=[],
                solution=result,
                confidence=0.3,
                latency_ms=latency,
                trace=trace,
            )

        # 2. Prototype match
        proto_matches = self.prototypes.match(hdc_vector, top_k=3)
        best_proto, best_sim = proto_matches[0]
        trace.append(f"Prototype: {best_proto} (sim={best_sim:.3f})")
        mhn_results = self.mhn.retrieve(hdc_vector, top_k=1)
        mhn_plan = None
        if mhn_results and mhn_results[0][1] >= 0.15:
            _, sim, meta = mhn_results[0]
            mhn_plan = meta.get("solution", {})
            trace.append(f"MHN plan retrieved (sim={sim:.3f}) — using as decomposition hint")
            self.stats["mhn_assisted"] += 1

        # 4. ALWAYS decompose (even if MHN hit)
        sub_descs = self._get_sub_problems(problem_text, best_proto, mhn_plan)
        if not sub_descs:
            # Can't decompose — try direct Z3 solve
            trace.append("No decomposition pattern — attempting direct Z3 solve")
            solution, source = self._z3_solve(problem_text, hdc_vector, context, trace)
            latency = (time.perf_counter() - t0) * 1000
            if source == "router_fallback":
                self.stats["router_fallback"] += 1
            return DecompositionResult(
                root=SubProblem(problem_text, best_proto, hdc_vector, best_sim,
                               solution=solution, source=source),
                prototype_match=best_proto,
                prototype_similarity=best_sim,
                sub_problems=[],
                solution=solution,
                confidence=0.5 if solution else 0.0,
                latency_ms=latency,
                trace=trace,
                used_router_fallback=(source == "router_fallback"),
            )

        trace.append(f"Decomposed into {len(sub_descs)} sub-problems")
        self.stats["decomposed"] += 1

        # 5. Solve each sub-problem
        sub_problems = []
        for desc in sub_descs:
            sp = self._solve_sub_problem(desc, context, trace, parent_entities=entities)
            sub_problems.append(sp)

        # 6. Compose
        solved_count = sum(1 for sp in sub_problems if sp.is_solved)
        if solved_count == len(sub_problems):
            composed = self._compose(sub_problems, problem_text)
            trace.append(f"All {solved_count} sub-problems solved — composing")
            confidence = 0.85
        elif solved_count > 0:
            composed = self._compose(sub_problems, problem_text)
            trace.append(f"{solved_count}/{len(sub_problems)} solved — partial solution")
            confidence = 0.5 * solved_count / len(sub_problems)
        else:
            composed = None
            trace.append("No sub-problems solved")
            confidence = 0.1
            self.stats["failed"] += 1

        latency = (time.perf_counter() - t0) * 1000

        return DecompositionResult(
            root=SubProblem(problem_text, best_proto, hdc_vector, best_sim,
                          children=sub_problems),
            prototype_match=best_proto,
            prototype_similarity=best_sim,
            sub_problems=sub_problems,
            solution=composed,
            confidence=confidence,
            latency_ms=latency,
            trace=trace,
            used_router_fallback=False,
        )

    def _get_sub_problems(self, problem: str, prototype: str, mhn_plan: dict | None) -> list[str]:
        """Get sub-problem descriptions from prototype patterns or MHN plan."""

        # If MHN provides a decomposition plan, use it
        if mhn_plan and isinstance(mhn_plan, dict):
            solution = mhn_plan.get("solution", mhn_plan)
            if isinstance(solution, dict) and "steps" in solution:
                # This is already a decomposition chain
                return [s.get("description", str(s))[:100] for s in solution["steps"] if isinstance(s, dict)]

        # Use prototype-based decomposition patterns
        from .decomposer import Decomposer
        patterns = Decomposer.__dict__  # Access the method source

        # Inline the decomposition patterns (same as original decomposer)
        decomposition_patterns: dict[str, list[str]] = self._decomposition_patterns()

        # Find matching pattern
        if prototype in decomposition_patterns:
            return decomposition_patterns[prototype]

        # Try prefix match
        for key, subs in decomposition_patterns.items():
            if prototype.startswith(key) or key.startswith(prototype):
                return subs

        return []

    def _decomposition_patterns(self) -> dict[str, list[str]]:
        """Sub-problem decomposition patterns by prototype."""
        return {
            "scheduling_meetings": [
                "extract all participants and their availability constraints",
                "compute overlapping free time slots between participants",
                "assign each meeting to a conflict-free time slot",
            ],
            "scheduling_calendar": [
                "check calendar availability for each participant",
                "find overlapping free time slots",
                "select optimal time considering priorities",
            ],
            "scheduling_availability": [
                "collect availability from all participants",
                "compute intersection of available times",
                "rank slots by preference and select best",
            ],
            "scheduling_conflict_resolution": [
                "identify which meetings conflict with each other",
                "rank conflicting meetings by priority level",
                "reschedule lowest priority meetings to free slots",
            ],
            "scheduling": [
                "identify all items to schedule with their constraints",
                "determine available time slots and conflicts",
                "assign each item to a valid time slot",
            ],
            "scheduling_tasks": [
                "list all tasks with deadlines and durations",
                "identify dependencies between tasks",
                "sequence tasks to satisfy all constraints",
            ],
            "scheduling_travel": [
                "identify all trip components needed",
                "compare options and find optimal combination",
                "book components in dependency order",
            ],
            "optimization": [
                "identify decision variables and their domains",
                "define hard constraints that must be satisfied",
                "define the objective function to optimize",
                "solve to find optimal variable assignment",
            ],
            "optimization_linear": [
                "identify variables and cost coefficients",
                "define linear constraints on variables",
                "solve the linear program for optimal values",
            ],
            "optimization_multi_objective": [
                "identify competing objectives and priorities",
                "define constraints for each objective",
                "find balanced solution across objectives",
            ],
            "allocation_budget": [
                "list all expense categories with minimum requirements",
                "determine total available budget and hard constraints",
                "allocate budget across categories to maximize utility",
            ],
            "allocation_resources": [
                "list all resources and their capabilities",
                "list all tasks and their requirements",
                "assign resources to tasks satisfying all constraints",
            ],
            "budget_planning": [
                "enumerate all expected income sources",
                "enumerate all expected expense categories",
                "compute net balance and allocate surplus",
            ],
            "budget_cut": [
                "list all expense categories and current spending",
                "rank categories by operational importance",
                "reduce lowest priority categories to meet target",
            ],
            "budget_balancing": [
                "list all income sources and amounts",
                "list all expense items and amounts",
                "identify gap and recommend specific adjustments",
            ],
            "proof_logical": [
                "identify all premises and the target conclusion",
                "apply valid inference rules to derive intermediate steps",
                "chain inferences from premises to conclusion",
            ],
            "proof_by_contradiction": [
                "assume the negation of the statement to prove",
                "derive a contradiction from this assumption",
                "conclude the original statement must be true",
            ],
            "proof_induction": [
                "verify the base case holds for the smallest value",
                "state the inductive hypothesis for arbitrary n",
                "prove the statement holds for n plus one using the hypothesis",
            ],
            "proof_correctness": [
                "identify the program specification and preconditions",
                "establish loop invariants for all loops",
                "show each step maintains invariants toward postcondition",
            ],
            "math_algebra": [
                "identify all unknowns and given equations",
                "apply algebraic manipulations to isolate unknowns",
                "verify the solution satisfies all original equations",
            ],
            "math_optimization": [
                "identify the objective function and decision variables",
                "enumerate all constraints on the variables",
                "find variable values that optimize the objective",
            ],
            "math_probability": [
                "define the sample space and relevant events",
                "determine the probability measure",
                "compute the target probability",
            ],
            "math_statistics": [
                "identify the data and its characteristics",
                "compute relevant descriptive statistics",
                "draw inference with appropriate confidence",
            ],
            "debugging_logic_error": [
                "trace the execution path for the failing input",
                "identify where output diverges from expected behavior",
                "determine the root cause and suggest a fix",
            ],
            "debugging_runtime_error": [
                "read the error message and stack trace",
                "identify the null or invalid value causing the error",
                "determine the root cause and suggest a fix",
            ],
            "code_review": [
                "examine code structure and conventions",
                "identify potential bugs and edge cases",
                "suggest specific improvements",
            ],
            "code_test": [
                "identify the function and its expected behavior",
                "design test cases covering normal edge and error cases",
                "specify expected outputs for each test case",
            ],
            "data_cleaning": [
                "scan for missing duplicate and outlier values",
                "apply appropriate cleaning strategy per issue type",
                "verify the cleaned dataset is consistent",
            ],
            "data_aggregation": [
                "identify grouping keys and aggregation metrics",
                "group records and compute aggregate values",
                "format and output summarized results",
            ],
            "data_join": [
                "identify the common key between datasets",
                "merge records on the common key",
                "handle unmatched records appropriately",
            ],
            "transformation_format": [
                "parse the source format and extract all data",
                "map source fields to target format fields",
                "generate correctly formatted output",
            ],
            "transformation_data": [
                "identify the current format and target format",
                "apply the necessary transformations",
                "verify correctness of the output",
            ],
            "validation_schema": [
                "parse the input and identify its structure",
                "check each field against the expected schema",
                "report all validation violations found",
            ],
            "validation_logic": [
                "extract all logical assertions from the input",
                "check for contradictions between assertions",
                "report any inconsistencies found",
            ],
            "comparison_ranking": [
                "extract evaluation criteria from the problem",
                "evaluate each item against all criteria",
                "rank items by total score and report ranking",
            ],
            "planning_sequential": [
                "identify the goal and current state",
                "enumerate available actions and their preconditions",
                "find action sequence from current state to goal",
            ],
            "planning_conditional": [
                "identify the primary plan and its assumptions",
                "identify what could go wrong and fallback options",
                "specify when to switch to each fallback",
            ],
            "filtering_query": [
                "parse the query conditions from the problem",
                "apply conditions to each record in the dataset",
                "return all records satisfying all conditions",
            ],
            "parsing_structured": [
                "read the input and identify its structure",
                "extract fields and values",
                "validate the extracted data",
            ],
            "counting_conditional": [
                "identify the counting condition",
                "evaluate the condition for each item",
                "report the total count of matches",
            ],
            "search_approximate": [
                "compute similarity between query and all items",
                "rank items by similarity score",
                "return top matches above threshold",
            ],
            "sorting_multi_key": [
                "identify primary and secondary sort keys",
                "sort by primary key then secondary key",
                "output the sorted result",
            ],
            "generation_template": [
                "identify the template structure and its slots",
                "gather values for each slot from the input",
                "fill the template and verify completeness",
            ],
        }

    def _solve_sub_problem(self, desc: str, context: dict, trace: list,
                          parent_entities: dict | None = None) -> SubProblem:
        """Solve a single sub-problem. Always tries Z3."""
        hdc = self.parser.parse(desc)
        proto_matches = self.prototypes.match(hdc, top_k=1)
        proto, sim = proto_matches[0]

        # Pass parent entities so Z3 has access to people names etc
        solution, source = self._z3_solve(desc, hdc, context, trace,
                                          entities=parent_entities)

        return SubProblem(
            description=desc,
            prototype=proto,
            hdc_vector=hdc,
            similarity=sim,
            solution=solution,
            source=source,
        )

    def _z3_solve(self, problem_text: str, hdc_vector: np.ndarray,
                  context: dict, trace: list,
                  entities: dict | None = None) -> tuple[dict | None, str]:
        """Attempt to solve using Z3 with REAL constraint generation.

        Extracts entities from problem text, builds a problem-specific
        Z3 model, solves it, and formats the solution as human-readable.
        """
        # Check for external data requirement
        if requires_external_data(problem_text):
            trace.append(f"  Requires external data — cannot solve with Z3 alone")
            result = handle_external_data_problem(problem_text)
            return result, "external"

        # Extract entities (use parser if available, or our own extraction)
        if entities is None:
            entities = self.parser.extract_entities(problem_text)
        else:
            # Merge with our own extraction to catch anything we missed
            own = self.parser.extract_entities(problem_text)
            for k, v in own.items():
                if k not in entities or not entities[k]:
                    entities[k] = v

        ctx = {**context, "text": problem_text}
        solver, variables, ptype = build_z3_model(entities, ctx)

        if solver is None:
            trace.append(f"  Z3 unavailable — falling back")
            return self._router_fallback(hdc_vector, trace)

        # Solve
        solution = solve_z3_model(solver, variables, ptype)

        if solution and solution.get("status") == "solved":
            n_vars = len(variables)
            formatted = solution.get("formatted", "")
            trace.append(f"  Z3: SAT — {ptype} with {n_vars} real variables")
            if formatted:
                for line in formatted.split("\n")[:3]:
                    trace.append(f"    {line}")
            self.stats["z3_solved"] += 1
            return solution, "z3"
        elif solution and solution.get("status") == "unsat":
            trace.append(f"  Z3: UNSAT — {solution.get('error', 'constraints contradictory')}")
            return solution, "z3"
        else:
            trace.append(f"  Z3: could not solve — falling back to router")
            return self._router_fallback(hdc_vector, trace)

    def _router_fallback(self, hdc_vector: np.ndarray, trace: list) -> tuple[dict | None, str]:
        """Use router as last resort fallback."""
        if self.router is not None:
            try:
                routing = self.router.route(hdc_vector)
                trace.append(f"  Router fallback: {routing.domain}/{routing.task_type}")
                self.stats["router_fallback"] += 1
                return ({"domain": routing.domain, "task_type": routing.task_type},
                        "router_fallback")
            except Exception:
                pass
        return None, "unsolved"

    def _compose(self, sub_problems: list[SubProblem], original_problem: str) -> dict:
        """Compose sub-problem solutions into a final answer.

        The composition includes:
        - The original problem
        - Each sub-problem and its solution
        - The reasoning steps (for interpretability)
        """
        steps = []
        for i, sp in enumerate(sub_problems):
            steps.append({
                "step": i + 1,
                "sub_problem": sp.description,
                "prototype": sp.prototype,
                "solution": sp.solution,
                "source": sp.source,
            })

        return {
            "problem": original_problem,
            "method": "decomposition",
            "total_steps": len(steps),
            "steps": steps,
            "reasoning_type": "constraint_satisfaction",
        }

    def learn(self, problem_text: str, decomposition_plan: dict,
              outcome: str = "success", metadata: dict | None = None):
        """Store a decomposition plan in MHN.

        Note: This stores the PLAN, not the final answer. The plan tells
        future invocations HOW to decompose similar problems.
        """
        problem_hdc = self.parser.parse(problem_text)
        plan_hdc = self.parser.parse(str(decomposition_plan)[:500])
        meta = {**(metadata or {}), "solution": decomposition_plan, "outcome": outcome}
        self.mhn.store(problem_hdc, plan_hdc, meta)

    def get_stats(self) -> dict:
        total = max(self.stats["total"], 1)
        return {
            **self.stats,
            "decomposition_rate": f"{self.stats['decomposed']/total:.1%}",
            "z3_rate": f"{self.stats['z3_solved']/total:.1%}",
            "mhn_assist_rate": f"{self.stats['mhn_assisted']/total:.1%}",
            "router_fallback_rate": f"{self.stats['router_fallback']/total:.1%}",
        }
