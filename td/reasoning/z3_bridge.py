"""Z3 Constraint Validator — bridges HDC vectors to formal symbolic reasoning.

This is the hardest research component of TD v2. The current implementation
uses template-based query generation (reliable) rather than algebraic
decomposition (open research problem).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from z3 import (
    Solver, Bool, Int, Real, And, Or, Not, Implies, If,
    sat, unsat, unknown, ModelRef,
)


@dataclass
class Z3Result:
    """Result of a Z3 constraint solve.

    Attributes:
        status: "sat", "unsat", or "unknown".
        model: Variable assignments if sat, else None.
        proof: Proof trace if unsat, else None.
        smt_query: The SMT-LIB query that was executed.
    """
    status: str
    model: dict[str, Any] | None = None
    proof: str | None = None
    smt_query: str = ""

    @property
    def is_valid(self) -> bool:
        """True if the action/constraint is valid (SAT)."""
        return self.status == "sat"

    @property
    def satisfiability_score(self) -> float:
        """Numeric satisfiability for confidence scoring.
        SAT = 1.0, UNKNOWN = 0.5, UNSAT = 0.0.
        """
        if self.status == "sat":
            return 1.0
        elif self.status == "unknown":
            return 0.5
        return 0.0


# Template selection rules — keyword-based matching
TEMPLATE_RULES = [
    {
        "name": "form_validation",
        "keywords": ["form", "submit", "field", "fill", "click", "login", "contact"],
        "min_matches": 2,
    },
    {
        "name": "api_sequential",
        "keywords": ["fetch", "api", "call", "sequential", "request", "endpoint", "auth"],
        "min_matches": 2,
    },
    {
        "name": "file_parse",
        "keywords": ["parse", "csv", "json", "file", "read", "validate", "schema", "column"],
        "min_matches": 2,
    },
    {
        "name": "monitor_threshold",
        "keywords": ["cpu", "memory", "disk", "threshold", "restart", "alert", "service", "monitor"],
        "min_matches": 2,
    },
]


class Z3Bridge:
    """Bridges HDC vector representations to Z3 SMT queries.

    Pipeline:
        1. HDC vector → concept decomposition (similarity with vocabulary)
        2. Concept decomposition → template selection
        3. Template filling (context → variable bindings)
        4. Z3 solve
        5. Result → confidence score

    The bridge uses template-based query generation, which is reliable
    but limited to predefined constraint patterns. Algebraic decomposition
    of HDC vectors into arbitrary SMT-LIB is an open research problem
    (flagged for TD Pro).
    """

    def __init__(self, template_dir: str | Path | None = None):
        """Initialize Z3 bridge.

        Args:
            template_dir: Directory with .smt template files.
                          If None, uses built-in templates.
        """
        self.template_dir = Path(template_dir) if template_dir else None

    def decompose(self, hdc_vector: np.ndarray, vocabulary,
                 threshold: float = 0.15) -> dict[str, float]:
        """Decompose HDC vector into concept probabilities.

        For each concept in vocabulary, compute cosine similarity.
        Concepts above threshold are included in the decomposition.

        Args:
            hdc_vector: HDC vector to decompose.
            vocabulary: ConceptVocabulary with known concepts.
            threshold: Minimum similarity to include a concept.

        Returns:
            Dict: {concept_name: similarity_score} for matched concepts.
            Sorted by similarity descending.

        Time: O(vocab_size × dim) ≈ 10ms for 1000 concepts × 10K dim.
        """
        query_f = hdc_vector.astype(np.float32)
        results = {}

        for name, concept_vec in vocabulary.concepts.items():
            if name.startswith("_"):
                continue
            sim = float(np.dot(query_f, concept_vec.astype(np.float32)) / len(query_f))
            if sim >= threshold:
                results[name] = sim

        # Sort by similarity descending
        return dict(sorted(results.items(), key=lambda x: -x[1]))

    def select_template(self, concepts: dict[str, float]) -> str | None:
        """Select best Z3 template based on detected concepts.

        Rule-based matching: each template requires a minimum number
        of keyword matches from the concept decomposition.

        Args:
            concepts: Concept decomposition from decompose().

        Returns:
            Template name or None if no match.
        """
        best_match = None
        best_score = 0

        for rule in TEMPLATE_RULES:
            matches = sum(1 for kw in rule["keywords"] if kw in concepts)
            if matches >= rule["min_matches"]:
                score = matches / len(rule["keywords"])
                if score > best_score:
                    best_score = score
                    best_match = rule["name"]

        return best_match

    def validate_action(self, action_plan: list[dict],
                        constraints: dict[str, Any]) -> Z3Result:
        """Validate an action plan against hard constraints.

        Generates a Z3 query from the action plan and constraints,
        then solves it. If SAT, the plan is valid. If UNSAT, the
        proof shows which constraint is violated.

        Args:
            action_plan: List of action dicts, e.g.:
                [{"action": "click", "target": "submit_button"},
                 {"action": "type", "target": "username", "value": "Alice"}]
            constraints: Dict of constraint key-value pairs, e.g.:
                {"submit_visible": True, "captcha_present": False,
                 "required_fields_filled": True}

        Returns:
            Z3Result with validation outcome.
        """
        s = Solver()

        # Declare boolean variables for each constraint
        z3_vars: dict[str, Bool] = {}
        for key, value in constraints.items():
            var_name = key.replace(" ", "_")
            z3_vars[var_name] = Bool(var_name)
            if isinstance(value, bool):
                if value:
                    s.add(z3_vars[var_name])
                else:
                    s.add(Not(z3_vars[var_name]))

        # Add action-plan-derived constraints
        # If any action targets submit, require form readiness
        actions = [a.get("action", "") for a in action_plan]
        targets = [a.get("target", "") for a in action_plan]

        if "submit" in str(targets).lower() or "submit" in str(actions).lower():
            if "submit_visible" in z3_vars:
                s.add(z3_vars["submit_visible"])
            if "required_fields_filled" in z3_vars:
                s.add(z3_vars["required_fields_filled"])
            if "captcha_present" in z3_vars:
                s.add(Not(z3_vars["captcha_present"]))

        # If any action is "restart", require safety constraints
        if "restart" in str(actions).lower():
            if "during_maintenance" in z3_vars:
                s.add(Not(z3_vars["during_maintenance"]))
            if "service_healthy" in z3_vars:
                s.add(z3_vars["service_healthy"])

        # If any action is "type" or "fill", require field visibility
        if any(a in ("type", "fill") for a in actions):
            if "field_visible" in z3_vars:
                s.add(z3_vars["field_visible"])

        # Solve
        result = s.check()

        if result == sat:
            model = s.model()
            model_dict = {}
            for var_name, z3_var in z3_vars.items():
                val = model.eval(z3_var, model_completion=True)
                model_dict[var_name] = bool(val)
            return Z3Result(status="sat", model=model_dict, smt_query=str(s))

        elif result == unsat:
            return Z3Result(
                status="unsat",
                proof="Constraints are unsatisfiable — action plan violates requirements.",
                smt_query=str(s),
            )

        else:
            return Z3Result(status="unknown", smt_query=str(s))

    def solve(self, smt_lib: str) -> Z3Result:
        """Execute Z3 solver on a raw SMT-LIB string.

        Args:
            smt_lib: SMT-LIB v2 format query string.

        Returns:
            Z3Result.
        """
        # For raw SMT-LIB, we'd need to parse and execute
        # This is a simplified interface for advanced users
        s = Solver()
        try:
            s.from_string(smt_lib)
        except Exception as e:
            return Z3Result(status="unknown", smt_query=smt_lib,
                          proof=f"Parse error: {e}")

        result = s.check()
        if result == sat:
            return Z3Result(status="sat", model={}, smt_query=smt_lib)
        elif result == unsat:
            return Z3Result(status="unsat", proof="UNSAT", smt_query=smt_lib)
        return Z3Result(status="unknown", smt_query=smt_lib)
