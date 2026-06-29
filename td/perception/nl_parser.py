"""Natural Language → HDC vector parser — FIXED VERSION.

Extracts entities, problem types, and goals. NOT just action/target/context.

Pipeline:
    1. Tokenize + entity extraction (numbers, people, quantities, goals)
    2. Problem type detection (concrete_plan, advice, proof, explanation, info_request)
    3. Concept matching (keyword → vocabulary lookup)
    4. HDC composition: bundle(bind(role, concept), bind(goal, constraint), ...)
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from .hdc import (
    ConceptVocabulary,
    bind,
    bundle,
    generate_hypervector,
    permute,
    similarity,
)


# ── PROBLEM TYPE DETECTION ──
# These keywords determine HOW the system should respond, not just WHAT domain.

ADVICE_KEYWORDS = {
    "how to", "how do i", "how can i", "tips", "advice", "strategy",
    "best way", "better way", "without", "avoid", "prevent", "stop",
    "technique", "method", "approach", "habit", "routine", "practice",
    "procrastination", "focus", "concentrate", "motivation", "productivity",
    "efficiency", "workflow", "time management", "study habits",
}

PROOF_KEYWORDS = {
    "prove", "proof", "theorem", "lemma", "axiom", "corollary",
    "induction", "contradiction", "direct proof", "by induction",
    "implies", "if and only if", "necessary and sufficient",
    "for all", "there exists", "qed",
}

EXPLANATION_KEYWORDS = {
    "why", "how does", "what is", "explain", "reason", "cause",
    "because", "due to", "purpose", "meaning", "difference between",
    "compare", "contrast", "versus", "vs",
}

INFO_REQUEST_KEYWORDS = {
    "cheapest", "price", "cost", "current", "today", "now", "live",
    "weather", "stock", "news", "latest", "recent", "update",
    "find", "search", "lookup", "where is", "when is", "who is",
}

CONCRETE_PLAN_KEYWORDS = {
    "schedule", "book", "reserve", "allocate", "assign", "plan",
    "arrange", "organize", "coordinate", "set up", "make",
}

# ── ENTITY EXTRACTION ──

# People names (expandable)
KNOWN_NAMES = {
    "alice", "bob", "carol", "dave", "eve", "frank", "grace",
    "heidi", "ivan", "judy", "karen", "leo", "mallory", "nancy",
    "oscar", "peggy", "quinn", "rupert", "sybil", "ted", "ursula",
    "victor", "wendy", "xavier", "yvonne", "zack",
    # Roles that function as people
    "interviewer", "interviewers", "candidate", "candidates",
    "participant", "participants", "attendee", "attendees",
    "member", "members", "staff", "team", "manager", "employee",
    "student", "teacher", "client", "customer", "vendor",
}

# Quantities and units
NUMBER_PATTERN = re.compile(r"\b(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|dozen|hundred|thousand)\b")

# Time references
TIME_KEYWORDS = {
    "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday", "weekend", "weekday",
    "morning", "afternoon", "evening", "night",
    "am", "pm", "o\'clock", "hour", "hours", "minute", "minutes",
    "day", "days", "week", "weeks", "month", "months",
    "today", "tomorrow", "next week", "this week",
}

# Goals / constraints (what the user wants to achieve or avoid)
GOAL_KEYWORDS = {
    "without", "avoid", "prevent", "no", "never", "dont", "don\'t",
    "minimize", "maximize", "optimize", "reduce", "increase",
    "fastest", "cheapest", "best", "most", "least", "asap",
    "deadline", "budget", "constraint", "limit", "capacity",
    "procrastination", "delay", "postpone", "rush", "urgent",
}

# ── ORIGINAL KEYWORDS (kept for backward compat) ──

ACTION_KEYWORDS = {
    "click", "type", "scroll", "submit", "select", "hover", "drag",
    "navigate", "extract", "fill", "check", "open", "close", "fetch",
    "post", "call", "read", "write", "create", "parse", "validate",
    "convert", "alert", "restart", "monitor", "wait", "refresh",
    "delete", "copy", "move", "export", "import", "search", "filter",
    "schedule", "allocate", "optimize", "minimize", "maximize", "prove",
    "solve", "rank", "sort", "count", "compare", "transform", "generate",
    "debug", "explain", "predict", "plan", "balance", "distribute",
    "arrange", "assign", "book", "reserve",
}

TARGET_KEYWORDS = {
    "button", "form", "field", "input", "dropdown", "checkbox",
    "link", "table", "modal", "tab", "menu", "page", "url",
    "endpoint", "api", "response", "request", "file", "csv", "json",
    "schema", "column", "service", "process", "cpu", "memory", "disk",
    "meeting", "appointment", "calendar", "deadline", "budget", "cost",
    "task", "resource", "team", "staff", "department", "constraint",
    "variable", "objective", "solution", "proof", "theorem", "hypothesis",
    "bug", "error", "exception", "code", "function", "algorithm",
    "data", "dataset", "record", "query", "result", "schedule",
    "plan", "itinerary", "route", "assignment", "allocation",
    "interviewer", "candidate", "panel", "examiner", "proctor",
    "participant", "attendee", "guest", "speaker", "presenter",
    "customer", "vendor", "client", "stakeholder", "manager",
    "employee", "volunteer", "student", "teacher", "researcher",
    "task", "tasks", "project", "projects", "item", "items",
}

CONTEXT_KEYWORDS = {
    "login", "contact", "registration", "checkout", "search",
    "dashboard", "settings", "profile", "admin", "home", "landing",
    "conflict", "priority", "deadline", "vacation", "availability",
    "client", "morning", "afternoon", "week", "month", "tuesday",
    "monday", "wednesday", "thursday", "friday", "weekend",
    "linear", "quadratic", "exponential", "optimal", "feasible",
    "consistent", "contradiction", "premise", "conclusion", "inference",
    "negative", "positive", "integer", "float", "string", "boolean",
    "ascending", "descending", "alphabetical", "chronological",
}

COMPOUND_CONCEPTS = {
    "submit button": "submit_button",
    "login form": "login_form",
    "contact form": "contact_form",
    "search bar": "search_bar",
    "api key": "api_key",
    "auth token": "auth_token",
    "status code": "status_code",
    "time slot": "time_slot",
    "time slots": "time_slot",
    "meeting room": "meeting_room",
    "team member": "team_member",
    "team members": "team_member",
    "client call": "client_call",
    "deadline conflict": "deadline_conflict",
    "scheduling conflict": "scheduling_conflict",
    "resource allocation": "resource_allocation",
    "task assignment": "task_assignment",
    "budget allocation": "budget_allocation",
    "cost optimization": "cost_optimization",
    "expense category": "expense_category",
    "budget constraint": "budget_constraint",
    "if and only if": "if_and_only_if",
    "logical proof": "logical_proof",
    "formal proof": "formal_proof",
    "logic error": "logic_error",
    "runtime error": "runtime_error",
    "null pointer": "null_pointer",
    "stack trace": "stack_trace",
    "execution path": "execution_path",
}


class NLParser:
    """Lightweight natural language → HDC encoder with entity extraction.

    Extracts:
        - problem_type: concrete_plan | advice | proof | explanation | info_request
        - entities: who, what, how_many, when, goals
        - action, target, context (backward compat)

    Time: ~0.2ms per parse.
    """

    def __init__(self, vocabulary: ConceptVocabulary):
        self.vocab = vocabulary

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s\-]", " ", text)
        return text.split()

    def _extract_compound_concepts(self, text: str) -> list[str]:
        found = []
        lower = text.lower()
        for phrase, concept in COMPOUND_CONCEPTS.items():
            if phrase in lower:
                found.append(concept)
        return found

    def detect_problem_type(self, text: str) -> str:
        """Detect if user wants a plan, advice, proof, explanation, or info."""
        lower = text.lower()

        # Check advice first ("how to" is very common)
        if any(kw in lower for kw in ADVICE_KEYWORDS):
            return "advice"

        # Check proof
        if any(kw in lower for kw in PROOF_KEYWORDS):
            return "proof"

        # Check explanation
        if any(kw in lower for kw in EXPLANATION_KEYWORDS):
            return "explanation"

        # Check info request (needs external data)
        if any(kw in lower for kw in INFO_REQUEST_KEYWORDS):
            return "info_request"

        # Default: concrete plan (schedule, allocate, book, etc.)
        if any(kw in lower for kw in CONCRETE_PLAN_KEYWORDS):
            return "concrete_plan"

        return "unknown"

    def extract_entities(self, text: str) -> dict[str, Any]:
        """Extract rich entities from text.

        Returns dict with:
            - problem_type: str
            - action: str | None
            - target: str | None
            - context: str | None
            - who: list[str] (people/roles mentioned)
            - what: list[str] (tasks/objects)
            - how_many: int | None
            - when: list[str] (time references)
            - goals: list[str] (constraints, objectives, things to avoid)
        """
        tokens = self._tokenize(text)
        lower = text.lower()
        result: dict[str, Any] = {}

        # 1. Problem type (CRITICAL — determines reasoning path)
        result["problem_type"] = self.detect_problem_type(text)

        # 2. Extract numbers
        numbers = re.findall(r"\b\d+\b", text)
        if numbers:
            result["how_many"] = int(numbers[0])

        # 3. Extract people (capitalized words + known names)
        people = []
        # Check known names
        for token in tokens:
            if token in KNOWN_NAMES:
                people.append(token)
        # Check capitalized words (potential names)
        raw_tokens = text.split()
        for token in raw_tokens:
            clean = re.sub(r"[^\w]", "", token)
            if clean and clean[0].isupper() and clean.lower() not in {"i", "a"}:
                if clean.lower() not in people:
                    people.append(clean.lower())
        result["who"] = list(set(people))

        # 4. Extract "what" (tasks, objects, targets)
        what = []
        for token in tokens:
            if token in TARGET_KEYWORDS:
                what.append(token)
        result["what"] = list(set(what))

        # 5. Extract time references
        when = []
        for token in tokens:
            if token in TIME_KEYWORDS:
                when.append(token)
        # Check compound time phrases
        for phrase in ["next week", "this week", "next month", "tomorrow morning"]:
            if phrase in lower:
                when.append(phrase.replace(" ", "_"))
        result["when"] = list(set(when))

        # 6. Extract goals/constraints (what user wants to achieve or avoid)
        goals = []
        for token in tokens:
            if token in GOAL_KEYWORDS:
                goals.append(token)
        # Check phrases like "without procrastination"
        if "without" in lower:
            after = lower.split("without")[-1].strip().split()[0] if len(lower.split("without")) > 1 else ""
            if after:
                goals.append(f"avoid_{after}")
        if "avoid" in lower:
            after = lower.split("avoid")[-1].strip().split()[0] if len(lower.split("avoid")) > 1 else ""
            if after:
                goals.append(f"avoid_{after}")
        result["goals"] = list(set(goals))

        # 7. Legacy action/target/context extraction (for backward compat)
        compounds = self._extract_compound_concepts(text)

        for token in tokens:
            if token in ACTION_KEYWORDS:
                result["action"] = token
                break

        target_found = False
        for compound in compounds:
            result["target"] = compound
            target_found = True
            break
        if not target_found:
            for token in tokens:
                if token in TARGET_KEYWORDS:
                    result["target"] = token
                    target_found = True
                    break

        for token in tokens:
            if token in CONTEXT_KEYWORDS:
                result["context"] = token
                break

        return result

    def parse(self, text: str) -> np.ndarray:
        """Parse natural language into HDC vector with full entity encoding.

        Encodes:
            - problem_type (permute to distinguish from other roles)
            - action, target, context (legacy)
            - who, what, how_many, when, goals (new entities)

        If no concepts detected, returns random vector (triggers escalation).
        """
        entities = self.extract_entities(text)

        if not entities or entities.get("problem_type") == "unknown":
            return generate_hypervector(self.vocab.dim)

        # Build HDC vector from all extracted entities
        vectors = []

        # Problem type (most important — determines reasoning path)
        if "problem_type" in entities:
            pt = entities["problem_type"]
            if not self.vocab.has(pt):
                self.vocab.add_concept(pt)
            if not self.vocab.has("problem_type"):
                self.vocab.add_concept("problem_type")
            vectors.append(bind(self.vocab.get("problem_type"), self.vocab.get(pt)))

        # Action (legacy)
        if "action" in entities:
            a = entities["action"]
            if not self.vocab.has(a):
                self.vocab.add_concept(a)
            if not self.vocab.has("action"):
                self.vocab.add_concept("action")
            vectors.append(bind(self.vocab.get("action"), self.vocab.get(a)))

        # Target (legacy)
        if "target" in entities:
            t = entities["target"]
            if not self.vocab.has(t):
                self.vocab.add_concept(t)
            if not self.vocab.has("target"):
                self.vocab.add_concept("target")
            vectors.append(bind(self.vocab.get("target"), self.vocab.get(t)))

        # Who (people/roles)
        if entities.get("who"):
            who_vectors = []
            for person in entities["who"]:
                if not self.vocab.has(person):
                    self.vocab.add_concept(person)
                who_vectors.append(self.vocab.get(person))
            if not self.vocab.has("who"):
                self.vocab.add_concept("who")
            vectors.append(bind(self.vocab.get("who"), bundle(*who_vectors)))

        # What (tasks/objects)
        if entities.get("what"):
            what_vectors = []
            for item in entities["what"]:
                if not self.vocab.has(item):
                    self.vocab.add_concept(item)
                what_vectors.append(self.vocab.get(item))
            if not self.vocab.has("what"):
                self.vocab.add_concept("what")
            vectors.append(bind(self.vocab.get("what"), bundle(*what_vectors)))

        # How many (quantity)
        if "how_many" in entities:
            n = str(entities["how_many"])
            if not self.vocab.has(n):
                self.vocab.add_concept(n)
            if not self.vocab.has("how_many"):
                self.vocab.add_concept("how_many")
            vectors.append(bind(self.vocab.get("how_many"), self.vocab.get(n)))

        # When (time references)
        if entities.get("when"):
            when_vectors = []
            for t in entities["when"]:
                if not self.vocab.has(t):
                    self.vocab.add_concept(t)
                when_vectors.append(self.vocab.get(t))
            if not self.vocab.has("when"):
                self.vocab.add_concept("when")
            vectors.append(bind(self.vocab.get("when"), bundle(*when_vectors)))

        # Goals (constraints, objectives, things to avoid)
        if entities.get("goals"):
            goal_vectors = []
            for g in entities["goals"]:
                if not self.vocab.has(g):
                    self.vocab.add_concept(g)
                goal_vectors.append(self.vocab.get(g))
            if not self.vocab.has("goals"):
                self.vocab.add_concept("goals")
            vectors.append(bind(self.vocab.get("goals"), bundle(*goal_vectors)))

        if not vectors:
            return generate_hypervector(self.vocab.dim)

        return bundle(*vectors)

    def parse_with_extra(self, text: str, **extra: str) -> np.ndarray:
        """Parse text and merge with additional key-value context."""
        # First get the base vector
        base = self.parse(text)

        if not extra:
            return base

        # Encode extra context
        extra_vectors = []
        for role, filler in extra.items():
            if not self.vocab.has(role):
                self.vocab.add_concept(role)
            if not self.vocab.has(filler):
                self.vocab.add_concept(filler)
            extra_vectors.append(bind(self.vocab.get(role), self.vocab.get(filler)))

        if extra_vectors:
            return bundle([base, bundle(*extra_vectors)])
        return base
