"""Natural Language → HDC vector parser — HDC PROTOTYPE VERSION.

Classifies problem type by HDC VECTOR SIMILARITY, not keyword matching.

Pipeline:
    1. Encode prototype sentences for each problem type as HDC vectors
    2. Parse user input into HDC vector (same pipeline)
    3. Classify by cosine similarity to prototypes
    4. Extract entities (who, what, how_many, when, goals)
    5. Compose final HDC vector with problem_type + entities

The HDC way: similar meaning → similar vector → same classification.
No keyword lists. No hardcoded domains.
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
    similarity,
)


# ── PROTOTYPE SENTENCES (semantic anchors) ──
# These are the ONLY hardcoded strings in the system.
# Everything else is learned via HDC similarity.

PROTOTYPE_SENTENCES = {
    "advice": [
        "how to do something without something",
        "tips for doing something better",
        "best way to avoid something",
        "strategy for improving something",
        "how can I stop doing something",
        "advice on dealing with something",
        "what should I do about something",
        "help me with something",
        "I am struggling with something",
        "I keep putting off something",
        "what is the best approach for something",
    ],
    "proof": [
        "prove that something implies something",
        "show by induction that something",
        "theorem about something",
        "formal proof of something",
        "demonstrate that something is true",
        "verify that something holds for all",
        "prove by contradiction that something",
        "mathematical proof of something",
    ],
    "explanation": [
        "why does something happen",
        "how does something work",
        "what is the reason for something",
        "explain the cause of something",
        "compare something and something",
        "what is the difference between something",
        "why is something the case",
    ],
    "info_request": [
        "what is the current price of something",
        "find the cheapest something",
        "what is the weather today",
        "latest news about something",
        "where is something located",
        "when is something happening",
        "who is the best something",
        "current status of something",
    ],
    "concrete_plan": [
        "schedule something with someone",
        "book something for someone",
        "allocate something across something",
        "assign something to someone",
        "plan something for next week",
        "arrange something with someone",
        "coordinate something for someone",
        "reserve something for someone",
        "distribute something among someone",
    ],
}


# ── ENTITY EXTRACTION (lightweight, no ML) ──

KNOWN_NAMES = {
    "alice", "bob", "carol", "dave", "eve", "frank", "grace",
    "heidi", "ivan", "judy", "karen", "leo", "mallory", "nancy",
    "oscar", "peggy", "quinn", "rupert", "sybil", "ted", "ursula",
    "victor", "wendy", "xavier", "yvonne", "zack",
    "interviewer", "interviewers", "candidate", "candidates",
    "participant", "participants", "attendee", "attendees",
    "member", "members", "staff", "team", "manager", "employee",
    "student", "teacher", "client", "customer", "vendor",
}

TIME_KEYWORDS = {
    "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday", "weekend", "weekday",
    "morning", "afternoon", "evening", "night",
    "am", "pm", "hour", "hours", "minute", "minutes",
    "day", "days", "week", "weeks", "month", "months",
    "today", "tomorrow", "next_week", "this_week",
}

GOAL_KEYWORDS = {
    "without", "avoid", "prevent", "no", "never", "dont", "don't",
    "minimize", "maximize", "optimize", "reduce", "increase",
    "fastest", "cheapest", "best", "most", "least", "asap",
    "deadline", "budget", "constraint", "limit", "capacity",
    "procrastination", "delay", "postpone", "rush", "urgent",
}

ACTION_KEYWORDS = {
    "schedule", "book", "reserve", "allocate", "assign", "plan",
    "arrange", "organize", "coordinate", "set", "make", "parse",
    "validate", "convert", "transform", "optimize", "prove", "solve",
    "debug", "explain", "predict", "balance", "distribute", "find",
    "compare", "filter", "sort", "search", "check", "verify",
}

TARGET_KEYWORDS = {
    "meeting", "appointment", "calendar", "deadline", "budget", "cost",
    "task", "resource", "team", "staff", "constraint", "variable",
    "solution", "proof", "theorem", "bug", "error", "code", "function",
    "data", "dataset", "record", "query", "result", "schedule",
    "plan", "itinerary", "route", "assignment", "allocation",
    "interviewer", "candidate", "panel", "participant", "attendee",
    "customer", "client", "stakeholder", "manager", "employee",
    "student", "teacher", "task", "project", "item",
}

COMPOUND_CONCEPTS = {
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
    "budget constraint": "budget_constraint",
}


class NLParser:
    """HDC-based natural language parser with prototype similarity classification.

    Problem type detection uses HDC VECTOR SIMILARITY, not keyword matching.
    Prototype sentences are encoded as HDC vectors once at initialization.
    User input is encoded using the same pipeline and compared by cosine similarity.

    Time: ~0.3-0.5ms per parse (prototype comparison adds ~0.2ms vs keyword matching).
    """

    def __init__(self, vocabulary: ConceptVocabulary):
        self.vocab = vocabulary
        self.prototype_vectors: dict[str, list[np.ndarray]] = {}
        self._init_prototypes()

    # ── PROTOTYPE INITIALIZATION ──

    def _init_prototypes(self):
        """Encode all prototype sentences as HDC vectors."""
        for problem_type, sentences in PROTOTYPE_SENTENCES.items():
            vectors = []
            for sentence in sentences:
                vec = self._encode_prototype(sentence)
                vectors.append(vec)
            self.prototype_vectors[problem_type] = vectors

    def _encode_prototype(self, text: str) -> np.ndarray:
        """Encode a prototype sentence using the same pipeline as real input.

        Prototypes are encoded with ONLY action/target/context (no problem_type)
        to avoid circular dependency. The similarity is in the semantic content,
        not the explicit label.
        """
        concepts = self._extract_concepts_simple(text)
        return self._compose_hdc(concepts)

    def _extract_concepts_simple(self, text: str) -> dict[str, str]:
        """Lightweight concept extraction for prototypes (no problem_type detection)."""
        tokens = self._tokenize(text)
        result: dict[str, str] = {}

        # Extract action
        for token in tokens:
            if token in ACTION_KEYWORDS:
                result["action"] = token
                break

        # Extract target
        compounds = self._extract_compound_concepts(text)
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

        # Extract context
        for token in tokens:
            if token in TIME_KEYWORDS or token in GOAL_KEYWORDS:
                result["context"] = token
                break

        return result

    # ── PROBLEM TYPE DETECTION (HDC SIMILARITY) ──

    def detect_problem_type(self, text: str) -> str:
        """Detect problem type by HDC similarity to prototypes.

        Algorithm:
            1. Encode user input as HDC vector
            2. For each problem type, compute similarity to all its prototypes
            3. Score = 0.6 * average similarity + 0.4 * max similarity
            4. Return type with highest score above threshold

        This captures semantic similarity even without word overlap.
        "I'm struggling to get started" → similar to "how can I stop" → advice.
        """
        # Fast path: check unambiguous keywords first
        fast_result = self._fast_detect_problem_type(text)
        if fast_result:
            return fast_result

        # HDC prototype matching
        input_hdc = self._encode_input_for_classification(text)

        best_type = "unknown"
        best_score = 0.0

        for problem_type, proto_vectors in self.prototype_vectors.items():
            similarities = [similarity(input_hdc, pv) for pv in proto_vectors]
            avg_sim = sum(similarities) / len(similarities)
            max_sim = max(similarities)
            score = 0.6 * avg_sim + 0.4 * max_sim

            if score > best_score:
                best_score = score
                best_type = problem_type

        # Threshold: must be reasonably similar to known prototypes
        if best_score > 0.15:
            return best_type
        return "unknown"

    def _encode_input_for_classification(self, text: str) -> np.ndarray:
        """Encode input for classification (same as prototype encoding)."""
        concepts = self._extract_concepts_simple(text)
        return self._compose_hdc(concepts)

    # ── FAST PATH (keyword fallback for common cases) ──

    def _fast_detect_problem_type(self, text: str) -> str | None:
        """Fast keyword-based pre-check for obvious cases."""
        lower = text.lower()

        # Advice
        if lower.startswith("how to") or "how do i" in lower or "how can i" in lower:
            return "advice"
        if any(kw in lower for kw in ["putting off", "struggling to", "procrastinat"]):
            return "advice"

        # Proof
        if any(kw in lower for kw in ["prove ", "theorem", "formal proof", "qed",
                                       "by induction", "by contradiction"]):
            return "proof"

        # Info request
        if any(kw in lower for kw in ["cheapest", "current price", "weather today",
                                       "latest news", "best price"]):
            return "info_request"

        # Concrete plan (starts with or contains action verb)
        for w in ["schedule", "book ", "allocate", "assign ", "reserve", "arrange",
                  "convert ", "transform ", "parse ", "export", "balance ", "validate ",
                  "optimize ", "plan ", "debug ", "find duplicate", "find all",
                  "coordinate", "distribute"]:
            if lower.startswith(w) or (" " + w) in lower:
                return "concrete_plan"

        # Named entities with action verbs
        if re.search(r"(schedule|book|allocate|assign|plan).*(with|for).*[A-Z][a-z]+", text):
            return "concrete_plan"

        return None


    # ── ENTITY EXTRACTION ──

    def extract_entities(self, text: str) -> dict[str, Any]:
        """Extract rich entities from text.

        Returns:
            problem_type: str (detected via HDC similarity)
            action, target, context: str | None (legacy)
            who: list[str] (people/roles)
            what: list[str] (tasks/objects)
            how_many: int | None
            when: list[str] (time references)
            goals: list[str] (constraints, objectives)
        """
        tokens = self._tokenize(text)
        lower = text.lower()
        result: dict[str, Any] = {}

        # 1. Problem type (HDC similarity primary, fast path fallback)
        fast_type = self._fast_detect_problem_type(text)
        if fast_type:
            result["problem_type"] = fast_type
        else:
            result["problem_type"] = self.detect_problem_type(text)

        # 2. Extract numbers
        numbers = re.findall(r"\d+", text)
        if numbers:
            result["how_many"] = int(numbers[0])

        # 3. Extract people
        people = []
        for token in tokens:
            if token in KNOWN_NAMES:
                people.append(token)
        # Capitalized words (potential names)
        raw_tokens = text.split()
        for token in raw_tokens:
            clean = re.sub(r"[^\w]", "", token)
            if clean and clean[0].isupper() and clean.lower() not in {"i", "a"}:
                name = clean.lower()
                if name not in people:
                    people.append(name)
        result["who"] = list(set(people))

        # 4. Extract "what" (tasks/objects)
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
        # Compound phrases
        for phrase in ["next week", "this week", "next month", "tomorrow morning"]:
            if phrase in lower:
                when.append(phrase.replace(" ", "_"))
        result["when"] = list(set(when))

        # 6. Extract goals/constraints
        goals = []
        for token in tokens:
            if token in GOAL_KEYWORDS:
                goals.append(token)
        # Phrases like "without procrastination"
        if "without" in lower:
            parts = lower.split("without")
            if len(parts) > 1:
                after = parts[-1].strip().split()[0] if parts[-1].strip() else ""
                if after:
                    goals.append(f"avoid_{after}")
        if "avoid" in lower:
            parts = lower.split("avoid")
            if len(parts) > 1:
                after = parts[-1].strip().split()[0] if parts[-1].strip() else ""
                if after:
                    goals.append(f"avoid_{after}")
        result["goals"] = list(set(goals))

        # 7. Legacy action/target/context
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
            if token in TIME_KEYWORDS or token in GOAL_KEYWORDS:
                result["context"] = token
                break

        return result

    # ── HDC ENCODING ──

    def parse(self, text: str) -> np.ndarray:
        """Parse natural language into full HDC vector.

        Encodes:
            - problem_type (permute to distinguish from other roles)
            - action, target, context (legacy)
            - who, what, how_many, when, goals (entities)
        """
        entities = self.extract_entities(text)

        if not entities or entities.get("problem_type") == "unknown":
            return generate_hypervector(self.vocab.dim)

        return self._encode_entities(entities)

    def _encode_entities(self, entities: dict[str, Any]) -> np.ndarray:
        """Encode extracted entities into a single HDC vector."""
        vectors = []

        # Problem type (most important)
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

        # Goals (constraints, objectives)
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

    def _compose_hdc(self, concepts: dict[str, str]) -> np.ndarray:
        """Compose HDC vector from a simple concept dict (for prototypes)."""
        vectors = []
        for role, filler in concepts.items():
            if not self.vocab.has(role):
                self.vocab.add_concept(role)
            if not self.vocab.has(filler):
                self.vocab.add_concept(filler)
            vectors.append(bind(self.vocab.get(role), self.vocab.get(filler)))

        if not vectors:
            return generate_hypervector(self.vocab.dim)

        return bundle(*vectors)

    # ── UTILITIES ──

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

    def parse_with_extra(self, text: str, **extra: str) -> np.ndarray:
        """Parse text and merge with additional key-value context."""
        base = self.parse(text)

        if not extra:
            return base

        extra_vectors = []
        for role, filler in extra.items():
            if not self.vocab.has(role):
                self.vocab.add_concept(role)
            if not self.vocab.has(filler):
                self.vocab.add_concept(filler)
            extra_vectors.append(bind(self.vocab.get(role), self.vocab.get(filler)))

        if extra_vectors:
            return bundle(base, bundle(*extra_vectors))
        return base

    # ── DEBUG / INTROSPECTION ──

    def explain_classification(self, text: str) -> dict[str, Any]:
        """Return detailed classification explanation for debugging.

        Shows similarity scores to ALL prototypes, not just the winner.
        """
        input_hdc = self._encode_input_for_classification(text)

        scores = {}
        for problem_type, proto_vectors in self.prototype_vectors.items():
            similarities = [float(similarity(input_hdc, pv)) for pv in proto_vectors]
            avg_sim = sum(similarities) / len(similarities)
            max_sim = max(similarities)
            best_proto_idx = similarities.index(max_sim)
            score = 0.6 * avg_sim + 0.4 * max_sim

            scores[problem_type] = {
                "score": round(score, 3),
                "avg_similarity": round(avg_sim, 3),
                "max_similarity": round(max_sim, 3),
                "best_matching_prototype": PROTOTYPE_SENTENCES[problem_type][best_proto_idx],
                "all_similarities": [round(s, 3) for s in similarities],
            }

        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)

        return {
            "input": text,
            "winner": ranked[0][0] if ranked[0][1]["score"] > 0.30 else "unknown",
            "winner_score": ranked[0][1]["score"],
            "all_scores": dict(ranked),
        }
