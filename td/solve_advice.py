"""Advice mode solver for Thinking Dust v2.

When the user asks "how to", "tips", "advice", "without procrastination", etc.,
TD enters advice mode instead of Z3 constraint solving.

Retrieves behavioral strategies from MHN, composes actionable advice,
validates relevance, and formats human-readable output.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from .perception.hdc import similarity, bind, bundle


# ── STRATEGY DATABASE (seed patterns, learned from MHN at runtime) ──

BEHAVIORAL_STRATEGIES = {
    # Productivity / Anti-procrastination
    "pomodoro": {
        "title": "Pomodoro Technique",
        "description": "Work in 25-minute focused bursts, then take a 5-minute break. After 4 cycles, take a longer 15-30 minute break.",
        "tags": ["focus", "time_management", "procrastination", "study"],
        "effectiveness": 0.85,
    },
    "eat_the_frog": {
        "title": "Eat the Frog",
        "description": "Tackle your hardest, most important task first thing in the morning. Everything after feels easier.",
        "tags": ["priority", "procrastination", "morning_routine", "willpower"],
        "effectiveness": 0.80,
    },
    "distraction_free": {
        "title": "Eliminate Distractions",
        "description": "Phone in another room. Block social media. Use website blockers. Create a dedicated workspace.",
        "tags": ["focus", "environment", "procrastination", "deep_work"],
        "effectiveness": 0.75,
    },
    "time_blocking": {
        "title": "Time Blocking",
        "description": "Assign every task to a specific time slot on your calendar. Treat it like a meeting you cannot miss.",
        "tags": ["scheduling", "planning", "time_management", "commitment"],
        "effectiveness": 0.82,
    },
    "two_minute_rule": {
        "title": "Two-Minute Rule",
        "description": "If a task takes less than 2 minutes, do it immediately. Prevents small tasks from piling up.",
        "tags": ["productivity", "quick_wins", "momentum", "habit"],
        "effectiveness": 0.78,
    },
    "accountability_partner": {
        "title": "Accountability Partner",
        "description": "Tell someone your plan and deadline. The social commitment makes you more likely to follow through.",
        "tags": ["social", "commitment", "procrastination", "motivation"],
        "effectiveness": 0.72,
    },
    "energy_management": {
        "title": "Match Tasks to Energy Levels",
        "description": "Do creative/hard work when energy is high. Do admin/easy work when energy is low. Track your circadian rhythm.",
        "tags": ["energy", "timing", "productivity", "self_awareness"],
        "effectiveness": 0.80,
    },

    # Learning / Study
    "active_recall": {
        "title": "Active Recall",
        "description": "Don't re-read. Close the book and write down everything you remember from memory. Test yourself actively.",
        "tags": ["learning", "memory", "study", "retention"],
        "effectiveness": 0.90,
    },
    "spaced_repetition": {
        "title": "Spaced Repetition",
        "description": "Review material at increasing intervals (1 day, 3 days, 7 days, 14 days). Use Anki or similar tools.",
        "tags": ["learning", "memory", "long_term", "study"],
        "effectiveness": 0.88,
    },
    "interleaving": {
        "title": "Interleaving",
        "description": "Mix different topics/skills in one study session instead of blocking. Forces deeper discrimination.",
        "tags": ["learning", "discrimination", "study", "skill_building"],
        "effectiveness": 0.75,
    },
    "teaching_others": {
        "title": "Teach What You Learn",
        "description": "Explain the concept to someone else (or a rubber duck). If you can't explain it simply, you don't understand it.",
        "tags": ["learning", "understanding", "feynman", "study"],
        "effectiveness": 0.85,
    },

    # Debugging / Coding
    "reproduce_first": {
        "title": "Reproduce Consistently",
        "description": "Before fixing, make the bug happen every time. If you can't reproduce it, you can't verify the fix.",
        "tags": ["debugging", "testing", "verification", "methodical"],
        "effectiveness": 0.92,
    },
    "isolate_scope": {
        "title": "Isolate the Scope",
        "description": "Comment out half the code. Does the bug still happen? Binary search until you find the minimal reproducing case.",
        "tags": ["debugging", "binary_search", "minimal_reproduction", "efficiency"],
        "effectiveness": 0.88,
    },
    "rubber_duck": {
        "title": "Rubber Duck Debugging",
        "description": "Explain your code line by line to an inanimate object. The act of verbalizing often reveals the bug.",
        "tags": ["debugging", "verbalization", "insight", "cognitive"],
        "effectiveness": 0.78,
    },
    "binary_search_bugs": {
        "title": "Binary Search for Bugs",
        "description": "Git bisect between last known good commit and current bad commit. Find the exact commit that introduced the bug.",
        "tags": ["debugging", "git", "version_control", "efficiency"],
        "effectiveness": 0.85,
    },

    # Code Quality
    "small_functions": {
        "title": "Small Functions",
        "description": "Each function should do one thing and do it well. If you need a comment to explain what it does, it's too big.",
        "tags": ["code_quality", "readability", "maintenance", "simplicity"],
        "effectiveness": 0.82,
    },
    "meaningful_names": {
        "title": "Meaningful Names",
        "description": "A variable name should tell you WHY it exists and HOW it's used. Avoid `data`, `temp`, `x`.",
        "tags": ["code_quality", "readability", "naming", "clarity"],
        "effectiveness": 0.80,
    },
    "dry_principle": {
        "title": "DRY Principle",
        "description": "Don't Repeat Yourself. If you see the same code twice, extract it into a function or constant.",
        "tags": ["code_quality", "maintenance", "refactoring", "efficiency"],
        "effectiveness": 0.85,
    },
    "single_responsibility": {
        "title": "Single Responsibility",
        "description": "A class/module should have one reason to change. If it does two things, split it.",
        "tags": ["code_quality", "architecture", "modularity", "design"],
        "effectiveness": 0.83,
    },

    # General Problem Solving
    "first_principles": {
        "title": "First Principles Thinking",
        "description": "Break the problem down to fundamental truths. Rebuild from there instead of reasoning by analogy.",
        "tags": ["problem_solving", "creativity", "innovation", "deep_thinking"],
        "effectiveness": 0.85,
    },
    "inversion": {
        "title": "Inversion",
        "description": "Instead of asking 'how do I succeed?', ask 'how do I guarantee failure?' Then avoid those things.",
        "tags": ["problem_solving", "strategy", "risk_management", "clarity"],
        "effectiveness": 0.80,
    },
    "occams_razor": {
        "title": "Occam's Razor",
        "description": "The simplest explanation is usually correct. Don't add complexity unless necessary.",
        "tags": ["problem_solving", "simplicity", "debugging", "design"],
        "effectiveness": 0.82,
    },
}


class AdviceSolver:
    """Solver for advice-type problems.

    Retrieves relevant behavioral strategies from MHN or seed database,
    composes actionable advice tailored to the user's specific constraints.
    """

    def __init__(self, mhn, hdc_vocab, parser):
        self.mhn = mhn
        self.vocab = hdc_vocab
        self.parser = parser

    def solve(self, entities: dict[str, Any]) -> dict[str, Any]:
        """Generate advice based on extracted entities.

        Args:
            entities: Output from NLParser.extract_entities()
                - problem_type: 'advice'
                - what: list of tasks/objects
                - how_many: quantity
                - goals: list of constraints/goals (e.g., 'avoid_procrastination')
                - who: list of people/roles

        Returns:
            dict with:
                - type: 'advice'
                - strategies: list of actionable tips
                - confidence: float (0-1)
                - caveat: str (disclaimer about heuristic nature)
                - reasoning_trace: list of steps
        """
        trace = []

        # 1. Encode the problem for MHN retrieval
        problem_hdc = self._encode_problem(entities)
        trace.append("Encoded problem as HDC vector")

        # 2. Retrieve relevant strategies from MHN
        retrieved = self.mhn.retrieve(problem_hdc, top_k=1)
        if retrieved:
            mhn_sim = retrieved[0][1]
            mhn_meta = retrieved[0][2]
            trace.append(f"MHN retrieval: similarity={mhn_sim:.3f}")
        else:
            mhn_sim = 0.0
            mhn_meta = {}
            trace.append("MHN retrieval: no match")

        strategies = []

        if mhn_sim > 0.6 and mhn_meta:
            # Use retrieved decomposition as strategy hints
            strategy_keys = self._decode_strategy_keys(mhn_meta)
            trace.append(f"Retrieved {len(strategy_keys)} strategy hints from MHN")
        else:
            # Fallback: match strategies by tags
            strategy_keys = self._match_strategies_by_tags(entities)
            trace.append(f"Matched {len(strategy_keys)} strategies by tags")

        # 3. Compose actionable advice
        for key in strategy_keys:
            if key in BEHAVIORAL_STRATEGIES:
                strategy = BEHAVIORAL_STRATEGIES[key]
                strategies.append({
                    "title": strategy["title"],
                    "description": strategy["description"],
                    "effectiveness": strategy["effectiveness"],
                })

        # 4. Validate: do strategies address the user's goals?
        validated_strategies = self._validate_relevance(strategies, entities)
        trace.append(f"Validated {len(validated_strategies)} strategies against user goals")

        # 5. Calculate confidence
        confidence = self._calculate_confidence(validated_strategies, entities, mhn_sim)
        trace.append(f"Confidence: {confidence:.2f}")

        return {
            "type": "advice",
            "strategies": validated_strategies,
            "confidence": confidence,
            "caveat": "Behavioral advice based on known techniques. Results vary by individual. Not formally proven.",
            "reasoning_trace": trace,
            "source": "MHN" if mhn_sim > 0.6 else "tag_matching",
        }

    def _encode_problem(self, entities: dict) -> np.ndarray:
        """Encode problem entities into HDC vector for MHN retrieval."""
        vectors = []

        # Encode problem type
        if not self.vocab.has("problem_type"):
            self.vocab.add_concept("problem_type")
        if not self.vocab.has("advice"):
            self.vocab.add_concept("advice")
        vectors.append(bind(self.vocab.get("problem_type"), self.vocab.get("advice")))

        # Encode what (tasks/objects)
        if entities.get("what"):
            what_vectors = []
            for item in entities["what"]:
                if not self.vocab.has(item):
                    self.vocab.add_concept(item)
                what_vectors.append(self.vocab.get(item))
            if not self.vocab.has("what"):
                self.vocab.add_concept("what")
            vectors.append(bind(self.vocab.get("what"), bundle(*what_vectors)))

        # Encode goals (constraints, things to avoid)
        if entities.get("goals"):
            goal_vectors = []
            for g in entities["goals"]:
                if not self.vocab.has(g):
                    self.vocab.add_concept(g)
                goal_vectors.append(self.vocab.get(g))
            if not self.vocab.has("goals"):
                self.vocab.add_concept("goals")
            vectors.append(bind(self.vocab.get("goals"), bundle(*goal_vectors)))

        # Encode how_many
        if "how_many" in entities:
            n = str(entities["how_many"])
            if not self.vocab.has(n):
                self.vocab.add_concept(n)
            if not self.vocab.has("how_many"):
                self.vocab.add_concept("how_many")
            vectors.append(bind(self.vocab.get("how_many"), self.vocab.get(n)))

        return bundle(*vectors)

    def _decode_strategy_keys(self, meta: dict) -> list[str]:
        """Extract strategy keys from MHN metadata."""
        if isinstance(meta, dict):
            sol = meta.get("solution", meta)
            if isinstance(sol, dict):
                action = str(sol.get("action", ""))
                if "debug" in action:
                    return ["reproduce_first", "isolate_scope", "rubber_duck", "binary_search_bugs"]
                elif "schedule" in action or "plan" in action:
                    return ["pomodoro", "eat_the_frog", "time_blocking", "two_minute_rule"]
        return []

    def _match_strategies_by_tags(self, entities: dict) -> list[str]:
        """Match strategies by tag overlap with user goals and context."""
        user_tags = set()

        # Extract tags from goals
        for goal in entities.get("goals", []):
            user_tags.add(goal.replace("avoid_", ""))
            user_tags.add("procrastination")
            user_tags.add("focus")

        # Extract tags from what
        for what in entities.get("what", []):
            user_tags.add(what)

        # Score each strategy by tag overlap
        scored = []
        for key, strategy in BEHAVIORAL_STRATEGIES.items():
            overlap = len(set(strategy["tags"]) & user_tags)
            if overlap > 0:
                scored.append((key, overlap + strategy["effectiveness"]))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [k for k, _ in scored[:5]]

    def _validate_relevance(self, strategies: list[dict], entities: dict) -> list[dict]:
        """Ensure strategies actually address user goals."""
        raw_goals = entities.get("goals", [])
        # Clean goals: strip punctuation, remove filler words
        goals = []
        for g in raw_goals:
            clean = g.replace("avoid_", "").replace("?", "").replace("without", "").strip()
            if clean and len(clean) > 2:  # Skip tiny fragments
                goals.append(clean)

        if not goals:
            return strategies  # No goals to validate against — accept all

        validated = []
        for s in strategies:
            desc = s["description"].lower()
            title = s["title"].lower()
            relevant = any(
                goal in desc or goal in title
                for goal in goals
            )
            if relevant:
                validated.append(s)

        # If nothing matched, return top strategies anyway (better than nothing)
        if not validated and strategies:
            return strategies[:3]

        return validated

    def _calculate_confidence(self, strategies: list[dict], entities: dict, mhn_similarity: float) -> float:
        """Calculate confidence score."""
        if not strategies:
            return 0.3

        # Base confidence from strategy effectiveness
        avg_effectiveness = sum(s["effectiveness"] for s in strategies) / len(strategies)

        # Boost if MHN had a good match
        mhn_boost = min(mhn_similarity * 0.2, 0.15)

        # Penalize if few strategies
        coverage_penalty = max(0, (5 - len(strategies)) * 0.05)

        confidence = avg_effectiveness + mhn_boost - coverage_penalty
        return min(confidence, 0.95)  # Cap at 0.95 (advice is never 100%)

    def format_output(self, result: dict[str, Any]) -> str:
        """Format advice result as human-readable text."""
        lines = []
        lines.append("Here's a strategy based on known techniques:\n")

        for i, strategy in enumerate(result["strategies"], 1):
            title = strategy['title']
            desc = strategy['description']
            lines.append(f"{i}. **{title}**")
            lines.append(f"   {desc}\n")

        conf = result['confidence']
        caveat = result['caveat']
        lines.append(f"\n*Confidence: {conf:.0%}*")
        lines.append(f"*{caveat}*")

        return "\n".join(lines)
