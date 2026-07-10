"""Preprocessing Layer for TD v2.

Transforms messy human input into clean, structured sentences
that the parser can extract triples from.

Architecture: User → preprocess() → TD v2 think()/teach()

This is Layer 0 in the TD v2 architecture. TD v2 is a reasoning engine,
NOT an NLP engine. This layer handles the messy human language.

Reference: ARCHITECTURE.md §2 — Layer 0: Input Preprocessing
Reference: PREPROCESSING_PROMPT.md — LLM prompt design
"""

from __future__ import annotations

import json
import re
from typing import Optional


# Default preprocessing prompt (loaded from PREPROCESSING_PROMPT.md)
PREPROCESSING_PROMPT = """You are a sentence simplifier for a knowledge graph engine.
Your job: break complex human input into SIMPLE atomic sentences.

## Rules

1. ONE fact per sentence. No compound sentences.
2. ALWAYS repeat the subject. Never use pronouns (he/she/it/they/that).
3. Remove filler words: "so", "like", "you know", "I mean", "basically",
   "actually", "well", "right", "I was wondering", "can you tell me".
4. Use simple SVO structure: Subject Verb Object.
5. For "is a" facts, use: "X is_a Y" (with underscore).
6. For relations, use underscored compound: "X tool_for Y", "X part_of Y".
7. Questions: rewrite as declarative if possible.
8. Output ONLY valid JSON. No explanation.

## Output Format

{"sentences": ["sentence 1", "sentence 2", ...]}

## Examples

INPUT: "So I was curious, what's the deal with matches and fire?"
OUTPUT: {"sentences": ["what is match used for"]}

INPUT: "I was wondering if you could tell me what a match is used for starting fires?"
OUTPUT: {"sentences": ["what is match used for fire"]}

INPUT: "The cell in biology, what's it made of vs the one in prison?"
OUTPUT: {"sentences": ["what is cell in biology made of", "what is cell in prison"]}

INPUT: "Like, cell phones and stuff, they use towers right?"
OUTPUT: {"sentences": ["cell phone uses tower"]}

INPUT: "Paris is the capital of France and France is in the EU"
OUTPUT: {"sentences": ["Paris is the capital of France", "France is in the EU"]}

INPUT: "Alice and Bob went to the store"
OUTPUT: {"sentences": ["Alice went to the store", "Bob went to the store"]}

INPUT: "the thing that you strike to make fire, what's it called"
OUTPUT: {"sentences": ["what is the thing that is struck to make fire"]}

INPUT: "Python is a programming language that's used for data science and web development"
OUTPUT: {"sentences": ["Python is_a programming language", "Python is used for data science", "Python is used for web development"]}

INPUT: "what about prison?"
OUTPUT: {"sentences": ["what about cell in prison"]}

INPUT: "cranes are large wading birds with long legs that migrate thousands of miles"
OUTPUT: {"sentences": ["cranes are large wading birds", "cranes have long legs", "cranes migrate thousands of miles"]}

INPUT: "seals are marine mammals that live in cold waters along the Atlantic coast and they haul out on rocks to rest"
OUTPUT: {"sentences": ["seals are marine mammals", "seals live in cold waters", "seals live along Atlantic coast", "seals haul out on rocks"]}
"""


class Preprocessor:
    """Preprocesses messy human input into clean sentences for TD v2.

    Supports two modes:
    1. LLM-based: calls an LLM with the preprocessing prompt (bootstrap/demo)
    2. Rule-based: applies simple rules (production, zero-cost)

    Usage:
        # LLM mode
        preprocessor = Preprocessor(llm_client=my_client)
        sentences = preprocessor.preprocess("So like, what's the deal with matches?")

        # Rule-based mode (no LLM)
        preprocessor = Preprocessor()
        sentences = preprocessor.preprocess("So like, what's the deal with matches?")
    """

    # Filler words to strip in rule-based mode
    FILLERS = frozenset({
        "so", "like", "you know", "i mean", "basically", "actually",
        "well", "right", "i was wondering", "can you tell me",
        "could you tell me", "do you know", "what about",
    })

    def __init__(self, llm_client=None, prompt: str = PREPROCESSING_PROMPT):
        """Initialize the preprocessor.

        Args:
            llm_client: LLM client with a .complete(prompt) method.
                        If None, uses rule-based preprocessing.
            prompt: The preprocessing prompt (default: PREPROCESSING_PROMPT).
        """
        self.llm_client = llm_client
        self.prompt = prompt

    def preprocess(self, user_input: str) -> list[str]:
        """Preprocess user input into clean sentences.

        Args:
            user_input: Raw user input (messy, informal, complex).

        Returns:
            List of clean, simple sentences for TD v2.
        """
        if not user_input or not user_input.strip():
            return []

        if self.llm_client:
            return self._preprocess_llm(user_input)
        return self._preprocess_rules(user_input)

    def _preprocess_llm(self, user_input: str) -> list[str]:
        """LLM-based preprocessing (bootstrap/demo mode)."""
        full_prompt = f"{self.prompt}\n\nINPUT: \"{user_input}\"\nOUTPUT:"

        try:
            response = self.llm_client.complete(full_prompt)
            # Parse JSON response
            data = json.loads(response.strip())
            sentences = data.get("sentences", [])
            # Validate: each sentence must be a non-empty string
            return [s.strip() for s in sentences if s.strip()]
        except (json.JSONDecodeError, KeyError, AttributeError):
            # Fallback to rule-based if LLM output is invalid
            return self._preprocess_rules(user_input)

    def _preprocess_rules(self, user_input: str) -> list[str]:
        """Rule-based preprocessing (production mode, zero-cost).

        Applies simple rules:
        1. Strip filler words
        2. Split on coordinating conjunctions (and, but, or)
        3. Split on semicolons and periods
        4. Remove leading/trailing whitespace
        """
        text = user_input.strip()

        # Strip filler words (case-insensitive)
        for filler in self.FILLERS:
            text = re.sub(rf'\b{re.escape(filler)}\b', '', text, flags=re.IGNORECASE)

        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        if not text:
            return []

        # Split on coordinating conjunctions with commas
        # "Paris is in France, and France is in the EU" → 2 sentences
        parts = re.split(r',\s*(?:and|but|or)\s+|\s*;\s*', text)

        # Also split on " and " when it connects two independent clauses
        # (has a subject on both sides)
        sentences = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            # Try to split on " and " if both sides look like clauses
            and_splits = re.split(r'\s+and\s+', part)
            if len(and_splits) > 1:
                # Check if each split looks like a clause (has at least 2 words)
                if all(len(s.split()) >= 2 for s in and_splits):
                    sentences.extend(s.strip() for s in and_splits if s.strip())
                else:
                    sentences.append(part)
            else:
                sentences.append(part)

        return sentences if sentences else [text]


def preprocess_query(user_input: str, llm_client=None) -> list[str]:
    """Convenience function for preprocessing.

    Args:
        user_input: Raw user input.
        llm_client: Optional LLM client.

    Returns:
        List of clean sentences for TD v2.
    """
    preprocessor = Preprocessor(llm_client=llm_client)
    return preprocessor.preprocess(user_input)
