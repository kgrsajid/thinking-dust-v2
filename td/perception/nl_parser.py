"""Natural Language → HDC vector parser.

NOT an LLM. NOT a transformer. Just keyword extraction + concept lookup.

Pipeline:
    1. Tokenize (split + lowercase + strip punctuation)
    2. Concept matching (keyword → vocabulary lookup)
    3. Role detection (action words, targets, conditions)
    4. HDC composition: bundle(bind(role, concept), ...)
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


# Keywords that map to roles in encode_record
ACTION_KEYWORDS = {
    "click", "type", "scroll", "submit", "select", "hover", "drag",
    "navigate", "extract", "fill", "check", "open", "close", "fetch",
    "post", "call", "read", "write", "create", "parse", "validate",
    "convert", "alert", "restart", "monitor", "wait", "refresh",
    "delete", "copy", "move", "export", "import", "search", "filter",
}

TARGET_KEYWORDS = {
    "button", "form", "field", "input", "dropdown", "checkbox",
    "link", "table", "modal", "tab", "menu", "page", "url",
    "endpoint", "api", "response", "request", "file", "csv", "json",
    "schema", "column", "service", "process", "cpu", "memory", "disk",
}

CONTEXT_KEYWORDS = {
    "login", "contact", "registration", "checkout", "search",
    "dashboard", "settings", "profile", "admin", "home", "landing",
}

# Multi-word concept mappings
COMPOUND_CONCEPTS = {
    "submit button": "submit_button",
    "login form": "login_form",
    "contact form": "contact_form",
    "search bar": "search_bar",
    "api key": "api_key",
    "auth token": "auth_token",
    "status code": "status_code",
}


class NLParser:
    """Lightweight natural language → HDC encoder.

    Uses keyword extraction and concept vocabulary lookup.
    No model inference required — pure dictionary lookups.

    Time: ~0.1ms per parse.
    """

    def __init__(self, vocabulary: ConceptVocabulary):
        """Initialize with a concept vocabulary.

        Args:
            vocabulary: Pre-populated ConceptVocabulary.
        """
        self.vocab = vocabulary

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization: lowercase, strip punctuation, split."""
        text = text.lower().strip()
        # Replace non-alphanumeric (except hyphens/underscores) with space
        text = re.sub(r"[^\w\s\-]", " ", text)
        tokens = text.split()
        return tokens

    def _extract_compound_concepts(self, text: str) -> list[str]:
        """Detect multi-word concepts like 'submit button' → 'submit_button'."""
        found = []
        lower = text.lower()
        for phrase, concept in COMPOUND_CONCEPTS.items():
            if phrase in lower:
                found.append(concept)
        return found

    def extract_concepts(self, text: str) -> dict[str, str]:
        """Extract structured concepts from text.

        Identifies action, target, and context from a natural language
        command using keyword matching.

        Args:
            text: Natural language input.

        Returns:
            Dict with keys like "action", "target", "context" mapping
            to concept names. Only includes keys for detected concepts.

        Example:
            >>> parser.extract_concepts("Click the submit button on the login form")
            {"action": "click", "target": "submit_button", "context": "login"}
        """
        tokens = self._tokenize(text)
        result: dict[str, str] = {}

        # Detect compound concepts first
        compounds = self._extract_compound_concepts(text)

        # Extract action
        for token in tokens:
            if token in ACTION_KEYWORDS:
                result["action"] = token
                break

        # Extract target (check compounds first, then single tokens)
        target_found = False
        for compound in compounds:
            if compound in ("submit_button", "login_form", "contact_form",
                            "search_bar", "api_key", "auth_token", "status_code"):
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
            if token in CONTEXT_KEYWORDS:
                result["context"] = token
                break
        # Check compound concepts for context
        if "context" not in result:
            for compound in compounds:
                if "form" in compound or "login" in compound:
                    base = compound.replace("form", "").replace("_", "").strip()
                    if base and base in CONTEXT_KEYWORDS:
                        result["context"] = base
                        break

        return result

    def parse(self, text: str) -> np.ndarray:
        """Parse natural language into HDC vector.

        Extracts concepts and composes them into a single hypervector
        using role-filler binding:

            bundle(bind(action_role, action_concept),
                   bind(target_role, target_concept),
                   bind(context_role, context_concept))

        If no concepts are detected, generates a random vector (which
        will have low similarity to all stored patterns → triggers
        escalation).

        Args:
            text: Natural language input.

        Returns:
            np.ndarray[int8] of shape (dim,).

        Time: ~0.1ms.
        """
        concepts = self.extract_concepts(text)

        if not concepts:
            # No concepts detected → random vector (will trigger escalation)
            return generate_hypervector(self.vocab.dim)

        # Ensure all roles and fillers exist in vocabulary
        for role, filler in concepts.items():
            if not self.vocab.has(role):
                self.vocab.add_concept(role)
            if not self.vocab.has(filler):
                self.vocab.add_concept(filler)

        return self.vocab.encode_record(**concepts)

    def parse_with_extra(self, text: str, **extra: str) -> np.ndarray:
        """Parse text and merge with additional key-value context.

        Useful for adding metadata like "domain", "priority", etc.

        Args:
            text: Natural language input.
            **extra: Additional key-value pairs to bind into the vector.

        Returns:
            Bipolar hypervector.
        """
        concepts = self.extract_concepts(text)
        concepts.update(extra)

        if not concepts:
            return generate_hypervector(self.vocab.dim)

        for role, filler in concepts.items():
            if not self.vocab.has(role):
                self.vocab.add_concept(role)
            if not self.vocab.has(filler):
                self.vocab.add_concept(filler)

        return self.vocab.encode_record(**concepts)
