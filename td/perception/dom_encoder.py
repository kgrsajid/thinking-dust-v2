"""Web DOM → HDC vector encoder.

Pipeline: DOM tree → feature extraction → CA Reservoir → HDC vector
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter

import numpy as np

from .ca_reservoir import CAReservoir
from .hdc import ConceptVocabulary, bind, bundle


# Tags we care about for feature extraction
TAG_TYPES = {
    "div", "span", "form", "input", "button", "a", "select",
    "textarea", "table", "tr", "td", "img", "ul", "li",
    "h1", "h2", "h3", "p", "label", "option",
}

INPUT_TYPES = {"text", "email", "password", "checkbox", "radio", "submit", "hidden"}


class DOMEncoder:
    """Encodes web DOM state into HDC vector via CA Reservoir.

    Feature extraction from DOM:
    - Tag type counts (normalized)
    - Input field types
    - Form structure
    - Visibility hints
    - Text content keywords (matched against vocabulary)

    All features are projected to binary, fed through CA Reservoir,
    and output as 10K-dim bipolar HDC vector.

    Time: ~0.5ms (parsing) + ~0.1ms (CA + HDC) ≈ 0.6ms.
    """

    def __init__(self, vocabulary: ConceptVocabulary, ca: CAReservoir | None = None):
        """Initialize DOM encoder.

        Args:
            vocabulary: Concept vocabulary for keyword matching.
            ca: Optional pre-configured CA Reservoir. Creates one matching vocab dim if None.
        """
        self.vocab = vocabulary
        if ca is None:
            from .ca_reservoir import CAConfig
            ca = CAReservoir(CAConfig(input_dim=vocabulary.dim))
        self.ca = ca

    def _extract_features(self, html: str) -> dict[str, float]:
        """Extract normalized features from HTML.

        Returns:
            Dict mapping feature name → value in [0, 1].
        """
        features: dict[str, float] = {}
        html_lower = html.lower()

        # Tag counts (normalized by max count)
        tag_counts = Counter(re.findall(r"<(\w+)", html_lower))
        max_count = max(tag_counts.values()) if tag_counts else 1

        for tag, count in tag_counts.items():
            if tag in TAG_TYPES:
                features[f"tag_{tag}"] = count / max_count

        # Input types
        for inp_type in INPUT_TYPES:
            pattern = rf'type=["\']{inp_type}["\']'
            matches = len(re.findall(pattern, html_lower))
            if matches > 0:
                features[f"input_{inp_type}"] = min(matches / 5.0, 1.0)

        # Form detection
        form_count = len(re.findall(r"<form", html_lower))
        if form_count > 0:
            features["has_form"] = 1.0
            features["form_count"] = min(form_count / 3.0, 1.0)

        # Required fields
        required = len(re.findall(r"required", html_lower))
        if required > 0:
            features["has_required_fields"] = 1.0
            features["required_count"] = min(required / 5.0, 1.0)

        # Visibility hints
        if "display:none" in html_lower or "hidden" in html_lower:
            features["has_hidden_elements"] = 1.0
        if "display:block" in html_lower or "display: flex" in html_lower:
            features["has_visible_elements"] = 1.0

        # Captcha detection
        if "captcha" in html_lower or "recaptcha" in html_lower or "hcaptcha" in html_lower:
            features["has_captcha"] = 1.0

        # Submit button
        if re.search(r'type=["\']submit["\']', html_lower) or "<button" in html_lower:
            features["has_submit"] = 1.0

        return features

    def _features_to_binary(self, features: dict[str, float]) -> np.ndarray:
        """Convert features to binary array for CA Reservoir input.

        Each feature above threshold 0.5 contributes a 1, others 0.
        Uses deterministic SHA-256 hashing (NOT Python's salted hash()).
        """
        binary = np.zeros(200, dtype=np.uint8)
        for name, value in features.items():
            pos = int(hashlib.sha256(name.encode()).hexdigest(), 16) % 200
            binary[pos] = 1 if value > 0.5 else 0
        return binary

    def encode(self, dom_html: str) -> np.ndarray:
        """Encode DOM HTML into HDC vector.

        Combines CA Reservoir (structural features) with concept
        binding (keyword extraction) for a unified representation.

        Args:
            dom_html: HTML string.

        Returns:
            np.ndarray[int8] of shape (10_000,).
        """
        # Structural features → CA Reservoir → HDC
        features = self._extract_features(dom_html)
        binary = self._features_to_binary(features)
        ca_vector = self.ca.evolve(binary)

        # Keyword extraction → concept binding
        concept_parts = []
        html_lower = dom_html.lower()
        for concept_name in self.vocab.concepts:
            if len(concept_name) > 2 and concept_name in html_lower:
                concept_parts.append(self.vocab.get(concept_name))

        if concept_parts:
            # Bundle concept vectors
            concept_vec = concept_parts[0]
            for cv in concept_parts[1:]:
                concept_vec = bundle(concept_vec, cv)
            # Combine CA structural + concept semantic
            return bundle(ca_vector, concept_vec)
        else:
            return ca_vector
