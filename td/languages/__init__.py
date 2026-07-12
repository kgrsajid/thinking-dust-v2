"""Language-specific configurations for Thinking Dust.

Each language has its own module with word sets, stop words,
discourse deixis verbs, and other language-specific data.

The main code loads from this registry — no hardcoded English
words in the core logic.

To add a new language:
1. Copy td/languages/en.py → td/languages/xx.py
2. Replace all English words with target language equivalents
3. Register in td/languages/__init__.py
4. See DEVELOPMENT.md "Adding a New Language" for detailed guide

Reference: Universal Dependencies (Nivre et al., 2016)
Reference: spaCy multilingual support (20+ languages)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Set


@dataclass
class LanguageConfig:
    """Configuration for a specific language.

    Contains all language-specific word sets used by the parser.
    The core logic accesses these through the registry — never
    directly from hardcoded sets.

    Attributes:
        code: ISO 639-1 language code (e.g., "en", "de", "fr")
        name: Human-readable language name
        stop_words: Stop words for fallback (when spaCy unavailable)
        prepositions: Prepositions for regex fallback (when spaCy unavailable)
        possessive_pronouns: Possessive pronouns (for coreference)
        entity_pronouns: Personal/entity pronouns (for coreference)
        discourse_deixis_verbs: Abstract verbs for discourse deixis
        discourse_deixis_it_verbs: Subset where "it" is discourse deixis
        relation_prototypes: HDC-encoded phrases for constraint type detection
            (dict of relation_name -> list of synonym phrases)
    """
    code: str
    name: str
    stop_words: FrozenSet[str] = frozenset()
    prepositions: FrozenSet[str] = frozenset()
    possessive_pronouns: FrozenSet[str] = frozenset()
    entity_pronouns: FrozenSet[str] = frozenset()
    discourse_deixis_verbs: FrozenSet[str] = frozenset()
    discourse_deixis_it_verbs: FrozenSet[str] = frozenset()
    relation_prototypes: Dict[str, str] = field(default_factory=dict)
    # Copula verbs for pattern matching (e.g., "is", "are", "was")
    # Used by _merge_* methods. Replace with spaCy 'cop' dep when possible.
    copula_verbs: FrozenSet[str] = frozenset()
    # Articles for entity name cleaning (e.g., "the", "a", "an")
    # Used by _get_chunk_text. Replace with spaCy 'det' dep when possible.
    articles: FrozenSet[str] = frozenset()
    # Genitive markers (e.g., "of") — entity-internal, not a relation
    genitive_markers: FrozenSet[str] = frozenset()
    # Demonstrative pronouns for discourse deixis detection.
    # Used with Jauhar et al. (2015) two-stage approach.
    # "this"/"that"/"it" as subject of abstract verb → discourse deixis.
    demonstrative_pronouns: FrozenSet[str] = frozenset()
    # BEAGLE-specific stop words for word vector training.
    # Superset of parser stop words: adds pronouns, interrogatives,
    # and function words that should not accumulate context vectors.
    # Used by td/perception/word_vectors.py — NEVER hardcoded.
    beagle_stop_words: FrozenSet[str] = frozenset()
    # Copula-to-is_a mapping for relation canonicalization.
    # Clause segmenter produces (X, copula, Y), dep parser produces (X, is_a, Y).
    # Both represent the same copular relationship.
    # Key: copula word (e.g., "is"), Value: canonical relation (e.g., "is_a")
    # Reference: UD `attr` — nominal predicate of copular construction
    copula_to_isa: Dict[str, str] = field(default_factory=dict)

    # ── Relation Property Detection Words ──────────────────────────
    # Used by detect_relation_properties() in td/kg/__init__.py
    # Language-specific words that indicate relation semantics.

    # Stative/spatial verbs that indicate transitivity
    # "located in", "contained in", "residing in" → transitive
    # Reference: Levin (1993), "English Verb Classes and Alternations"
    stative_verbs: FrozenSet[str] = frozenset()

    # Event verbs that are NOT transitive even with prepositions
    # "born in", "died in", "happened in" → NOT transitive
    event_verbs: FrozenSet[str] = frozenset()

    # Prepositions that indicate transitivity when combined with stative verbs
    transitive_preps: FrozenSet[str] = frozenset()

    # Words that indicate symmetric relations
    # "borders", "adjacent to", "married to", "sibling of" → symmetric
    symmetric_words: FrozenSet[str] = frozenset()

    # Default relation properties for this language (bootstrap)
    # These are pre-seeded when the KG is created.
    # Format: {relation_name: {property_set}}
    default_relation_properties: Dict[str, set] = field(default_factory=dict)


# ── Language Registry ─────────────────────────────────────────────
_REGISTRY: Dict[str, LanguageConfig] = {}


def register_language(config: LanguageConfig):
    """Register a language configuration.

    Example:
        register_language(LanguageConfig(
            code="de",
            name="German",
            stop_words={"der", "die", "das", "ein", "eine", ...},
            prepositions={"auf", "aus", "bei", "mit", ...},
            ...
        ))
    """
    _REGISTRY[config.code] = config


def get_language(code: str) -> LanguageConfig:
    """Get language configuration by code. Returns English if not found.

    Logs a warning when falling back to English.
    """
    if code in _REGISTRY:
        return _REGISTRY[code]
    if code != "en":
        import warnings
        warnings.warn(
            f"Language '{code}' not registered in td/languages/. "
            f"Falling back to English. Available: {get_available_languages()}",
            stacklevel=2,
        )
    return _REGISTRY.get("en")


def get_available_languages() -> list[str]:
    """List registered language codes."""
    return list(_REGISTRY.keys())


# ── Load built-in languages ───────────────────────────────────────
from . import en  # noqa: E402  — English is always available
