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
