"""Temporal connective definitions for multiple languages.

Extensible registry of temporal discourse connectives mapped to
Allen's interval algebra relations. Currently supports English,
with slots for other languages.

Architecture:
- AllenRelation and SemanticType enums for type safety
- LanguageProfile groups connectives + conditionals + time nouns
- register_language() for adding new languages
- Builder pattern instead of global mutation

References:
- TimeML (Pustejovsky et al., 2003)
- PDTB 3.0 (Webber et al., 2019)
- CICLING — "On the Identification of Temporal Clauses"
- Allen (1983) — "Maintaining Knowledge about Temporal Intervals"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class AllenRelation(str, Enum):
    """Allen's interval algebra relations (subset used in TD v2)."""
    BEFORE = "before"       # X ends before Y starts
    AFTER = "after"         # X starts after Y ends
    MEETS = "meets"         # X ends exactly when Y starts
    OVERLAPS = "overlaps"   # X starts before Y, ends during Y
    DURING = "during"       # X is contained within Y
    EQUALS = "equals"       # X and Y are co-extensive


class SemanticType(str, Enum):
    """Semantic classification of temporal connectives."""
    SEQUENTIAL = "sequential"       # "then", "next", "finally"
    SIMULTANEOUS = "simultaneous"   # "while", "meanwhile", "during"
    PRECEDENCE = "precedence"       # "before", "earlier", "previously"
    SUCCESSION = "succession"       # "after", "since", "once"
    CONDITIONAL = "conditional"     # "once" (conditional-temporal hybrid)


@dataclass
class TemporalConnective:
    """A temporal discourse connective with its Allen relation."""
    word: str
    allen_relation: AllenRelation
    semantic_type: SemanticType
    dep_pattern: str            # Expected spaCy dep: "advmod", "mark", "prep"
    is_multiword: bool = False
    is_conditional_context: bool = False  # "then" can be conditional


@dataclass
class LanguageProfile:
    """Complete temporal profile for a language."""
    connectives: dict[str, TemporalConnective] = field(default_factory=dict)
    conditional_markers: frozenset[str] = field(default_factory=frozenset)
    time_nouns: frozenset[str] = field(default_factory=frozenset)


# ─── English Temporal Connectives ───────────────────────────────────
# Comprehensive list from TimeML, PDTB 3.0, and CICLING.

def _build_english_profile() -> LanguageProfile:
    """Build the English temporal connective profile.

    Uses builder pattern instead of global mutation.
    """
    connectives: dict[str, TemporalConnective] = {}

    def _en(word: str, rel: AllenRelation, type_: SemanticType,
            dep: str = "advmod", multi: bool = False,
            conditional: bool = False):
        connectives[word] = TemporalConnective(
            word, rel, type_, dep, multi, conditional
        )

    # Sequential (forward-looking: event1 happened, THEN event2)
    _en("then",           AllenRelation.BEFORE,    SemanticType.SEQUENTIAL, conditional=True)
    _en("subsequently",   AllenRelation.BEFORE,    SemanticType.SEQUENTIAL)
    _en("afterwards",     AllenRelation.BEFORE,    SemanticType.SEQUENTIAL)
    _en("afterward",      AllenRelation.BEFORE,    SemanticType.SEQUENTIAL)
    _en("next",           AllenRelation.BEFORE,    SemanticType.SEQUENTIAL)
    _en("first",          AllenRelation.BEFORE,    SemanticType.SEQUENTIAL)
    _en("finally",        AllenRelation.BEFORE,    SemanticType.SEQUENTIAL)
    _en("later",          AllenRelation.BEFORE,    SemanticType.SEQUENTIAL)
    _en("soon",           AllenRelation.BEFORE,    SemanticType.SEQUENTIAL)
    _en("eventually",     AllenRelation.BEFORE,    SemanticType.SEQUENTIAL)
    _en("thereafter",     AllenRelation.BEFORE,    SemanticType.SEQUENTIAL)
    _en("henceforth",     AllenRelation.BEFORE,    SemanticType.SEQUENTIAL)
    _en("subsequent",     AllenRelation.BEFORE,    SemanticType.SEQUENTIAL)
    _en("ultimately",     AllenRelation.BEFORE,    SemanticType.SEQUENTIAL)
    _en("presently",      AllenRelation.BEFORE,    SemanticType.SEQUENTIAL)

    # Simultaneous (events at the same time)
    _en("meanwhile",      AllenRelation.OVERLAPS,  SemanticType.SIMULTANEOUS)
    _en("meantime",       AllenRelation.OVERLAPS,  SemanticType.SIMULTANEOUS)
    _en("at the same time", AllenRelation.EQUALS,  SemanticType.SIMULTANEOUS, multi=True)
    _en("in the meantime",  AllenRelation.OVERLAPS, SemanticType.SIMULTANEOUS, multi=True)

    # Precedence (backward-looking: event2 references event1 in the past)
    _en("previously",     AllenRelation.AFTER,     SemanticType.PRECEDENCE)
    _en("earlier",        AllenRelation.AFTER,     SemanticType.PRECEDENCE)
    _en("beforehand",     AllenRelation.AFTER,     SemanticType.PRECEDENCE)
    _en("prior",          AllenRelation.AFTER,     SemanticType.PRECEDENCE)
    _en("already",        AllenRelation.AFTER,     SemanticType.PRECEDENCE)

    # Subordinating conjunctions (dep=mark)
    _en("after",          AllenRelation.AFTER,     SemanticType.SUCCESSION, dep="mark")
    _en("before",         AllenRelation.BEFORE,    SemanticType.PRECEDENCE, dep="prep")
    _en("since",          AllenRelation.AFTER,     SemanticType.SUCCESSION, dep="mark")
    _en("once",           AllenRelation.BEFORE,    SemanticType.CONDITIONAL, dep="mark")
    _en("until",          AllenRelation.MEETS,     SemanticType.SIMULTANEOUS, dep="mark")
    _en("till",           AllenRelation.MEETS,     SemanticType.SIMULTANEOUS, dep="mark")
    _en("when",           AllenRelation.OVERLAPS,  SemanticType.SIMULTANEOUS, dep="mark")
    _en("whenever",       AllenRelation.OVERLAPS,  SemanticType.SIMULTANEOUS, dep="mark")
    _en("while",          AllenRelation.DURING,    SemanticType.SIMULTANEOUS, dep="mark")
    _en("whilst",         AllenRelation.DURING,    SemanticType.SIMULTANEOUS, dep="mark")
    _en("as",             AllenRelation.OVERLAPS,  SemanticType.SIMULTANEOUS, dep="mark")

    # Multi-word subordinating
    _en("as soon as",     AllenRelation.MEETS,     SemanticType.SUCCESSION, dep="mark", multi=True)
    _en("as long as",     AllenRelation.DURING,    SemanticType.SIMULTANEOUS, dep="mark", multi=True)
    _en("so long as",     AllenRelation.DURING,    SemanticType.SIMULTANEOUS, dep="mark", multi=True)
    _en("by the time",    AllenRelation.BEFORE,    SemanticType.PRECEDENCE, dep="mark", multi=True)
    _en("now that",       AllenRelation.AFTER,     SemanticType.SUCCESSION, dep="mark", multi=True)
    _en("every time",     AllenRelation.OVERLAPS,  SemanticType.SIMULTANEOUS, dep="mark", multi=True)

    # Prepositional (dep=prep)
    _en("during",         AllenRelation.DURING,    SemanticType.SIMULTANEOUS, dep="prep")
    _en("throughout",     AllenRelation.DURING,    SemanticType.SIMULTANEOUS, dep="prep")
    _en("within",         AllenRelation.DURING,    SemanticType.SIMULTANEOUS, dep="prep")
    _en("between",        AllenRelation.DURING,    SemanticType.SIMULTANEOUS, dep="prep")

    conditional_markers = frozenset({
        "if", "when", "whenever", "unless", "provided",
        "assuming", "suppose", "given that", "in case",
    })

    time_nouns = frozenset({
        "noon", "midnight", "dawn", "dusk", "sunrise", "sunset",
        "morning", "afternoon", "evening", "night",
        "today", "tomorrow", "yesterday",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "year", "month", "week", "day", "hour", "minute", "second",
        "years", "months", "weeks", "days", "hours", "minutes", "seconds",
        "christmas", "easter",
    })

    return LanguageProfile(
        connectives=connectives,
        conditional_markers=conditional_markers,
        time_nouns=time_nouns,
    )


# ─── Language Registry ──────────────────────────────────────────────

LANGUAGE_PROFILES: dict[str, LanguageProfile] = {
    "en": _build_english_profile(),
    # Placeholder slots for other languages:
    # "zh": _build_chinese_profile(),
    # "de": _build_german_profile(),
    # "fr": _build_french_profile(),
    # "es": _build_spanish_profile(),
    # "ja": _build_japanese_profile(),
    # "ko": _build_korean_profile(),
    # "ru": _build_russian_profile(),
    # "ar": _build_arabic_profile(),
}


def register_language(lang: str, profile: LanguageProfile) -> None:
    """Register temporal connectives for a new language.

    Usage:
        profile = LanguageProfile(
            connectives={"然后": TemporalConnective("然后", AllenRelation.BEFORE, ...)},
            conditional_markers=frozenset({"如果"}),
            time_nouns=frozenset({"今天", "明天"}),
        )
        register_language("zh", profile)
    """
    LANGUAGE_PROFILES[lang] = profile


def get_connectives(lang: str = "en") -> dict[str, TemporalConnective]:
    """Get temporal connectives for a language."""
    profile = LANGUAGE_PROFILES.get(lang)
    if profile is None:
        profile = LANGUAGE_PROFILES.get("en")
    return profile.connectives


def get_conditional_markers(lang: str = "en") -> frozenset:
    """Get conditional markers for a language."""
    profile = LANGUAGE_PROFILES.get(lang)
    if profile is None:
        profile = LANGUAGE_PROFILES.get("en")
    return profile.conditional_markers


def get_time_nouns(lang: str = "en") -> frozenset:
    """Get time nouns for a language."""
    profile = LANGUAGE_PROFILES.get(lang)
    if profile is None:
        profile = LANGUAGE_PROFILES.get("en")
    return profile.time_nouns
