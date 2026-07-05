"""Temporal connective definitions for multiple languages.

Extensible registry of temporal discourse connectives mapped to
Allen's interval algebra relations. Currently supports English,
with slots for other languages.

Architecture:
- Each language has a CONNECTIVES dict: word → Allen relation
- Multi-word connectives supported (e.g., "as soon as")
- Connectives are classified by semantic type (sequential, simultaneous, etc.)
- The extractor uses this registry to detect temporal ordering

References:
- TimeML (Pustejovsky et al., 2003)
- PDTB 3.0 (Webber et al., 2019)
- CICLING — "On the Identification of Temporal Clauses"
- Allen (1983) — "Maintaining Knowledge about Temporal Intervals"
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class TemporalConnective:
    """A temporal discourse connective with its Allen relation."""
    word: str               # The connective word/phrase
    allen_relation: str     # Allen's relation: "before", "after", "overlaps", "during", "meets"
    semantic_type: str      # "sequential", "precedence", "succession", "simultaneous", "conditional"
    dep_pattern: str        # Expected spaCy dep: "advmod", "mark", "prep", "cc"
    is_multiword: bool = False


# Allen's interval algebra relations (subset used in TD v2):
# before  — X ends before Y starts
# after   — X starts after Y ends
# meets   — X ends exactly when Y starts
# overlaps — X starts before Y, ends during Y
# during  — X is contained within Y
# equals  — X and Y are co-extensive

# ─── English Temporal Connectives ───────────────────────────────────
# Comprehensive list from TimeML, PDTB 3.0, and CICLING.

ENGLISH_CONNECTIVES: dict[str, TemporalConnective] = {}

def _en(word, rel, type_, dep="advmod", multi=False):
    ENGLISH_CONNECTIVES[word] = TemporalConnective(word, rel, type_, dep, multi)

# Sequential (forward-looking: event1 happened, THEN event2)
_en("then",           "before",  "sequential")
_en("subsequently",   "before",  "sequential")
_en("afterwards",     "before",  "sequential")
_en("afterward",      "before",  "sequential")
_en("next",           "before",  "sequential")
_en("first",          "before",  "sequential")
_en("finally",        "before",  "sequential")
_en("later",          "before",  "sequential")
_en("soon",           "before",  "sequential")
_en("eventually",     "before",  "sequential")
_en("thereafter",     "before",  "sequential")
_en("henceforth",     "before",  "sequential")
_en("subsequent",     "before",  "sequential")
_en("ultimately",     "before",  "sequential")
_en("presently",      "before",  "sequential")

# Simultaneous (events at the same time — Allen's overlaps/during/equals)
_en("meanwhile",      "overlaps", "simultaneous")
_en("meantime",       "overlaps", "simultaneous")
_en("at the same time", "equals", "simultaneous", multi=True)
_en("in the meantime",  "overlaps", "simultaneous", multi=True)

# Precedence (backward-looking: event2 references event1 in the past)
_en("previously",     "after",   "precedence")
_en("earlier",        "after",   "precedence")
_en("beforehand",     "after",   "precedence")
_en("prior",          "after",   "precedence")
_en("already",        "after",   "precedence")

# Subordinating conjunctions (dep=mark or dep=prep)
_en("after",          "after",   "succession",   dep="mark")
_en("before",         "before",  "precedence",   dep="prep")
_en("since",          "after",   "succession",   dep="mark")
_en("once",           "before",  "conditional",  dep="mark")
_en("until",          "meets",   "simultaneous", dep="mark")
_en("till",           "meets",   "simultaneous", dep="mark")
_en("when",           "overlaps","simultaneous", dep="mark")
_en("whenever",       "overlaps","simultaneous", dep="mark")
_en("while",          "during",  "simultaneous", dep="mark")
_en("whilst",         "during",  "simultaneous", dep="mark")
_en("as",             "overlaps","simultaneous", dep="mark")

# Multi-word subordinating
_en("as soon as",     "meets",   "succession",   dep="mark", multi=True)
_en("as long as",     "during",  "simultaneous", dep="mark", multi=True)
_en("so long as",     "during",  "simultaneous", dep="mark", multi=True)
_en("by the time",    "before",  "precedence",   dep="mark", multi=True)
_en("now that",       "after",   "succession",   dep="mark", multi=True)
_en("every time",     "overlaps","simultaneous", dep="mark", multi=True)

# Prepositional (dep=prep)
_en("during",         "during",  "simultaneous", dep="prep")
_en("throughout",     "during",  "simultaneous", dep="prep")
_en("within",         "during",  "simultaneous", dep="prep")
_en("between",        "during",  "simultaneous", dep="prep")

# Conditional "then" markers (NOT temporal — these make "then" conditional)
ENGLISH_CONDITIONAL_MARKERS = frozenset({
    "if", "when", "whenever", "unless", "provided",
    "assuming", "suppose", "given that", "in case",
})

# Time nouns — "before noon" is time reference, not event ordering
ENGLISH_TIME_NOUNS = frozenset({
    "noon", "midnight", "dawn", "dusk", "sunrise", "sunset",
    "morning", "afternoon", "evening", "night",
    "today", "tomorrow", "yesterday",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "year", "month", "week", "day", "hour", "minute", "second",
    "years", "months", "weeks", "days", "hours", "minutes", "seconds",
    "noon", "midnight", "christmas", "easter",
})

# ─── Placeholder for other languages ────────────────────────────────
# Architecture: each language defines its own CONNECTIVES dict.
# The extractor selects the appropriate dict based on the spaCy model language.

LANGUAGE_REGISTRIES: dict[str, dict[str, TemporalConnective]] = {
    "en": ENGLISH_CONNECTIVES,
    # "zh": CHINESE_CONNECTIVES,   # TODO
    # "de": GERMAN_CONNECTIVES,    # TODO
    # "fr": FRENCH_CONNECTIVES,    # TODO
    # "es": SPANISH_CONNECTIVES,   # TODO
    # "ja": JAPANESE_CONNECTIVES,  # TODO
    # "ko": KOREAN_CONNECTIVES,    # TODO
    # "ru": RUSSIAN_CONNECTIVES,   # TODO
    # "ar": ARABIC_CONNECTIVES,    # TODO
}

LANGUAGE_CONDITIONAL_MARKERS: dict[str, frozenset] = {
    "en": ENGLISH_CONDITIONAL_MARKERS,
}

LANGUAGE_TIME_NOUNS: dict[str, frozenset] = {
    "en": ENGLISH_TIME_NOUNS,
}


def get_connectives(lang: str = "en") -> dict[str, TemporalConnective]:
    """Get temporal connectives for a language."""
    return LANGUAGE_REGISTRIES.get(lang, ENGLISH_CONNECTIVES)


def get_conditional_markers(lang: str = "en") -> frozenset:
    """Get conditional markers for a language."""
    return LANGUAGE_CONDITIONAL_MARKERS.get(lang, ENGLISH_CONDITIONAL_MARKERS)


def get_time_nouns(lang: str = "en") -> frozenset:
    """Get time nouns for a language."""
    return LANGUAGE_TIME_NOUNS.get(lang, ENGLISH_TIME_NOUNS)


def register_language(lang: str,
                      connectives: dict[str, TemporalConnective],
                      conditional_markers: frozenset = None,
                      time_nouns: frozenset = None):
    """Register temporal connectives for a new language.

    Usage:
        CHINESE_CONNECTIVES = {
            "然后": TemporalConnective("然后", "before", "sequential", "advmod"),
            "之后": TemporalConnective("之后", "before", "sequential", "advmod"),
            "同时": TemporalConnective("同时", "overlaps", "simultaneous", "advmod"),
        }
        register_language("zh", CHINESE_CONNECTIVES)
    """
    LANGUAGE_REGISTRIES[lang] = connectives
    if conditional_markers:
        LANGUAGE_CONDITIONAL_MARKERS[lang] = conditional_markers
    if time_nouns:
        LANGUAGE_TIME_NOUNS[lang] = time_nouns
