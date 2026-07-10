"""English language configuration for Thinking Dust.

Contains all English-specific word sets used by the parser:
- Stop words (fallback when spaCy unavailable)
- Prepositions (fallback when spaCy unavailable)
- Pronouns (for coreference resolution)
- Discourse deixis verbs (Jauhar et al. 2015)

To extend to a new language:
1. Copy this file → td/languages/xx.py
2. Replace all English words with target language equivalents
3. Register in td/languages/__init__.py
4. See DEVELOPMENT.md "Adding a New Language" for detailed guide

Reference: Universal Dependencies (Nivre et al., 2016)
Reference: Jauhar et al. (2015), *SEM, pp. 299-308
"""

from . import LanguageConfig, register_language

# ── English Stop Words ────────────────────────────────────────────
# Used ONLY when spaCy is not available (fallback path).
# When spaCy is available, use token.is_stop (language-agnostic).
#
# Source: spaCy English stop word list (en_core_web_sm)
# Reference: Honnibal & Montani (2017), spaCy 2
STOP_WORDS = frozenset({
    "the", "a", "an", "and", "or", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "can", "shall", "it", "its", "this",
    "that", "these", "those", "there", "their", "they", "them", "of", "to",
    "in", "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "above", "below", "between", "under",
    "again", "further", "then", "once", "here", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "but", "because", "until", "while", "about", "against", "down",
    "out", "off", "over",
})

# ── English Prepositions ──────────────────────────────────────────
# Used ONLY in regex fallback path (when spaCy unavailable).
# When spaCy is available, use token.pos_ == "ADP" (language-agnostic).
#
# Source: Universal POS Tags — ADP category
# Reference: Nivre et al. (2016), Universal Dependencies 2.0
PREPOSITIONS = frozenset({
    "on", "in", "at", "from", "to", "by", "for", "with", "into",
    "through", "during", "before", "after", "about", "near", "over",
    "above", "below", "between", "under", "against", "of",
})

# ── English Pronouns ──────────────────────────────────────────────
# Used for coreference resolution.
# When spaCy is available, use token.pos_ == "PRON" + morph features.
#
# Source: Universal POS Tags — PRON category
# Reference: Nivre et al. (2016), Universal Dependencies 2.0
POSSESSIVE_PRONOUNS = frozenset({
    "its", "his", "her", "their", "theirs", "our", "your", "my", "mine",
})

ENTITY_PRONOUNS = frozenset({
    "he", "she", "it", "they", "him", "her", "them", "his", "its",
    "their", "theirs",
})

# ── English Discourse Deixis Verbs ────────────────────────────────
# Abstract/cognitive verbs for discourse deixis detection.
# Used with Jauhar et al. (2015) two-stage approach:
#   Stage 1: Is pronoun "this"/"that"/"it" the subject (nsubj) of
#            a verb in this set? → discourse deixis → skip
#
# The syntactic check (dep=nsubj) is language-agnostic (Universal
# Dependencies). This verb set is English-specific.
#
# Reference: Jauhar et al. (2015), *SEM, pp. 299-308
# Reference: Webber (1988), "Discourse Deixis." ACL.
DISCOURSE_DEIXIS_VERBS = frozenset({
    # Verbs of demonstration/implication
    "show", "prove", "mean", "suggest", "indicate", "demonstrate",
    "reveal", "confirm", "imply", "illustrate", "reflect",
    # Verbs of causation/result
    "result", "lead", "cause", "enable", "allow", "prevent",
    "require", "involve", "affect",
    # Verbs of emotional response
    "surprise", "shock", "please", "anger", "upset", "annoy",
})

# Subset: verbs where "it" is likely discourse deixis.
# "It shows X" → discourse deixis. "It surprises me" → NOT deixis.
# Only "this"/"that" trigger the full set; "it" only triggers this subset.
DISCOURSE_DEIXIS_IT_VERBS = frozenset({
    "show", "prove", "mean", "suggest", "indicate", "demonstrate",
    "reveal", "confirm", "imply", "illustrate", "reflect",
})

# ── English Relation Prototypes (HDC-encoded) ─────────────────────
# HDC-encoded phrases for constraint type detection.
# Used for fuzzy similarity matching in the HDC layer.
# Actual relation detection uses spaCy dependency labels.
#
# For a new language: translate the phrases to the target language.
# The HDC encoding is language-specific — different words = different vectors.
#
# Reference: Kanerva (2009), "Hyperdimensional Computing"
RELATION_PROTOTYPES = {
    "different": "different distinct separate unique",
    "before": "before earlier precedes first",
    "after": "after later follows second",
    "excludes": "excludes cannot together forbidden",
    "limited": "limited bounded maximum minimum",
    "grouped": "grouped together category partition",
    "sum_to": "sum total adds equals",
    "implies": "implies if then requires means",
    "overlap": "overlap conflict cannot same time",
    "precedence": "precedence must before chain ordered",
    "ratio": "ratio proportional divided times",
    "count": "count exactly at least how many",
    "equivalent": "equivalent same equal identical",
    "optimize": "optimize maximize minimize best",
}

# ── English Copula Verbs ──────────────────────────────────────────
# Used by _merge_* pattern methods for string matching.
# When spaCy is available, use token.dep_ == "cop" (language-agnostic).
COPULA_VERBS = frozenset({
    "is", "are", "was", "were", "be", "been", "being",
    "become", "became", "seem", "seemed",
})

# ── English Articles ──────────────────────────────────────────────
# Used by _get_chunk_text for entity name cleaning.
# When spaCy is available, use token.pos_ == "DET" (language-agnostic).
ARTICLES = frozenset({"the", "a", "an"})

# ── English Genitive Markers ──────────────────────────────────────
# "of" is entity-internal (genitive), not a relation.
# "capital of France" → entity: "capital of France"
# When spaCy is available, use token.dep_ == "case" (language-agnostic).
GENITIVE_MARKERS = frozenset({"of"})

# ── English Demonstrative Pronouns ────────────────────────────────
# Used for discourse deixis detection.
# "this"/"that"/"it" as subject of abstract verb → discourse deixis.
# Reference: Jauhar et al. (2015), *SEM — two-stage approach
DEMONSTRATIVE_PRONOUNS = frozenset({"this", "that", "it"})

# ── English Copula → is_a Mapping ────────────────────────────────
# Clause segmenter produces (X, "is", Y), dep parser produces (X, "is_a", Y).
# Both represent the same copular relationship. Canonicalize to "is_a".
# Reference: UD `attr` — nominal predicate of copular construction
COPULA_TO_ISA = {
    "is": "is_a",
    "are": "is_a",
    "was": "is_a",
    "were": "is_a",
}

# ── English BEAGLE Stop Words ─────────────────────────────────────
# Superset of parser stop words for word vector training.
# Removes function words, pronouns, interrogatives, and auxiliaries
# before co-occurrence counting. Only content words accumulate context.
#
# Used by td/perception/word_vectors.py — NEVER hardcoded in core logic.
# When adding a new language, define an equivalent set in td/languages/xx.py.
#
# Reference: Jones & Mewhort (2007), Psychological Review 114(1): 1-37.
#   "High-frequency function words are removed before counting."
BEAGLE_STOP_WORDS = STOP_WORDS | frozenset({
    # Pronouns (personal + possessive)
    "i", "you", "he", "she", "we", "me", "him", "her", "us",
    "my", "your", "our", "mine", "yours", "hers", "ours",
    # Interrogatives
    "what", "which", "who", "whom", "whose", "where", "when", "how", "why",
    # Demonstratives (already in STOP_WORDS but explicit)
    "this", "that", "these", "those",
    # Relative pronouns
    "who", "whom", "whose", "which", "that",
    # Common auxiliaries (already in STOP_WORDS but explicit)
    "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "must", "can", "shall",
    # Determiners
    "the", "a", "an", "some", "any", "no", "every", "each",
    # Conjunctions
    "and", "or", "but", "nor", "yet", "so",
    # Prepositions (already in STOP_WORDS but explicit)
    "of", "to", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "about",
    "against", "between", "under", "over", "above", "below",
    # Other function words
    "if", "because", "until", "while", "then", "once", "here", "there",
    "not", "very", "just", "also", "still", "already", "even",
})

# ── Register English ──────────────────────────────────────────────
register_language(LanguageConfig(
    code="en",
    name="English",
    stop_words=STOP_WORDS,
    prepositions=PREPOSITIONS,
    possessive_pronouns=POSSESSIVE_PRONOUNS,
    entity_pronouns=ENTITY_PRONOUNS,
    discourse_deixis_verbs=DISCOURSE_DEIXIS_VERBS,
    discourse_deixis_it_verbs=DISCOURSE_DEIXIS_IT_VERBS,
    relation_prototypes=RELATION_PROTOTYPES,
    copula_verbs=COPULA_VERBS,
    articles=ARTICLES,
    genitive_markers=GENITIVE_MARKERS,
    demonstrative_pronouns=DEMONSTRATIVE_PRONOUNS,
    beagle_stop_words=BEAGLE_STOP_WORDS,
    copula_to_isa=COPULA_TO_ISA,
))
