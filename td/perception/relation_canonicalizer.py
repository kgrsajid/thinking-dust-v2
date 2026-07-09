"""Relation canonicalization for triple deduplication.

Solves the duplicate triple problem caused by two extraction paths:
- Dependency extraction: verb + prep → "went_to", "invested_in"
- Clause segmenter: bare verb → "went", "invest"

Both produce valid triples for the same fact with different relation names.
Post-extraction canonicalization normalizes both to the same canonical form,
enabling deduplication while preserving the richer relation.

Approach: Option B from EDC framework (Zhang & Soh, 2024).
- Extract both paths
- Canonicalize relations by stripping preposition suffix and lemmatizing
- Deduplicate on (subject, canonical_relation, object)
- Keep the more specific original relation when duplicates found

References:
- Zhang & Soh (2024), "Extract, Define, Canonicalize" — arXiv:2404.03868
- UDASTE (ScienceDirect, 2023) — "restrictive triple relation types"
- KGGen (arXiv, Feb 2025) — "variations normalized"
- Stanford OpenIE (2015) — clause splitting pipeline
"""

from __future__ import annotations

# Preposition suffixes commonly attached to verbs in dependency extraction.
# "went_to" = went + to, "invested_in" = invested + in
PREPOSITION_SUFFIXES = frozenset({
    "_to", "_in", "_at", "_for", "_with", "_from", "_on", "_of",
    "_into", "_onto", "_about", "_through", "_over", "_under",
    "_up", "_down", "_out", "_off", "_away", "_back",
    "_between", "_among", "_against", "_toward", "_towards",
})

# Relations that look like verb+prep but are actually compound relations.
# "capital_of" is NOT "capital" + "of" — it's a compound.
# BUT: "invested_in" IS "invested" + "in" — verb+prep, canonicalize it.
COMPOUND_RELATIONS = frozenset({
    "capital_of", "part_of", "member_of", "type_of", "kind_of",
    "sort_of", "pair_of", "piece_of", "set_of", "group_of",
    "depends_on", "relied_on", "based_on", "focused_on",
    "born_in",
    "married_to", "compared_to", "related_to", "similar_to",
    "connected_to", "devoted_to", "committed_to", "dedicated_to",
    "made_by", "created_by", "founded_by", "owned_by",
})

# Cache for lemmatized verbs
_lemma_cache: dict[str, str] = {}

# Lazy-loaded spaCy model for lemmatization
_nlp_singleton = None


def _get_nlp():
    """Get or create a spaCy model for lemmatization."""
    global _nlp_singleton
    if _nlp_singleton is None:
        try:
            import spacy
            _nlp_singleton = spacy.load("en_core_web_sm")
        except (ImportError, OSError):
            _nlp_singleton = False
    return _nlp_singleton if _nlp_singleton is not False else None


def _lemmatize_verb(verb: str, nlp=None) -> str:
    """Lemmatize a verb string using spaCy.

    'went' → 'go', 'invested' → 'invest', 'running' → 'run'
    Uses provided nlp model, falls back to global singleton.
    """
    if verb in _lemma_cache:
        return _lemma_cache[verb]

    # Try provided model first, then global singleton
    model = nlp or _get_nlp()
    if model is not None:
        doc = model(verb)
        if doc:
            result = doc[0].lemma_
            _lemma_cache[verb] = result
            return result

    # No spaCy available — return as-is
    _lemma_cache[verb] = verb
    return verb


def canonicalize_relation(relation: str, nlp=None) -> str:
    """Canonicalize a relation by stripping preposition suffix and lemmatizing.

    This is the core deduplication function. Two relations that produce the
    same canonical form are considered duplicates.

    Examples:
        "went_to"       → "go"     (strip "_to", lemmatize "went")
        "invested_in"   → "invest" (strip "_in", lemmatize "invested")
        "went"          → "go"     (no suffix, lemmatize "went")
        "capital_of"    → "capital_of" (compound relation, keep as-is)
        "in"            → "in"     (not a verb+prep, keep as-is)
        "depends_on"    → "depends_on" (compound relation, keep as-is)
    """
    rel = relation.lower().strip()

    # Compound relations are kept as-is
    if rel in COMPOUND_RELATIONS:
        return rel

    # Check for preposition suffix
    for suffix in PREPOSITION_SUFFIXES:
        if rel.endswith(suffix):
            verb_part = rel[:-len(suffix)]
            if len(verb_part) >= 2 and verb_part not in COMPOUND_RELATIONS:
                return _lemmatize_verb(verb_part, nlp)

    # No preposition suffix — just lemmatize
    return _lemmatize_verb(rel, nlp)


def relation_specificity(relation: str) -> int:
    """Score relation specificity for deduplication preference.

    Higher = more specific/informative. When two relations canonicalize
    to the same form, the more specific one is kept.
    """
    rel = relation.lower().strip()
    if rel in COMPOUND_RELATIONS:
        return 3  # Compound relations are most specific
    if "_" in rel:
        return 2  # Verb+prep is more specific
    return 1  # Bare verb is least specific


# Articles stripped from entity strings during dedup key computation.
# Aligns clause segmenter output (which now strips determiners) with
# main parser output (which always strips them).
_ARTICLES = frozenset({"the", "a", "an"})


def _normalize_entity(text: str) -> str:
    """Strip leading articles from entity text for dedup key."""
    words = text.lower().split()
    while words and words[0] in _ARTICLES:
        words = words[1:]
    return " ".join(words)


def deduplicate_triples(triples: list[tuple[str, str, str]],
                        nlp=None) -> list[tuple[str, str, str]]:
    """Deduplicate triples using relation canonicalization + entity normalization.

    When two triples have the same (subject, canonical_relation, object),
    keeps the one with the more specific original relation.
    Entity strings are normalized (leading articles stripped) before
    comparison to catch near-duplicates from the dual extraction paths.
    """
    seen: dict[tuple[str, str, str], tuple[tuple[str, str, str], int]] = {}

    for triple in triples:
        s, r, o = triple
        canonical = canonicalize_relation(r, nlp)
        key = (_normalize_entity(s), canonical, _normalize_entity(o))
        spec = relation_specificity(r)

        if key not in seen:
            seen[key] = (triple, spec)
        else:
            existing_triple, existing_spec = seen[key]
            if spec > existing_spec:
                seen[key] = (triple, spec)

    return [triple for triple, _ in seen.values()]
