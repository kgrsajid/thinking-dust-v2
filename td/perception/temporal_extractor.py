"""Temporal ordering extraction from discourse connectives.

Detects temporal connectives ("then", "after", "before", "subsequently",
"while", "meanwhile", "as soon as", etc.) in text and extracts temporal
ordering triples between events using Allen's interval algebra.

Architecture:
- Temporal connectives defined in temporal_connectives.py (multilingual registry)
- Extractor uses spaCy dependency parsing to detect connective patterns
- Three extraction patterns: coordinated verbs, subordinating, prepositional
- Allen's interval algebra for temporal relations (before, after, overlaps, etc.)

References:
- Allen, J.F. (1983). "Maintaining Knowledge about Temporal Intervals."
- TimeML (Pustejovsky et al., 2003) — temporal annotation standard
- PDTB 3.0 (Webber et al., 2019) — Penn Discourse Treebank
- Chambers et al. — unsupervised temporal ordering extraction
- Consistent Discourse-level TRE (EMNLP 2025)
- ATOMIC-2020 — common sense knowledge with isBefore/isAfter relations
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .temporal_connectives import (
    get_connectives,
    get_conditional_markers,
    get_time_nouns,
    TemporalConnective,
    AllenRelation,
    SemanticType,
)


@dataclass
class TemporalOrdering:
    """A temporal ordering between two events."""
    event1_description: str  # e.g., "alice went to paris"
    event2_description: str  # e.g., "alice invested in stocks"
    relation: str            # Allen's relation: "before", "after", "overlaps", "during"
    connective: str          # The connective word: "then", "after", etc.
    semantic_type: str       # "sequential", "precedence", "simultaneous", etc.
    confidence: float = 0.8


def extract_temporal_orderings(doc, lang: str = "en") -> list[TemporalOrdering]:
    """Extract temporal orderings from a spaCy Doc.

    Detects temporal connectives between coordinated clauses and
    returns temporal ordering triples.

    Handles three patterns:
    1. Coordinated verbs with "then": "X and then Y" → X BEFORE Y
    2. Subordinating "after/before": "After X, Y" → X BEFORE Y
    3. Prepositional "before/after": "X before Y" → X BEFORE Y

    Args:
        doc: spaCy Doc object
        lang: Language code (default "en")

    Returns:
        List of TemporalOrdering objects
    """
    connectives = get_connectives(lang)
    conditional_markers = get_conditional_markers(lang)
    time_nouns = get_time_nouns(lang)
    orderings = []

    for sent in doc.sents:
        tokens = list(sent)

        for i, token in enumerate(tokens):
            word = token.text.lower()
            if word not in connectives:
                continue

            conn = connectives[word]

            # Skip conditional context (e.g., "then" in "If X then Y")
            if conn.is_conditional_context:
                tokens_before = tokens[:i]
                is_conditional = any(t.text.lower() in conditional_markers for t in tokens_before)
                if is_conditional:
                    conditional_token = next(
                        (t for t in tokens_before if t.text.lower() in conditional_markers), None
                    )
                    if conditional_token and conditional_token.head == token.head:
                        continue

            # Pattern 1: Sequential connectives as advmod of verb
            # "Alice went to Paris and then invested" → went BEFORE invested
            if conn.semantic_type == SemanticType.SEQUENTIAL and conn.dep_pattern == "advmod":
                if token.dep_ == "advmod" and token.head.pos_ in ("VERB", "AUX"):
                    head_verb = token.head
                    other_verb = _find_conjunct_verb(head_verb)
                    if other_verb is None:
                        other_verb = _find_verb_in_previous_sentence(sent, doc)
                    if other_verb:
                        event1 = _describe_verb_event(other_verb, doc)
                        event2 = _describe_verb_event(head_verb, doc)
                        if event1 and event2:
                            orderings.append(TemporalOrdering(
                                event1_description=event1,
                                event2_description=event2,
                                relation=conn.allen_relation,
                                connective=word,
                                semantic_type=conn.semantic_type,
                            ))

            # Pattern 2: Subordinating conjunctions (mark)
            # "After Alice went to Paris, she invested" → went BEFORE invested
            elif conn.dep_pattern == "mark" and token.dep_ == "mark":
                subordinate_verb = token.head
                main_verb = _find_main_clause_verb(sent, subordinate_verb)
                if main_verb:
                    event_sub = _describe_verb_event(subordinate_verb, doc)
                    event_main = _describe_verb_event(main_verb, doc)
                    if event_sub and event_main:
                        # Determine event ordering based on connective semantics
                        if conn.allen_relation == AllenRelation.AFTER:
                            # "After X, Y" → X BEFORE Y
                            orderings.append(TemporalOrdering(
                                event1_description=event_sub,
                                event2_description=event_main,
                                relation="before",
                                connective=word,
                                semantic_type=conn.semantic_type,
                            ))
                        elif conn.allen_relation == AllenRelation.BEFORE:
                            # "Before X, Y" → Y BEFORE X
                            orderings.append(TemporalOrdering(
                                event1_description=event_main,
                                event2_description=event_sub,
                                relation="before",
                                connective=word,
                                semantic_type=conn.semantic_type,
                            ))
                        elif conn.allen_relation in (AllenRelation.OVERLAPS, AllenRelation.DURING, AllenRelation.EQUALS):
                            # "While X, Y" → X OVERLAPS Y
                            orderings.append(TemporalOrdering(
                                event1_description=event_sub,
                                event2_description=event_main,
                                relation=conn.allen_relation,
                                connective=word,
                                semantic_type=conn.semantic_type,
                            ))
                        elif conn.allen_relation == AllenRelation.MEETS:
                            # "Until X, Y" → Y MEETS X
                            orderings.append(TemporalOrdering(
                                event1_description=event_main,
                                event2_description=event_sub,
                                relation="meets",
                                connective=word,
                                semantic_type=conn.semantic_type,
                            ))
                        else:
                            # Default: use the connective's Allen relation
                            orderings.append(TemporalOrdering(
                                event1_description=event_sub,
                                event2_description=event_main,
                                relation=conn.allen_relation,
                                connective=word,
                                semantic_type=conn.semantic_type,
                            ))

            # Pattern 3: Prepositional connectives
            # "Alice went to Paris before investing" → went BEFORE investing
            # Also handles subordinating connectives that spaCy parses as prep
            elif (conn.dep_pattern == "prep" or conn.dep_pattern == "mark") and token.dep_ == "prep":
                main_verb = token.head
                if main_verb.pos_ in ("VERB", "AUX"):
                    pobj = None
                    for child in token.children:
                        if child.dep_ in ("pobj", "pcomp"):
                            pobj = child
                            break
                    if pobj:
                        # Skip time expressions
                        if pobj.text.lower() in time_nouns:
                            continue
                        if pobj.pos_ in ("NUM",) or pobj.like_num:
                            continue

                        event_main = _describe_verb_event(main_verb, doc)
                        event_sub = _get_subtree_text(pobj)
                        if event_main and event_sub:
                            if conn.allen_relation == AllenRelation.BEFORE:
                                orderings.append(TemporalOrdering(
                                    event1_description=event_main,
                                    event2_description=event_sub,
                                    relation="before",
                                    connective=word,
                                    semantic_type=conn.semantic_type,
                                ))
                            elif conn.allen_relation in (AllenRelation.DURING, AllenRelation.OVERLAPS):
                                orderings.append(TemporalOrdering(
                                    event1_description=event_main,
                                    event2_description=event_sub,
                                    relation=conn.allen_relation,
                                    connective=word,
                                    semantic_type=conn.semantic_type,
                                ))
                            else:
                                orderings.append(TemporalOrdering(
                                    event1_description=event_sub,
                                    event2_description=event_main,
                                    relation="before",
                                    connective=word,
                                    semantic_type=conn.semantic_type,
                                ))

    return orderings


def _find_conjunct_verb(verb) -> Optional[object]:
    """Find the verb that is coordinated with this verb."""
    if verb.dep_ == "conj":
        head = verb.head
        if head.pos_ in ("VERB", "AUX"):
            return head
    for child in verb.children:
        if child.dep_ == "conj" and child.pos_ in ("VERB", "AUX"):
            return child
    return None


def _find_main_clause_verb(sent, subordinate_verb) -> Optional[object]:
    """Find the main clause verb when a subordinate clause exists."""
    for token in sent:
        if token.dep_ == "ROOT" and token.pos_ in ("VERB", "AUX"):
            if token != subordinate_verb:
                return token
    return None


def _find_verb_in_previous_sentence(current_sent, doc) -> Optional[object]:
    """Find the main verb in the sentence immediately before the current one."""
    sents = list(doc.sents)
    for i, sent in enumerate(sents):
        if sent == current_sent and i > 0:
            prev_sent = sents[i - 1]
            for token in prev_sent:
                if token.dep_ == "ROOT" and token.pos_ in ("VERB", "AUX"):
                    return token
    return None


def _describe_verb_event(verb, doc) -> Optional[str]:
    """Create a description of the event around a verb."""
    parts = []
    for child in verb.children:
        if child.dep_ in ("nsubj", "nsubjpass"):
            subj_text = _get_subtree_text(child)
            parts.append(subj_text)
            break

    parts.append(verb.text.lower())

    for child in verb.children:
        if child.dep_ in ("dobj", "attr", "prep", "advmod", "prt"):
            if child.dep_ == "advmod" and child.text.lower() in get_connectives("en"):
                continue
            obj_text = _get_subtree_text(child)
            if obj_text:
                parts.append(obj_text)

    return " ".join(parts) if parts else verb.text.lower()


def _get_subtree_text(token) -> str:
    """Get the text of a token and its subtree, excluding punctuation."""
    words = []
    for t in token.subtree:
        if not t.is_punct:
            words.append(t.text.lower())
    return " ".join(words)


def temporal_triples_from_text(text: str, nlp=None, lang: str = "en") -> list[tuple[str, str, str]]:
    """Extract temporal ordering triples from text."""
    if nlp is None:
        import spacy
        nlp = spacy.load("en_core_web_sm")

    doc = nlp(text)
    orderings = extract_temporal_orderings(doc, lang)

    triples = []
    for ordering in orderings:
        triples.append((
            ordering.event1_description,
            ordering.relation,
            ordering.event2_description,
        ))

    return triples
