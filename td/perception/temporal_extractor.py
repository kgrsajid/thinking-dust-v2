"""Temporal ordering extraction from discourse connectives.

Detects temporal connectives ("then", "after", "before", "subsequently")
in text and extracts temporal ordering triples between events.

When "Alice went to Paris and then invested in stocks" is parsed:
- Event 1: Alice went to Paris
- Event 2: Alice invested in stocks
- Temporal: Event 1 BEFORE Event 2

This uses Allen's interval algebra (already implemented in TD v2) to
store temporal relations as triples with the "before" relation.

References:
- Allen, J.F. (1983). "Maintaining Knowledge about Temporal Intervals."
- TimeML (Pustejovsky et al., 2003) — temporal annotation standard
- Chambers et al. — unsupervised temporal ordering extraction
- Consistent Discourse-level TRE (EMNLP 2025)
- ATOMIC-2020 — isBefore/isAfter relations
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# Temporal connectives and their Allen's relation mappings.
# "then" → BEFORE (event before the "then" clause happened first)
# "after" → AFTER (event after "after" happened later)
# "before" → BEFORE (event after "before" happened earlier)
# "subsequently" → BEFORE
# "first...then" → BEFORE
TEMPORAL_CONNECTIVES = {
    # connective: (Allen relation, which event is first)
    # "before" means: the clause WITHOUT the connective happened BEFORE the clause WITH it
    "then": "before",
    "subsequently": "before",
    "afterwards": "before",
    "after": "after",
    "before": "before",
    "next": "before",
    "first": "before",
    "finally": "before",
    "subsequent": "before",
    "previously": "after",
    "earlier": "after",
    "later": "before",
}

# Conditional "then" — NOT temporal. Distinguished by context.
# "If X then Y" — "then" is conditional, not temporal.
CONDITIONAL_MARKERS = {"if", "when", "whenever", "unless", "provided"}


@dataclass
class TemporalOrdering:
    """A temporal ordering between two events."""
    event1_description: str  # e.g., "alice went to paris"
    event2_description: str  # e.g., "alice invested in stocks"
    relation: str            # Allen's relation: "before", "after"
    connective: str          # The connective word: "then", "after", etc.
    confidence: float = 0.8


def extract_temporal_orderings(doc) -> list[TemporalOrdering]:
    """Extract temporal orderings from a spaCy Doc.

    Detects temporal connectives between coordinated clauses and
    returns temporal ordering triples.

    Handles three patterns:
    1. Coordinated verbs with "then": "X and then Y" → X BEFORE Y
    2. Subordinating "after/before": "After X, Y" → X BEFORE Y
    3. Prepositional "before/after": "X before Y" → X BEFORE Y

    Args:
        doc: spaCy Doc object

    Returns:
        List of TemporalOrdering objects
    """
    orderings = []

    for sent in doc.sents:
        tokens = list(sent)

        for i, token in enumerate(tokens):
            word = token.text.lower()
            if word not in TEMPORAL_CONNECTIVES:
                continue

            # Skip conditional "then"
            if word == "then":
                is_conditional = any(t.text.lower() in CONDITIONAL_MARKERS for t in tokens)
                if is_conditional:
                    continue

            # Pattern 1: "then" as advmod of a verb (coordinated verbs)
            # "Alice went to Paris and then invested" → went BEFORE invested
            if word in ("then", "subsequently", "afterwards", "next", "first", "finally", "later"):
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
                                relation="before",
                                connective=word,
                            ))

            # Pattern 2: "after/before" as subordinating conjunction (mark)
            # "After Alice went to Paris, she invested" → went BEFORE invested
            elif word in ("after", "before") and token.dep_ == "mark":
                # token.head is the verb of the subordinate clause
                subordinate_verb = token.head
                # Find the main clause verb (the ROOT or the verb in the main clause)
                main_verb = _find_main_clause_verb(sent, subordinate_verb)
                if main_verb:
                    event_sub = _describe_verb_event(subordinate_verb, doc)
                    event_main = _describe_verb_event(main_verb, doc)
                    if event_sub and event_main:
                        if word == "after":
                            # "After X, Y" → X happened before Y
                            orderings.append(TemporalOrdering(
                                event1_description=event_sub,
                                event2_description=event_main,
                                relation="before",
                                connective=word,
                            ))
                        else:  # before
                            # "Before X, Y" → Y happened before X
                            orderings.append(TemporalOrdering(
                                event1_description=event_main,
                                event2_description=event_sub,
                                relation="before",
                                connective=word,
                            ))

            # Pattern 3: "before/after" as preposition (prep)
            # "Alice went to Paris before investing" → went BEFORE investing
            elif word in ("before", "after") and token.dep_ == "prep":
                main_verb = token.head
                if main_verb.pos_ in ("VERB", "AUX"):
                    # Find the object of the preposition
                    pobj = None
                    for child in token.children:
                        if child.dep_ in ("pobj", "pcomp"):
                            pobj = child
                            break
                    if pobj:
                        event_main = _describe_verb_event(main_verb, doc)
                        event_sub = _get_subtree_text(pobj)
                        if event_main and event_sub:
                            if word == "before":
                                orderings.append(TemporalOrdering(
                                    event1_description=event_main,
                                    event2_description=event_sub,
                                    relation="before",
                                    connective=word,
                                ))
                            else:  # after
                                orderings.append(TemporalOrdering(
                                    event1_description=event_sub,
                                    event2_description=event_main,
                                    relation="before",
                                    connective=word,
                                ))

    return orderings


def _find_parent_verb(token) -> Optional[object]:
    """Walk up the dependency tree to find the parent verb."""
    current = token.head
    for _ in range(5):  # max 5 hops
        if current.pos_ in ("VERB", "AUX"):
            return current
        if current == current.head:
            break
        current = current.head
    return None


def _find_conjunct_verb(verb) -> Optional[object]:
    """Find the verb that is coordinated with this verb via 'and'/'but'/'or'.

    'Alice went to Paris and then invested' →
    verb=invested, conjunct=went
    """
    # Check if this verb is a conjunct of another verb
    if verb.dep_ == "conj":
        head = verb.head
        if head.pos_ in ("VERB", "AUX"):
            return head

    # Check if another verb is a conjunct of this verb
    for child in verb.children:
        if child.dep_ == "conj" and child.pos_ in ("VERB", "AUX"):
            return child

    return None


def _find_main_clause_verb(sent, subordinate_verb) -> Optional[object]:
    """Find the main clause verb when a subordinate clause exists.

    'After Alice went to Paris, she invested in stocks'
    subordinate_verb = went, main_verb = invested
    """
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
    """Create a description of the event around a verb.

    'went' with subject 'Alice' and prep 'to Paris' → 'alice went to paris'
    """
    parts = []

    # Find subject
    for child in verb.children:
        if child.dep_ in ("nsubj", "nsubjpass"):
            subj_text = _get_subtree_text(child)
            parts.append(subj_text)
            break

    # Add the verb
    parts.append(verb.text.lower())

    # Add objects and prep phrases
    for child in verb.children:
        if child.dep_ in ("dobj", "attr", "prep", "advmod", "prt"):
            if child.dep_ == "advmod" and child.text.lower() in TEMPORAL_CONNECTIVES:
                continue  # Skip temporal connectives
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


def temporal_triples_from_text(text: str, nlp=None) -> list[tuple[str, str, str]]:
    """Extract temporal ordering triples from text.

    Convenience function that processes text and returns triples
    in the format (event1, "before"/"after", event2).

    Args:
        text: Input text
        nlp: spaCy model (loaded lazily if None)

    Returns:
        List of (event1, relation, event2) tuples
    """
    if nlp is None:
        import spacy
        nlp = spacy.load("en_core_web_sm")

    doc = nlp(text)
    orderings = extract_temporal_orderings(doc)

    triples = []
    for ordering in orderings:
        triples.append((
            ordering.event1_description,
            ordering.relation,
            ordering.event2_description,
        ))

    return triples
