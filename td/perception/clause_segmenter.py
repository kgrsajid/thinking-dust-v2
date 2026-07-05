"""Clause segmentation via spaCy dependency tree.

Splits compound/complex sentences into simple clauses, each producing
at most one SVO triple. Uses verb-based sentence splitting following
the approach from Sahaj Software (2023).

Algorithm:
1. Find all verbs in the sentence (ROOT, conj, relcl, xcomp, ccomp)
2. For each verb, find its subject(s) and object(s)
3. Walk conj chains to expand coordinated subjects/objects
4. Generate one simple clause per (subject, verb, object) combination

References:
- Sahaj Software (2023): "Knowledge graphs from complex text"
  https://www.sahaj.ai/knowledge-graphs-from-complex-text/
- Manning & Schütze (1999): "Foundations of Statistical NLP", Ch. 5
- spaCy dependency labels: https://universaldependencies.org/u/dep/
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SimpleClause:
    """A simple SVO clause extracted from a complex sentence."""
    subject: str
    relation: str
    obj: str
    source_text: str  # original clause text
    confidence: float = 1.0

    def __repr__(self):
        return f"({self.subject}, {self.relation}, {self.obj})"


def _get_conj_chain(token) -> list:
    """Walk conj chain to find all coordinated elements.

    'music, a story and visuals' → [music, story, visuals]
    """
    chain = [token]
    for child in token.children:
        if child.dep_ == "conj":
            chain.extend(_get_conj_chain(child))
    return chain


def _get_subjects(verb) -> list:
    """Get all subjects of a verb, including coordinated subjects.

    'Alice and Bob went' → [Alice, Bob]
    """
    subjects = []
    for child in verb.children:
        if child.dep_ in ("nsubj", "nsubjpass"):
            subjects.extend(_get_conj_chain(child))
    return subjects


def _get_objects(verb) -> list:
    """Get all objects of a verb, including coordinated objects.

    'have music, a story and visuals' → [music, story, visuals]
    """
    objects = []
    for child in verb.children:
        if child.dep_ in ("dobj", "attr", "pobj", "oprd"):
            objects.extend(_get_conj_chain(child))
        # Handle prep chains: 'aggregate into whole' → object is 'whole'
        elif child.dep_ == "prep":
            for grandchild in child.children:
                if grandchild.dep_ == "pobj":
                    objects.extend(_get_conj_chain(grandchild))
    return objects


def _get_verb_text(verb) -> str:
    """Get the full verb phrase including auxiliaries.

    'may have' → 'may have'
    'is running' → 'is running'
    """
    parts = []
    for child in verb.children:
        if child.dep_ in ("aux", "auxpass", "neg"):
            parts.append(child.text)
    parts.append(verb.text)
    return " ".join(parts)


def _get_object_text(obj) -> str:
    """Get the full noun phrase for an object.

    'a story' → 'a story'
    'functional whole' → 'functional whole'
    """
    parts = []
    for child in obj.children:
        if child.dep_ in ("det", "amod", "compound", "nummod", "poss"):
            parts.append(child.text)
    parts.append(obj.text)
    return " ".join(parts)


def _get_subject_text(subj) -> str:
    """Get the full noun phrase for a subject."""
    parts = []
    for child in subj.children:
        if child.dep_ in ("det", "amod", "compound", "nummod", "poss"):
            parts.append(child.text)
    parts.append(subj.text)
    return " ".join(parts)


def segment_clauses(doc) -> list[SimpleClause]:
    """Segment a spaCy Doc into simple SVO clauses.

    Args:
        doc: spaCy Doc object

    Returns:
        List of SimpleClause objects, each with subject, relation, object.
    """
    clauses = []

    for sent in doc.sents:
        # Find all verbs in the sentence
        verbs = _find_verbs(sent)

        for verb_info in verbs:
            verb = verb_info["verb"]
            subjects = _get_subjects(verb)
            objects = _get_objects(verb)

            # If no subjects found, try to inherit from parent verb
            if not subjects and verb_info.get("parent_verb"):
                subjects = _get_subjects(verb_info["parent_verb"])

            # If still no subjects, skip
            if not subjects:
                continue

            # Generate one clause per (subject, object) pair
            verb_text = _get_verb_text(verb)
            for subj in subjects:
                subj_text = _get_subject_text(subj)

                if objects:
                    for obj in objects:
                        obj_text = _get_object_text(obj)
                        clauses.append(SimpleClause(
                            subject=subj_text.lower().strip(),
                            relation=verb_text.lower().strip(),
                            obj=obj_text.lower().strip(),
                            source_text=f"{subj_text} {verb.text} {obj_text}",
                        ))
                else:
                    # Verb with no object (intransitive)
                    clauses.append(SimpleClause(
                        subject=subj_text.lower().strip(),
                        relation=verb_text.lower().strip(),
                        obj="",
                        source_text=f"{subj_text} {verb.text}",
                    ))

    return clauses


def _find_verbs(sent) -> list[dict]:
    """Find all verbs in a sentence with their hierarchy.

    Returns list of dicts with:
    - verb: the verb token
    - parent_verb: the parent verb (for inherited subjects)
    - dep: the dependency relation
    """
    verbs = []

    for token in sent:
        if token.pos_ in ("VERB", "AUX") and token.dep_ in ("ROOT", "conj", "relcl", "advcl", "xcomp", "ccomp"):
            parent = None
            if token.dep_ in ("conj", "relcl", "advcl", "xcomp", "ccomp"):
                # Walk up to find the parent verb
                head = token.head
                while head.pos_ not in ("VERB", "AUX") and head != head.head:
                    head = head.head
                if head.pos_ in ("VERB", "AUX"):
                    parent = head

            verbs.append({
                "verb": token,
                "parent_verb": parent,
                "dep": token.dep_,
            })

    return verbs


def segment_text(text: str, nlp=None) -> list[SimpleClause]:
    """Convenience function: segment text into simple clauses.

    Args:
        text: Input text (one or more sentences)
        nlp: spaCy language model (loaded lazily if None)

    Returns:
        List of SimpleClause objects.
    """
    if nlp is None:
        import spacy
        nlp = spacy.load("en_core_web_sm")

    doc = nlp(text)
    return segment_clauses(doc)
