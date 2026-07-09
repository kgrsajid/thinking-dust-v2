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

    For relative clauses ('which is the capital'), resolves the
    relative pronoun to its antecedent (the noun the clause modifies).

    Reference: Universal Dependencies — acl:relcl dependency
    The relativizer can be understood as an anaphor whose antecedent
    is the head of the relative clause.
    """
    subjects = []
    for child in verb.children:
        if child.dep_ in ("nsubj", "nsubjpass"):
            # Check for relative pronoun (which, who, that)
            if child.tag_ in ("WDT", "WP") and verb.dep_ == "relcl":
                # Resolve: the antecedent is the noun the relcl modifies
                antecedent = verb.head
                subjects.extend(_get_conj_chain(antecedent))
            else:
                subjects.extend(_get_conj_chain(child))
    return subjects


def _get_objects(verb) -> list[tuple]:
    """Get all objects of a verb with their preposition context.

    Returns list of (token, prep_label_or_None) tuples.
    'divide through mitosis and meiosis'
      → [(mitosis, 'through'), (meiosis, 'through')]
    'have music, a story and visuals'
      → [(music, None), (story, None), (visuals, None)]

    Reference: Bohnet et al. (2020) — coordination expansion must
    preserve the dependency path from ROOT to the head of the coordination.
    """
    objects = []
    for child in verb.children:
        if child.dep_ in ("dobj", "attr", "pobj", "oprd"):
            for tok in _get_conj_chain(child):
                objects.append((tok, None))
        # Handle prep chains: 'aggregate into whole' → object is 'whole'
        elif child.dep_ == "prep":
            for grandchild in child.children:
                if grandchild.dep_ == "pobj":
                    for tok in _get_conj_chain(grandchild):
                        objects.append((tok, child.lemma_))
    return objects


def _get_verb_text(verb) -> str:
    """Get the bare verb text, excluding ALL auxiliaries.

    Both aux (can, may, should) and auxpass (be, is) are syntactic markers.
    The main parser uses bare token.text for the same reason — these don't
    carry semantic content for relation extraction. "can be compressed" and
    "compressed" should produce the same relation string.

    Only negation ("not") is kept — it carries semantic meaning (NOT_ prefix).

    This aligns clause segmenter output with main parser output, enabling
    deduplication via relation_canonicalizer.py.

    Reference: Universal Dependencies — aux/auxpass dependency labels
    Reference: de Marneffe et al. (2014), "Universal Stanford Dependencies"
    """
    parts = []
    for child in verb.children:
        if child.dep_ == "neg":
            parts.append(child.text)
        # Skip all auxiliaries (aux + auxpass) — syntactic markers only
    parts.append(verb.text)
    return " ".join(parts)


def _get_object_text(obj) -> str:
    """Get the full noun phrase for an object (determiners stripped).

    Recursively collects modifier text to handle nested amod/advmod chains.
    'jelly-like substance' → 'jelly-like substance' (not just 'substance')
    'a story' → 'story' (determiners stripped to match main parser output)

    Determiners are stripped so clause segmenter and main parser produce
    the same entity strings, enabling deduplication.

    Reference: UDASTE (ScienceDirect, 2023) — compound expansion via
    dependency subtree collects amod, compound, nummod children.
    Reference: Min et al. (2025) — confidence scores penalize shallow extraction.
    """
    parts = []
    for child in obj.children:
        if child.dep_ == "det":
            continue  # Strip determiners — match main parser behavior
        elif child.dep_ in ("amod", "compound", "nummod", "poss", "advmod"):
            # Recurse: collect modifiers of modifiers
            # "jelly-like substance" → like is amod of substance,
            # jelly is advmod of like → collect both
            sub_text = _collect_modifiers_recursive(child)
            parts.append(sub_text)
    parts.append(obj.text)
    return " ".join(parts)


def _collect_modifiers_recursive(token) -> str:
    """Recursively collect all modifier text for a token.

    Handles nested modifier chains:
    'jelly-like' → advmod(jelly) → amod(like) → 'jelly like'
    'membrane-bound' → npadvmod(membrane) → amod(bound) → 'membrane bound'
    """
    parts = []
    for child in token.children:
        if child.dep_ in ("amod", "compound", "nummod", "poss", "advmod", "npadvmod"):
            sub = _collect_modifiers_recursive(child)
            if sub:
                parts.append(sub)
            parts.append(child.text)
    parts.append(token.text)
    return " ".join(parts)


def _get_subject_text(subj) -> str:
    """Get the full noun phrase for a subject (determiners stripped).

    Same recursive approach as _get_object_text. Determiners stripped
    to match main parser output for deduplication.
    """
    parts = []
    for child in subj.children:
        if child.dep_ == "det":
            continue  # Strip determiners — match main parser behavior
        elif child.dep_ in ("amod", "compound", "nummod", "poss", "advmod"):
            sub_text = _collect_modifiers_recursive(child)
            parts.append(sub_text)
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
            # BUT NOT for ccomp/xcomp or their conj children.
            # spaCy sometimes attaches conj directly to ROOT instead of
            # to the ccomp verb. So we also check if any sibling verb
            # in the same conj chain is ccomp/xcomp.
            #
            # Example: "controls what enters and exits the cell"
            # spaCy: enters=ccomp, exits=conj(head=controls)
            # exits.parent = controls (ROOT) → passes filter
            # But enters is ccomp sibling → should skip
            if not subjects and verb_info.get("parent_verb"):
                dep = verb_info["dep"]
                parent = verb_info["parent_verb"]
                if dep not in ("ccomp", "xcomp"):
                    # Check parent dep
                    if parent.dep_ not in ("ccomp", "xcomp"):
                        # Check sibling verbs — is any conj sibling a ccomp?
                        has_ccomp_sibling = False
                        for sibling in parent.children:
                            if (sibling.dep_ in ("ccomp", "xcomp")
                                    and sibling.pos_ in ("VERB", "NOUN")):
                                has_ccomp_sibling = True
                                break
                        if not has_ccomp_sibling:
                            subjects = _get_subjects(parent)

            # If still no subjects, skip
            if not subjects:
                continue

            # Generate one clause per (subject, object) pair
            verb_text = _get_verb_text(verb)
            for subj in subjects:
                subj_text = _get_subject_text(subj)

                if objects:
                    for obj_token, prep_label in objects:
                        obj_text = _get_object_text(obj_token)
                        # Include prep in relation: "divide" + "through" → "divide_through"
                        if prep_label:
                            rel = f"{verb_text}_{prep_label}".lower().strip()
                        else:
                            rel = verb_text.lower().strip()
                        clauses.append(SimpleClause(
                            subject=subj_text.lower().strip(),
                            relation=rel,
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
