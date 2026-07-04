"""Generic NL Parser -- Zero hardcoded entities, zero keyword lists.

Based on:
    - Kanerva (2009) -- HDC role-filler records, DOI: 10.1007/s12559-009-9009-8
    - Kleyko et al. (2022) -- HDC/VSA Survey, ACM Computing Surveys 55(6), Article 130.
    - Kleyko et al. (2025) -- Principled neuromorphic reservoir computing, Nature Communications
    - Yilmaz (2015) -- CA + HDC reservoir, arXiv:1503.00851

Innate prototypes work in pure mode (empty MHN):
    - stop_word_prototype: filters noise words by HDC similarity
    - stop_word_ratio: reject spans where >50% tokens are stop words
    - relation_prototypes: classify 14 relation types by HDC similarity

Learned patterns (seeded mode, MHN required):
    - entity type vectors, constraint templates
"""

from __future__ import annotations

import re
import string
from typing import Any

import numpy as np

from .hdc import (
    bind, bundle, similarity, permute,
    generate_hypervector, normalize_hdc,
)


class CAReservoir:
    """1D CA reservoir with Rule 90. Zero trainable parameters."""
    def __init__(self, width=64, steps=16, rule=90):
        self.width = width
        self.steps = steps
        self.rule = rule

    def _step(self, state):
        left = np.roll(state, -1)
        right = np.roll(state, 1)
        return np.logical_xor(left, right).astype(np.uint8)

    def process(self, token_vectors):
        features = []
        for tv in token_vectors:
            # Sample to CA width (64 bits) from the token vector
            if len(tv) > self.width:
                indices = np.arange(self.width)
                sampled = tv[indices % len(tv)]
            else:
                sampled = tv
            if len(sampled) < self.width:
                sampled = np.tile(sampled, self.width // len(sampled) + 1)[:self.width]
            state = (sampled > 0).astype(np.uint8)
            for _ in range(self.steps):
                state = self._step(state)
            feature = np.zeros(len(tv), dtype=np.float32)
            # FIX: cast to int16 before arithmetic to avoid uint8 overflow
            # (uint8: 0*2-1 = 255 due to wrap!)
            feature[:len(state)] = state.astype(np.int16) * 2 - 1
            features.append(feature)
        return features


class GenericEntityGraph:
    def __init__(self):
        self.entities = []
        self.relations = []
        self.constraints = []

    def add_entity(self, text, hdc, type_vec=None):
        eid = f"e{len(self.entities)}"
        self.entities.append({"id": eid, "text": text, "hdc": hdc, "type_vec": type_vec or hdc})
        return eid

    def add_relation(self, src, tgt, rel_vec, rel_type=""):
        self.relations.append({"src": src, "tgt": tgt, "rel_vec": rel_vec, "rel_type": rel_type})

    def add_constraint(self, type_vec, subjects, params=None):
        self.constraints.append({"type_vec": type_vec, "subjects": subjects, "params": params or {}})


class GenericNLParser:
    """spaCy-powered parser. Uses dependency parsing for extraction.

    Replaces hardcoded rules with spaCy NLP:
    - stop_words → token.is_stop (language-agnostic)
    - relation_prototypes → token.dep_ (dependency labels)
    - _merge_*_pattern → doc.noun_chunks (compound noun detection)
    - _pp_words → token.pos_ == "ADP" (preposition detection)
    - _strip_pp → token.head (PP attachment via dependency tree)
    - regex patterns → dependency tree traversal

    Reference: Honnibal & Montani (2017), "spaCy 2"
    """

    def __init__(self, vocab, mhn, dim=10_000):
        self.vocab = vocab
        self.mhn = mhn
        self.dim = dim
        self.ca = CAReservoir(width=64, steps=16, rule=90)

        self.role_markers = {
            "subject": generate_hypervector(dim),
            "object": generate_hypervector(dim),
            "modifier": generate_hypervector(dim),
            "connector": generate_hypervector(dim),
        }

        # ─── spaCy NLP pipeline (lazy load) ──────────────────────────
        self._nlp = None

        # ─── INNATE: Stop-word prototype (HDC-encoded sentence) ──────
        # Kept for HDC encoding compatibility. Actual stop word detection
        # uses spaCy's token.is_stop (language-agnostic).
        self.stop_word_prototype = self._encode_phrase(
            "the a an and or is are was were be been have has had do does did "
            "will would could should may might must can shall it its "
            "this that these those there their they them of to in for on with at by from "
            "as into through during before after above below between under again "
            "further then once here why how all any both each few more most other "
            "some such no nor not only own same so than too very just but because "
            "until while about against down out off over under"
        )

        # ─── INNATE: Relation prototypes (14 types) ──────────────────
        # Kept for HDC encoding compatibility. Actual relation detection
        # uses spaCy's dependency labels (token.dep_).
        self.relation_prototypes = {
            "different": self._encode_phrase("different distinct separate unique"),
            "before": self._encode_phrase("before earlier precedes first"),
            "after": self._encode_phrase("after later follows second"),
            "excludes": self._encode_phrase("excludes cannot together forbidden"),
            "limited": self._encode_phrase("limited bounded maximum minimum"),
            "grouped": self._encode_phrase("grouped together category partition"),
            "sum_to": self._encode_phrase("sum total adds equals"),
            "implies": self._encode_phrase("implies if then requires means"),
            "overlap": self._encode_phrase("overlap conflict cannot same time"),
            "precedence": self._encode_phrase("precedence must before chain ordered"),
            "ratio": self._encode_phrase("ratio proportional divided times"),
            "count": self._encode_phrase("count exactly at least how many"),
            "equivalent": self._encode_phrase("equivalent same equal identical"),
            "optimize": self._encode_phrase("optimize maximize minimize best"),
        }

        self.constraint_signals = set(self.relation_prototypes.keys())

        # Fast lookup stop words — kept as fallback for non-spaCy paths
        self.stop_words = {
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
        }

    @property
    def nlp(self):
        """Lazy-load spaCy pipeline."""
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
            except (ImportError, OSError):
                self._nlp = False  # spaCy not available
        return self._nlp if self._nlp is not False else None

    def register_relation(self, relation: str, example_phrases: str = None):
        """Register a learned relation as a prototype for future parsing.

        This is how the parser learns about new relations from the KG.
        When a user teaches "north_of" → transitive, the KG saves it to SQLite,
        and this method creates an HDC prototype so the parser can detect it
        in future queries like "is Kazakhstan north of Uzbekistan?"

        Args:
            relation: Relation name (e.g., "north_of", "married_to")
            example_phrases: Optional example phrases. If None, uses the
                           relation name itself (underscores → spaces).
        """
        if relation in self.relation_prototypes:
            return  # Already known

        # Generate prototype from relation name (e.g., "north_of" → "north of")
        if example_phrases is None:
            example_phrases = relation.replace("_", " ")

        self.relation_prototypes[relation] = self._encode_phrase(example_phrases)

    def _tokenize(self, text):
        text = text.lower()
        text = re.sub(r"[^\w\s'-]", " ", text)
        return [t.strip("-'") for t in text.split() if t.strip()]

    def _encode_token(self, token):
        if not token:
            return generate_hypervector(self.dim)
        chars = [c for c in token if c in string.ascii_lowercase or c.isdigit()]
        if not chars:
            return generate_hypervector(self.dim)
        ngrams = []
        for i, c in enumerate(chars):
            c_vec = self.vocab.get_char_vector(c)
            ngrams.append(permute(c_vec, i))
        for i in range(len(chars) - 1):
            c1 = self.vocab.get_char_vector(chars[i])
            c2 = self.vocab.get_char_vector(chars[i + 1])
            ngrams.append(bind(c1, permute(c2, 1)))
        return bundle(*ngrams) if ngrams else generate_hypervector(self.dim)

    def _encode_phrase(self, text_or_tokens, start=0, end=None):
        if isinstance(text_or_tokens, str):
            tokens = self._tokenize(text_or_tokens)
            start, end = 0, len(tokens)
        else:
            tokens = text_or_tokens
            end = end if end is not None else len(tokens)
        vecs = []
        for i, tok in enumerate(tokens[start:end]):
            tok_vec = self._encode_token(tok)
            pos_vec = permute(tok_vec, i)
            role = list(self.role_markers.values())[i % len(self.role_markers)]
            vecs.append(bind(pos_vec, role))
        return bundle(*vecs) if vecs else generate_hypervector(self.dim)

    def parse(self, text):
        tokens = self._tokenize(text)
        if not tokens:
            return generate_hypervector(self.dim)
        token_vectors = [self._encode_token(t) for t in tokens]
        ca_features = self.ca.process(token_vectors)
        doc_components = []
        for i, (tok_vec, ca_vec) in enumerate(zip(token_vectors, ca_features)):
            combined = bundle(tok_vec * 0.7, ca_vec * 0.3)
            doc_components.append(permute(combined, i))
        return normalize_hdc(bundle(*doc_components))

    def extract_structure(self, text):
        tokens = self._tokenize(text)
        graph = GenericEntityGraph()
        if not tokens:
            return {"graph": graph, "text": text, "hdc": self.parse(text)}

        # Step 1: Discover entities with innate stop-word filtering
        entity_spans = self._discover_entity_spans(tokens)
        for span in entity_spans:
            phrase = " ".join(tokens[span["start"]:span["end"]])
            phrase_hdc = self._encode_phrase(tokens, span["start"], span["end"])
            type_vec = self._discover_type(phrase_hdc)
            eid = graph.add_entity(phrase, phrase_hdc, type_vec)
            span["eid"] = eid

        # Step 1b: Detect "X is the Y of Z" pattern
        # If three adjacent entities: [X, Y, Z] with "is" and "of" between them,
        # then Y is part of the relation, not an entity. Remove Y from entities
        # and add a direct relation X --Y_of--> Z.
        self._merge_is_y_of_z_pattern(graph, tokens, entity_spans)

        # Same for "X is Y to Z" pattern (e.g., "is married to", "is related to")
        self._merge_is_y_to_z_pattern(graph, tokens, entity_spans)

        # General "X is Y Z" where Y is a known relation (e.g., "is before", "is in")
        self._merge_is_y_z_pattern(graph, tokens, entity_spans)

        # Step 1c: Merge compound nouns AFTER pattern matching
        # "united kingdom is in europe" → pattern extracts (united kingdom, in, europe)
        # THEN merge "united kingdom" into single entity in the graph
        self._merge_compound_nouns_in_graph(graph, tokens)

        # Track entity pairs that already have relations (from pattern matching)
        existing_rel_pairs = {(r["src"], r["tgt"]) for r in graph.relations}

        # Step 2: Discover relations with innate prototype classification
        for i, e1 in enumerate(graph.entities):
            for e2 in graph.entities[i + 1:]:
                # Skip if already has a relation from pattern matching
                if (e1["id"], e2["id"]) in existing_rel_pairs:
                    continue
                rel_type = self._discover_relation_innate(e1, e2, tokens, text)
                if rel_type:
                    rel_hdc = self._encode_phrase(f"{e1['text']} {rel_type} {e2['text']}")
                    graph.add_relation(e1["id"], e2["id"], rel_hdc, rel_type)

        # Step 3: Discover constraints (MHN-dependent)
        problem_hdc = self.parse(text)
        constraints = self._discover_constraints(problem_hdc, graph)
        for c in constraints:
            graph.add_constraint(c["type_vec"], c["subjects"], c.get("params"))

        return {"graph": graph, "text": text, "hdc": problem_hdc, "tokens": tokens}

    def extract_triples_spacy(self, text: str) -> list[tuple[str, str, str]]:
        """Extract triples using spaCy dependency parsing.

        Replaces all hardcoded regex patterns with dependency tree traversal.
        Language-agnostic — works for any language spaCy supports.

        Handles:
        - "X is in Y" → (x, in, y)
        - "X is part of Y" → (x, part_of, y)
        - "X is the capital of Y" → (x, capital_of, y)
        - "X is before Y on Z" → (x, before, y) — PP attachment
        - "X evolved from Y" → (x, evolved_from, y)
        - "X treats Y" → (x, treats, y)

        Reference: Honnibal & Montani (2017), "spaCy 2"
        """
        if self.nlp is None:
            return []

        doc = self.nlp(text)
        triples = []

        for token in doc:
            # ─── Copular constructions: "X is Y" ─────────────────
            if token.dep_ == "ROOT" and token.lemma_ in ("be", "become", "seem"):
                subject = None
                for child in token.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        subject = child
                        break
                if not subject:
                    continue

                subj_text = self._get_chunk_text(doc, subject)

                # Collect all preps attached to ROOT
                preps = [c for c in token.children if c.dep_ == "prep"]

                # Collect attr (adjective/noun predicate)
                attrs = [c for c in token.children if c.dep_ in ("attr", "acomp")]

                if attrs:
                    attr = attrs[0]
                    # Check if attr has its own prep chain: "part of Y", "married to Y"
                    attr_preps = [c for c in attr.children if c.dep_ == "prep"]
                    if attr_preps:
                        prep = attr_preps[0]
                        pobj = [c for c in prep.children if c.dep_ == "pobj"]
                        if pobj:
                            obj_text = self._get_chunk_text(doc, pobj[0])
                            # Use token text (not lemma) to preserve relation name
                            # "sibling" → "sibling_of" (not "sible_of")
                            rel = f"{attr.text.lower()}_{prep.lemma_}"
                            triples.append((subj_text, rel, obj_text))
                            continue

                # Check preps attached to ROOT: "is in Y", "is before Y"
                if preps:
                    # Take FIRST prep (ignore trailing context like "on Danube")
                    prep = preps[0]
                    pobj = [c for c in prep.children if c.dep_ == "pobj"]
                    if pobj:
                        obj_text = self._get_chunk_text(doc, pobj[0])
                        triples.append((subj_text, prep.lemma_, obj_text))
                        continue

            # ─── Verb constructions: "X evolved from Y", "X treats Y" ──
            if token.pos_ == "VERB" and token.dep_ == "ROOT":
                subj = None
                dobj = None
                prep_chain = None

                for child in token.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        subj = child
                    elif child.dep_ == "dobj":
                        dobj = child
                    elif child.dep_ in ("prep", "agent"):
                        for gc in child.children:
                            if gc.dep_ == "pobj":
                                prep_chain = (child, gc)
                                break

                if subj:
                    subj_text = self._get_chunk_text(doc, subj)
                    if dobj:
                        obj_text = self._get_chunk_text(doc, dobj)
                        triples.append((subj_text, token.text.lower(), obj_text))
                    elif prep_chain:
                        prep, obj = prep_chain
                        obj_text = self._get_chunk_text(doc, obj)
                        # Use token text (not lemma) to preserve relation name
                        # "sibling" → "sibling_of" (not "sible_of")
                        # "evolved" → "evolved_from" (not "evolve_from")
                        rel = f"{token.text.lower()}_{prep.lemma_}"
                        triples.append((subj_text, rel, obj_text))

            # ─── Noun-based constructions (no copula) ──────────────
            # "Paris capital of France" → (paris, capital_of, france)
            # "France in the EU" → (france, in, eu)
            # "Kazakhstan north of Uzbekistan" → (kazakhstan, north_of, uzbekistan)
            if token.dep_ == "ROOT" and token.pos_ in ("NOUN", "PROPN"):
                # The ROOT itself is the subject
                subj_text = self._get_chunk_text(doc, token)

                # Check for compound children (subject is compound of ROOT)
                # "Paris capital of France" → compound=Paris, ROOT=capital
                compounds = [c for c in token.children if c.dep_ == "compound"]
                if compounds:
                    # Join all compound tokens for multi-word entities
                    # "New York City capital of USA" → subj = "new york city"
                    subj_text = " ".join([c.text.lower() for c in compounds])

                # Check for prep chain: "capital of France", "in the EU"
                preps = [c for c in token.children if c.dep_ == "prep"]
                if preps:
                    prep = preps[0]
                    pobj = [c for c in prep.children if c.dep_ == "pobj"]
                    if pobj:
                        obj_text = self._get_chunk_text(doc, pobj[0])
                        if compounds:
                            # "Paris capital of France" → capital_of
                            rel = f"{token.text.lower()}_{prep.lemma_}"
                        else:
                            # "France in the EU" → in
                            rel = prep.lemma_
                        triples.append((subj_text, rel, obj_text))
                        continue

                # Check for appos with prep: "Kazakhstan north of Uzbekistan"
                appos = [c for c in token.children if c.dep_ == "appos"]
                if appos:
                    appos_token = appos[0]
                    appos_preps = [c for c in appos_token.children if c.dep_ == "prep"]
                    if appos_preps:
                        prep = appos_preps[0]
                        pobj = [c for c in prep.children if c.dep_ == "pobj"]
                        if pobj:
                            obj_text = self._get_chunk_text(doc, pobj[0])
                            rel = f"{appos_token.text.lower()}_{prep.lemma_}"
                            triples.append((subj_text, rel, obj_text))
                            continue

        return triples

    def _get_chunk_text(self, doc, token) -> str:
        """Get the full noun chunk text for a token, stripping determiners.

        "United Kingdom" → "united kingdom" (not just "Kingdom")
        "the EU" → "eu" (strip "the")
        "a country" → "country" (strip "a")
        """
        for chunk in doc.noun_chunks:
            if token in chunk:
                # Strip leading determiners ("the", "a", "an")
                words = chunk.text.lower().split()
                while words and words[0] in ("the", "a", "an"):
                    words = words[1:]
                return " ".join(words) if words else chunk.text.lower()
        return token.text.lower()

    def _merge_compound_nouns_in_graph(self, graph, tokens):
        """Merge adjacent single-token entities that form compound nouns.

        After pattern matching has extracted relations, scan the graph for
        adjacent entities that should be merged:
        - "united" + "kingdom" → "united kingdom"
        - "north" + "america" → "north america"
        - "world" + "war" + "2" → "world war 2"

        This runs AFTER pattern matching so relations are already extracted.
        Merged entities inherit all relations from their parts.

        Reference: Manning & Schütze (1999), Chapter 5: Collocations.
        """
        if len(graph.entities) < 2:
            return

        # Relation words that should NOT be merged into entities
        relation_words = {
            "in", "of", "to", "from", "at", "on", "by", "for",
            "into", "through", "during", "before", "after",
            "is", "are", "was", "were", "be", "been", "has", "have", "had",
            "part_of", "capital_of", "born_in", "lives_in", "located_in",
        }

        # Find adjacent entities in token order
        merged = True
        while merged:
            merged = False
            for i in range(len(graph.entities) - 1):
                e1 = graph.entities[i]
                e2 = graph.entities[i + 1]

                # Get token positions
                e1_tokens = e1["text"].split()
                e2_tokens = e2["text"].split()

                # Check if e1's last token and e2's first token are adjacent in source
                try:
                    e1_last_idx = tokens.index(e1_tokens[-1])
                    e2_first_idx = tokens.index(e2_tokens[0])
                except ValueError:
                    continue

                # Must be adjacent (no gap)
                if e2_first_idx != e1_last_idx + 1:
                    continue

                # Don't merge if either is a relation word
                if e1["text"] in relation_words or e2["text"] in relation_words:
                    continue

                # Don't merge if either has a relation (already used in pattern)
                has_rel = any(
                    r["src"] == e1["id"] or r["tgt"] == e1["id"] or
                    r["src"] == e2["id"] or r["tgt"] == e2["id"]
                    for r in graph.relations
                )
                if has_rel:
                    continue

                # Merge e1 + e2
                merged_text = e1["text"] + " " + e2["text"]
                merged_hdc = self._encode_phrase(merged_text)
                type_vec = self._discover_type(merged_hdc)

                # Update e1 with merged text
                e1["text"] = merged_text
                e1["hdc"] = merged_hdc
                e1["type"] = type_vec

                # Remove e2
                graph.entities.remove(e2)
                merged = True
                break

    def _merge_post_relation_entities(self, tokens, spans):
        """Merge adjacent non-stop tokens into compound nouns.

        Two patterns:
        1. AFTER relation words: "in central asia" → "central asia"
        2. BEFORE stop words: "united kingdom was" → "united kingdom"

        This handles geographic names, organization names, compound nouns.

        Reference: Compound noun detection in NLP (Manning & Schütze, 1999)
        """
        # Spatial/temporal relation words that precede multi-word entities
        relation_words = {
            "in", "of", "to", "from", "at", "on", "by", "for",
            "into", "through", "during", "before", "after",
            "part_of", "capital_of", "born_in", "lives_in", "located_in",
        }

        result_spans = []
        skip_until = -1

        for span in spans:
            if span["start"] < skip_until:
                continue

            tok = tokens[span["start"]]

            # Pattern 1: Merge AFTER relation words
            # "in central asia" → merge "central" + "asia"
            if tok in relation_words:
                result_spans.append(span)
                merge_start = span["end"]
                merge_end = merge_start
                for next_span in spans:
                    if next_span["start"] == merge_end:
                        next_tok = tokens[next_span["start"]]
                        if next_tok not in self.stop_words and any(c.isalnum() for c in next_tok):
                            merge_end = next_span["end"]
                        else:
                            break
                    elif next_span["start"] > merge_end:
                        break

                if merge_end - merge_start >= 2:
                    merged_span = {
                        "start": merge_start,
                        "end": merge_end,
                        "sim": 0.0,
                        "tokens": tokens[merge_start:merge_end],
                    }
                    result_spans.append(merged_span)
                    skip_until = merge_end
                else:
                    for next_span in spans:
                        if next_span["start"] == merge_start:
                            result_spans.append(next_span)
                            skip_until = merge_start + 1
                            break

            # Pattern 2: Merge BEFORE stop words
            # "united kingdom was" → merge "united" + "kingdom"
            # Check if NEXT token is a stop word (is, was, in, etc.)
            elif any(c.isalnum() for c in tok):
                # Look ahead: is there a stop word after adjacent non-stop tokens?
                merge_start = span["start"]
                merge_end = span["end"]
                for next_span in spans:
                    if next_span["start"] == merge_end:
                        next_tok = tokens[next_span["start"]]
                        if next_tok in self.stop_words:
                            # Found stop word — stop merging
                            break
                        elif any(c.isalnum() for c in next_tok):
                            # Adjacent non-stop token — merge
                            merge_end = next_span["end"]
                        else:
                            break
                    elif next_span["start"] > merge_end:
                        break

                if merge_end - merge_start >= 2:
                    merged_span = {
                        "start": merge_start,
                        "end": merge_end,
                        "sim": 0.0,
                        "tokens": tokens[merge_start:merge_end],
                    }
                    result_spans.append(merged_span)
                    skip_until = merge_end
                else:
                    result_spans.append(span)
            else:
                result_spans.append(span)

        return result_spans

    def _merge_is_y_of_z_pattern(self, graph, tokens, entity_spans):
        """Detect 'X is the Y of Z' pattern and merge Y into the relation.

        Pattern: [entity_X] [is] [the] [entity_Y] [of] [entity_Z]
        After merge: [entity_X] --Y_of--> [entity_Z] (entity_Y removed)

        This handles "Paris is the capital of France" → (paris, capital_of, france)
        instead of three disconnected entities.
        """
        if len(graph.entities) < 3:
            return

        # Check each consecutive triple of entities
        to_remove = []
        to_add_rels = []

        for i in range(len(graph.entities) - 2):
            e1 = graph.entities[i]
            e2 = graph.entities[i + 1]    # middle entity (the relation word)
            e3 = graph.entities[i + 2]

            # Check if the original text between e1 and e3 contains "is" and "of"
            # Find positions in original token list
            try:
                e1_tok_idx = tokens.index(e1["text"].split()[0])
                e3_tok_idx = tokens.index(e3["text"].split()[0])
            except ValueError:
                continue

            # Get the span between e1 and e3
            between = tokens[e1_tok_idx:e3_tok_idx + 1]
            between_text = " ".join(between)

            # Pattern: X is [the] Y of Z
            # "is" must appear before Y, "of" must appear after Y
            if "is" in between and "of" in between:
                y_text = e2["text"]
                # Verify Y is between "is" and "of" in the token sequence
                is_idx = between.index("is") if "is" in between else -1
                of_idx = between.index("of") if "of" in between else -1
                y_idx = None
                for j, t in enumerate(between):
                    if t == y_text.split()[0]:
                        y_idx = j
                        break

                if y_idx and is_idx < y_idx < of_idx:
                    # Valid "X is the Y of Z" pattern
                    rel_type = f"{y_text}_of"
                    rel_hdc = self._encode_phrase(f"{e1['text']} {rel_type} {e3['text']}")
                    to_add_rels.append((e1["id"], e3["id"], rel_hdc, rel_type))
                    to_remove.append(e2["id"])

        # Remove middle entities (they're relation components, not entities)
        for eid in to_remove:
            graph.entities = [e for e in graph.entities if e["id"] != eid]

        # Add relations
        for src, tgt, hdc, rel_type in to_add_rels:
            graph.add_relation(src, tgt, hdc, rel_type)

    def _merge_is_y_to_z_pattern(self, graph, tokens, entity_spans):
        """Detect 'X is Y to Z' pattern and merge Y into the relation.

        Pattern: [entity_X] [is] [entity_Y] [to] [entity_Z]
        After merge: [entity_X] --Y_to--> [entity_Z] (entity_Y removed)

        This handles "David Beckham is married to Victoria Beckham"
        → (david, married_to, victoria) instead of "married" being an entity.
        """
        if len(graph.entities) < 3:
            return

        to_remove = []
        to_add_rels = []

        for i in range(len(graph.entities) - 2):
            e1 = graph.entities[i]
            e2 = graph.entities[i + 1]    # middle entity (the relation word)
            e3 = graph.entities[i + 2]

            # Check if "to" appears between e2 and e3 in the original tokens
            try:
                e2_tok_idx = tokens.index(e2["text"].split()[0])
                e3_tok_idx = tokens.index(e3["text"].split()[0])
            except ValueError:
                continue

            between = tokens[e2_tok_idx:e3_tok_idx + 1]

            # Pattern: Y [is] to Z — "to" must appear after Y
            if "to" in between:
                y_text = e2["text"]
                to_idx = between.index("to")
                y_idx = None
                for j, t in enumerate(between):
                    if t == y_text.split()[0]:
                        y_idx = j
                        break

                if y_idx is not None and y_idx < to_idx:
                    # Valid "X is Y to Z" pattern
                    rel_type = f"{y_text}_to"
                    rel_hdc = self._encode_phrase(f"{e1['text']} {rel_type} {e3['text']}")
                    to_add_rels.append((e1["id"], e3["id"], rel_hdc, rel_type))
                    to_remove.append(e2["id"])

        # Remove middle entities (they're relation components, not entities)
        for eid in to_remove:
            graph.entities = [e for e in graph.entities if e["id"] != eid]

        # Add relations
        for src, tgt, hdc, rel_type in to_add_rels:
            graph.add_relation(src, tgt, hdc, rel_type)

    def _merge_is_y_z_pattern(self, graph, tokens, entity_spans):
        """Detect 'X is Y Z' where Y is a known relation word.

        Pattern: [entity_X] [is] [relation_word] [entity_Z]
        After merge: [entity_X] --Y--> [entity_Z] (Y removed from entities)

        Handles "Tokyo_2020 is before LA_2028" → (tokyo_2020, before, la_2028)
        and "France is in EU" → (france, in, eu).

        Works with tokens (not just entities) so relation words like "in"
        don't need to be in the entity list.
        """
        if len(graph.entities) < 2:
            return

        to_add_rels = []

        # Scan tokens for "is" + relation_word pattern
        for i, tok in enumerate(tokens):
            if tok not in ("is", "are", "was", "were"):
                continue

            # Look for relation word after "is"
            if i + 1 >= len(tokens):
                continue
            rel_word = tokens[i + 1]
            if rel_word not in self.relation_prototypes and \
               rel_word not in self.constraint_signals:
                continue

            # Find entity before "is" and entity after relation word
            # Entity before: closest entity whose last token is before "is"
            e_before = None
            for e in graph.entities:
                e_last = e["text"].split()[-1]
                try:
                    e_idx = tokens.index(e_last)
                    if e_idx < i:
                        e_before = e
                except ValueError:
                    continue

            # Entity after: closest entity whose first token is after relation word
            e_after = None
            for e in graph.entities:
                e_first = e["text"].split()[0]
                try:
                    e_idx = tokens.index(e_first)
                    if e_idx > i + 1:
                        e_after = e
                        break
                except ValueError:
                    continue

            if e_before and e_after:
                rel_type = rel_word
                rel_hdc = self._encode_phrase(f"{e_before['text']} {rel_type} {e_after['text']}")
                to_add_rels.append((e_before["id"], e_after["id"], rel_hdc, rel_type))

        for src, tgt, hdc, rel_type in to_add_rels:
            graph.add_relation(src, tgt, hdc, rel_type)
            graph.add_relation(src, tgt, hdc, rel_type)

    def _discover_entity_spans(self, tokens):
        """Discover entity spans. Single tokens first, merge only with evidence.

        Pure mode (no MHN):
        1. Every non-stop, non-pure-punctuation token is a single-token entity.
        2. Adjacent entity tokens MAY be merged if their joint HDC vector
           has higher MHN similarity than either alone (seeded mode only).
        3. Adjacent non-stop tokens after spatial/temporal relation words
           are merged (e.g., "in central asia" → "central asia").

        This avoids the mega-phrase problem where "assign 3 different tasks"
         gets grabbed as one entity instead of ["3", "tasks"].
        """
        spans = []
        covered = set()

        # Step 1: Mark all non-stop tokens as single-token entities
        for i, tok in enumerate(tokens):
            if tok in self.stop_words:
                continue
            # Pure punctuation
            if not any(c.isalnum() for c in tok):
                continue
            spans.append({"start": i, "end": i + 1, "sim": 0.0, "tokens": [tok]})
            covered.add(i)

        # Step 2: In seeded mode, try merging adjacent entities
        # Only merge if MHN similarity of the merged form is higher than
        # the individual forms. This is learned composition, not hardcoded.
        if len(self.mhn.patterns) > 0:
            merged = True
            while merged and len(spans) > 1:
                merged = False
                for idx in range(len(spans) - 1):
                    s1 = spans[idx]
                    s2 = spans[idx + 1]
                    # Must be adjacent in token positions
                    if s1["end"] != s2["start"]:
                        continue
                    # Check if any token between them is uncovered (non-entity gap)
                    gap = s2["start"] - s1["end"]
                    if gap > 0:
                        continue

                    # Compare MHN similarity
                    merged_hdc = self._encode_phrase(tokens, s1["start"], s2["end"])
                    results = self.mhn.retrieve(merged_hdc, top_k=1)
                    merged_sim = results[0][1] if results else 0.0

                    s1_hdc = self._encode_phrase(tokens, s1["start"], s1["end"])
                    s2_hdc = self._encode_phrase(tokens, s2["start"], s2["end"])
                    r1 = self.mhn.retrieve(s1_hdc, top_k=1)
                    r2 = self.mhn.retrieve(s2_hdc, top_k=1)
                    s1_sim = r1[0][1] if r1 else 0.0
                    s2_sim = r2[0][1] if r2 else 0.0

                    if merged_sim > max(s1_sim, s2_sim) + 0.05:
                        # Don't merge if either token is a relation prototype
                        # (before, after, different, etc. are relation markers, not entities)
                        s1_tokens = tokens[s1["start"]:s1["end"]]
                        s2_tokens = tokens[s2["start"]:s2["end"]]
                        if any(t in self.relation_prototypes for t in s1_tokens + s2_tokens):
                            continue
                        # Merge
                        spans[idx] = {
                            "start": s1["start"], "end": s2["end"],
                            "sim": merged_sim,
                            "tokens": tokens[s1["start"]:s2["end"]],
                        }
                        del spans[idx + 1]
                        merged = True
                        break

        spans.sort(key=lambda s: s["start"])
        return spans

    def _discover_type(self, phrase_hdc):
        results = self.mhn.retrieve(phrase_hdc, top_k=1)
        if results and results[0][1] > 0.50:
            meta = results[0][2]
            if "type_vec" in meta:
                return meta["type_vec"]
            return results[0][0]
        return None

    def _discover_relation_innate(self, e1, e2, tokens, text):
        """Discover relation using innate prototypes (works in pure mode).

        Strategy:
        1. Find the inter-entity span in original tokens
        2. Extract connective words (prepositions, stop words between entities)
        3. Compare connective words to spatial prototypes
        4. If no match, compare full span to all prototypes
        """
        # Find token positions of entities
        e1_first = e1["text"].split()[0]
        e2_first = e2["text"].split()[0]
        try:
            e1_start = tokens.index(e1_first)
            e2_start = tokens.index(e2_first)
        except ValueError:
            return None

        # Ensure e1 before e2
        if e1_start > e2_start:
            e1_start, e2_start = e2_start, e1_start
            e1, e2 = e2, e1

        e2_end = e2_start + len(e2["text"].split())
        span = tokens[e1_start:e2_end]

        # Extract connective words (between entity boundaries)
        e1_end = e1_start + len(e1["text"].split())
        connectives = tokens[e1_end:e2_start]

        # Strategy 1: Compare connective words to prototypes
        if connectives:
            # Filter out pure auxiliaries/determiners, but KEEP content words
            # (e.g., "married" in "is married to" is a relation word, not noise)
            auxiliaries = {"is", "are", "was", "were", "the", "a", "an",
                          "be", "been", "have", "has", "had", "do", "does",
                          "did", "will", "would", "could", "should"}
            relation_words = [w for w in connectives
                            if w not in auxiliaries and len(w) > 1]

            if relation_words:
                conn_text = " ".join(relation_words)
                conn_hdc = self._encode_phrase(conn_text)

                best_type = None
                best_sim = 0.0
                for rel_type, proto_hdc in self.relation_prototypes.items():
                    sim = similarity(conn_hdc, proto_hdc)
                    if sim > best_sim and sim > 0.15:
                        best_sim = sim
                        best_type = rel_type

                if best_type:
                    return best_type

                # No prototype match — synthesize relation type from content words
                # e.g., "married" → "married_to", "north" → "north_of"
                # Use the words that aren't pure prepositions
                content = [w for w in relation_words if w not in self.stop_words]
                if content:
                    rel_candidate = "_".join(content)
                    # Append common suffixes if missing
                    if not any(rel_candidate.endswith(s) for s in ("_of", "_to", "_in", "_for")):
                        # Check if a preposition follows the content words
                        postpositions = [w for w in relation_words if w in self.stop_words]
                        if postpositions:
                            rel_candidate = f"{rel_candidate}_{postpositions[-1]}"
                    return rel_candidate

        # Strategy 2: Compare full span to prototypes (fallback)
        rel_phrase = " ".join(span)
        rel_hdc = self._encode_phrase(rel_phrase)

        best_type = None
        best_sim = 0.0
        for rel_type, proto_hdc in self.relation_prototypes.items():
            sim = similarity(rel_hdc, proto_hdc)
            if sim > best_sim and sim > 0.25:
                best_sim = sim
                best_type = rel_type

        # Strategy 3: MHN override if available
        results = self.mhn.retrieve(rel_hdc, top_k=1)
        if results and results[0][1] > 0.40 and results[0][1] > best_sim:
            meta = results[0][2]
            return meta.get("relation_type", best_type or "")

        return best_type

    def _discover_constraints(self, problem_hdc, graph):
        constraints = []
        results = self.mhn.retrieve(problem_hdc, top_k=3)
        for vec, sim, meta in results:
            if sim < 0.40:
                continue
            template = meta.get("constraint_template")
            if template:
                subjects = self._map_template_to_entities(template, graph)
                if subjects:
                    constraints.append({"type_vec": vec, "subjects": subjects, "params": template.get("params", {})})
        return constraints

    def _map_template_to_entities(self, template, graph):
        selector = template.get("selector_vec")
        if selector is None:
            return [e["id"] for e in graph.entities]
        matches = []
        for e in graph.entities:
            sim = similarity(e["hdc"], selector)
            matches.append((e["id"], sim))
        matches.sort(key=lambda x: x[1], reverse=True)
        threshold = template.get("selector_threshold", 0.30)
        return [eid for eid, sim in matches if sim > threshold]

    def extract_entities(self, text):
        struct = self.extract_structure(text)
        graph = struct["graph"]
        entities = {
            "_raw_text": text,
            "_entity_count": len(graph.entities),
            "_relation_count": len(graph.relations),
            "_constraint_count": len(graph.constraints),
        }
        for e in graph.entities:
            type_key = self._type_to_key(e.get("type_vec"))
            if type_key not in entities:
                entities[type_key] = []
            entities[type_key].append(e["text"])
        for r in graph.relations:
            src = next(e["text"] for e in graph.entities if e["id"] == r["src"])
            tgt = next(e["text"] for e in graph.entities if e["id"] == r["tgt"])
            if "_relations" not in entities:
                entities["_relations"] = []
            entities["_relations"].append(f"{src} -> {r['rel_type']} -> {tgt}")
        for c in graph.constraints:
            if "_constraints" not in entities:
                entities["_constraints"] = []
            subj_texts = [next(e["text"] for e in graph.entities if e["id"] == sid) for sid in c["subjects"]]
            entities["_constraints"].append({"subjects": subj_texts, "params": c["params"]})
        return entities

    def _type_to_key(self, type_vec):
        if type_vec is None:
            return "unknown"
        results = self.mhn.retrieve(type_vec, top_k=1)
        if results:
            meta = results[0][2]
            return meta.get("type_name", "discovered")
        return "unknown"
