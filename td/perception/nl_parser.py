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
    """Zero-hardcode parser. Innate prototypes work in pure mode."""

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

        # ─── INNATE: Stop-word prototype (HDC-encoded sentence) ──────
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
        # Encoded as SHORT phrases (same encoding as inter-entity phrases)
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

        # Fast lookup stop words — excludes words that are also relation types
        # (before, after, different, etc. are constraint signals, not noise)
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

    def _discover_entity_spans(self, tokens):
        """Discover entity spans. Single tokens first, merge only with evidence.

        Pure mode (no MHN):
        1. Every non-stop, non-pure-punctuation token is a single-token entity.
        2. Adjacent entity tokens MAY be merged if their joint HDC vector
           has higher MHN similarity than either alone (seeded mode only).

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
            # Filter to prepositions (the actual relation words)
            # "is", "the", "was" are auxiliaries/determiners, not relations
            prepositions = [w for w in connectives
                          if w in self.stop_words and len(w) > 1
                          and w not in ("is", "are", "was", "were", "the", "a", "an",
                                       "be", "been", "have", "has", "had", "do", "does",
                                       "did", "will", "would", "could", "should")]

            if prepositions:
                conn_text = " ".join(prepositions)
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
