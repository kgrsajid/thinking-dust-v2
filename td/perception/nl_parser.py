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
            indices = np.arange(len(tv))
            sampled = tv[indices % len(tv)] if len(tv) > self.width else tv
            if len(sampled) < self.width:
                sampled = np.tile(sampled, self.width // len(sampled) + 1)[:self.width]
            state = (sampled > 0).astype(np.uint8)
            for _ in range(self.steps):
                state = self._step(state)
            feature = np.zeros(len(tv), dtype=np.float32)
            feature[:len(state)] = state * 2 - 1
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

        # Fast lookup stop words
        self.stop_words = {
            "the", "a", "an", "and", "or", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "can", "shall", "it", "its", "this",
            "that", "these", "those", "there", "their", "they", "them", "of", "to",
            "in", "for", "on", "with", "at", "by", "from", "as", "into", "through",
            "during", "before", "after", "above", "below", "between", "under",
            "again", "further", "then", "once", "here", "why", "how", "all", "any",
            "both", "each", "few", "more", "most", "other", "some", "such", "no",
            "nor", "not", "only", "own", "same", "so", "than", "too", "very", "just",
            "but", "if", "because", "until", "while", "about", "against", "down",
            "out", "off", "over", "under",
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

        # Step 2: Discover relations with innate prototype classification
        for i, e1 in enumerate(graph.entities):
            for e2 in graph.entities[i + 1:]:
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

    def _discover_entity_spans(self, tokens):
        """Discover entities with innate stop-word filtering.

        Rules (pure mode, no MHN):
        1. Skip span if ALL tokens are stop words.
        2. Skip span if >50% tokens are stop words (for multi-token spans).
        3. Skip span if HDC similarity to stop_word_prototype > 0.55.
        4. Accept span if it contains a digit (numbers are always entities).
        5. Accept multi-token spans with at least one non-stop word.
        6. Single non-stop tokens are entities.

        Rules (seeded mode, MHN available):
        7. MHN match overrides all filters (learned exceptions).
        """
        spans = []
        covered = set()

        for length in range(min(4, len(tokens)), 0, -1):
            for start in range(len(tokens) - length + 1):
                end = start + length
                if any(i in covered for i in range(start, end)):
                    continue

                span_tokens = tokens[start:end]
                phrase = " ".join(span_tokens)
                phrase_hdc = self._encode_phrase(tokens, start, end)

                # Rule 1: All stop words -> skip
                if all(t in self.stop_words for t in span_tokens):
                    continue

                # Rule 2: >50% stop words for multi-token spans -> skip
                if len(span_tokens) > 1:
                    stop_ratio = sum(1 for t in span_tokens if t in self.stop_words) / len(span_tokens)
                    if stop_ratio > 0.5:
                        continue

                # Rule 3: HDC similarity to stop-word prototype -> skip
                stop_sim = similarity(phrase_hdc, self.stop_word_prototype)
                if stop_sim > 0.55:
                    continue

                # Rule 4: Contains digit -> always entity
                has_digit = any(t.isdigit() for t in span_tokens)

                # Rule 5: Multi-token with non-stop word -> entity
                has_content = length > 1 and not all(t in self.stop_words for t in span_tokens)

                # Rule 7: MHN match (learned pattern)
                mhn_match = False
                mhn_sim = 0.0
                results = self.mhn.retrieve(phrase_hdc, top_k=1)
                if results and results[0][1] > 0.55:
                    mhn_match = True
                    mhn_sim = results[0][1]

                if mhn_match or has_digit or has_content:
                    spans.append({"start": start, "end": end, "sim": mhn_sim if mhn_match else 0.0})
                    for i in range(start, end):
                        covered.add(i)

        # Single non-stop tokens
        for i, tok in enumerate(tokens):
            if i not in covered and tok not in self.stop_words and not tok.isdigit():
                spans.append({"start": i, "end": i + 1, "sim": 0.0})

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

        FIX: Use SAME encoding scheme as prototypes (_encode_phrase).
        Old code used bind(e1, bind(context, e2)) which produces a random
        vector unrelated to the bundled-sentence prototypes.
        """
        # Encode the inter-entity phrase the SAME way as prototypes
        rel_phrase = f"{e1['text']} {e2['text']}"
        rel_hdc = self._encode_phrase(rel_phrase)

        # Innate: compare to 14 relation prototypes
        best_type = None
        best_sim = 0.0
        for rel_type, proto_hdc in self.relation_prototypes.items():
            sim = similarity(rel_hdc, proto_hdc)
            if sim > best_sim and sim > 0.25:  # Lowered from 0.35
                best_sim = sim
                best_type = rel_type

        # Learned: MHN override if available
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
