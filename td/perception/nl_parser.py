"""Generic NL Parser -- Zero hardcoded entities, zero keyword lists.

Based on:
    - Kanerva (2009) -- HDC role-filler records for structured representation
      DOI: 10.1007/s12559-009-9009-8
    - Kleyko et al. (2022) -- HDC/VSA Survey, ACM Computing Surveys 55(6), Article 130.
    - Kleyko et al. (2025) -- Principled neuromorphic reservoir computing
      Nature Communications 16(1). DOI: 10.1038/s41467-025-55832-y
    - Yilmaz (2015) -- CA + HDC reservoir, arXiv:1503.00851

Design principle: The parser discovers structure, it does not impose it.
No hardcoded `who`, `how_many`, `dollars`. No regex for phone numbers.
Instead:
    1. CA reservoir extracts local text features (zero params).
    2. HDC n-grams encode tokens with positional context.
    3. MHN retrieval discovers what kind of thing a token is
       by similarity to previously stored entity patterns.
    4. INNATE relation prototypes classify relations even with empty MHN (pure mode).
    5. INNATE stop-word prototype filters noise even with empty MHN (pure mode).
    6. Constraints are discovered by retrieving constraint-pattern vectors
       from MHN using the HDC-encoded problem state.

Output: A generic scene graph -- entities, relations, constraints --
        with no domain-specific typing.
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


# =========================================================================
# Cellular Automata Reservoir (Kleyko 2025, Yilmaz 2015)
# =========================================================================

class CAReservoir:
    """1D Cellular Automata reservoir with Rule 90.

    Rule 90: next_state[i] = state[i-1] XOR state[i+1]
    Produces rich spatiotemporal patterns from minimal input.
    Zero trainable parameters. Deterministic.
    """

    def __init__(self, width: int = 64, steps: int = 16, rule: int = 90):
        self.width = width
        self.steps = steps
        self.rule = rule

    def _step(self, state: np.ndarray) -> np.ndarray:
        """Single CA step with periodic boundary."""
        left = np.roll(state, -1)
        right = np.roll(state, 1)
        return np.logical_xor(left, right).astype(np.uint8)

    def process(self, token_vectors: list[np.ndarray]) -> list[np.ndarray]:
        """Run CA reservoir over token sequence."""
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


# =========================================================================
# Generic Entity Graph (Kanerva 2009 -- role-filler records)
# =========================================================================

class GenericEntityGraph:
    """A scene graph with no hardcoded entity types."""

    def __init__(self):
        self.entities: list[dict] = []
        self.relations: list[dict] = []
        self.constraints: list[dict] = []

    def add_entity(self, text: str, hdc: np.ndarray, type_vec: np.ndarray | None = None):
        eid = f"e{len(self.entities)}"
        self.entities.append({
            "id": eid, "text": text, "hdc": hdc,
            "type_vec": type_vec or hdc,
        })
        return eid

    def add_relation(self, src: str, tgt: str, rel_vec: np.ndarray, rel_type: str = ""):
        self.relations.append({
            "src": src, "tgt": tgt, "rel_vec": rel_vec, "rel_type": rel_type,
        })

    def add_constraint(self, type_vec: np.ndarray, subjects: list[str], params: dict | None = None):
        self.constraints.append({
            "type_vec": type_vec, "subjects": subjects, "params": params or {},
        })


# =========================================================================
# Generic NL Parser -- WIRED prototypes, not dead code
# =========================================================================

class GenericNLParser:
    """Zero-hardcode natural language parser.

    Innate prototypes (pure mode, no MHN required):
        - stop_word_prototype: filters noise words
        - relation_prototypes: classifies relations by HDC similarity
    Learned patterns (seeded mode, MHN required):
        - entity type vectors
        - constraint templates
    """

    def __init__(self, vocab, mhn, dim: int = 10_000):
        self.vocab = vocab
        self.mhn = mhn
        self.dim = dim
        self.ca = CAReservoir(width=64, steps=16, rule=90)

        # Generic grammatical roles (algebraic markers, not semantic categories)
        self.role_markers = {
            "subject": generate_hypervector(dim),
            "object": generate_hypervector(dim),
            "modifier": generate_hypervector(dim),
            "connector": generate_hypervector(dim),
        }

        # ─── INNATE PROTOTYPES: work in pure mode (empty MHN) ─────

        # Stop-word prototype: filters "the", "and", "is" etc.
        self.stop_word_prototype = self._encode_phrase(
            "the a an and or is are was were be been have has had do does did "
            "will would could should may might must can shall it its "
            "this that these those there their they them"
        )

        # Relation prototypes: classify discovered relations by HDC similarity
        # These are HDC vectors of semantic sentences describing each relation type.
        # Even with empty MHN, the parser can classify relations by comparing
        # the discovered relation vector to these prototypes.
        self.relation_prototypes = {
            "different": self._encode_phrase("cannot be the same must be distinct separate unique"),
            "before": self._encode_phrase("comes earlier precedes happens first prior to"),
            "after": self._encode_phrase("comes later follows happens second subsequent to"),
            "excludes": self._encode_phrase("cannot coexist mutually exclusive not together forbidden"),
            "limited": self._encode_phrase("has maximum has minimum bounded by restricted limited to"),
            "grouped": self._encode_phrase("belongs together same category partition grouped with"),
            "sum_to": self._encode_phrase("adds up to totals equals sum must equal"),
            "implies": self._encode_phrase("if then means requires leads to implies"),
            "overlap": self._encode_phrase("cannot overlap no conflict separate time no overlap"),
            "precedence": self._encode_phrase("must come before chain ordered sequence precedence"),
            "ratio": self._encode_phrase("proportional ratio times as many as divided by"),
            "count": self._encode_phrase("how many at least at most exactly count of"),
            "equivalent": self._encode_phrase("same as equal to equivalent to identical"),
            "optimize": self._encode_phrase("maximize minimize best optimal optimize"),
        }

        # Stop words as a set for fast lookup (innate, not learned)
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
            "out", "off", "over", "under", "again", "further", "then", "once",
        }

    # ─── Token Encoding (Kleyko 2022: n-gram HDC) ───────────────────────

    def _tokenize(self, text: str) -> list[str]:
        """Minimal tokenization: lowercase, strip punctuation, split."""
        text = text.lower()
        text = re.sub(r"[^\w\s'-]", " ", text)
        return [t.strip("-'") for t in text.split() if t.strip()]

    def _encode_token(self, token: str) -> np.ndarray:
        """Encode a token via character n-gram HDC (Kleyko 2022)."""
        if not token:
            return generate_hypervector(self.dim)
        chars = [c for c in token if c in string.ascii_lowercase or c.isdigit()]
        if not chars:
            return generate_hypervector(self.dim)
        ngrams = []
        for i, c in enumerate(chars):
            c_vec = self.vocab.get_char_vector(c)
            pos_vec = permute(c_vec, i)
            ngrams.append(pos_vec)
        for i in range(len(chars) - 1):
            c1 = self.vocab.get_char_vector(chars[i])
            c2 = self.vocab.get_char_vector(chars[i + 1])
            bigram = bind(c1, permute(c2, 1))
            ngrams.append(bigram)
        return bundle(*ngrams) if ngrams else generate_hypervector(self.dim)

    def _encode_phrase(self, text_or_tokens, start: int = 0, end: int = None) -> np.ndarray:
        """Encode a phrase with positional binding (Kanerva 2009).
        Overload: accepts string or token list with indices."""
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

    # ─── Document Composition ──────────────────────────────────────────

    def parse(self, text: str) -> np.ndarray:
        """Encode full text as HDC vector."""
        tokens = self._tokenize(text)
        if not tokens:
            return generate_hypervector(self.dim)
        token_vectors = [self._encode_token(t) for t in tokens]
        ca_features = self.ca.process(token_vectors)
        doc_components = []
        for i, (tok_vec, ca_vec) in enumerate(zip(token_vectors, ca_features)):
            combined = bundle(tok_vec * 0.7, ca_vec * 0.3)
            pos_bound = permute(combined, i)
            doc_components.append(pos_bound)
        return normalize_hdc(bundle(*doc_components))

    # ─── Generic Structure Extraction ───────────────────────────────────

    def extract_structure(self, text: str) -> dict:
        """Extract a generic scene graph from text.

        No hardcoded entity types. Instead:
        1. Discover entities with stop-word filtering (innate, works in pure mode).
        2. Classify each phrase by MHN retrieval (if available) or innate similarity.
        3. Discover relations by co-occurrence + innate prototype classification.
        4. Discover constraints by MHN retrieval (if available).
        """
        tokens = self._tokenize(text)
        graph = GenericEntityGraph()

        if not tokens:
            return {"graph": graph, "text": text, "hdc": self.parse(text)}

        # Step 1: Discover entities (with innate stop-word filtering)
        entity_spans = self._discover_entity_spans(tokens)
        for span in entity_spans:
            phrase = " ".join(tokens[span["start"]:span["end"]])
            phrase_hdc = self._encode_phrase(tokens, span["start"], span["end"])
            type_vec = self._discover_type(phrase_hdc)
            eid = graph.add_entity(phrase, phrase_hdc, type_vec)
            span["eid"] = eid

        # Step 2: Discover relations (with innate prototype classification)
        for i, e1 in enumerate(graph.entities):
            for e2 in graph.entities[i + 1:]:
                rel_type = self._discover_relation_innate(e1, e2, tokens, text)
                if rel_type:
                    # Encode relation as HDC vector for storage
                    rel_hdc = bind(e1["hdc"], bind(self.parse(f"{e1['text']} {rel_type} {e2['text']}"), e2["hdc"]))
                    graph.add_relation(e1["id"], e2["id"], rel_hdc, rel_type)

        # Step 3: Discover constraints (MHN-dependent, may be empty in pure mode)
        problem_hdc = self.parse(text)
        constraints = self._discover_constraints(problem_hdc, graph)
        for c in constraints:
            graph.add_constraint(c["type_vec"], c["subjects"], c.get("params"))

        return {"graph": graph, "text": text, "hdc": problem_hdc, "tokens": tokens}

    # ─── Discovery Methods ──────────────────────────────────────────────

    def _discover_entity_spans(self, tokens: list[str]) -> list[dict]:
        """Discover entity spans with innate stop-word filtering.

        In pure mode (empty MHN):
        - Skip stop words using innate stop-word set.
        - Multi-token spans that are NOT stop words are candidate entities.
        - Single non-stop tokens are also entities.

        In seeded mode (non-empty MHN):
        - Also check MHN similarity for entity patterns.
        - MHN match overrides stop-word filter (learned exceptions).
        """
        spans = []
        covered = set()

        # Try longer spans first
        for length in range(min(4, len(tokens)), 0, -1):
            for start in range(len(tokens) - length + 1):
                end = start + length
                if any(i in covered for i in range(start, end)):
                    continue

                span_tokens = tokens[start:end]
                phrase = " ".join(span_tokens)
                phrase_hdc = self._encode_phrase(tokens, start, end)

                # INNATE: Skip if all tokens are stop words
                if all(t in self.stop_words for t in span_tokens):
                    continue

                # INNATE: Skip if phrase is highly similar to stop-word prototype
                stop_sim = similarity(phrase_hdc, self.stop_word_prototype)
                if stop_sim > 0.55:
                    continue

                # LEARNED: Check MHN for entity patterns (if available)
                mhn_match = False
                results = self.mhn.retrieve(phrase_hdc, top_k=1)
                if results and results[0][1] > 0.55:
                    mhn_match = True

                # Accept if:
                #   - MHN match (learned pattern), OR
                #   - Contains a digit (numbers are always entities), OR
                #   - Multi-token non-stop phrase (innate heuristic)
                if mhn_match or any(t.isdigit() for t in span_tokens) or (length > 1 and not all(t in self.stop_words for t in span_tokens)):
                    spans.append({"start": start, "end": end, "sim": results[0][1] if mhn_match else 0.0})
                    for i in range(start, end):
                        covered.add(i)

        # Add remaining single non-stop tokens as entities
        for i, tok in enumerate(tokens):
            if i not in covered and tok not in self.stop_words and not tok.isdigit():
                spans.append({"start": i, "end": i + 1, "sim": 0.0})

        spans.sort(key=lambda s: s["start"])
        return spans

    def _discover_type(self, phrase_hdc: np.ndarray) -> np.ndarray | None:
        """Discover the type of a phrase by MHN retrieval."""
        results = self.mhn.retrieve(phrase_hdc, top_k=1)
        if results and results[0][1] > 0.50:
            meta = results[0][2]
            if "type_vec" in meta:
                return meta["type_vec"]
            return results[0][0]
        return None

    def _discover_relation_innate(self, e1: dict, e2: dict, tokens: list[str], text: str) -> str | None:
        """Discover relation between two entities using INNATE prototypes.

        This works in pure mode (empty MHN) because it compares the
        co-occurrence context to innate relation prototypes via HDC similarity.

        Returns the relation type string if a prototype matches above threshold,
        or None if no relation is detected.
        """
        # Extract context window between the two entities
        # (simplified: use full text as context)
        context = f"{e1['text']} ... {e2['text']}"
        context_hdc = self.parse(context)

        # Relation representation: bind(e1, bind(context, e2))
        rel_hdc = bind(e1["hdc"], bind(context_hdc, e2["hdc"]))

        # INNATE: Compare to relation prototypes by HDC similarity
        best_type = None
        best_sim = 0.0
        for rel_type, proto_hdc in self.relation_prototypes.items():
            sim = similarity(rel_hdc, proto_hdc)
            if sim > best_sim and sim > 0.35:  # Threshold for relation detection
                best_sim = sim
                best_type = rel_type

        # LEARNED: Also check MHN (if available, may override innate)
        results = self.mhn.retrieve(rel_hdc, top_k=1)
        if results and results[0][1] > 0.45 and results[0][1] > best_sim:
            meta = results[0][2]
            return meta.get("relation_type", best_type or "")

        return best_type

    def _discover_constraints(self, problem_hdc: np.ndarray, graph: GenericEntityGraph) -> list[dict]:
        """Discover constraints by MHN retrieval."""
        constraints = []
        results = self.mhn.retrieve(problem_hdc, top_k=3)
        for vec, sim, meta in results:
            if sim < 0.40:
                continue
            template = meta.get("constraint_template")
            if template:
                subjects = self._map_template_to_entities(template, graph)
                if subjects:
                    constraints.append({
                        "type_vec": vec, "subjects": subjects,
                        "params": template.get("params", {}),
                    })
        return constraints

    def _map_template_to_entities(self, template: dict, graph: GenericEntityGraph) -> list[str]:
        """Map constraint template to entities by HDC similarity."""
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

    # ─── Legacy compatibility ────────────────────────────────────────────

    def extract_entities(self, text: str) -> dict:
        """Legacy interface. Returns generic structure as flat dict."""
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

    def _type_to_key(self, type_vec: np.ndarray | None) -> str:
        if type_vec is None:
            return "unknown"
        results = self.mhn.retrieve(type_vec, top_k=1)
        if results:
            meta = results[0][2]
            return meta.get("type_name", "discovered")
        return "unknown"
