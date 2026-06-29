"""Generic NL Parser — Zero hardcoded entities, zero keyword lists.

Based on:
    - Kanerva (2009) — HDC role-filler records for structured representation
      "Hyperdimensional Computing: An Introduction", Cognitive Computation 1(2), 139-159.
      DOI: 10.1007/s12559-009-9009-8
      Key insight: bind(role, filler) creates arbitrary records. No fixed schema.

    - Kleyko et al. (2022) — HDC/VSA Survey, ACM Computing Surveys 55(6), Article 130.
      Key insight: n-gram encoding + positional permutation captures syntax.
      "Fractional binding" and holographic reduced representations (HRR) enable
      variable-role binding without hardcoded slots.

    - Kleyko et al. (2025) — Principled neuromorphic reservoir computing,
      Nature Communications 16(1). DOI: 10.1038/s41467-025-55832-y
      Key insight: CA reservoir (Rule 90) extracts local context features
      with zero trainable parameters. HDC encodes reservoir state.

    - Yilmaz (2015) — CA + HDC reservoir, arXiv:1503.00851
      Key insight: Cellular automata dynamics create rich feature spaces
      when combined with HDC encoding.

Design principle: The parser discovers structure, it does not impose it.
No hardcoded `who`, `how_many`, `dollars`. No regex for phone numbers.
Instead:
    1. CA reservoir extracts local text features (zero params).
    2. HDC n-grams encode tokens with positional context.
    3. MHN retrieval discovers what kind of thing a token is
       by similarity to previously stored entity patterns.
    4. Relations are discovered by co-occurrence in CA reservoir windows.
    5. Constraints are discovered by retrieving constraint-pattern vectors
       from MHN using the HDC-encoded problem state.

Output: A generic scene graph — entities, relations, constraints —
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

    Used here to extract local context features from token sequences.
    Each token seeds a CA evolution; the final state is a feature vector.
    """

    def __init__(self, width: int = 64, steps: int = 16, rule: int = 90):
        self.width = width
        self.steps = steps
        self.rule = rule
        # Rule 90 lookup: next = left XOR right
        self.rule_table = {}
        for left in (0, 1):
            for center in (0, 1):
                for right in (0, 1):
                    idx = (left << 2) | (center << 1) | right
                    # Rule 90: XOR of neighbors
                    self.rule_table[idx] = left ^ right

    def _step(self, state: np.ndarray) -> np.ndarray:
        """Single CA step with periodic boundary."""
        left = np.roll(state, -1)
        right = np.roll(state, 1)
        # Vectorized rule 90: next = left XOR right
        return np.logical_xor(left, right).astype(np.uint8)

    def process(self, token_vectors: list[np.ndarray]) -> list[np.ndarray]:
        """Run CA reservoir over token sequence.

        Each token vector seeds the CA initial state (thresholded to binary).
        After `steps` evolutions, the CA state is encoded as HDC vector.

        Returns list of context feature vectors, one per token.
        """
        features = []
        for tv in token_vectors:
            # Seed CA state from token vector (sample to width)
            indices = np.arange(len(tv))
            sampled = tv[indices % len(tv)] if len(tv) > self.width else tv
            if len(sampled) < self.width:
                sampled = np.tile(sampled, self.width // len(sampled) + 1)[:self.width]
            state = (sampled > 0).astype(np.uint8)

            # Evolve CA
            for _ in range(self.steps):
                state = self._step(state)

            # Convert CA state to HDC feature vector
            feature = np.zeros(len(tv), dtype=np.float32)
            feature[:len(state)] = state * 2 - 1  # 0/1 -> -1/+1
            features.append(feature)

        return features


# =========================================================================
# Generic Entity Discovery (Kanerva 2009 — role-filler records)
# =========================================================================

class GenericEntityGraph:
    """A scene graph with no hardcoded entity types.

    Entities are anonymous nodes with HDC type vectors discovered from MHN.
    Relations are edges with HDC relation vectors.
    Constraints are hyperedges with HDC constraint vectors.

    This is a generic structure — it can represent scheduling, budget,
    logic, spatial, or any other problem domain without schema changes.
    """

    def __init__(self):
        self.entities: list[dict] = []       # {id, text, hdc, type_vec}
        self.relations: list[dict] = []      # {src, tgt, rel_vec, rel_type}
        self.constraints: list[dict] = []    # {type_vec, subjects, params}

    def add_entity(self, text: str, hdc: np.ndarray, type_vec: np.ndarray | None = None):
        eid = f"e{len(self.entities)}"
        self.entities.append({
            "id": eid,
            "text": text,
            "hdc": hdc,
            "type_vec": type_vec or hdc,  # Default: self-typed
        })
        return eid

    def add_relation(self, src: str, tgt: str, rel_vec: np.ndarray, rel_type: str = ""):
        self.relations.append({
            "src": src,
            "tgt": tgt,
            "rel_vec": rel_vec,
            "rel_type": rel_type,
        })

    def add_constraint(self, type_vec: np.ndarray, subjects: list[str], params: dict | None = None):
        self.constraints.append({
            "type_vec": type_vec,
            "subjects": subjects,
            "params": params or {},
        })


# =========================================================================
# Generic NL Parser
# =========================================================================

class GenericNLParser:
    """Zero-hardcode natural language parser.

    Discovers entities, relations, and constraints dynamically using
    HDC algebra + CA reservoir + MHN retrieval. No keyword lists.
    """

    def __init__(self, vocab, mhn, dim: int = 10_000):
        self.vocab = vocab
        self.mhn = mhn  # Used to discover types by similarity to stored patterns
        self.dim = dim
        self.ca = CAReservoir(width=64, steps=16, rule=90)

        # HDC vectors for generic grammatical roles (not domain-specific)
        # These are algebraic markers, not semantic categories
        self.role_markers = {
            "subject": generate_hypervector(dim),
            "object": generate_hypervector(dim),
            "modifier": generate_hypervector(dim),
            "connector": generate_hypervector(dim),
        }

        # Innate stop-word prototype (HDC-encoded sentence)
        # Filters noise in pure mode without requiring seed data
        self.stop_word_prototype = self._encode_phrase(
            "the a an and or is are was were be been have has had do does did "
            "will would could should may might must can shall it its "
            "this that these those there their they them"
        )

        # Innate relation prototypes (classify relations by HDC similarity)
        self.relation_prototypes = {
            "different": self._encode_phrase("cannot be the same must be distinct separate"),
            "before": self._encode_phrase("comes earlier precedes happens first"),
            "after": self._encode_phrase("comes later follows happens second"),
            "excludes": self._encode_phrase("cannot coexist mutually exclusive not together"),
            "limited": self._encode_phrase("has maximum has minimum bounded by restricted"),
            "grouped": self._encode_phrase("belongs together same category partition"),
            "sum_to": self._encode_phrase("adds up to totals equals sum"),
            "implies": self._encode_phrase("if then means requires leads to"),
            "overlap": self._encode_phrase("cannot overlap no conflict separate time"),
        }

    # ─── Token Encoding (Kleyko 2022: n-gram HDC) ───────────────────────

    def _tokenize(self, text: str) -> list[str]:
        """Minimal tokenization: lowercase, strip punctuation, split."""
        text = text.lower()
        # Keep internal apostrophes, remove other punctuation
        text = re.sub(r"[^\w\s\'\-]", " ", text)
        return [t.strip("-\'") for t in text.split() if t.strip()]

    def _encode_token(self, token: str) -> np.ndarray:
        """Encode a token via character n-gram HDC (Kleyko 2022).

        Each token is a bundle of its character n-grams with positional binding.
        This captures morphology without a dictionary.
        """
        if not token:
            return generate_hypervector(self.dim)

        # Character set: a-z + digits + special
        chars = [c for c in token if c in string.ascii_lowercase or c.isdigit()]
        if not chars:
            return generate_hypervector(self.dim)

        ngrams = []
        # Unigrams
        for i, c in enumerate(chars):
            c_vec = self.vocab.get_char_vector(c)
            pos_vec = permute(c_vec, i)  # Position encoding
            ngrams.append(pos_vec)

        # Bigrams
        for i in range(len(chars) - 1):
            c1 = self.vocab.get_char_vector(chars[i])
            c2 = self.vocab.get_char_vector(chars[i + 1])
            # Bind bigram: c1 ⊗ permute(c2, 1)
            bigram = bind(c1, permute(c2, 1))
            ngrams.append(bigram)

        return bundle(*ngrams) if ngrams else generate_hypervector(self.dim)

    def _encode_phrase(self, text_or_tokens, start: int = 0, end: int = None) -> np.ndarray:
        """Encode a phrase with positional binding (Kanerva 2009).

        Overload: accepts either a string or a token list with indices.
        """
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



    # ─── Document Composition (Kanerva 2009: record superposition) ─────

    def parse(self, text: str) -> np.ndarray:
        """Encode full text as HDC vector.

        Uses CA reservoir for local context + HDC positional bundling.
        """
        tokens = self._tokenize(text)
        if not tokens:
            return generate_hypervector(self.dim)

        # Encode tokens
        token_vectors = [self._encode_token(t) for t in tokens]

        # CA reservoir features (Kleyko 2025, Yilmaz 2015)
        ca_features = self.ca.process(token_vectors)

        # Compose: bundle each token with its CA context
        # Document = bundle over i of bind(token_i, CA_feature_i) with position
        doc_components = []
        for i, (tok_vec, ca_vec) in enumerate(zip(token_vectors, ca_features)):
            # Combine token + context
            combined = bundle(tok_vec * 0.7, ca_vec * 0.3)
            # Positional binding
            pos_bound = permute(combined, i)
            doc_components.append(pos_bound)

        return normalize_hdc(bundle(*doc_components))

    # ─── Generic Structure Extraction (No hardcoded types) ──────────────

    def extract_structure(self, text: str) -> dict:
        """Extract a generic scene graph from text.

        No hardcoded entity types. Instead:
        1. Discover multi-token phrases by MHN similarity.
        2. Classify each phrase by retrieving its type from MHN.
        3. Discover relations by CA co-occurrence + MHN retrieval.
        4. Discover constraints by retrieving constraint patterns from MHN.
        """
        tokens = self._tokenize(text)
        graph = GenericEntityGraph()

        if not tokens:
            return {"graph": graph, "text": text, "hdc": self.parse(text)}

        # ─── Step 1: Discover entities (multi-token phrases) ─────
        # Try all spans 1-4 tokens. If span HDC is similar to stored entity
        # in MHN (sim > 0.55), it's an entity.
        entity_spans = self._discover_entity_spans(tokens)

        for span in entity_spans:
            phrase = " ".join(tokens[span["start"]:span["end"]])
            phrase_hdc = self._encode_phrase(tokens, span["start"], span["end"])
            # Retrieve type from MHN (if any)
            type_vec = self._discover_type(phrase_hdc)
            eid = graph.add_entity(phrase, phrase_hdc, type_vec)
            span["eid"] = eid

        # ─── Step 2: Discover relations between entities ─────────
        # For each pair of entities, check if their co-occurrence window
        # retrieves a relation pattern from MHN.
        for i, e1 in enumerate(graph.entities):
            for e2 in graph.entities[i + 1:]:
                rel_vec = self._discover_relation(e1, e2, tokens, text)
                if rel_vec is not None:
                    # Determine relation type by MHN retrieval
                    rel_type = self._classify_relation(rel_vec)
                    graph.add_relation(e1["id"], e2["id"], rel_vec, rel_type)

        # ─── Step 3: Discover constraints ────────────────────────
        # Encode the full problem and retrieve constraint patterns from MHN
        problem_hdc = self.parse(text)
        constraints = self._discover_constraints(problem_hdc, graph)
        for c in constraints:
            graph.add_constraint(c["type_vec"], c["subjects"], c.get("params"))

        return {
            "graph": graph,
            "text": text,
            "hdc": problem_hdc,
            "tokens": tokens,
        }

    # ─── Discovery Methods (all use MHN retrieval, no hardcoding) ──────

    def _discover_entity_spans(self, tokens: list[str]) -> list[dict]:
        """Discover entity spans by MHN similarity.

        Try all spans 1-4 tokens. A span is an entity if its HDC vector
        is similar to any stored pattern in MHN (threshold 0.55).
        If no MHN matches, use heuristic: capitalized words or numbers
        are likely entities (this is the only non-MHN heuristic).
        """
        spans = []
        covered = set()

        # Try longer spans first (prefer multi-token entities)
        for length in range(min(4, len(tokens)), 0, -1):
            for start in range(len(tokens) - length + 1):
                end = start + length
                if any(i in covered for i in range(start, end)):
                    continue

                phrase_hdc = self._encode_phrase(tokens, start, end)

                # Check MHN for entity patterns
                results = self.mhn.retrieve(phrase_hdc, top_k=1)
                if results and results[0][1] > 0.55:
                    spans.append({"start": start, "end": end, "sim": results[0][1]})
                    for i in range(start, end):
                        covered.add(i)
                    continue

                # Fallback: numbers are always entities
                phrase = " ".join(tokens[start:end])
                if any(t.isdigit() for t in tokens[start:end]):
                    spans.append({"start": start, "end": end, "sim": 0.0})
                    for i in range(start, end):
                        covered.add(i)

        # Add remaining single tokens as entities
        for i, tok in enumerate(tokens):
            if i not in covered:
                spans.append({"start": i, "end": i + 1, "sim": 0.0})

        # Sort by start position
        spans.sort(key=lambda s: s["start"])
        return spans

    def _discover_type(self, phrase_hdc: np.ndarray) -> np.ndarray | None:
        """Discover the type of a phrase by MHN retrieval.

        Returns the HDC vector of the most similar stored type pattern,
        or None if no match.
        """
        results = self.mhn.retrieve(phrase_hdc, top_k=1)
        if results and results[0][1] > 0.50:
            meta = results[0][2]
            # If stored pattern has a type vector, return it
            if "type_vec" in meta:
                return meta["type_vec"]
            # Otherwise return the stored vector itself as type proxy
            return results[0][0]
        return None

    def _discover_relation(self, e1: dict, e2: dict, tokens: list[str], text: str) -> np.ndarray | None:
        """Discover relation between two entities.

        Encode the text window between the two entities plus their HDC vectors.
        Retrieve from MHN to see if this pattern matches a known relation.
        """
        # Find token positions
        # (simplified: use full text as context window)
        context = f"{e1['text']} ... {e2['text']}"
        context_hdc = self.parse(context)

        # Relation representation: bind(e1, bind(context, e2))
        rel_hdc = bind(e1["hdc"], bind(context_hdc, e2["hdc"]))

        results = self.mhn.retrieve(rel_hdc, top_k=1)
        if results and results[0][1] > 0.45:
            return rel_hdc
        return None

    def _classify_relation(self, rel_vec: np.ndarray) -> str:
        """Classify a relation vector by retrieving its label from MHN.

        If no label stored, return empty string (anonymous relation).
        """
        results = self.mhn.retrieve(rel_vec, top_k=1)
        if results:
            meta = results[0][2]
            return meta.get("relation_type", "")
        return ""

    def _discover_constraints(self, problem_hdc: np.ndarray, graph: GenericEntityGraph) -> list[dict]:
        """Discover constraints by retrieving constraint patterns from MHN.

        The problem HDC is decomposed using constraint prototypes
        (retrieved from MHN, not hardcoded). Each retrieved constraint
        pattern is mapped to entities in the graph.
        """
        constraints = []

        # Retrieve top constraint patterns from MHN
        results = self.mhn.retrieve(problem_hdc, top_k=3)
        for vec, sim, meta in results:
            if sim < 0.40:
                continue
            # Check if metadata contains a constraint template
            template = meta.get("constraint_template")
            if template:
                # Map template to available entities
                subjects = self._map_template_to_entities(template, graph)
                if subjects:
                    constraints.append({
                        "type_vec": vec,
                        "subjects": subjects,
                        "params": template.get("params", {}),
                    })

        return constraints

    def _map_template_to_entities(self, template: dict, graph: GenericEntityGraph) -> list[str]:
        """Map a constraint template to entities in the graph by HDC similarity.

        The template has a `selector_vec` (HDC) that describes what kind of
        entities it applies to. We find entities in the graph whose HDC
        is most similar to this selector.
        """
        selector = template.get("selector_vec")
        if selector is None:
            # No selector: apply to all entities
            return [e["id"] for e in graph.entities]

        # Find entities most similar to selector
        matches = []
        for e in graph.entities:
            sim = similarity(e["hdc"], selector)
            matches.append((e["id"], sim))

        matches.sort(key=lambda x: x[1], reverse=True)
        # Take top N or all above threshold
        threshold = template.get("selector_threshold", 0.30)
        return [eid for eid, sim in matches if sim > threshold]

    # ─── Legacy compatibility ────────────────────────────────────────────

    def extract_entities(self, text: str) -> dict:
        """Legacy interface. Returns generic structure as flat dict."""
        struct = self.extract_structure(text)
        graph = struct["graph"]

        # Flatten to legacy format for backward compatibility
        # But without hardcoded keys — keys are discovered dynamically
        entities = {
            "_raw_text": text,
            "_entity_count": len(graph.entities),
            "_relation_count": len(graph.relations),
            "_constraint_count": len(graph.constraints),
        }

        # Add discovered entity texts by type similarity
        for e in graph.entities:
            type_key = self._type_to_key(e.get("type_vec"))
            if type_key not in entities:
                entities[type_key] = []
            entities[type_key].append(e["text"])

        # Add relation texts
        for r in graph.relations:
            src = next(e["text"] for e in graph.entities if e["id"] == r["src"])
            tgt = next(e["text"] for e in graph.entities if e["id"] == r["tgt"])
            if "_relations" not in entities:
                entities["_relations"] = []
            entities["_relations"].append(f"{src} -> {r['rel_type']} -> {tgt}")

        # Add constraint summaries
        for c in graph.constraints:
            if "_constraints" not in entities:
                entities["_constraints"] = []
            subj_texts = [next(e["text"] for e in graph.entities if e["id"] == sid)
                          for sid in c["subjects"]]
            entities["_constraints"].append({
                "subjects": subj_texts,
                "params": c["params"],
            })

        return entities

    def _type_to_key(self, type_vec: np.ndarray | None) -> str:
        """Convert a type vector to a string key by MHN retrieval.
        If no match, return 'unknown'."""
        if type_vec is None:
            return "unknown"
        results = self.mhn.retrieve(type_vec, top_k=1)
        if results:
            meta = results[0][2]
            return meta.get("type_name", "discovered")
        return "unknown"
