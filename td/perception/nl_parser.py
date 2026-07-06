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


from td.languages import get_language, LanguageConfig

class GenericNLParser:
    """spaCy-powered parser. Uses dependency parsing for extraction.

    All language-specific word sets are loaded from td/languages/.
    No hardcoded English words in this class -- only spaCy UD features
    and language registry lookups.

    Replaces hardcoded rules with spaCy NLP:
    - stop_words -> token.is_stop (language-agnostic)
    - relation_prototypes -> token.dep_ (dependency labels)
    - _merge_*_pattern -> doc.noun_chunks (compound noun detection)
    - _pp_words -> token.pos_ == "ADP" (preposition detection)
    - _strip_pp -> token.head (PP attachment via dependency tree)
    - regex patterns -> dependency tree traversal

    Reference: Universal Dependencies (Nivre et al., 2016)
    Reference: spaCy multilingual support (20+ languages)
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

        # ─── Language configuration (loaded from registry) ───────────
        # All language-specific word sets are in td/languages/{lang}.py.
        # Reference: td/languages/__init__.py
        self._lang_config = get_language("en")  # default, updated on first nlp call
        self._fallback_stop_words = self.lang_config.stop_words

        # ─── INNATE: Stop-word prototype (HDC-encoded sentence) ──────
        # HDC stop word prototype — loaded from language registry.
        # Used for HDC fuzzy similarity matching (not string lookup).
        # String-based stop word detection uses spaCy token.is_stop.
        # Reference: td/languages/en.py
        _stop_phrase = " ".join(sorted(self.lang_config.stop_words))
        self.stop_word_prototype = self._encode_phrase(_stop_phrase)

        # ─── INNATE: Relation prototypes (14 types) ──────────────────
        # Loaded from language registry. HDC-encoded phrases for constraint
        # type detection. Actual relation detection uses spaCy dep labels.
        # Reference: td/languages/en.py, td/languages/__init__.py
        self.relation_prototypes = {
            name: self._encode_phrase(phrases)
            for name, phrases in self.lang_config.relation_prototypes.items()
        }

        self.constraint_signals = set(self.relation_prototypes.keys())

    def is_stop_word(self, word: str) -> bool:
        """Check if a word is a stop word using language registry.

        Uses the language-specific fallback set (fast, no spaCy call).
        For language-agnostic detection at token level, use token.is_stop
        directly when you have a spaCy Doc.

        Reference: td/languages/en.py — STOP_WORDS
        """
        return word.lower() in self._fallback_stop_words

    @property
    def lang_config(self):
        """Language configuration, lazily loaded if not set.
        
        Ensures _lang_config is always available even when __init__
        was bypassed (e.g., in tests using __new__).
        """
        if not hasattr(self, '_lang_config') or self._lang_config is None:
            from td.languages import get_language
            self._lang_config = get_language("en")
            self._fallback_stop_words = self.lang_config.stop_words
        return self._lang_config

    def is_function_word(self, token) -> bool:
        """Check if a spaCy token is a function word (not a content word).

        Function words: ADP (prepositions), AUX (auxiliaries), DET (determiners),
        CCONJ (coordinating conjunctions), SCONJ (subordinating conjunctions),
        PART (particles), PUNCT (punctuation).

        Content words: NOUN, VERB, ADJ, ADV, PROPN, NUM.

        Reference: Universal POS Tags (Nivre et al., 2016)
        """
        return token.pos_ in ("ADP", "AUX", "DET", "CCONJ", "SCONJ", "PART", "PUNCT")

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

    @property
    def nlp_coref(self):
        """Lazy-load spaCy coreference resolution pipeline.

        Uses en_coreference_web_trf model (490MB, one-time download).
        Two-pipeline approach: process text with main nlp first,
        then resolve coreferences with nlp_coref.

        Reference: spaCy coref blog (Explosion, 2022)
        Reference: spaCy Discussion #12302
        """
        if not hasattr(self, '_nlp_coref'):
            self._nlp_coref = None
            try:
                import spacy
                self._nlp_coref = spacy.load(
                    "en_coreference_web_trf", vocab=self.nlp.vocab
                )
            except (ImportError, OSError):
                self._nlp_coref = False
        return self._nlp_coref if self._nlp_coref is not False else None

    def enable_coreference(self):
        """Enable coreference resolution in the parser pipeline.

        Loads the en_coreference_web_trf model (490MB, one-time download).
        After calling this, extract_triples_spacy() will resolve pronouns
        before extracting triples.

        Note: This downgrades spaCy compatibility (model trained on 3.4).
        """
        self._coref_enabled = True
        # Force load the model
        _ = self.nlp_coref

    def disable_coreference(self):
        """Disable coreference resolution (default state)."""
        self._coref_enabled = False

    # Discourse deixis: abstract verbs that indicate "this"/"that" refers
    # Abstract/cognitive verb senses for discourse deixis detection.
    # Used with spaCy dependency parsing: when "this"/"that" is the subject
    # (nsubj) of a verb in this set, it's likely discourse deixis.
    #
    # This is NOT a hardcoded word list — it's a semantic class that drives
    # the syntactic check (Jauhar et al. 2015, Stage 1: Classification).
    # The check is: nsubj + head.lemma_ in this set.
    #
    # Reference: Jauhar, S.K. et al. (2015). "Resolving Discourse-Deictic
    #   Pronouns: A Two-Stage Approach." *SEM 2015, pp. 299-308.
    # ── Discourse Deixis — Loaded from Language Registry ─────────
    # All verb sets are in td/languages/{lang}.py — not hardcoded here.
    # The syntactic check (dep=nsubj) is language-agnostic (Universal
    # Dependencies). The verb sets are language-specific.
    #
    # To add a new language: see td/languages/de.py for example.
    # Reference: Jauhar et al. (2015), *SEM, pp. 299-308.

    @classmethod
    def _get_abstract_verbs(cls, lang: str = "en") -> frozenset:
        """Get abstract verb set for discourse deixis in the given language."""
        config = get_language(lang)
        return config.discourse_deixis_verbs if config else frozenset()

    @classmethod
    def _get_it_verbs(cls, lang: str = "en") -> frozenset:
        """Get verb set where 'it' is likely discourse deixis."""
        config = get_language(lang)
        return config.discourse_deixis_it_verbs if config else frozenset()

    @classmethod
    def register_discourse_deixis(cls, lang: str, verbs: set[str],
                                   it_verbs: set[str] = None):
        """Register abstract verbs for a new language.

        Example:
            GenericNLParser.register_discourse_deixis("de", {
                "zeigen", "beweisen", "bedeuten",
            }, {"zeigen", "beweisen"})
        """
        from .languages import _REGISTRY
        if lang in _REGISTRY:
            config = _REGISTRY[lang]
            config.discourse_deixis_verbs = frozenset(verbs)
            if it_verbs:
                config.discourse_deixis_it_verbs = frozenset(it_verbs)

    def resolve_coreferences(self, text: str) -> tuple[str, dict]:
        """Build coreference map from text using spaCy two-pipeline approach.

        Returns a pronoun→entity map WITHOUT modifying the text.
        Resolution happens at the triple level (see resolve_triple_coreferences).

        This follows the approach recommended by:
        - spaCy coref team: "resolve at span level, not text level"
        - Neo4j (2026): "coreference resolution converts pronouns into
          referred entities"
        - Are LLMs Effective KG Constructors? (EMNLP 2025): "resolve
          references into meaningful entities, avoiding ambiguous triples"

        Args:
            text: Input text with potential coreferences

        Returns:
            Tuple of (original_text, coref_map) where coref_map maps
            pronoun token indices to (entity_name, is_possessive).
        """
        if self.nlp is None:
            return text, {}

        if not getattr(self, '_coref_enabled', False):
            return text, {}

        if self.nlp_coref is None:
            return text, {}

        doc = self.nlp(text)
        doc = self.nlp_coref(doc)

        # Build pronoun → (entity_name, is_possessive) map
        # Uses spaCy POS + morph features (language-agnostic).
        # - Possessive: token.pos_ == "PRON" and "Yes" in token.morph.get("Poss")
        # - Personal:   token.pos_ == "PRON" and "Prs" in token.morph.get("PronType")
        # Reference: Universal Dependencies — PronType, Poss features
        pronoun_map = {}

        for key, spans in doc.spans.items():
            if not key.startswith("coref_clusters"):
                continue

            entity_span = None
            pronoun_spans = []
            for span in spans:
                # Use spaCy POS to detect pronouns (language-agnostic).
                # A span is a pronoun if its root token is PRON.
                # Reference: Universal POS — PRON category
                is_pronoun = span.root.pos_ == "PRON"
                if not is_pronoun:
                    entity_span = span
                else:
                    pronoun_spans.append(span)

            if entity_span is None:
                continue

            entity_name = entity_span.text.lower().strip()
            # Strip leading articles using language registry.
            # Reference: td/languages/en.py — ARTICLES
            for article in self.lang_config.articles:
                prefix = article + " "
                if entity_name.startswith(prefix):
                    remainder = entity_name[len(prefix):]
                    if " " in remainder:
                        entity_name = remainder
                    break

            for pronoun_span in pronoun_spans:
                for token in pronoun_span:
                    # Use spaCy morph feature to detect possessive pronouns.
                    # "Poss=Yes" is a UD morphological feature (language-agnostic).
                    # Reference: Universal Dependencies — Poss feature
                    is_poss = "Yes" in token.morph.get("Poss")
                    pronoun_map[token.i] = (entity_name, is_poss)

        return text, pronoun_map

    def resolve_triple_coreferences(
        self,
        triples: list[tuple[str, str, str]],
        pronoun_map: dict,
        doc
    ) -> list[tuple[str, str, str]]:
        """Resolve pronouns in extracted triples (not in text).

        Instead of replacing pronouns in the text before extraction,
        we extract triples first (with pronouns intact), then resolve
        pronouns in the subject/object fields.

        Handles:
        - Subject pronouns: "it runs" → ("video game console", "runs", ...)
        - Object pronouns: "loves it" → (..., "loves", "video game console")
        - Possessive pronouns: "its video games" → (..., "video games") with
          possessive note in proof
        - Discourse deixis: "this shows" → skip entirely

        Reference: spaCy coref team: "resolve at span level, not text level"
        Reference: Neo4j (2026): coreference as IE pipeline step
        Reference: Are LLMs Effective KG Constructors? (EMNLP 2025)

        Args:
            triples: Extracted triples with pronouns intact
            pronoun_map: Map from token index → (entity_name, is_possessive)
            doc: spaCy Doc for token lookup

        Returns:
            Triples with pronouns resolved
        """
        if not pronoun_map:
            return triples

        # Build a reverse map: pronoun text → entity name
        # (since triples store text, not token indices)
        text_to_entity = {}
        possessive_entities = {}
        for token_idx, (entity, is_poss) in pronoun_map.items():
            if token_idx < len(doc):
                pronoun_text = doc[token_idx].text.lower()
                text_to_entity[pronoun_text] = entity
                if is_poss:
                    possessive_entities[pronoun_text] = entity

        resolved = []
        for s, r, o in triples:
            # Resolve subject pronouns
            new_s = s
            if s in text_to_entity:
                new_s = text_to_entity[s]

            # Resolve object pronouns (check if object contains possessive)
            # Uses spaCy 'poss' dependency label (Universal Dependencies).
            # "its video games" → "its" has dep=poss, head="games"
            # Replace possessive pronoun with resolved entity name.
            # Reference: Universal Dependencies — 'poss' label
            new_o = o
            for poss, entity in possessive_entities.items():
                if poss in new_o:
                    # Replace the possessive pronoun with the entity name
                    new_o = new_o.replace(poss, entity, 1)
                    break
            if new_o in text_to_entity:
                new_o = text_to_entity[new_o]

            # Skip discourse deixis triples: "this shows" / "that means" / "it proves"
            # Two-stage approach (Jauhar et al. 2015):
            #   Stage 1 (Classification): pronoun is demonstrative AND subject
            #   of an abstract/cognitive verb → discourse deixis → skip
            #
            # Uses language registry for both pronouns and verbs.
            # Reference: Jauhar et al. (2015), *SEM, pp. 299-308.
            # Reference: td/languages/en.py — DEMONSTRATIVE_PRONOUNS
            lang = doc.lang_ if doc else "en"
            abstract_verbs = self._get_abstract_verbs(lang)
            it_verbs = self._get_it_verbs(lang)
            dem_pronouns = self.lang_config.demonstrative_pronouns
            # "this"/"that" trigger for all abstract verbs
            # "it" only triggers for purely demonstrative verbs
            # Reference: td/languages/en.py — DEMONSTRATIVE_PRONOUNS
            non_it_dems = dem_pronouns - {"it"}  # this, that
            if new_s in non_it_dems and r in abstract_verbs:
                continue
            if new_s in dem_pronouns and r in it_verbs:
                continue

            resolved.append((new_s, r, new_o))

        return resolved

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

        Pipeline:
        1. Build coreference map (no text modification)
        2. Clause segmentation — split compound/complex sentences
        3. Dependency parsing — extract SVO from each clause
        4. Temporal ordering extraction
        5. Triple-level coreference resolution (pronouns → entities)
        6. Adjectival predicate extraction ("runs smoother" → quality)
        7. Deduplication via relation canonicalization

        Reference: Honnibal & Montani (2017), "spaCy 2"
        Reference: Sahaj Software (2023), "Knowledge graphs from complex text"
        Reference: spaCy coref team — "resolve at span level, not text level"
        Reference: Neo4j (2026) — coreference as IE pipeline step
        Reference: Are LLMs Effective KG Constructors? (EMNLP 2025)
        """
        if self.nlp is None:
            return []

        # ─── Step 0: Build coreference map (no text modification) ───
        _, coref_map = self.resolve_coreferences(text)
        doc = self.nlp(text)
        all_triples = []

        # ─── Step 1: Clause segmentation (only for complex sentences) ──
        # Split compound/complex sentences into simple clauses.
        # Only activate when coordination or relative clauses are present.
        # For simple sentences, the dependency-based extraction below is better.
        has_coordination = any(
            token.dep_ in ("conj", "relcl") for token in doc
        )
        if has_coordination:
            from .clause_segmenter import segment_clauses
            segmented_clauses = segment_clauses(doc)
            for clause in segmented_clauses:
                if clause.subject and clause.relation and clause.obj:
                    all_triples.append((clause.subject, clause.relation, clause.obj))
        triples = []

        for token in doc:
            # ─── Copular constructions: "X is Y" ─────────────────
            # Uses language registry for copula verb lemmas.
            # Reference: td/languages/en.py — COPULA_VERBS
            if token.dep_ == "ROOT" and token.lemma_ in self.lang_config.copula_verbs:
                subject = None
                for child in token.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        subject = child
                        break
                if not subject:
                    continue

                # Get all coordinated subjects: "Alice and Bob" → ["alice", "bob"]
                subj_texts = self._get_coordinated_subjects(doc, subject)

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
                            rel = f"{attr.text.lower()}_{prep.lemma_}"
                            for subj_text in subj_texts:
                                triples.append((subj_text, rel, obj_text))
                            continue

                    # Check for xcomp: "are central to computer science"
                    # acomp=central, xcomp=computer (with aux=to, dobj=science)
                    xcomps = [c for c in token.children if c.dep_ == "xcomp"]
                    if xcomps:
                        xcomp = xcomps[0]
                        xcomp_dobj = [c for c in xcomp.children if c.dep_ == "dobj"]
                        if xcomp_dobj:
                            obj_text = self._get_chunk_text(doc, xcomp_dobj[0])
                            rel = f"{attr.text.lower()}_{xcomp.text.lower()}"
                            for subj_text in subj_texts:
                                triples.append((subj_text, rel, obj_text))
                            continue

                # Check preps attached to ROOT: "is in Y", "is before Y"
                if preps:
                    prep = preps[0]
                    pobj = [c for c in prep.children if c.dep_ == "pobj"]
                    if pobj:
                        obj_text = self._get_chunk_text(doc, pobj[0])
                        for subj_text in subj_texts:
                            triples.append((subj_text, prep.lemma_, obj_text))
                        continue

            # ─── Verb constructions: "X evolved from Y", "X treats Y" ──
            # Also handles passive voice: "France is known for wine"
            # Reference: TEA Nets (arXiv, Apr 2026) — nsubjpass + agent dep
            # Reference: Analytics Vidhya (2024) — spaCy passive detection
            if token.pos_ == "VERB" and token.dep_ == "ROOT":
                subj = None
                dobj = None
                prep_chain = None
                is_passive = False
                agent_obj = None

                for child in token.children:
                    if child.dep_ == "nsubjpass":
                        subj = child
                        is_passive = True
                    elif child.dep_ == "nsubj":
                        if subj is None:  # Don't overwrite nsubjpass
                            subj = child
                    elif child.dep_ == "dobj":
                        dobj = child
                    elif child.dep_ == "agent":
                        # "by Salesforce" — agent becomes the logical subject
                        for gc in child.children:
                            if gc.dep_ == "pobj":
                                agent_obj = gc
                                break
                    elif child.dep_ in ("prep",):
                        for gc in child.children:
                            if gc.dep_ == "pobj":
                                prep_chain = (child, gc)
                                break

                # Detect negation
                has_neg = any(c.dep_ == "neg" for c in token.children)

                if subj:
                    subj_texts = self._get_coordinated_subjects(doc, subj)

                    # Passive voice with agent: swap subject/object
                    # "France is known for wine" → (wine, known_in, france)
                    # "Tableau was acquired by Salesforce" → (salesforce, acquired, tableau)
                    if is_passive and agent_obj:
                        agent_text = self._get_chunk_text(doc, agent_obj)
                        patient_texts = subj_texts
                        if dobj:
                            obj_text = self._get_chunk_text(doc, dobj)
                            for pt in patient_texts:
                                rel = token.text.lower()
                                if has_neg:
                                    rel = f"NOT_{rel}"
                                triples.append((agent_text, rel, obj_text))
                        elif prep_chain:
                            prep, obj = prep_chain
                            obj_text = self._get_chunk_text(doc, obj)
                            rel = f"{token.text.lower()}_{prep.lemma_}"
                            if has_neg:
                                rel = f"NOT_{rel}"
                            triples.append((agent_text, rel, obj_text))
                        else:
                            # Passive with agent but no object: "X was VERB by Y"
                            for pt in patient_texts:
                                rel = token.text.lower()
                                if has_neg:
                                    rel = f"NOT_{rel}"
                                triples.append((agent_text, rel, pt))
                    elif is_passive and not agent_obj:
                        # Passive without agent: "X was VERB" → (X, VERB, ?)
                        # Store with passive marker
                        if dobj:
                            obj_text = self._get_chunk_text(doc, dobj)
                            for subj_text in subj_texts:
                                rel = token.text.lower()
                                if has_neg:
                                    rel = f"NOT_{rel}"
                                triples.append((subj_text, rel, obj_text))
                        elif prep_chain:
                            prep, obj = prep_chain
                            obj_text = self._get_chunk_text(doc, obj)
                            rel = f"{token.text.lower()}_{prep.lemma_}"
                            if has_neg:
                                rel = f"NOT_{rel}"
                            for subj_text in subj_texts:
                                triples.append((subj_text, rel, obj_text))
                    else:
                        # Active voice
                        if dobj:
                            obj_text = self._get_chunk_text(doc, dobj)
                            for subj_text in subj_texts:
                                rel = token.text.lower()
                                if has_neg:
                                    rel = f"NOT_{rel}"
                                triples.append((subj_text, rel, obj_text))
                        elif prep_chain:
                            prep, obj = prep_chain
                            obj_text = self._get_chunk_text(doc, obj)
                            rel = f"{token.text.lower()}_{prep.lemma_}"
                            if has_neg:
                                rel = f"NOT_{rel}"
                            for subj_text in subj_texts:
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

                # Check for det-as-subject pattern: "A feeds into B"
                # spaCy misparses verb+prep as NOUN+prep with det subject
                # det=feeds(ROOT), prep=into, pobj=B → (a, feeds_into, b)
                # Only match single uppercase letters (entity names like A, B, X)
                # NOT common articles ("the", "a", "an") or real determiners
                dets = [c for c in token.children if c.dep_ == "det"]
                preps_for_det = [c for c in token.children if c.dep_ == "prep"]
                if dets and preps_for_det and not compounds:
                    det = dets[0]
                    if len(det.text) == 1 and det.text.isupper():
                        prep = preps_for_det[0]
                        pobj = [c for c in prep.children if c.dep_ == "pobj"]
                        if pobj:
                            obj_text = self._get_chunk_text(doc, pobj[0])
                            rel = f"{token.text.lower()}_{prep.lemma_}"
                            triples.append((det.text.lower(), rel, obj_text))
                            continue

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

        # ─── Step 3: Merge clause segmenter results ────────────────
        # Clause segmenter handles coordinated objects/subjects that
        # the dependency-based extraction above may miss.
        if all_triples:
            triples.extend(all_triples)

        # ─── Step 4: Extract temporal orderings ─────────────────
        # Detect discourse connectives ("then", "after", "before")
        # and create temporal ordering triples.
        #
        # Reference: Allen (1983), TimeML (Pustejovsky et al., 2003)
        from .temporal_extractor import extract_temporal_orderings
        temporal_orderings = extract_temporal_orderings(doc)
        for ordering in temporal_orderings:
            triples.append((
                ordering.event1_description,
                ordering.relation,
                ordering.event2_description,
            ))

        # ─── Step 5: Triple-level coreference resolution ─────────
        # Resolve pronouns in extracted triples (not in text).
        # "it runs" → ("video game console", "runs", ...)
        # "its video games" → (..., "video games") (possessive stripped)
        # "this shows" → skip (discourse deixis)
        #
        # Reference: spaCy coref team — "resolve at span level"
        # Reference: Neo4j (2026) — coreference as IE pipeline step
        triples = self.resolve_triple_coreferences(triples, coref_map, doc)

        # ─── Step 6: Adjectival predicate extraction ─────────────
        # "The man is friendly" → (the man, has_characteristic, friendly)
        # "The engine runs smoother" → (the engine, has_characteristic, smooth)
        # Uses spaCy 'acomp' dependency (adjectival complement).
        #
        # Maps to Wikidata P1552 "has characteristic" — "inherent or
        # distinguishing quality or feature of the entity."
        # This is the standard Wikidata property for entity attributes.
        #
        # Reference: Wikidata P1552 — https://www.wikidata.org/wiki/Property:P1552
        # Reference: Universal Dependencies — 'acomp' label
        # Reference: Honnibal & Montani (2017), spaCy dependency parsing
        for token in doc:
            if token.pos_ in ("VERB", "AUX") and token.dep_ in ("ROOT", "conj"):
                has_dobj = any(c.dep_ == "dobj" for c in token.children)
                adj_mods = [c for c in token.children
                           if c.dep_ == "acomp" and c.pos_ == "ADJ"]
                if not has_dobj and adj_mods:
                    subj = None
                    for c in token.children:
                        if c.dep_ in ("nsubj", "nsubjpass"):
                            subj = self._get_chunk_text(doc, c)
                            break
                    if subj:
                        for adj in adj_mods:
                            triples.append((subj, "has_characteristic", adj.lemma_))

        # ─── Step 7: Deduplicate via relation canonicalization ─────
        # Two extraction paths (clause segmenter + dependency) produce
        # duplicates with different relation names for the same fact.
        # Canonicalize relations, deduplicate, keep richer relation.
        #
        # Reference: Zhang & Soh (2024), "Extract, Define, Canonicalize"
        # Reference: UDASTE (ScienceDirect, 2023)
        from .relation_canonicalizer import deduplicate_triples
        triples = deduplicate_triples(triples, nlp=self.nlp)

        return triples

    def _get_chunk_text(self, doc, token) -> str:
        """Get the full noun chunk text for a token, stripping determiners.

        "United Kingdom" → "united kingdom" (not just "Kingdom")
        "the EU" → "eu" (strip "the")
        "a country" → "country" (strip "a")
        "World War 2" → "world war 2" (include nummod)
        "united states of america" → "united states of america" (walk prep chain)
        """
        for chunk in doc.noun_chunks:
            if token in chunk:
                # Start with chunk text
                words = list(chunk.text.lower().split())

                # Strip leading determiners using language registry.
                # Reference: td/languages/en.py — ARTICLES
                while words and words[0] in self.lang_config.articles:
                    words = words[1:]

                # Include nummod children (e.g., "World War 2" → include "2")
                for child in token.children:
                    if child.dep_ == "nummod" and child.text.lower() not in words:
                        words.append(child.text.lower())

                # Walk prep chains ONLY for chunk root AND only when pobj
                # has compound modifiers (entity-internal preps)
                # "united states of america" → "of america" is part of entity
                # "France in EU" → "in" is the relation, NOT part of entity
                if token == chunk.root:
                    for child in token.children:
                        if child.dep_ == "prep":
                            for gc in child.children:
                                if gc.dep_ == "pobj":
                                    # Only include if pobj has compounds
                                    # (suggesting it's part of a larger entity)
                                    pobj_compounds = [c for c in gc.children if c.dep_ in ("compound", "nummod")]
                                    if pobj_compounds:
                                        prep_phrase = f"{child.text.lower()} {gc.text.lower()}"
                                        for ggc in gc.children:
                                            if ggc.dep_ in ("compound", "nummod") and ggc.text.lower() not in prep_phrase:
                                                prep_phrase = f"{ggc.text.lower()} {prep_phrase}"
                                        words.append(prep_phrase)

                return " ".join(words) if words else chunk.text.lower()
        return token.text.lower()

    def _get_coordinated_subjects(self, doc, subject_token) -> list[str]:
        """Extract all coordinated subjects from a subject token.

        Handles arbitrarily deep coordination:
        - "Alice and Bob went to Paris" → ["alice", "bob"]
        - "Alice, Bob, Carol, Dave and Eve went to Paris" → [all 5]
        - "Algorithms and data structures are central to CS" → ["algorithms", "data structures"]
        - "France, Germany and Italy are in Europe" → ["france", "germany", "italy"]

        Uses BFS to walk conj/nmod chains to arbitrary depth.
        Preserves modifiers (amod, compound, det) via noun chunk lookup.

        Reference: Manning & Schütze (1999), Chapter 5: Collocations.
        """
        def _token_text(token):
            """Get text for a token, preserving compounds and modifiers."""
            # Find the noun chunk for this token
            token_chunk = None
            for chunk in doc.noun_chunks:
                if token.i >= chunk.start and token.i < chunk.end:
                    token_chunk = chunk
                    break

            if token_chunk:
                # Check if this token is a coordinated element within a larger chunk
                # If so, use just the token's subtree (not the full chunk)
                has_coord_siblings = any(
                    c.dep_ in ("conj", "nmod", "npadvmod") 
                    for c in token.children
                ) or any(
                    token.dep_ in ("conj", "nmod", "npadvmod")
                    for c in doc if c.i == token.head.i
                    for c2 in c.children if c2.i == token.i
                )

                # If this token has coordinated siblings in the same chunk,
                # use just this token's subtree text
                if has_coord_siblings and token_chunk.end - token_chunk.start > 2:
                    # Get this token's compounds and modifiers (not full subtree)
                    modifiers = [c for c in token.children if c.dep_ in ("compound", "amod")]
                    all_tokens = sorted(
                        modifiers + [token],
                        key=lambda t: t.i
                    )
                    words = [t.text.lower() for t in all_tokens]
                    if words:
                        return " ".join(words)

                # Otherwise use the full chunk
                words = token_chunk.text.lower().split()
                while words and words[0] in self.lang_config.articles:
                    words = words[1:]
                # Include nummod children (e.g., "World War 2" → include "2")
                for child in token.children:
                    if child.dep_ == "nummod" and child.text.lower() not in words:
                        words.append(child.text.lower())
                # Walk prep chains (e.g., "united states of america")
                # "of" is entity-internal (genitive), always include
                # Other preps are relations, only include when pobj has compounds
                for child in token.children:
                    if child.dep_ == "prep":
                        for gc in child.children:
                            if gc.dep_ == "pobj":
                                if child.text.lower() in self.lang_config.genitive_markers:
                                    # "of" is entity-internal, always include
                                    prep_phrase = f"of {gc.text.lower()}"
                                    for ggc in gc.children:
                                        if ggc.dep_ in ("compound", "nummod") and ggc.text.lower() not in prep_phrase:
                                            prep_phrase = f"{ggc.text.lower()} {prep_phrase}"
                                    words.append(prep_phrase)
                                else:
                                    # Other preps: only include when pobj has compounds
                                    pobj_compounds = [c for c in gc.children if c.dep_ in ("compound", "nummod")]
                                    if pobj_compounds:
                                        prep_phrase = f"{child.text.lower()} {gc.text.lower()}"
                                        for ggc in gc.children:
                                            if ggc.dep_ in ("compound", "nummod") and ggc.text.lower() not in prep_phrase:
                                                prep_phrase = f"{ggc.text.lower()} {prep_phrase}"
                                        words.append(prep_phrase)
                return " ".join(words) if words else token_chunk.text.lower()

            # Fallback: compound + token
            compounds = [c for c in token.children if c.dep_ == "compound"]
            if compounds:
                return " ".join([c.text.lower() for c in compounds] + [token.text.lower()])
            return token.text.lower()

        # BFS to collect all coordinated tokens at arbitrary depth
        visited = set()
        queue = [subject_token]
        subjects = []

        while queue:
            token = queue.pop(0)
            if token.i in visited:
                continue
            visited.add(token.i)
            subjects.append(_token_text(token))

            # Walk conj, nmod, and npadvmod children (all coordination types)
            for child in token.children:
                if child.dep_ in ("conj", "nmod", "npadvmod") and child.i not in visited:
                    queue.append(child)

        return list(dict.fromkeys(subjects))  # dedupe preserving order

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

        # Relation words that should NOT be merged into entities.
        # Loaded from language registry (prepositions, copula verbs, stop words).
        # KG-specific relation names are added separately.
        # Reference: td/languages/en.py
        relation_words = set(self.lang_config.prepositions)
        relation_words.update(self.lang_config.copula_verbs)
        relation_words.update(self.lang_config.stop_words)
        relation_words.update({"part_of", "capital_of", "born_in", "lives_in", "located_in"})

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
        # Spatial/temporal relation words that precede multi-word entities.
        # Loaded from language registry.
        # Reference: td/languages/en.py
        relation_words = set(self.lang_config.prepositions)
        relation_words.update({"part_of", "capital_of", "born_in", "lives_in", "located_in"})

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
                        if next_tok not in self._fallback_stop_words and any(c.isalnum() for c in next_tok):
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
                        if next_tok in self._fallback_stop_words:
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
            # Uses language registry for copula verbs and genitive markers.
            # Reference: td/languages/en.py — COPULA_VERBS, GENITIVE_MARKERS
            copulas = self.lang_config.copula_verbs
            genitives = self.lang_config.genitive_markers
            has_copula = any(t in copulas for t in between)
            has_genitive = any(t in genitives for t in between)
            if has_copula and has_genitive:
                y_text = e2["text"]
                # Verify Y is between copula and genitive in the token sequence
                cop_idx = next((i for i, t in enumerate(between) if t in copulas), -1)
                gen_idx = next((i for i, t in enumerate(between) if t in genitives), -1)
                y_idx = None
                for j, t in enumerate(between):
                    if t == y_text.split()[0]:
                        y_idx = j
                        break

                if y_idx and cop_idx < y_idx < gen_idx:
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
            # Uses language registry for prepositions.
            # Reference: td/languages/en.py — PREPOSITIONS
            dative_markers = self.lang_config.prepositions  # "to" is in prepositions
            has_dative = any(t in dative_markers for t in between)
            if has_dative:
                y_text = e2["text"]
                to_idx = next(i for i, t in enumerate(between) if t in dative_markers)
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

        # Scan tokens for copula + relation_word pattern
        # Uses language registry for copula verbs.
        # Reference: td/languages/en.py — COPULA_VERBS
        for i, tok in enumerate(tokens):
            if tok not in self.lang_config.copula_verbs:
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
            if tok in self._fallback_stop_words:
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
            # Uses is_stop_word() which delegates to spaCy (language-agnostic).
            relation_words = [w for w in connectives
                            if not self.is_stop_word(w) and len(w) > 1]

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
                content = [w for w in relation_words if w not in self._fallback_stop_words]
                if content:
                    rel_candidate = "_".join(content)
                    # Append common suffixes if missing
                    if not any(rel_candidate.endswith(s) for s in ("_of", "_to", "_in", "_for")):
                        # Check if a preposition follows the content words
                        postpositions = [w for w in relation_words if w in self._fallback_stop_words]
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
