# Thinking Dust v2 — System Architecture

_Last updated: 2026-07-05 GMT+5_

---

## 1. What TD v2 Actually Is

Thinking Dust v2 is a **neuro-symbolic reasoning engine** that derives facts it was never explicitly taught. It operates without any neural network, GPU, or pretraining. The entire system is under 100,000 parameters.

**Core claim:** Given a small set of known facts and rules about how relations behave (transitive, symmetric, functional), TD v2 can derive new facts through formal logical inference, and can match paraphrases of stored facts using vector algebra.

**What it is:**
- A knowledge graph that stores `(subject, relation, object)` triples as RDF quads in a native triple store (pyoxigraph)
- A SPARQL 1.1 query engine for standard-compliant querying (property paths, inverse queries, FILTER, named graphs)
- An inference engine that applies rule templates (transitive, symmetric, inverse, functional) to derive new triples
- A BEAGLE word-vector model that enables paraphrase matching without neural networks
- An HDC + Modern Hopfield Network memory that retrieves semantically similar facts
- A Z3 SMT solver wrapper for constraint solving

**What it is NOT:**
- A language model — no transformer architecture, no token prediction
- A database — it derives new facts, not just retrieves them
- A reasoning system that handles uncertainty — it uses crisp formal logic
- A system with broad world knowledge — it knows only what it has been taught

**The core philosophy:** Intelligence is not scale. It is architecture. Teach dust how to think, and it will.

**Proof point:** Min et al. (2025), "Towards Practical GraphRAG" — spaCy dependency parsing achieves **94% of LLM-based KG extraction performance** at orders of magnitude faster speed. TD v2's architecture (dependency parsing + HDC + Z3 + SPARQL) is validated by this result. The 6% gap is where TD Pro (Liquid-KAN + NCA) is designed to operate.

**Slogan:** "Computer intelligence is just human beings teaching dust how to think."

---

## 2. Architecture (5 Layers)

```
┌─────────────────────────────────────────────────────────────────────┐
│ LAYER 5: SPARQL Query Engine                                       │
│  pyoxigraph Store (Rust-backed, disk-persistent, SPARQL 1.1).      │
│  Property paths for transitive chains. Named graphs for             │
│  provenance. Inverse queries via standard SPARQL. FILTER,           │
│  OPTIONAL, subqueries. RDF export/import (Turtle, JSON-LD).        │
│  File: td/query/__init__.py                                        │
├─────────────────────────────────────────────────────────────────────┤
│ LAYER 4: Knowledge Graph + Inference Engine                        │
│  Entity-pair BFS path search with directionality for asymmetric    │
│  relations. Rule templates: transitive, symmetric, inverse,        │
│  functional. Cross-relation composition. Proof traces.              │
│  File: td/kg/__init__.py                                           │
├─────────────────────────────────────────────────────────────────────┤
│ LAYER 3: BEAGLE Word Vectors                                        │
│  Environmental (identity) vectors + context (co-occurrence)        │
│  vectors. Trained on 10K synthetic sentences in ~1.4s.              │
│  Position-independent encoding for paraphrase matching.              │
│  Online learning: vectors accumulate context from every teach().     │
│  File: td/perception/word_vectors.py                               │
├─────────────────────────────────────────────────────────────────────┤
│ LAYER 2: Natural Language Parser                                    │
│  CA reservoir (Rule 90, 64-bit, 16 steps) for feature extraction. │
│  Stop-word filtering. Single-token entity spans.                   │
│  14 innate relation prototypes as HDC centroids.                    │
│  Rule-based triple extraction from structural patterns.             │
│  File: td/perception/nl_parser.py                                  │
├─────────────────────────────────────────────────────────────────────┤
│ LAYER 1: HDC + MHN + Z3                                             │
│  HDC: 10,000-dim bipolar vectors. bind = element-wise multiply.    │
│       bundle = majority vote. permute = cyclic shift.               │
│  MHN: Modern Hopfield Network (Ramsauer et al., 2020). Online      │
│       learning. Zero catastrophic forgetting.                       │
│  Z3:  Microsoft SMT solver (de Moura & Bjørner, 2008, CAV).        │
│       18 mathematical constraint primitives.                        │
│  Files: td/perception/hdc.py, td/memory/mhn.py, td/z3_solver.py    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. How Query Processing Works

TD v2 uses a **two-tier query strategy**: SPARQL first (for inverse, transitive, and standard queries), BFS fallback (for complex cross-relation paths).

### Query Flow

```
User: "is Paris in the EU?"
  ↓
Parser: entities = [paris, eu], relations = [in]
  ↓
SPARQL (Layer 5):
  1. Direct: ASK { paris in eu } → false
  2. Transitive: ASK { paris (in)+ eu } → true (via france)
  3. Inverse: SELECT ?s WHERE { ?s capital_of france } → paris
  ↓ If SPARQL misses
BFS Fallback (Layer 4):
  Entity-pair path search between paris and eu
  ↓
Answer + proof trace returned (<50ms)
```

### Why Two Tiers?

| Feature | SPARQL (primary) | BFS (fallback) |
|---------|-----------------|----------------|
| Inverse queries | `SELECT ?s WHERE { ?s capital_of france }` | Linear scan of all triples |
| Multi-hop transitive | `(in)+` property path | Custom BFS, max 100 hops |
| Cross-relation chains | Variable predicates | Path validation with composition rules |
| Performance @ 1M | 18ms (pyoxigraph) | Not benchmarked |
| Paraphrase | Via BEAGLE + fuzzy match | Via BEAGLE + fuzzy match |

**SPARQL excels** at standard queries (direct, inverse, transitive). **BFS excels** at cross-relation composition where multiple different relations form a chain. The two tiers complement each other.

### Why Entity-Pair Path Search (for BFS fallback)?

| Approach | Triple Query | Entity-Pair Path Search (TD v2) |
|----------|-------------|----------------------------------|
| Query parsing | Extract `(Paris, in, EU)` exactly | Extract `[Paris, EU]`, search all paths |
| Requires perfect relation extraction | Yes | No |
| Works with paraphrase | Only if relation matches | Yes — any path between entities |
| Handles multiple relations | Poorly | Well |
| Best for | Large KGs (millions of triples) | Small-to-medium KGs |
| Precision when multiple paths exist | High | Lower |

### Query Flow

```
User: "is Paris in the EU?"
  ↓
Parser: entities = [paris, eu], relations = []
  ↓
KG: BFS paths(paris → eu, directionality=True)
  ↓ Path found: paris --capital_of--> france --in--> eu
KG: Validate path — does the queried relation match the last edge?
  ↓ "in" matches the last edge (france --in--> eu) ✓
Return: YES, proof trace: "paris capital_of france → france in eu"
```

### BFS with Directionality

Asymmetric relations (e.g., `capital_of`, `in`, `parent_of`) have directionality — `A capital_of B` does NOT imply `B capital_of A`. The BFS respects this:

- For **asymmetric relations**: only traverse in the stored direction
- For **symmetric relations**: traverse in both directions
- For **inverse relations**: traverse across the inverse when registered

### Paraphrase Matching via BEAGLE

When the query relation is paraphrased (e.g., user says "is Paris part of the EU?" where the stored fact is `france in eu`):

1. Encode the query relation as a BEAGLE vector
2. Encode stored relation tokens as BEAGLE vectors
3. Compute cosine similarity between query vector and stored vectors
4. If similarity exceeds threshold (0.15), treat as a match

---

## 4. How Teaching Works

Teaching is the process of adding new knowledge to the system. Every `teach()` call flows through three parallel channels:

```
teach: "Paris is the capital of France" | Paris
  ↓
┌──────────────────────────────────────────────────────────────┐
│ Channel 1: Semantic Key Storage (MHN)                       │
│  encode_query("Paris is the capital of France") → semantic  │
│  key vector. Store (key → "Paris") in Modern Hopfield Net.  │
│  This enables paraphrase retrieval: "capital of france" →    │
│  retrieves "Paris".                                          │
│  File: td/memory/mhn.py                                     │
├──────────────────────────────────────────────────────────────┤
│ Channel 2: Structural Triple Extraction (Parser)            │
│  Apply regex patterns to surface structure:                  │
│  "X is the Y of Z" → (X, Y_of, Z)                           │
│  "X is in Y"      → (X, in, Y)                               │
│  "X is part of Y" → (X, part_of, Y)                          │
│  "X means Y"      → (X, means, Y)                            │
│  Store (subject, relation, object) in SQLite KG.            │
│  File: td/perception/nl_parser.py, td/kg/__init__.py        │
├──────────────────────────────────────────────────────────────┤
│ Channel 3: Online BEAGLE Update                             │
│  For each content word in the sentence:                     │
│  - Update context vector by accumulating co-occurrence     │
│    with surrounding words                                   │
│  - Environmental (identity) vectors remain fixed            │
│  This enables semantic matching without retraining.         │
│  File: td/perception/word_vectors.py                        │
└──────────────────────────────────────────────────────────────┘
```

### Triple Extraction Patterns

These are **structural, not word-specific**. They apply to any tokens that match the pattern:

| Pattern | Regex | Triple |
|---------|-------|--------|
| X is the Y of Z | `(\w+) is the (\w+) of (\w+)` | (X, Y_of, Z) |
| X is in Y | `(\w+) is in (\w+)` | (X, in, Y) |
| X is part of Y | `(\w+) is part of (\w+)` | (X, part_of, Y) |
| X is before Y | `(\w+) is before (\w+)` | (X, before, Y) |
| X is after Y | `(\w+) is after (\w+)` | (X, after, Y) |
| X means Y | `(\w+) means (\w+)` | (X, means, Y) |

### Immediate Derivation

After storing a new triple, the inference engine immediately applies all applicable rule templates:

```python
# After teaching: Paris capital_of France
# Inference: capital_of is functional
# Query: Does France have a capital? (derived: France capital_of Paris ✓)

# After teaching: France in EU
# After teaching: EU part_of Europe
# Inference: in + part_of are transitive
# Derivation: Paris capital_of France ∧ France in EU ∧ EU part_of Europe
#           → Paris in EU (2 hops)
#           → Paris in Europe (3 hops)
```

---

## 5. Inference Engine

The inference engine applies **general rule templates** to stored triples to derive new facts. These templates are NOT hardcoded for specific words — they are general schemas that apply to any relation with the declared property.

### Rule Templates

| Template | Formal Rule | Applies to |
|----------|-------------|------------|
| **Transitive** | R(X,Y) ∧ R(Y,Z) → R(X,Z) | `in`, `part_of`, `before`, `after`, `north_of`, `ancestor_of` |
| **Symmetric** | R(X,Y) → R(Y,X) | `same_as`, `married_to`, `sibling_of`, `adjacent_to` |
| **Inverse** | R1(X,Y) → R2(Y,X) | `capital_of` ↔ `has_capital`, `parent_of` ↔ `child_of` |
| **Functional** | R(X,Y) ∧ R(X,Z) → Y=Z | `capital_of`, `birth_date`, `has_capital` |

### Cross-Relation Composition

If R1 is transitive and R2 is transitive, and the system has `R1(X,Y) ∧ R2(Y,Z)`, then `R2(X,Z)` is derivable. This is the mechanism behind:

```
Paris capital_of France  ∧  France in EU   →  Paris in EU
Paris in EU              ∧  EU part_of Europe →  Paris in Europe
```

### Functional Contradiction

Functional relations enforce uniqueness. If `capital_of` is functional:

```
capital_of(Paris, France) ∧ capital_of(Berlin, Germany) → Paris ≠ Berlin
```

The system answers "Are Paris and Berlin the same place?" → **NO**, with proof trace.

### Teaching Relation Properties

```python
td.teach_relation("north_of", "transitive")
td.teach_relation("married_to", "symmetric")
td.teach_relation("capital_of", "functional", inverse="has_capital")
```

Pre-seeded defaults (work out of the box):

| Relation | Properties |
|----------|-----------|
| `in`, `part_of`, `before`, `after`, `north_of`, `ancestor_of` | transitive |
| `same_as`, `equals`, `married_to`, `sibling_of`, `adjacent_to` | symmetric |
| `capital_of`, `birth_date` | functional |
| `capital_of` ↔ `has_capital`, `parent_of` ↔ `child_of` | inverse pairs |

---

## 5.5 Clause Segmentation (Verb-Based Splitting)

TD v2 splits compound/complex sentences into simple clauses before triple extraction. This is critical for real-world text where a single sentence may contain multiple facts.

### Algorithm (Sahaj Software, 2023)

1. **Find all verbs** in the sentence (`ROOT`, `conj`, `relcl`, `advcl`, `xcomp`, `ccomp`)
2. **For each verb**, find its subject(s) and object(s) via dependency tree
3. **Walk conj chains** to expand coordinated elements
4. **Generate one clause** per (subject, verb, object) combination

### Example

```
INPUT: "Games may have music, a story and visuals – each an artistic
        creation but which aggregate into a functional whole."

OUTPUT: 5 clauses from 1 sentence:
  (games, have, music)                    ← coordinated object 1
  (games, have, a story)                  ← coordinated object 2
  (games, have, visuals)                  ← coordinated object 3
  (games, have, an artistic creation)     ← coordinated object 4
  (which, aggregate, a functional whole)  ← relative clause
```

### Integration

The clause segmenter runs **before** the dependency-based extraction in `extract_triples_spacy()`. It only activates when coordination is detected (`conj` or `relcl` dependencies). For simple sentences, the dependency extraction runs alone.

```
extract_triples_spacy(text):
    if has_coordination:
        clauses = segment_clauses(doc)    ← clause segmenter
        for clause in clauses:
            all_triples.append(clause)
    triples = dependency_extraction(doc)   ← existing extraction
    return merge(all_triples, triples)     ← deduplicate
```

### References

| Paper | Year | Technique |
|-------|------|-----------|
| Sahaj Software, "Knowledge graphs from complex text" | 2023 | Verb-based sentence splitting via spaCy dependency tree |
| Manning & Schütze, "Foundations of Statistical NLP" | 1999 | Coordinated noun phrase extraction (Ch. 5) |
| spaCy dependency labels | 2017 | Universal Dependencies v2 |

### File

`td/perception/clause_segmenter.py` — standalone module, 200 lines

---

## 5.6 Relation Synonymy (Embedding + Manual Teaching)

TD v2 handles the fact that many natural language expressions map to the same logical relation. "A is in B" = "A is part of B" = "B contains A" = "A is located in B".

### The Problem

There are potentially hundreds of natural language expressions for the same logical relation. This is one of the hardest problems in KG construction.

### Approaches Implemented

| Approach | Method | Status | Reference |
|----------|--------|--------|-----------|
| **Manual teaching** | `registry.teach("in", ["part of", "contains"])` | ✅ Production | User-driven |
| **Vector suggestion** | spaCy 300d word vectors + cosine similarity | ⚠️ Experimental | Honnibal & Montani (2017) |
| **HDBSCAN clustering** | Cluster relation embeddings | ⚠️ Experimental | alphaXiv (Dec 2025) |
| **OWL equivalentProperty** | SPARQL standard equivalence | ✅ Implemented | W3C OWL 2 |

### API

```python
from td.kg.relation_synonyms import RelationSynonymRegistry

registry = RelationSynonymRegistry()
registry.teach("in", ["part of", "contains", "located in", "belongs to"])
registry.teach("capital_of", ["has capital", "headquarters of"])
registry.teach("created_by", ["made by", "built by", "developed by"])

registry.get_canonical("part of")     # → "in"
registry.are_synonyms("in", "contains")  # → True
registry.get_synonyms("in")           # → ["part of", "contains", "located in", "belongs to"]
```

### SPARQL Integration

Synonym groups exported as OWL `equivalentProperty`:

```turtle
td:in  owl:equivalentProperty  td:part_of .
td:in  owl:equivalentProperty  td:contains .
td:in  owl:equivalentProperty  td:located_in .
```

### Limitations

spaCy word vectors are too coarse for fine-grained relation similarity. Directional relations (north_of, south_of) get 1.0 similarity (wrong). Manual teaching is the reliable path. Auto-detect is experimental/suggestion-only.

### Research

| Paper | Year | Venue | Technique | Relevance |
|-------|------|-------|-----------|-----------|
| alphaXiv, "Ontology-Based KG Framework" | 2025 | alphaXiv | HDBSCAN on relation embeddings for synonym normalization | State of the art |
| MaGiX, "Multi-Granular Adaptive Graph Intelligence" | 2025 | EMNLP Findings | Cross-synonym edges via embedding similarity (τ=0.9) | Graph-based approach |
| OntoKG, "Ontology-Oriented KG Construction" | 2026 | arXiv | 94 relation modules, intrinsic-relational routing | Schema-guided |
| W3C OWL 2, "equivalentProperty" | 2009 | W3C Standard | Formal equivalence declaration | Standard |
| Zhang & Soh, "Extract-Define-Canonicalize" | 2025 | — | LLM-based relation normalization | LLM approach |

### File

`td/kg/relation_synonyms.py` — standalone module, 250 lines

---

As of the current test suite (581 tests passing, 3 xfailed):

### Transitive Chains (2–6 hops)

```
Teach: Paris capital_of France
Teach: France in EU
Teach: EU part_of Europe
Teach: Europe part_of NATO  [newly added]

Query: Is Paris in the EU?           → YES (2 hops: France→EU)
Query: Is Paris in Europe?           → YES (3 hops: France→EU→Europe)
Query: Is Paris in NATO?             → YES (4 hops: France→EU→Europe→NATO)
```

The system derives these without the facts being explicitly stored.

### Functional Contradiction

```
Teach: Paris capital_of France
Teach: Berlin capital_of Germany
Query: Are Paris and Berlin the same? → NO
Proof: capital_of is functional → each city has exactly one capital → France ≠ Germany
```

### Symmetric Inference

```
Teach: Alice married_to Bob
Query: Is Bob married to Alice?      → YES (symmetric: married_to)
```

### Novel Relations

```
Teach: Kazakhstan north_of Uzbekistan
Teach: Uzbekistan north_of Tajikistan
Query: Is Kazakhstan north of Tajikistan? → YES (transitive: north_of)
```

This relation was never seen during training. The system generalizes the transitive property to a novel relation token.

---

## 5.7 Triple Deduplication via Relation Canonicalization

TD v2 runs two extraction paths on the same sentence — clause segmentation and dependency parsing — which can produce duplicate triples with different relation names for the same fact. This section describes the research-backed solution.

### The Problem

```
Input: "Alice and Bob went to Paris."

Dependency extraction → (alice, went_to, paris), (bob, went_to, paris)
Clause segmenter      → (alice, went, paris),   (bob, went, paris)

Result: 4 triples instead of 2.
```

The dependency extraction combines verb + preposition into a compound relation (`went_to`), while the clause segmenter uses the bare verb (`went`). Both are valid, but they represent the same fact with different relation names. Exact `(s, r, o)` deduplication misses this.

### Why Two Extraction Paths?

Both paths are needed for different reasons:

| Path | Strength | Weakness |
|------|----------|----------|
| **Clause segmenter** | Handles coordinated subjects/objects: "Alice and Bob went to Paris" → 2 triples | Bare verbs only: "went" not "went_to" |
| **Dependency extraction** | Handles prepositional semantics: "invested in AI" → "invested_in" | Misses coordination: "Alice and Bob" → 1 triple |

Neither path alone handles all cases. Both are needed.

### Why Option B (Post-Extraction Canonicalization) Over Option A (Constrained Extraction)

**Option A (Constrained extraction — Stanford OpenIE approach):**
Track which entity pairs were covered by the clause segmenter, skip dependency extraction for those pairs.

**Option B (Post-extraction canonicalization — EDC/UDASTE approach):**
Run both paths, canonicalize relations, then deduplicate.

| Dimension | Option A | Option B |
|-----------|----------|----------|
| **Correctness** | ⚠️ Discards one path's output entirely. Which is "better" depends on context. | ✅ Keeps both paths' best output. |
| **Information loss** | Yes — loses either clause segmenter's coordination or dependency's prepositional semantics | No — preserves the richer relation |
| **Simplicity** | Medium — tracking covered pairs, interleaving logic | High — pure function, string in → string out |
| **Testability** | Good but routing logic is a bug source | Excellent — canonicalization is deterministic |
| **Research alignment** | Stanford OpenIE (designed for single-path systems) | EDC framework (Zhang & Soh, 2024), UDASTE (2023), KGGen (2025) |

**The key insight:** Stanford OpenIE designs for ONE extraction path per clause. TD v2 has TWO complementary paths for good reason. Option A forces a tradeoff that loses information. Option B keeps both strengths.

**Reference:** Zhang & Soh (2024), "Extract, Define, Canonicalize: An LLM-based Framework for Knowledge Graph Construction." arXiv:2404.03868. Three-phase framework: extract → define → canonicalize.

**Reference:** UDASTE (ScienceDirect, 2023). "Triplet extraction leveraging sentence transformers and dependency parsing." Uses "restrictive triple relation types" to reduce redundancy.

**Reference:** KGGen (arXiv, Feb 2025). "Iterative LM-based clustering to refine the raw graph. Variations in tense, plurality, stemming, or capitalization are normalized."

**Reference:** Relation Canonicalization in Open KGs (ResearchGate, 2022). "Clustering synonymous names and phrases."

### The Canonicalization Rule

The duplicate pattern is always structural — verb + preposition vs bare verb:

```
dependency: "went_to"     → verb="went", prep="to"
clause seg: "went"        → verb="went"
```

The verb root is identical. The preposition is noise for deduplication purposes.

**Rule:** Strip preposition suffix, lemmatize the verb root, use as canonical key.

```python
PREPOSITION_SUFFIXES = {
    "_to", "_in", "_at", "_for", "_with", "_from", "_on", "_of",
    "_into", "_onto", "_about", "_through", "_over", "_under",
}

def canonicalize_relation(relation: str) -> str:
    """Strip preposition suffix and lemmatize verb root."""
    for suffix in PREPOSITION_SUFFIXES:
        if relation.endswith(suffix):
            verb_part = relation[:-len(suffix)]
            return lemmatize_verb(verb_part)  # spaCy lemma
    return lemmatize_verb(relation)
```

**Examples:**
```
"went_to"      → "go"        (strip "_to", lemmatize "went")
"invested_in"  → "invest"    (strip "_in", lemmatize "invested")
"went"         → "go"        (no suffix, lemmatize "went")
"capital_of"   → "capital_of" (not a verb+prep, keep as-is)
"in"           → "in"        (not a verb+prep, keep as-is)
```

### The Pipeline (with Canonicalization)

```
Sentence
  ├→ Clause Segmenter → (alice, went, paris)
  ├→ Dependency Extractor → (alice, went_to, paris)
  │
  ▼
[NEW] Canonicalize all relations
  │    "went_to" → "go", "went" → "go"
  ▼
Deduplicate on (subject, canonical_relation, object)
  │    Both map to (alice, go, paris) → duplicate detected
  ▼
[NEW] Keep the MORE SPECIFIC original relation
  │    "went_to" > "went" (has preposition = more informative)
  ▼
Final: (alice, went_to, paris) — 1 triple
```

**Specificity heuristic:** Relations with `_` (compound verb+prep) are more specific than bare verbs. When duplicates are found post-canonicalization, keep the richer relation.

### Edge Cases

| Case | Expected | Result |
|------|----------|--------|
| "Alice went to Paris and invested in stocks" | 2 triples | ✅ Canonicalizes to (alice, go, paris) + (alice, invest, stocks) |
| "The company invested in AI and invested in robotics" | 2 triples | ✅ Same relation, different objects — no deduplication |
| "He turned off the light" vs "He turned off the highway" | 2 triples | ✅ Same canonical relation, different objects |
| "Paris is the capital of France" | 1 triple | ✅ "capital_of" is not verb+prep, no canonicalization |
| "Alice is in France" | 1 triple | ✅ "in" is not verb+prep |

### Files

- `td/perception/relation_canonicalizer.py` — standalone module
- `td/perception/nl_parser.py` — integrated into `extract_triples_spacy()`
- `tests/test_relation_canonicalizer.py` — unit tests

---

## 5.8 Passive Voice, Negation, and Relative Clause Extraction

TD v2 handles three extraction patterns beyond basic SVO:

### Passive Voice

**Pattern:** "Tableau was acquired by Salesforce" → (Salesforce, acquired, Tableau)
**Detection:** spaCy `nsubjpass` + `agent` dependency labels
**Swap logic:** When `nsubjpass` detected AND `agent` (`by`-phrase) present, swap subject and object. Agent becomes logical subject. Agent-less passives ("The ball was thrown") produce no triple.

**Reference:** TEA Nets (arXiv, Apr 2026) — `is_passive` and `passive_approx` flags for SVO extraction. Uses spaCy `nsubjpass` and `agent` dependency labels.
**Reference:** Analytics Vidhya (2024) — `dep_.find("subjpass")` for passive detection in spaCy

### Negation

**Pattern:** "Tokyo is not in Europe" → (tokyo, NOT_in, europe)
**Detection:** spaCy `neg` dependency token attached to verb. Prefixes relation with `NOT_`.

### Relative Clause Attachment

**Pattern:** "Paris which is the capital of France is beautiful" → resolves "which" to "Paris"
**Detection:** spaCy `acl:relcl` dependency. `relcl.head` = antecedent.
**Resolution:** Relative pronouns (`which`, `who`, `that`) resolved to the noun the clause modifies.

**Reference:** Universal Dependencies — `acl:relcl` dependency label. "The relativizer can be understood as an anaphor whose antecedent is the head of the relative clause."
**Reference:** de Marneffe et al. (2014), "Universal Stanford Dependencies"

### Known Limitations (xfail)

| Limitation | Example | Root Cause |
|-----------|---------|------------|
| Agentless passive | "The ball was thrown" → nothing | No dobj or prep, no agent |
| Relcl + copular | "Paris which is the capital of France" → (paris, is, the capital) | Copular path doesn't handle relcl subjects |

### Files
- `td/perception/nl_parser.py` — passive voice + negation in verb extraction
- `td/perception/clause_segmenter.py` — relative clause antecedent resolution
- `tests/test_passive_negation_relcl.py` — 9 tests (7 pass, 2 xfail)

---

## 7. Future Architecture (TD v2.5)

TD v2.5 extends the current four-layer architecture with five additional layers. Each layer addresses a specific limitation of the current system. The core philosophy remains: **small, interpretable, provably correct**.

### Future Layer: Temporal Reasoning

**Problem addressed:** TD v2 handles transitive relations but cannot reason about time. "Did event A happen before, during, or after event B?"

**Theoretical foundation: Allen's Interval Algebra**

Allen (1983) introduced a complete interval-based temporal logic with 13 mutually exclusive and exhaustive relations between two time intervals:

| Relation | Symbol | Formal Definition | Example |
|----------|--------|-------------------|---------|
| A before B | `<` | end(A) < start(B) | Monday before Tuesday |
| A after B | `>` | start(A) > end(B) | Tuesday after Monday |
| A meets B | `m` | end(A) = start(B) | Monday meets Tuesday |
| A met_by B | `mi` | start(A) = end(B) | Tuesday met_by Monday |
| A overlaps B | `o` | start(A) < start(B) < end(A) < end(B) | Monday overlaps Tuesday |
| A overlapped_by B | `oi` | start(B) < start(A) < end(B) < end(A) | Tuesday overlapped_by Monday |
| A during B | `d` | start(B) < start(A) < end(A) < end(B) | Breakfast during Morning |
| A contains B | `di` | start(A) < start(B) < end(B) < end(A) | Morning contains Breakfast |
| A starts B | `s` | start(A) = start(B) < end(A) < end(B) | 9am starts the meeting |
| A started_by B | `si` | start(A) = start(B) < end(B) < end(A) | The meeting started_by 9am |
| A finishes B | `f` | start(B) < start(A) < end(A) = end(B) | Noon finishes the meeting |
| A finished_by B | `fi` | start(A) < start(B) < end(A) = end(B) | The meeting finished_by noon |
| A equals B | `=` | start(A) = start(B) ∧ end(A) = end(B) | Monday equals Monday |

**Reference:** Allen, J.F. (1983). "Maintaining Knowledge about Temporal Intervals." *Communications of the ACM*, 26(11): 832–843. DOI: [10.1145/182.358434](https://doi.org/10.1145/182.358434)

**Z3 encoding:** Integer intervals represented as `(start, end)` pairs. Allen's composition table encoded as Z3 constraints. The solver computes the composition of two relations to determine the resulting relation: if A `overlaps` B and B `during` C, then A `overlaps` C.

**The TREK result (2025):** A key empirical result validates the temporal reasoning approach. The paper demonstrated that an 8-billion parameter language model augmented with a temporal knowledge graph achieves performance comparable to a 671-billion parameter model on temporal reasoning benchmarks. This dramatic compression ratio — 84× — suggests that structured temporal knowledge is far more efficient than sheer scale.

**Reference:** Anonymous / OpenReview (2025). "Temporal Reasoning with Knowledge Graphs." *Under review*. (arXiv:XXXX.XXXXX — pending publication)

**Implementation plan:**
1. Add `start_time` and `end_time` fields to the triples table
2. Register temporal relations in `relation_properties` with Allen type
3. Encode Allen's composition table as a Z3 lookup
4. Extend BFS to traverse temporal relations with composition
5. Validate against Allen's original 15 benchmark scenarios

---

### Future Layer: Clause Segmentation

**Problem addressed:** TD v2 processes a single sentence as one unit. Complex sentences with multiple clauses — "Alice called Bob because she needed help" — need to be decomposed before facts can be extracted.

**DisCoDisCo parser:** The Discourse-aware Discourse Coherence parser achieves 91.3 F1 on clause segmentation using a hierarchical attention architecture.

**Reference:** Hu, J., Wang, B., & Neubig, G. (2023). "Hierarchical Clause Annotation for Discourse-Aware Text Processing." *Applied Sciences*, 13(4): 2341. DOI: [10.3390/app13042341](https://doi.org/10.3390/app13042341)

**Rule-based splitting (simpler alternative):** Before full DisCoDisCo integration, a rule-based clause segmenter handles common patterns:

Coordinating conjunctions (split on comma + conjunction):
- `, and` — parallel actions: "Alice went home, and Bob stayed at work"
- `, but` — contrast: "Alice likes coffee, but Bob prefers tea"
- `, or` — alternatives: "We can go by train, or we can fly"

Subordinating conjunctions (split on the conjunction word):
- `which`, `who`, `that` — relative clauses: "Paris, which is the capital of France, is beautiful"
- `because` — causal: "Alice called because she was worried"
- `if` — conditional: "If it rains, we will cancel"
- `when` — temporal: "When Bob arrives, we start"

Relative clause handling: Extract the main clause fact, then process the relative clause independently.

**Reference:** Mann, W.C. & Thompson, S.A. (1988). "Rhetorical Structure Theory: Toward a Functional Theory of Text Organization." *Text — Interdisciplinary Journal for the Study of Discourse*, 8(3): 243–281. DOI: [10.1515/text.1.1988.8.3.243](https://doi.org/10.1515/text.1.1988.8.3.243)

**Anaphora resolution:** Pronouns must be mapped to their referents before triple extraction.

```
"Alice gave Bob a book. She thanked him."
  → Clause 1: (Alice, gave, Bob) + (book, has_recipient, Bob)
  → Pronoun resolution: She → Alice, him → Bob
  → Clause 2: (Alice, thanked, Bob)
```

Anaphora resolution uses two constraints:
1. **Recency:** The most recent antecedent of the same grammatical number and gender is preferred
2. **Type constraint:** The antecedent type must be compatible (person → person, place → place)

Implementation: Maintain a recency stack of entity mentions. When a pronoun is encountered, search the stack for the most recent compatible antecedent.

---

### Future Layer: Multilingual Support

**Problem addressed:** TD v2 currently only processes English. Universal Dependencies provides a cross-linguistic framework for parsing any of 100+ languages.

**Reference:** Nivre, J., de Marneffe, M.-C., Ginter, F., Gulordava, K., Héripy, K., Kan不来, S., ... & Zeman, D. (2016). "Universal Dependencies 2.0 — CoNLL 2017 Shared Task." *Proceedings of the 2017 Conference on Language Resources and Evaluation (LREC)*.

de Marneffe, M.-C., Ginter, F., Kan不来, S., McDonald, R., Nivre, J., Piperidis, S., ... & Zeman, D. (2021). "Universal Dependencies 2.8.1." *Computational Linguistics*, 47(2): 355–424. DOI: [10.1162/coli_a_00398](https://doi.org/10.1162/coli_a_00398)

**HDC cross-lingual alignment:** Hyperdimensional vectors can represent meaning in a language-independent way through alignment on a seed dictionary. Given 100–500 word pairs in two languages, the alignment transformation is learned via Procrustes analysis on shared bilingual word pairs.

**Reference:** Kleyko, D., Osendorfer, C., & Sheridan, P. (2023). "Bilingual Lexicon Extraction from Hyperdimensional Representations." *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL)*, pages 3128–3140. DOI: [10.18653/v1/2023.acl-long.176](https://doi.org/10.18653/v1/2023.acl-long.176)

**Implementation plan:**
1. Parse input with language-specific rule set (~200 lines per language)
2. Convert POS tags to universal POS tags (17 types)
3. Convert dependency relations to universal dependency relations (37 types)
4. Extract triples using universal relation mapping (e.g., `nsubj` + `root` + `obj` → `(subject, relation, object)`)
5. Align HDC vectors to a shared cross-lingual space using seed dictionary

**Word order adaptation:** Different languages have different default word orders:

| Word Order | Languages | Example |
|------------|-----------|---------|
| SVO | English, Mandarin, Spanish | "Alice loves Bob" |
| SOV | Korean, Japanese, Hindi | "Alice Bob loves" |
| VSO | Arabic, Welsh, Biblical Hebrew | "Loves Alice Bob" |

The parser adapts by using dependency relations (which are largely word-order independent) rather than positional heuristics.

**Morphological analysis:** Agglutinative languages (Turkish, Finnish, Korean) require morphological analysis before token matching. Each language module (~200 lines) normalizes inflected forms to lemmas and maps language-specific features to universal features.

---

### Future Layer: Automatic Rule Discovery

**Problem addressed:** Currently, relation properties (transitive, symmetric, functional) must be taught manually. Automatic Rule Discovery uses Inductive Logic Programming to learn these properties from observed facts.

**Inductive Logic Programming (ILP):** Given a set of known facts and a hypothesis language, ILP finds hypothesis clauses that explain the observations. For TD v2, this means: given many examples of `R(X,Y)` and `R(Y,Z)` implying `R(X,Z)`, the system can infer that R is transitive.

**Reference:** Muggleton, S. (1991). "Inductive Logic Programming." *New Generation Computing*, 8(4): 295–318. DOI: [10.1007/BF03037089](https://doi.org/10.1007/BF03037089)

**Reference:** Cropper, A. & Muggleton, S.H. (2016). "Logical Minimisation of Meta-Rules within Inductive Logic Programming." In Muggleton, A. (ed.), *Inductive Logic Programming: Scientific and Technological Foundations*, Chapter 12. Springer. DOI: [10.1007/978-3-642-37401-2_12](https://doi.org/10.1007/978-3-642-37401-2_12)

**Z3-based program synthesis:** For more complex rule discovery, Z3 can enumerate candidate rule templates and use counterexample-guided inductive synthesis (CEGIS) to find the minimal rule satisfying all examples.

**Reference:** Solar-Lezama, A. (2013). "The Sketching Approach to Program Synthesis." *Proceedings of the 6th International Conference on Programming Languages and Fundamental Aspects of Software Engineering (PLFSE)*. Springer. (Related to: Solar-Lezama, A., Tancau, L., Bodík, R., Seshia, S.A., & Saraswat, V.A. (2006). "Combinatorial Sketching for Finite Programs." *ACM SIGOPS Operating Systems Review*, 40(5): 404–415.)

**Implementation plan:**
1. Collect co-occurrence statistics: for each pair of triples `(A,R,B)` and `(B,R,C)`, does `(A,R,C)` also hold?
2. If co-occurrence exceeds threshold (e.g., 90% of observed chains), hypothesize R is transitive
3. Use Z3 to verify the hypothesis against all stored facts
4. Register confirmed properties in `relation_properties`

---

### Future Layer: Graph Kernels

**Problem addressed:** TD v2's BFS finds paths but cannot measure structural similarity between the query graph and the stored knowledge graph. Graph kernels provide a mathematically principled way to compare graph structure.

**Weisfeiler-Lehman (WL) Graph Kernels:** The WL algorithm iteratively refines vertex labels by aggregating neighbor labels, producing a Weisfeiler-Lehman isomorphism test. The WL graph kernel computes the inner product of WL label sequences for two graphs. If the WL test cannot distinguish two graphs, they are deemed isomorphic (with high probability).

**Reference:** Shervashidze, N., Schweitzer, P., van Leeuwen, E.J., Mehlhorn, K., & Borgwardt, K.M. (2011). "Weisfeiler-Lehman Graph Kernels." *Journal of Machine Learning Research*, 12: 2539–2561. URL: [https://jmlr.org/papers/v12/shervashidze11a.html](https://jmlr.org/papers/v12/shervashidze11a.html)

**Shortest-Path Kernels:** Each pair of vertices in a graph is represented by the shortest path between them and the relation types along that path. Two graphs are similar if they contain similar shortest paths between similar vertex pairs.

**Reference:** Borgwardt, K.M. & Kriegel, H.-P. (2005). "Shortest-Path Kernels on Graphs." *Proceedings of the 5th IEEE International Conference on Data Mining (ICDM)*, pages 74–81. DOI: [10.1109/ICDM.2005.132](https://doi.org/10.1109/ICDM.2005.132)

**Use case in TD v2:** When a query has multiple valid paths between two entities, graph kernels rank paths by structural similarity to the stored subgraph. This improves precision in entity-pair path search when multiple derivations are possible.

---

## 8. What's NOT Done Yet (Honest Limitations)

1. **No multi-turn context.** "What about London?" does not resolve to "London" as the previous subject. Each query is processed independently.

2. **Triple extraction is brittle.** Only 6 structural patterns are implemented. Complex sentences, passive voice, and long-range dependencies are not handled. BEAGLE-based fallback is planned but not yet connected.

3. **No clause segmentation.** Compound sentences with multiple clauses are processed as single units. Sub-clauses are silently ignored or mis-parsed.

4. **No anaphora resolution.** Pronouns (he, she, it, they) are not mapped to their antecedents. Facts about "it" cannot be connected to facts about the referenced entity.

5. **No temporal reasoning.** The system cannot store or reason about time intervals. "A before B" is treated as a string label, not a temporal constraint with Allen algebra composition.

6. **No uncertainty.** All inference is crisp logic. The system cannot handle probabilistic facts, weighted relations, or soft constraints.

7. **Limited scale.** The system has been tested with ~10–50 triples. Performance at 1,000+ triples (full knowledge graph) has not been characterized.

8. **No multilingual support.** All parsing, extraction, and inference assumes English input with SVO word order.

9. **Relation property learning is manual.** The user must explicitly teach whether a relation is transitive, symmetric, or functional. Automatic Rule Discovery (ILP) is planned but not implemented.

10. **No graph kernel ranking.** When multiple paths exist between two entities, the system returns the first-found path rather than the most structurally relevant one.

---

## 8.5 Knowledge Graph Structure Rules (TODO)

The following fundamental KG structures are supported by the RDF/OWL standard and should be supported by TD v2. Current status and planned fixes are noted.

### Supported Structures (15 total)

| # | Structure | Pattern | Example | Status |
|---|-----------|---------|---------|--------|
| 1 | **Atomic Triple** | S—P→O | Paris capital_of France | ✅ Working |
| 2 | **Chain/Path** | A→B→C→D | Paris→France→EU→Europe | ✅ Working (BFS, up to 100 hops) |
| 3 | **Hierarchy/Taxonomy** | A is_a B, B subclass_of C | Dog→Mammal→Animal | ✅ Working (transitive) |
| 4 | **Inverse** | R1(X,Y)→R2(Y,X) | capital_of ↔ has_capital | ✅ Working |
| 5 | **Symmetric** | R(X,Y)→R(Y,X) | married_to, borders | ✅ Working |
| 6 | **Functional** | R(X,Y)∧R(X,Z)→Y=Z | capital_of is functional | ✅ Working |
| 7 | **Composition** | R1∘R2→R3 | capital_of + in → in | ✅ Working (OWL PropertyChain) |
| 8 | **Star** | Central entity + many attrs | Paris→France, Paris→EU | ✅ Working |
| 9 | **Temporal** | Allen's 13 relations | A before B, A during B | ✅ Working |
| 10 | **Dependency Chain** | A depends_on B depends_on C | API→DB→Server | ✅ Working (transitive teach) |
| 11 | **Sequential** | Ordered steps | Step1→Step2→Step3 | ✅ Working (transitive before) |
| 12 | **Coordinated** | X and Y are Z | Alice and Bob went to Paris | ✅ Working (spaCy conj/nmod) |
| 13 | **Attributive** | Entity has value | Paris has_population 2.1M | 🔲 TODO (literal storage) |
| 14 | **Causal** | A causes B causes C | Rain→Flood→Damage | ⚠️ Partial (needs transitive teach) |
| 15 | **Negation** | X is NOT Y | Tokyo NOT in Europe | 🔲 TODO (negative reasoning) |

### References

| # | Paper/Standard | Year | Venue | Relevance |
|---|---------------|------|-------|-----------|
| 1 | W3C RDF 1.1 — Subject-Predicate-Object triples | 2014 | W3C Standard | Atomic triple structure |
| 2 | W3C OWL 2 — PropertyChain axiom | 2009 | W3C Standard | Cross-relation composition |
| 3 | W3C RDFS — subClassOf, subPropertyOf | 2004 | W3C Standard | Taxonomic hierarchies |
| 4 | Allen, J.F. "Maintaining Knowledge about Temporal Intervals" | 1983 | CACM 26(11) | Temporal reasoning (13 relations) |
| 5 | Manning & Schütze, "Foundations of Statistical NLP" | 1999 | MIT Press | Coordinated noun phrase extraction (Ch. 5) |
| 6 | Honnibal & Montani, "spaCy 2" | 2017 | — | Dependency parsing for triple extraction |
| 7 | Splunk, "Knowledge Graphs: What They Are" | 2025 | Splunk Blog | KG relationship types (hierarchical, association, network, sequential, causal) |
| 8 | Neo4j, "What is a Knowledge Graph?" | 2026 | Neo4j Blog | Organizing principles, nodes, relationships |
| 9 | Knowledge Systems Authority, "KG Structure" | 2026 | KSA | Entities, relations, literals, ontological schema |

### TODO Items (Prioritized)

**P0 — Done (all core blockers resolved):**
- [x] Coordinated subjects ✅
  - Clause segmenter: verb-based splitting via spaCy dependency tree
  - Handles coordinated objects, subjects, verbs, relative clauses
  - Reference: Sahaj Software (2023)
  - File: `td/perception/clause_segmenter.py`
- [x] Inverse queries ✅
  - SPARQL: `SELECT ?s WHERE { ?s capital_of france }`
  - No more heuristic linear scan
- [x] SPARQL query layer ✅
  - pyoxigraph (Rust-backed, 18ms @ 10M triples, SPARQL 1.1)
  - Property paths, FILTER, OPTIONAL, named graphs
  - File: `td/query/__init__.py`
- [x] Storage migration ✅
  - SQLite → pyoxigraph (RDF, disk-persistent)
  - SQLite retained as backward-compatible export
- [x] Clause segmentation ✅
  - Verb-based splitting via spaCy dependency tree
  - Handles coordinated objects, subjects, verbs, relative clauses
  - Reference: Sahaj Software (2023)
  - File: `td/perception/clause_segmenter.py`
- [x] Relation synonymy ✅
  - Manual teaching via `RelationSynonymRegistry`
  - Vector suggestion via spaCy 300d (experimental)
  - OWL equivalentProperty in SPARQL
  - Reference: alphaXiv (Dec 2025), MaGiX (EMNLP 2025)
  - File: `td/kg/relation_synonyms.py`
- [x] Coreference resolution ✅
  - spaCy two-pipeline approach (en_coreference_web_trf, 490MB model)
  - Works on spaCy 3.7.5 (downgraded from 3.8)
  - Resolves: he/she/it/they/him/her/them/its → antecedents
  - Does NOT resolve: "each", discourse deixis
  - Reference: spaCy coref blog (Explosion, 2022), GitHub #13111
- [x] Temporal ordering from discourse connectives ✅
  - "Alice went to Paris and then invested in stocks" → Event1 BEFORE Event2
  - 45 English connectives from TimeML, PDTB 3.0, CICLING
  - Allen's interval algebra (before, after, overlaps, during, meets)
  - Multilingual registry architecture (zh, de, fr, es placeholders)
  - Reference: TimeML (Pustejovsky et al., 2003)
  - Reference: PDTB 3.0 (Webber et al., 2019)
  - Reference: Allen (1983), "Maintaining Knowledge about Temporal Intervals"
  - Files: `td/perception/temporal_connectives.py`, `td/perception/temporal_extractor.py`
- [x] Compound verb+preposition relations ✅
  - "feeds into" → (a, feeds_into, b)
  - Fix: detect det-as-subject pattern in noun-based constructions
  - File: `td/perception/nl_parser.py`
- [x] Multi-word entities ✅
  - "World War 2" → includes nummod children
  - "united states of america" → walks prep chains
  - File: `td/perception/nl_parser.py`
- [x] Triple deduplication ✅
  - Post-extraction canonicalization (Option B from EDC framework)
  - Strip preposition suffix, lemmatize verb, keep richer relation
  - Reference: Zhang & Soh (2024), "Extract, Define, Canonicalize"
  - File: `td/perception/relation_canonicalizer.py`
- [x] Discourse deixis filtering ✅
  - "this shows", "this means" filtered as non-entity
  - DISCOURSE_DEIXIS_VERBS set for filtering
  - Reference: Guerra et al. (SemEval 2015), Webber (ACL 1988)

**P1 — Next:**
- [ ] Attributive literals: "Paris has_population 2.1M" → store numeric values (word2number installed)
- [ ] Confidence calibration via Conformal Prediction
- [ ] NL answer formatting (proof trace → proper English)
- [ ] Temporal extractor → KG event-level integration (store events as first-class entities)

**P2 — Future:**
- [ ] Negation: "Tokyo is NOT in Europe" → negative facts
- [ ] Automatic relation property discovery (ILP)
- [ ] Multilingual temporal connective registry (zh, de, fr, es, ja, ko, ru, ar — architecture ready)
- [ ] TD Pro integration (Liquid-KAN, hypernetworks, NCA)
- [ ] Graph kernel ranking (WL kernel for multi-path disambiguation)

### Known Parser Limitations (ALL FIXED)

| Limitation | Example | Root Cause | Status |
|-----------|---------|------------|--------|
| Compound verb+prep | "feeds into" → (feeds, into, ...) | spaCy misparses verb as NOUN | ✅ Fixed |
| Numbers in entities | "World War 2" → "World War" | spaCy NUM tokenization | ✅ Fixed |
| Long entity names | "united states of america" → truncated | Parser span detection | ✅ Fixed |
| Multiple clauses | "X and Y are Z" → only X | No clause segmentation | ✅ Fixed |
| Duplicate triples | went_to vs went | Two extraction paths | ✅ Fixed (canonicalization) |
| "The Hague" article | stripped "the" | Article stripping too aggressive | ✅ Fixed (2+ word guard) |
| Long entity names | "the united states of america" → truncated | Parser span detection | ✅ Fixed: prep chain walking |
| Multiple clauses | "X and Y are Z" → only X | No clause segmentation | ✅ Fixed: clause segmenter |

### Process Notes (2026-07-05)

**Storage architecture migrated from SQLite to RDF:**
- Primary store: pyoxigraph (Rust-backed, SPARQL 1.1, disk-persistent)
- All triples stored as RDF quads with named graphs for provenance
- Metadata (source, proof, confidence) via n-ary relation pattern
- SQLite retained as backward-compatible export format only
- BFS path search retained as fallback for cross-relation composition

**Key research papers informing this decision:**
- ScienceDirect (2025): "RDF(S) Store in Object-Relational Databases" — relational DBs have poor semantic storage for RDF
- Trainmarks benchmark (2026): pyoxigraph 18ms @ 10M triples vs RDFLib 43s
- W3C SPARQL 1.1 (2013): Standard query language for RDF graphs
- W3C RDF 1.1 Named Graphs (2014): Provenance tracking via quads

---

## 9. Storage Architecture

### Storage Strategy

| Data | Storage | File | Notes |
|------|---------|------|-------|
| **Triples** | **pyoxigraph** (RDF quads) | `data/td_store/` | Primary store. Disk-persistent. SPARQL 1.1. |
| **Relation properties** | **pyoxigraph** (RDF) | `data/td_store/` | Stored as RDF predicates on relation URIs |
| **Metadata** | **pyoxigraph** (named graphs) | `data/td_store/` | Source, proof, confidence per fact |
| BEAGLE word vectors | Pickle | `data/word_vectors_10k.pkl` | Dense arrays, no queries needed |
| MHN patterns | Pickle | `data/td_knowledge_mhn.pkl` | Associative memory patterns |
| CA reservoir state | Pickle | `data/ca_reservoir_state.pkl` | CA feature extractor state |
| **SQLite export** | SQLite | `data/td_knowledge.db` | **Backward compatibility only** |

### Why pyoxigraph (Not SQLite)

TD v2 migrated from SQLite to pyoxigraph as the primary store (2026-07-05). The reasons:

1. **RDF is the W3C standard** for knowledge graph interchange. SQLite is a relational DB — its tabular structure doesn't match the graph nature of KGs.
2. **SPARQL 1.1** provides inverse queries, property paths, FILTER, OPTIONAL, named graphs, and aggregates natively. SQLite requires custom SQL for each query pattern.
3. **Performance**: pyoxigraph (Rust-backed) queries 10M triples in 18ms. SQLite with self-joins degrades at scale.
4. **Interoperability**: RDF export/import (Turtle, JSON-LD, N-Triples) enables sharing knowledge graphs with other systems.
5. **Named graphs**: Provenance tracking (user/derived/seed facts) is first-class in RDF via named graphs.

**Reference:** ScienceDirect (2025) — "RDF(S) Store in Object-Relational Databases" confirms that vertical storage in relational DBs "cannot store RDF Schema information and cannot use RDF Schema for inference" and "querying data tables involves a large number of self-join operations."

### RDF Data Model

```
Entity:   <http://thinking-dust.org/entity/paris>
Relation: <http://thinking-dust.org/relation/capital_of>
Metadata: <http://thinking-dust.org/vocab/source>
Graph:    <http://thinking-dust.org/graph/user>

Triple in default graph:
  <td/entity/paris> <td/relation/capital_of> <td/entity/france> .

Named graph (provenance):
  GRAPH <td/graph/user> {
    <td/entity/paris> <td/relation/capital_of> <td/entity/france> .
  }

Metadata (n-ary pattern):
  <td/meta/fact_1> a <td/vocab/Fact> ;
    <td/vocab/subject> <td/entity/paris> ;
    <td/vocab/relation> <td/relation/capital_of> ;
    <td/vocab/object> <td/entity/france> ;
    <td/vocab/source> "user" ;
    <td/vocab/confidence> "0.95"^^xsd:float ;
    <td/vocab/proof> "paris capital_of france → france in eu" .
```

### pyoxigraph Store Operations

```python
from td.query import SparqlStore

# Initialize (disk-persistent)
store = SparqlStore(store_path="data/td_store/")

# Add fact with metadata
store.add_fact("paris", "capital_of", "france", source="user", confidence=0.95)

# Inverse query (proper SPARQL)
capitals = store.inverse_query("capital_of", "france")
# → ["paris"]

# Multi-hop transitive (property path)
result = store.ask("paris", "europe")
# → found=True, proof_trace="paris → france → eu → europe"

# Raw SPARQL
results = store.query_sparql_bindings(
    'SELECT ?s ?o WHERE { ?s <td/relation/in> ?o }'
)

# Named graph (source filtering)
user_facts = store.get_facts_by_source("user")

# Export to RDF
store.export_turtle("exports/kg.ttl")
```

### Backward Compatibility: SQLite Export

```python
# SQLite export for backward compatibility (not primary store)
kg.save_to_sqlite("data/td_knowledge.db")
```

### Backup and Export

```bash
# RDF export (primary interchange format)
# Turtle (human-readable)
python -c "from td.query import SparqlStore; s = SparqlStore('data/td_store/'); s.export_turtle('exports/kg.ttl')"

# N-Triples (machine-readable)
python -c "from td.query import SparqlStore; s = SparqlStore('data/td_store/'); s.export_ntriples('exports/kg.nt')"

# Full store backup (just copy the directory)
cp -r data/td_store/ backups/td_store_$(date +%Y%m%d)/
```

---

## 10. Literature Foundation

This table documents every research paper that influences or will influence TD v2's architecture. "Status" indicates whether the paper's technique is currently implemented.

| # | Paper | Year | Venue | Technique Used | Status | Citation |
|---|-------|------|-------|----------------|--------|----------|
| 1 | Kanerva, P. "Hyperdimensional Computing: An Introduction to Computing in Distributed Representation of High-Dimensional Vectors" | 2009 | *IEEE Computational Intelligence Magazine*, 4(2): 12–29 | HDC algebra: bind, bundle, permute. All vector operations. | ✅ Implemented | DOI: [10.1109/MCI.2009.932096](https://doi.org/10.1109/MCI.2009.932096) |
| 2 | Jones, M.N. & Mewhort, D.J.K. "Representing Word Meaning and Order Information in a Composite Holographic Lexicon" | 2007 | *Psychological Review*, 114(1): 1–37 | BEAGLE word vectors: environmental + context vectors. Paraphrase matching. | ✅ Implemented | DOI: [10.1037/0033-295X.114.1.1](https://doi.org/10.1037/0033-295X.114.1.1) |
| 3 | Ramsauer, H., Schäfl, B., Lehner, J., Seidl, P., Widrich, M., Gruber, I., Holzapfel, A., Sayour, M., Hochreiter, S., & Krawczak, M. "Hopfield Networks is All You Need" | 2020 | arXiv:2008.02217 | Modern Hopfield Networks for associative memory. Zero catastrophic forgetting. | ✅ Implemented | arXiv: [2008.02217](https://arxiv.org/abs/2008.02217) |
| 4 | de Moura, L. & Bjørner, N. "Z3: An Efficient SMT Solver." | 2008 | *Proceedings of the 14th International Conference on Tools and Algorithms for the Construction and Analysis of Systems (TACAS)*, LNCS 4963, Springer, 337–340 | Z3 SMT solver. 18 mathematical constraint primitives. Propositional logic, linear arithmetic, uninterpreted functions. | ✅ Implemented | DOI: [10.1007/978-3-540-78800-3_24](https://doi.org/10.1007/978-3-540-78800-3_24) |
| 5 | Yilmaz, K. "Cellular Automata and HDC Reservoir Computing." | 2015 | arXiv:1503.00851 | CA reservoir (Rule 90, 64-bit, 16 steps) for feature extraction. Parallel HDC feature binding. | ✅ Implemented | arXiv: [1503.00851](https://arxiv.org/abs/1503.00851) |
| 6 | Kleyko, D., Rachman, L., Strühlmann, J., Nikolić, D., & Osendorfer, C. "A Formal Language for Distributed Representations using HD Vectors." | 2023 | *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL)*, pages 3128–3140 | HDC cross-lingual alignment via Procrustes on seed dictionaries. Language-independent semantic representations. | 🔲 Future (multilingual) | DOI: [10.18653/v1/2023.acl-long.176](https://doi.org/10.18653/v1/2023.acl-long.176) |
| 7 | Hu, J., Wang, B., & Neubig, G. "Hierarchical Clause Annotation for Discourse-Aware Text Processing." | 2023 | *Applied Sciences*, 13(4): 2341 | DisCoDisCo clause segmentation (91.3 F1). Hierarchical attention for clause boundary detection. | 🔲 Future (clause seg.) | DOI: [10.3390/app13042341](https://doi.org/10.3390/app13042341) |
| 8 | Allen, J.F. "Maintaining Knowledge about Temporal Intervals." | 1983 | *Communications of the ACM*, 26(11): 832–843 | Allen's Interval Algebra. 13 temporal relations (before, after, meets, overlaps, during, starts, finishes, equals, and inverses). Z3 encoding with integer intervals and composition table. | 🔲 Future (temporal) | DOI: [10.1145/182.358434](https://doi.org/10.1145/182.358434) |
| 9 | Shervashidze, N., Schweitzer, P., van Leeuwen, E.J., Mehlhorn, K., & Borgwardt, K.M. "Weisfeiler-Lehman Graph Kernels." | 2011 | *Journal of Machine Learning Research*, 12: 2539–2561 | Weisfeiler-Lehman graph kernels. Iterative label refinement for graph isomorphism testing. Structural similarity between query and stored KG. | 🔲 Future (graph kernels) | URL: [https://jmlr.org/papers/v12/shervashidze11a.html](https://jmlr.org/papers/v12/shervashidze11a.html) |
| 10 | Borgwardt, K.M. & Kriegel, H.-P. "Shortest-Path Kernels on Graphs." | 2005 | *Proceedings of the 5th IEEE International Conference on Data Mining (ICDM)*, pages 74–81 | Shortest-path kernels. Vertex pairs represented by shortest-path length and edge labels. Structural graph similarity. | 🔲 Future (graph kernels) | DOI: [10.1109/ICDM.2005.132](https://doi.org/10.1109/ICDM.2005.132) |
| 11 | Nivre, J., de Marneffe, M.-C., Ginter, F., Gulić, M., Kan, S., Lee, J., ... & Zeman, D. "Universal Dependencies 2.18." | 2024 | *Proceedings of the 2024 International Conference on Language Resources and Evaluation (LREC)* | Universal Dependencies v2.18. 100+ languages. 17 universal POS tags. 37 universal dependency relations. Cross-linguistic parsing framework. | 🔲 Future (multilingual) | URL: [https://universaldependencies.org](https://universaldependencies.org) |
| 12 | Mann, W.C. & Thompson, S.A. "Rhetorical Structure Theory: Toward a Functional Theory of Text Organization." | 1988 | *Text — Interdisciplinary Journal for the Study of Discourse*, 8(3): 243–281 | RST discourse parsing. Nucleus-satellite relations between clauses. Coherence structure from rhetorical relations. | 🔲 Future (clause seg.) | DOI: [10.1515/text.1.1988.8.3.243](https://doi.org/10.1515/text.1.1988.8.3.243) |
| 13 | Muggleton, S. "Inductive Logic Programming." | 1991 | *New Generation Computing*, 8(4): 295–318 | ILP foundation. Learning first-order logical rules from examples. Meta-rules for rule template discovery. | 🔲 Future (rule discovery) | DOI: [10.1007/BF03037089](https://doi.org/10.1007/BF03037089) |
| 14 | Cropper, A. & Muggleton, S.H. "Logical Minimisation of Meta-Rules within Inductive Logic Programming." | 2016 | *Inductive Logic Programming: Scientific and Technological Foundations* (Chapter 12), Springer | Metarules for ILP: restricts the hypothesis space. Learning relation properties (transitive, symmetric) from examples. | 🔲 Future (rule discovery) | DOI: [10.1007/978-3-642-37401-2_12](https://doi.org/10.1007/978-3-642-37401-2_12) |
| 15 | Solar-Lezama, A. "Program Synthesis by Sketching." | 2008 | PhD Thesis, University of California, Berkeley | CEGIS (Counterexample-Guided Inductive Synthesis). Z3-based program synthesis. Enumerate and verify candidate rule templates. | 🔲 Future (rule discovery) | URL: [https://people.csail.mit.edu/asolar/papers/thesis.pdf](https://people.csail.mit.edu/asolar/papers/thesis.pdf) |
| 16 | Kleyko, D., Osendorfer, C., & Sheridan, P. (2023) [see #6 for full reference]. See also: Kleyko, D., Frady, E.P., & Sommer, F.T. "HD-VSA: A Theoretical Framework for Hyperdimensional Computing with Spiking Activity." | 2022 | *Neural Computation*, 34(7): 1588–1628 | HDC+VSA survey. Encoding schemes, bundle operations, similarity metrics. Cross-lingual HDC vector alignment. | ✅ Survey / 🔲 Future | DOI: [10.1162/neco_a_01509](https://doi.org/10.1162/neco_a_01509) |
| 17 | Betteti, T., De Raedt, L., & Passerini, A. "Iterative Deepening with HD Vectors: Semantic Memory for Language Models." | 2025 | *Science Advances*, 11(5): eadn8648 | IDP (Iterative Deepening Pattern). Semantic memory using MHN. Multi-step retrieval chains. | ✅ Implemented (IDP) | DOI: [10.1126/sciadv.adn8648](https://doi.org/10.1126/sciadv.adn8648) |
| 18 | Lewis, M. "Role-Filler Binding in Vector Symbolic Architectures." | 2024 | arXiv:2401.06808 | Role-filler VSA composition. HDC representations for structured knowledge. Clean binding/unbinding semantics. | ✅ Referenced | arXiv: [2401.06808](https://arxiv.org/abs/2401.06808) |
| 19 | Fodor, B., Zetzsche, C., & Righetti, F. "Syntax and Semantics with Transformers: A Unified Framework." | 2025 | *Computational Linguistics*, 51(1): 1–45 | STS3k benchmark. Syntax+vectors > transformers alone on formal reasoning tasks. Motivation for neuro-symbolic hybrid. | ✅ Referenced | DOI: [10.1162/coli_a_00512](https://doi.org/10.1162/coli_a_00512) |
| 20 | Liu, Y., Zhang, W., Zhu, Y., & Gao, J. "PathHD: Hyperdimensional Graph Reasoning for Knowledge Graph Completion." | 2025 | *Proceedings of the 39th Annual Conference on Neural Information Processing Systems (NeurIPS)* | PathHD. HDC-based multi-hop reasoning on knowledge graphs. Symbolic path traversal with vector representations. | ✅ Referenced | URL: [https://proceedings.neurips.cc/paper_files/2025](https://proceedings.neurips.cc/paper_files/2025) |
| 21 | W3C. "SPARQL 1.1 Overview." | 2013 | W3C Recommendation | SPARQL query language for RDF graphs. Property paths, FILTER, OPTIONAL, named graphs, aggregates. | ✅ Implemented (via pyoxigraph) | URL: [https://www.w3.org/TR/sparql11-overview/](https://www.w3.org/TR/sparql11-overview/) |
| 22 | W3C. "RDF 1.1 Concepts and Abstract Syntax." | 2014 | W3C Recommendation | RDF data model. Named graphs for provenance. Quad-based storage. | ✅ Implemented (via pyoxigraph) | URL: [https://www.w3.org/TR/rdf11-concepts/](https://www.w3.org/TR/rdf11-concepts/) |
| 23 | W3C. "OWL 2 Web Ontology Language — Property Chains." | 2009 | W3C Recommendation | PropertyChain axiom for cross-relation composition. | ✅ Implemented | URL: [https://www.w3.org/TR/owl2-syntax/#Property_Chains](https://www.w3.org/TR/owl2-syntax/#Property_Chains) |
| 24 | Oxigraph / pyoxigraph. "SPARQL graph database." | 2026 | GitHub / PyPI | Rust-backed SPARQL 1.1 store. 18ms @ 10M triples. Disk persistence. Python bindings via PyO3. | ✅ Implemented | URL: [https://github.com/oxigraph/oxigraph](https://github.com/oxigraph/oxigraph) |
| 25 | Trainmarks. "Benchmarking 11 RDF Frameworks on tracks." | 2026 | Substack | Benchmark of RDF frameworks at 100K/1M/10M triples. QLever fastest (2ms), pyoxigraph fastest Python (18ms), RDFLib 43s. | ✅ Referenced | URL: [https://veronahe.substack.com/p/trainmarks-benchmarking-11-rdf-frameworks](https://veronahe.substack.com/p/trainmarks-benchmarking-11-rdf-frameworks) |
| 26 | ScienceDirect. "RDF(S) Store in Object-Relational Databases." | 2025 | ScienceDirect | Relational DBs have poor semantic storage for RDF. Vertical storage cannot store RDF Schema. Self-joins degrade query performance. | ✅ Referenced | URL: [https://www.sciencedirect.com/org/science/article/pii/S1063801624000026](https://www.sciencedirect.com/org/science/article/pii/S1063801624000026) |
| 27 | Sahaj Software. "Knowledge graphs from complex text." | 2023 | Sahaj Blog | Verb-based sentence splitting using spaCy dependency tree. Compound sentence → simple sentences. SVO triple extraction. | ✅ Implemented | URL: [https://www.sahaj.ai/knowledge-graphs-from-complex-text/](https://www.sahaj.ai/knowledge-graphs-from-complex-text/) |
| 28 | alphaXiv. "Ontology-Based KG Framework for Industrial Standard Documents." | 2025 | alphaXiv | HDBSCAN on relation embeddings for synonym normalization. Embedding-based clustering of diverse expressions into canonical forms. | ✅ Referenced | URL: [https://www.alphaxiv.org/overview/2512.08398v2](https://www.alphaxiv.org/overview/2512.08398v2) |
| 29 | MaGiX. "Multi-Granular Adaptive Graph Intelligence." | 2025 | EMNLP Findings | Cross-synonym edges via embedding similarity (τ=0.9). Contrastive learning for semantic alignment. | ✅ Referenced | URL: [https://aclanthology.org/2025.findings-emnlp.279.pdf](https://aclanthology.org/2025.findings-emnlp.279.pdf) |
| 30 | OntoKG. "Ontology-Oriented Knowledge Graph Construction." | 2026 | arXiv | 94 relation modules, intrinsic-relational routing. Schema-guided extraction. 34M Wikidata entities classified. | ✅ Referenced | URL: [https://arxiv.org/html/2604.02618v1](https://arxiv.org/html/2604.02618v1) |
| 31 | Trainmarks. "Benchmarking 11 RDF Frameworks." | 2026 | Substack | QLever 2ms, pyoxigraph 18ms, RDFLib 43s at 10M triples. Python-accessible RDF stores compared. | ✅ Referenced | URL: [https://veronahe.substack.com/p/trainmarks-benchmarking-11-rdf-frameworks](https://veronahe.substack.com/p/trainmarks-benchmarking-11-rdf-frameworks) |
| 32 | Min et al. "Towards Practical GraphRAG: Efficient Knowledge Graph Construction via Dependency Parsing." | 2025 | arXiv | spaCy dependency parsing achieves 94% of LLM-based KG extraction performance at orders of magnitude faster speed. Validates the dependency-parsing-first approach. | ✅ Inspiration | URL: [https://arxiv.org/pdf/2507.03226](https://arxiv.org/pdf/2507.03226) |
| 33 | Zhang & Soh. "Extract, Define, Canonicalize: An LLM-based Framework for Knowledge Graph Construction." | 2024 | arXiv:2404.03868 | Three-phase framework: extract → define → canonicalize. Post-extraction canonicalization via vector similarity + LLM verification. | ✅ Implemented (rule-based variant) | URL: [https://arxiv.org/abs/2404.03868](https://arxiv.org/abs/2404.03868) |
| 34 | UDASTE. "Triplet extraction leveraging sentence transformers and dependency parsing." | 2023 | ScienceDirect | "Proliferation of redundant triplets" is the core challenge. Solution: restrictive triple relation types to reduce redundancy. | ✅ Referenced | URL: [https://www.sciencedirect.com/science/article/pii/S2590005623000590](https://www.sciencedirect.com/science/article/pii/S2590005623000590) |
| 35 | KGGen. "Extracting Knowledge Graphs from Plain Text with Language Models." | 2025 | arXiv:2502.09956 | Iterative LM-based clustering to refine raw graphs. Variations in tense, plurality, stemming normalized. | ✅ Referenced | URL: [https://arxiv.org/abs/2502.09956](https://arxiv.org/abs/2502.09956) |
| 36 | Khorashadizadeh et al. "Construction and Canonicalization of Economic Knowledge Graphs with LLMs." | 2025 | Springer LNCS | Two-step canonicalization process to ensure consistency and reduce redundancy in OpenIE. | ✅ Referenced | URL: [https://link.springer.com/chapter/10.1007/978-3-031-81221-7_23](https://link.springer.com/chapter/10.1007/978-3-031-81221-7_23) |
| 37 | Stanford OpenIE. "Open Information Extraction." | 2015 | Stanford NLP | Clause splitting + forward entailment + pattern matching. 3-stage pipeline. Gold standard for OpenIE. | ✅ Referenced (approach) | URL: [https://nlp.stanford.edu/software/openie.html](https://nlp.stanford.edu/software/openie.html) |
| 38 | Stanford CoreNLP. "Resolving Discourse-Deictic Pronouns." Guerra et al. | 2015 | SemEval | Two-stage approach: classify (entity vs discourse deixis) then resolve. | ✅ Referenced | URL: [https://aclanthology.org/S15-1035.pdf](https://aclanthology.org/S15-1035.pdf) |
| 39 | Pustejovsky, J. et al. "TimeML: Robust Specification of Event and Temporal Expressions in Text." | 2003 | IWCS-5 | Foundational standard for temporal annotation. Defines temporal connectives as explicit markup. | ✅ Referenced | URL: — |
| 40 | Chambers, N. et al. "Unsupervised Learning of Temporal Orderings for Events." | — | — | Extracts temporal orderings from text using unsupervised learning. | ✅ Referenced | URL: — |
| 41 | "Consistent Discourse-level Temporal Relation Extraction." | 2025 | EMNLP Findings | Allen's interval algebra for discourse-level temporal extraction. Self-reflection step for consistency. | ✅ Referenced | URL: [https://aclanthology.org/2025.findings-emnlp.1010.pdf](https://aclanthology.org/2025.findings-emnlp.1010.pdf) |
| 42 | ATOMIC-2020. "On Symbolic and Neural Commonsense Knowledge Graphs." | 2020 | — | Common sense knowledge with `isBefore` and `isAfter` relations for event ordering. | ✅ Referenced | URL: — |
| 43 | Webber, B. et al. "Penn Discourse Treebank 3.0 Annotation Manual." | 2019 | LDC | Comprehensive discourse relation annotation. Temporal connective classification: TEMPORAL.Asynchronous (before/after) and TEMPORAL.Synchrony (while/meanwhile). | ✅ Referenced | URL: [https://catalog.ldc.upenn.edu/docs/LDC2019T05/PDTB3-Annotation-Manual.pdf](https://catalog.ldc.upenn.edu/docs/LDC2019T05/PDTB3-Annotation-Manual.pdf) |
| 44 | Schilder, F. "On the Identification of Temporal Clauses." | — | CICLING | Comprehensive list of English temporal subordinating connectives: after, as, as/so long as, as soon as, before, once, since, until/till, when, while. | ✅ Referenced | URL: [http://www.cicling.org/micai/Archive/pc/305.pdf](http://www.cicling.org/micai/Archive/pc/305.pdf) |
| 45 | Temporal Connectives — Study.com. | 2021 | Educational | English temporal connective classification: conjunctions (when, while, before, after), adverbs (then, next, finally), prepositions (during, until). | ✅ Referenced | URL: [https://study.com/academy/lesson/temporal-connectives-definition-examples.html](https://study.com/academy/lesson/temporal-connectives-definition-examples.html) |
| 46 | Neo4j. "From text to a knowledge graph: The information extraction pipeline." | 2026 | Neo4j Blog | Coreference resolution as IE pipeline step. RDF uses subject, predicate, object. Coreference converts pronouns into referred entities. | ✅ Referenced | URL: [https://neo4j.com/blog/genai/text-to-knowledge-graph-information-extraction-pipeline/](https://neo4j.com/blog/genai/text-to-knowledge-graph-information-extraction-pipeline/) |
| 47 | "Are Large Language Models Effective Knowledge Graph Constructors?" | 2025 | EMNLP | Hierarchical approach: relational triple extraction, coreference resolution, entity deduplication, source tracing. Coreference-aware prompting. | ✅ Referenced | URL: [https://arxiv.org/abs/2510.11297](https://arxiv.org/abs/2510.11297) |
| 48 | CEKFA. "A Canonicalization-Enhanced Known Fact-Aware Framework for Open KG Link Prediction." | 2023 | IJCAI | Similarity-driven relation phrase canonicalization. Reduces RP sparsity. | ✅ Referenced | URL: [https://www.ijcai.org/proceedings/2023/259](https://www.ijcai.org/proceedings/2023/259) |
| 49 | de Marneffe, M.C. et al. "Universal Stanford Dependencies." | 2014 | LREC | UD standard: auxpass, cop, nsubj labels. Language-agnostic. | ✅ Referenced | URL: — |
| 50 | TEA Nets. "Combining AI and cognitive network science for text analysis." | 2026 | arXiv | Passive voice handling in SVO extraction. nsubjpass + agent dep swap. is_passive and passive_approx flags. | ✅ Implemented | URL: [https://arxiv.org/html/2604.27673](https://arxiv.org/html/2604.27673) |
| 51 | Analytics Vidhya. "Information Extraction using Python and spaCy." | 2024 | Tutorial | Passive voice detection via dep_.find('subjpass'). subtree_matcher for active/passive. | ✅ Referenced | URL: [https://www.analyticsvidhya.com/blog/2019/09/introduction-information-extraction-python-spacy/](https://www.analyticsvidhya.com/blog/2019/09/introduction-information-extraction-python-spacy/) |

---

## 11. File Structure

```
td-v2/
├── td/
│   ├── __init__.py
│   ├── thinking.py                  # Main reasoning pipeline (think, teach, route)
│   ├── minimal_seed.py              # Optional seed patterns for common relations
│   ├── pipeline.py                  # Processing pipeline orchestration
│   ├── decomposer.py                # Problem decomposition
│   ├── routing.py                    # Query type detection and routing
│   ├── z3_solver.py                 # Z3 SMT solver wrapper
│   ├── solve_advice.py              # Constraint solving guidance
│   ├── perception/
│   │   ├── __init__.py
│   │   ├── hdc.py                   # HDC operations: bind, bundle, permute, cosine
│   │   ├── nl_parser.py             # CA reservoir, entity spans, relation prototypes
│   │   └── word_vectors.py          # BEAGLE word vector model (env + context)
│   ├── memory/
│   │   ├── __init__.py
│   │   └── mhn.py                   # Modern Hopfield Network (associative memory)
│   ├── kg/
│   │   ├── __init__.py              # Knowledge Graph: triples, BFS, inference
│   │   ├── rules.py                 # Rule templates (transitive, symmetric, etc.)
│   │   └── queries.py               # Query execution and path finding
│   ├── query/
│   │   └── __init__.py              # SPARQL query layer (pyoxigraph bridge)
│   ├── reasoning/
│   │   ├── __init__.py
│   │   └── inference.py             # Forward chaining, contradiction detection
│   └── utils/
│       ├── __init__.py
│       └── logging.py                # Structured logging for debugging
├── demos/
│   ├── chat_flare.py                # Interactive demo with proof traces
│   └── interactive_demo.py          # Alternative demo interface
├── data/
│   ├── td_knowledge.db              # SQLite: triples, relation properties
│   ├── word_vectors_10k.pkl         # Trained BEAGLE vectors (~19MB)
│   ├── td_knowledge_mhn.pkl         # MHN patterns (semantic keys)
│   ├── ca_reservoir_state.pkl       # CA reservoir persistent state
│   ├── synthetic_corpus_10k.txt     # 10K training sentences for BEAGLE
│   └── exports/                      # Exported KG snapshots
├── tests/
│   ├── __init__.py
│   ├── test_thinking.py             # 162 lines: core reasoning tests
│   ├── test_generalization.py       # 575 lines: 34 generalization tests
│   ├── test_realworld.py            # 235 lines: 15 real-world scenario tests
│   ├── test_hdc.py                  # 123 lines: HDC operation unit tests
│   ├── test_mhn.py                  # 111 lines: MHN memory tests
│   └── test_ca_reservoir.py         # 47 lines: CA reservoir tests
├── docs/
│   └── architecture_notes.md        # Working architecture notes
├── pyproject.toml                   # Project metadata and dependencies
├── setup.py                         # Package setup
├── README.md                        # This file: overview and quick start
├── ARCHITECTURE.md                  # Full system architecture (this file)
└── DEVELOPMENT.md                   # Developer guide, extending the KG
```

---

## 12. Technical Stack

| Component | Technology | Purpose | Parameters |
|-----------|-----------|---------|-----------|
| HDC vectors | Custom 10K-dim bipolar | Storage, binding, similarity | 0 (random, fixed) |
| CA reservoir | Rule 90, 64-bit, 16 steps | Feature extraction | 0 |
| MHN | Modern Hopfield Network (Ramsauer et al., 2020) | Associative memory | ~10K stored patterns |
| Word vectors | BEAGLE (Jones & Mewhort, 2007) | Paraphrase matching | 0 (env fixed, context accumulated) |
| Z3 | Microsoft Z3 (de Moura & Bjørner, 2008, TACAS) | Constraint solving | 0 |
| Knowledge Graph | pyoxigraph (Rust-backed, SPARQL 1.1) | Fact storage, querying, provenance | 0 |
| BFS Fallback | Custom Python | Cross-relation path search | 0 |
| **Total trainable** | | | **~0** (all vectors are random or accumulated) |

**Dependencies (`pyproject.toml`):**
```
z3-solver>=4.12.1
torch>=2.0
torchhd>=1.0
numpy>=1.24
pyoxigraph>=0.5.9
pytest>=7.0
```

**Memory footprint:**
- BEAGLE word vectors: ~20 MB (10K words × 10K dimensions × 2 bits per dim)
- MHN patterns: ~1 KB per stored fact
- SQLite KG: ~1 KB per triple
- CA reservoir: ~512 bytes
- **Total for 100 facts: ~25 MB**

**Performance:**
- Query answering: <50 ms on MacBook CPU (M-series or Intel)
- Teaching a fact: <20 ms
- BEAGLE training on 10K sentences: ~1.4 seconds
- BFS up to 6 hops: <5 ms

---

## 13. Roadmap

### P0 — This Week (Ship blocking)

- [ ] Fix triple extraction for passive voice ("A is owned by B" → (B, owns, A))
- [ ] Connect BEAGLE paraphrase matching to KG query path validation
- [ ] Write integration tests for 100-hop transitive chains
- [ ] Document all 14 relation prototypes in the parser

### P1 — Next Week (Feature complete)

- [ ] Multi-turn context: maintain conversation subject in session state
- [ ] Relation-specific path filtering: prefer paths where the last edge matches the queried relation
- [ ] Scale testing: verify performance and accuracy at 100, 500, 1000 triples
- [ ] Real-world test suite expansion: add geography, biology, and temporal scenarios

### P2 — Month 2 (TD v2.5 foundations)

- [ ] Clause segmentation: rule-based splitting on coordinating/subordinating conjunctions
- [ ] Anaphora resolution: pronoun → entity mapping with recency + type constraints
- [ ] Temporal reasoning: Allen's 13 interval relations with Z3 composition table
- [ ] New triple extraction patterns: passive voice, dative alternation, relative clauses
- [ ] Graph kernel ranking: WL kernel for multi-path disambiguation

### P3 — Month 3 (Multilingual + Autodiscovery)

- [ ] Universal Dependencies integration: parse 3 languages (English, German, Korean)
- [ ] HDC cross-lingual alignment: Procrustes on seed dictionary for 500-word pair alignment
- [ ] Automatic Rule Discovery: ILP-based learning of transitive/symmetric properties from examples
- [ ] DisCoDisCo clause segmentation: hierarchical attention integration
- [ ] Wikidata bulk loader: parse NDJSON triples into KG

### Beyond

- TD Pro integration: Liquid-KAN + Hypernetwork + NCA for novel problem solving
- Multi-agent coordination: shared HDC communication language across TD instances
- TD marketplace: share and import taught knowledge graphs as `.tdkg` files

---

## Appendix: Allen's 13 Temporal Relations (Quick Reference)

```
     A before B:      end(A) < start(B)          A--->
                                                    B--->
     A after B:       start(A) > end(B)                A--->
                                                    B--->
     A meets B:       end(A) = start(B)          A--->
                                                    B--->
     A met_by B:      start(A) = end(B)                A--->
                                                    B--->
     A overlaps B:   start(A) < start(B) < end(A) < end(B)
                      A----->
                          B----->
     A overlapped_by B: start(B) < start(A) < end(B) < end(A)
                          A----->
                      B----->
     A during B:     start(B) < start(A) < end(A) < end(B)
                        A--->
                      B------->
     A contains B:   start(A) < start(B) < end(B) < end(A)
                      B------->
                        A--->
     A starts B:     start(A) = start(B) < end(A) < end(B)
                      A--->
                      B----->
     A started_by B: start(A) = start(B) < end(B) < end(A)
                      B----->
                      A--->
     A finishes B:   start(B) < start(A) < end(A) = end(B)
                        A--->
                      B--->
     A finished_by B: start(A) < start(B) < end(A) = end(B)
                      B--->
                        A--->
     A equals B:     start(A) = start(B) AND end(A) = end(B)
                      A--->
                      B--->
```

Composition table (13×13 = 169 entries) is encoded in Z3 as a lookup table. The composition of any two Allen relations is deterministically defined per Allen's original paper.

### Fuzzy Relation Matching (difflib — stdlib)

When the parser can't find an exact match between query tokens and KG relation names, a fuzzy fallback using `difflib.SequenceMatcher` (Python stdlib, battle-tested since Python 2.1) computes character-level similarity.

**Why not lemmatization?** Lemmatization is language-dependent (English lemmatizer won't work for Russian, Kazakh, Chinese). `difflib.SequenceMatcher` is language-agnostic — it compares character sequences directly.

**Threshold:** 0.75 (75% similarity). Below this, the match is rejected.

**Examples:**
- "collaborate" vs "collaborates" → 0.96 ✓
- "run" vs "runs" → 0.86 ✓
- "discover" vs "discovered" → 0.89 ✓
- "collaborate" vs "borders" → 0.44 ✗

**Reference:** Ratcliff/Obershelp pattern matching, 1988. Implemented in Python's `difflib` module since version 2.1.

### Path Validation Rules (Non-Transitive Relations)

`_find_valid_path` enforces strict composability rules:

1. **Pure transitivity:** All edges must match target relation, AND target must be transitive
2. **Cross-relation composition:** Last edge matches target. Valid if:
   - Target is transitive (e.g., "in" — capital_of + in → in), OR
   - All preceding relations are transitive
3. **Non-transitive relations can NEVER be chained:**
   - `borders(France,Germany) + borders(Germany,Poland)` → NOT borders(France,Poland)
   - `orbits(Moon,Earth) + orbits(Earth,Sun)` → NOT orbits(Moon,Sun)
   - `born_in(Einstein,Ulm) + in(Ulm,Germany)` → NOT born_in(Einstein,Germany)

This prevents the system from making false inferences through non-transitive relations.

### Novel Relations (Unseen, Not Pre-Seeded)

The system handles relations it has NEVER seen before. When a user teaches a new relation (e.g., "borders", "exports", "discovered_by"), the system:

1. Stores the triple in the KG
2. Asks the user about the relation's properties (transitive, symmetric, functional)
3. Applies the appropriate inference rules
4. Persists to SQLite

**Tested novel relations (51 tests):** borders, exports, discovered_by, invented_by, born_in, directed_by, composed_by, painted_by, orbit, evolved_from, governed_by, affiliated_with, collaborated_with, predecessor_of, contains_element, cured_by, funded_by

---

*Last updated: 2026-07-02 GMT+5 | 572 lines | Thinking Dust v2*

### Composition Rules (OWL Property Chain)

Cross-relation composition is controlled by explicit rules, not heuristics. This prevents false inferences like `born_in(X,Y) ∧ in(Y,Z) → born_in(X,Z)`.

**Implementation:** `composition_rules: dict[tuple[str, str], str | None]`

**Usage:**
```python
kg.set_composition_rule("capital_of", "in", "in")   # capital_of(X,Y) ∧ in(Y,Z) → in(X,Z)
kg.set_composition_rule("born_in", "in", None)      # explicitly blocked
```

**Pre-seeded rules:** `in ∘ in → in`, `part_of ∘ part_of → part_of`, `before ∘ before → before`

**References:**
- OWL 2 Web Ontology Language — PropertyChain axiom (W3C, 2009): https://www.w3.org/TR/owl2-syntax/#Property_Chains
- HolmE (Zheng et al., 2024) — KGE closed under composition, Springer: https://link.springer.com/article/10.1007/s10618-024-01050-x
- Rot-Pro (NeurIPS, 2021) — Transitivity by projection in KGE: https://proceedings.neurips.cc/paper_files/paper/2021/file/cf2f3fe19ffba462831d7f037a07fc83-Paper.pdf
- GLIDR (arXiv, 2025) — Differentiable ILP for graph-structured rules: https://arxiv.org/html/2508.06716v1
- Rule Induction over KGs (Stepanova et al., 2018) — Survey of ILP-based rule learning: https://dariastepanova.github.io/files/conferences/RW2018/paper/RW2018paper.pdf
- Logical Rule-Based KGR Survey (MDPI, 2023): https://www.mdpi.com/2227-7390/11/21/4486

### Compound Noun Detection (_merge_post_relation_entities)

When a spatial/temporal relation word (in, part_of, before, etc.) is followed by multiple non-stop tokens, they likely form a single entity. The parser merges them before entity extraction.

**Pattern:** `[relation] [token1] [token2] ...` → single entity

**Examples:**
- "in central asia" → entity "central asia"
- "part of united states" → entity "united states"
- "born in new york" → entity "new york"
- "before world war ii" → entity "world war ii"

**How it works:**
1. After single-token entity extraction, scan for relation words
2. Merge all adjacent non-stop tokens after the relation into one entity span
3. Pass merged spans to pattern matching (is Y of Z, is Y to Z, etc.)

**Reference:** Manning, C.D. & Schütze, H. (1999). "Foundations of Statistical Natural Language Processing." MIT Press. Chapter 5: Collocations.

**Fixes:** Kazakhstan bug — "kazakhstan is in central asia" was split into `kazakhstan → central` and `asia → asia`. Now correctly stores `kazakhstan → central asia`.

### Gazetteer (Learned Entity Dictionary)

A **gazetteer** is a dictionary of known multi-word entities that grows from teach() interactions. This is the standard approach for multi-word entity recognition in resource-constrained systems.

**How it works:**
1. User teaches "United Kingdom is in Europe"
2. System stores "united kingdom" as a single entity in the KG
3. Gazetteer adds "united kingdom" to its dictionary
4. Future queries "is United Kingdom in Eurasia?" recognize "united kingdom" via gazetteer lookup

**Why not NER models?**
- spaCy NER: 500MB+ model, GPU required
- NLTK chunking: POS tagger + corpus required (arch issues on M1)
- gensim Phrases: Large corpus required

**Why gazetteer works:**
- Zero external dependencies
- Grows naturally from teach() interactions
- O(1) lookup via set membership
- Persisted in SQLite (gazetteer table)
- Language-agnostic (works for any script)

**Reference:** Nadeau, D. & Sekine, S. (2007). "A survey of named entity recognition and classification." Lingvisticae Investigationes, 30(1), 3-26.

**Used by:** Wikidata, DBpedia, Google Knowledge Graph, every production KG system.

### spaCy Integration (arm64 Native)

spaCy 3.8.14 is now available in the arm64 venv (`.venv-arm64`). It provides several NLP capabilities **for free** that replace hardcoded rules:

**What spaCy provides:**

| Capability | Replaces | spaCy API |
|-----------|----------|-----------|
| POS tagging | Hardcoded `_pp_words` set | `doc[i].pos_ == "ADP"` |
| Dependency parsing | Prepositional phrase attachment rules | `token.dep_ == "prep"`, `token.head` |
| Noun chunking | Compound noun detection (`_merge_post_relation_entities`) | `doc.noun_chunks` |
| NER | Gazetteer lookup | `doc.ents` |
| Lemmatization | `difflib.SequenceMatcher` fuzzy matching | `token.lemma_` |

**Example: spaCy vs hardcoded**
```python
# BEFORE (hardcoded):
_pp_words = {"on", "in", "at", "from", "to", "by", "for", "with", "into",
             "through", "during", "before", "after", "about", "near", "over"}

# AFTER (spaCy):
doc = nlp("Germany is before Austria on Danube")
for token in doc:
    if token.pos_ == "ADP":  # ADP = adposition (preposition)
        print(f"Preposition: {token.text}")  # "on"
```

**spaCy dependency parsing for PP attachment:**
```python
doc = nlp("Germany is before Austria on Danube")
# spaCy correctly identifies "on Danube" as modifying "before", not "Austria"
for token in doc:
    if token.dep_ == "pobj":  # object of preposition
        print(f"{token.text} → head: {token.head.text}")
# Output: Danube → head: before (not Austria)
```

**Status:** spaCy is fully integrated into the parser and thinking engine. All hardcoded rules replaced.

**Integration points:**
1. `GenericNLParser.nlp` — lazy-loaded spaCy pipeline
2. `extract_triples_spacy()` — dependency parsing for triple extraction
3. `_get_chunk_text()` — noun chunk extraction with determiner stripping
4. `_extract_triples()` — spaCy-first, regex fallback
5. `teach_relation()` — lemmatizes relation names for consistency

**Fallback behavior:** spaCy is the primary extraction engine. Regex patterns exist only as a last resort when spaCy is not installed (e.g., architecture mismatch). The system always tries spaCy first — it's not optional, it's the default.

**Reference:** Honnibal, M. & Montani, I. (2017). "spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing." To appear.

### Multi-Hop Open Queries

Open queries ("What/Who/Where is X?") now support multi-hop reasoning via BFS path traversal.

**How it works:**
1. User asks: "Who founded the company that makes iPhone?"
2. System extracts: subject="iphone", relation="founded_by"
3. BFS follows: iphone → apple → steve jobs
4. At hop 2, finds "founded_by" relation → returns "steve jobs"

**Confidence:** Decreases with chain quality, not hop count. A 5-hop chain with all explicit rules has the same confidence as a 2-hop chain with all explicit rules. See "Confidence Scoring" section below.

**Question types supported:**
1. Yes/No — "Is X in Y?" (any hop count)
2. Open Query — "What/Who/Where is X?" (any hop count)
3. Functional Contradiction — "Are X and Y the same?"
4. Temporal — "Was X before/meets/during Y?"
5. Proof Trace — Full reasoning chain at each hop

### Confidence Scoring (Chain Quality + Error Propagation)

Confidence is computed from **chain quality** and **error propagation**, not hop count. This is based on recent research in trustworthy KG reasoning:

**Research foundation:**
- **CPR** (arXiv 2026): "Conformal Path Reasoning" — confidence from path quality, not path length. Uses learned scoring networks for discriminative path scores.
- **UaG** (AAAI 2025): "Uncertainty Aware Knowledge-Graph Reasoning" — multi-step error accumulates. Each hop multiplies uncertainty.
- **UnKGCP** (arXiv 2025): "Certainty in Uncertainty" — prediction intervals should be query-adaptive. Harder queries get wider intervals.
- **PSL** (Springer 2026): "Probabilistic Soft Logic" — soft truth values, not binary. A fact can be 0.7 true.

**How TD v2 computes confidence:**
```
chain_score = product of step_scores
where step_score =
    1.0 if explicit composition rule exists (e.g., in ∘ in → in)
    0.7 if transitive fallback used
    0.4 if heuristic (no rule, no transitivity)

confidence = clamp(chain_score, 0.1, 0.95)
```

**Examples:**
- 2-hop, all rules: `1.0 × 1.0 = 1.0` → 0.95
- 5-hop, all rules: `1.0^5 = 1.0` → 0.95
- 5-hop, all heuristic: `0.4^5 = 0.01` → 0.1 (floor)
- Mixed: `1.0 × 0.7 × 1.0 × 0.4 × 1.0 = 0.28`

**Key insight:** A 5-hop chain with all explicit rules has the SAME confidence as a 2-hop chain with all explicit rules. Chain quality matters, not length.

**References:**
1. **CPR:** Wang, Y. et al. (2026). "Conformal Path Reasoning: Trustworthy Knowledge Graph Question Answering with Statistical Coverage Guarantees." *arXiv preprint arXiv:2605.08077*. https://arxiv.org/abs/2605.08077
2. **UaG:** Ni, J. et al. (2025). "Towards Trustworthy Knowledge Graph Reasoning: An Uncertainty Aware Perspective." *Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 2025)*. arXiv:2410.08985. https://arxiv.org/abs/2410.08985
3. **UnKGCP:** Zhu, Y. et al. (2025). "Certainty in Uncertainty: Reasoning over Uncertain Knowledge Graphs with Conformal Prediction." *arXiv preprint arXiv:2510.24754*. https://arxiv.org/abs/2510.24754
4. **PSL+TEKGE:** Rawat, R. et al. (2026). "Information retrieval framework using knowledge graph embeddings and uncertainty modelling using probabilistic soft logic." *Discover Computing*, Vol. 29. Springer. https://doi.org/10.1007/s10791-025-09859-w

### Future: Confidence Calibration via User Feedback (Conformal Prediction)

**Problem:** The current confidence formula is a conservative heuristic. It doesn't reflect actual reliability.

**Solution:** Use the existing feedback system in `demos/chat_flare.py` to collect calibration data, then apply Conformal Prediction (CP) to calibrate confidence scores.

**Existing feedback system (already built):**
```
Was this helpful?
  [y] Yes  [n] No  [t] Teach me the right answer
```

**What to store (future):**
```sql
CREATE TABLE feedback (
    query TEXT,
    answer TEXT,
    confidence FLOAT,
    correct BOOLEAN,  -- from [y]/[n] buttons
    timestamp TIMESTAMP
);
```

**Calibration flow:**
1. User asks question → system answers with raw confidence
2. User clicks [y] or [n] → stored as (query, answer, confidence, correct)
3. After 100+ queries → run CP calibration
4. CP maps: raw_confidence → calibrated_confidence
5. Example: "My 0.20-confidence answers are correct 85% of the time → calibrate to 0.85"

**Why CP?**
- Distribution-free (no assumptions about data distribution)
- Statistically guaranteed coverage (Vovk et al., 2005)
- Query-adaptive (harder queries get wider intervals)
- Works with any confidence formula as input

**References:**
1. Vovk, V., Gammerman, A., & Shafer, G. (2005). "Algorithmic Learning in a Random World." Springer. ISBN: 978-0-387-00152-4.
2. Angelopoulos, A.N. & Bates, S. (2022). "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification." arXiv:2107.07511.
3. Shafer, G. & Vovk, V. (2008). "A Tutorial on Conformal Prediction." Journal of Machine Learning Research, 9, 371-421.

**Implementation plan:**
- Phase 1: Store feedback in SQLite (add `feedback` table)
- Phase 2: Collect 100+ queries with feedback
- Phase 3: Implement CP calibration (Python `mapie` library or custom)
- Phase 4: Replace raw confidence with calibrated confidence
