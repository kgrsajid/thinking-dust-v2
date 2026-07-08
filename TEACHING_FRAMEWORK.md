# TD v2 — Rigorous Teaching Framework

**Date:** 2026-07-08
**Status:** DRAFT — for TD v2 operational release
**Purpose:** Define how to teach TD v2 effectively, rigorously, and reproducibly

---

## 1. Philosophy

TD v2 starts from zero. It knows nothing until you teach it. This is a feature, not a bug — every fact in its knowledge graph has a provenance, a teaching moment, a human behind it.

But teaching a reasoning engine is not like teaching a database. The order matters. The specificity matters. The relation properties matter. A sloppy teaching session produces a confused knowledge graph. A rigorous one produces an engine that derives facts it was never taught.

This document defines the protocol.

---

## 2. Teaching Protocol

### 2.1 Phase 1: Declare Relation Properties (Before Facts)

Before teaching any facts, declare how each relation behaves. This is the "teaching dust how to think" step.

```
# Step 1: Declare relations
relation: capital_of functional inverse:has_capital
relation: in transitive
relation: part_of transitive
relation: before transitive
relation: after transitive
relation: married_to symmetric
relation: borders symmetric
relation: north_of transitive
```

**Why first:** Inference runs immediately when facts are stored. If `in` is declared transitive AFTER teaching "France in EU" and "EU in Europe", the system won't derive "France in Europe" until you manually trigger `derive_all()`. Declaring properties first ensures every new fact immediately participates in inference.

**Decision tree for relation properties:**

```
Is the relation describing...
├── A chain? (X→Y→Z means X→Z)
│   → transitive
│   Examples: in, part_of, before, after, north_of, ancestor_of,
│             contains, depends_on, larger_than, evolved_from
│
├── A mutual bond? (X→Y means Y→X)
│   → symmetric
│   Examples: married_to, sibling_of, adjacent_to, borders,
│             neighbor_of, equals, collaborated_with
│
├── A unique mapping? (X→Y, X→Z → Y=Z)
│   → functional
│   Examples: birth_date, capital_of, has_capital, unique_id,
│             discovered_by, invented_by, directed_by
│
└── An opposite direction? (X→Y ↔ Y→X)
    → inverse:R2
    Examples: parent_of↔child_of, owns↔owned_by,
              employs↔employed_by, teaches↔taught_by
```

### 2.2 Phase 2: Teach Core Facts (The Foundation)

Teach the fundamental facts that form the backbone of the knowledge graph. These are the facts that other facts will be derived from.

```
# Step 2: Teach core facts
teach: Paris is the capital of France
teach: France is in the EU
teach: EU is part of Europe
```

**Principles:**
- **Teach from general to specific.** Start with broad categories, then add details.
- **Teach in logical order.** If A depends on B, teach B first.
- **Use consistent relation names.** Don't mix "in" and "located_in" and "part of" for the same relation.

### 2.3 Phase 3: Teach Polysemous Entities (Word Sense Disambiguation)

When the same word has multiple meanings, teach `is_a` declarations FIRST to establish senses.

```
# Step 3: Establish senses via is_a
teach: cell is_a organelle        ← biology sense
teach: cell is_a room              ← prison sense (room ≠ organelle → new sense)
teach: cell is_a device            ← technology sense

# Then teach related facts (auto-routes to correct sense)
teach: cell is part of organism    → biology sense
teach: cell is part of prison      → prison sense
teach: cell connects to network    → technology sense
```

**Why `is_a` first:** The `is_a` object is the primary WSD signal. Different `is_a` objects = different senses. This is a logical principle backed by the type theory literature (OWL 2, W3C 2009).

**Research backing:**
- McInnes et al. (2012): BSC-WSD achieves 94.55% accuracy with HDC vectors
- Romanian WSD paper (2025): 5 example sentences per sense sufficient
- Sumanathilaka et al. (2026): Neighbour word analysis = critical WSD signal

### 2.4 Phase 4: Teach Domain-Specific Vocabulary

Expand the BEAGLE word vectors by teaching domain-specific sentences. This improves query-time sense resolution.

```
# Step 4: Domain vocabulary (for BEAGLE training)
teach: the cell membrane controls what enters and exits
teach: mitochondria are organelles that produce energy
teach: a prisoner was locked in a cold damp cell
teach: guards inspected each cell in the prison block
teach: cell phones connect to cellular networks
```

**Why:** BEAGLE learns word meaning from co-occurrence. Teaching domain-specific sentences builds the vocabulary needed for query-time disambiguation.

### 2.5 Phase 5: Verify and Explore

After teaching, verify the system can derive new facts and explore its knowledge.

```
# Step 5: Verify
ask: is Paris in Europe?           ← should derive: YES (3 hops)
ask: are Paris and Berlin same?    ← should derive: NO (functional)
ask: is cell part of organism?     ← should route to biology sense

# Explore
stats                               ← show KG statistics
trace                               ← enable proof traces
```

---

## 3. Teaching Patterns

### 3.1 The Chain Pattern

Teach a sequence of facts that form a transitive chain.

```
teach: A is in B
teach: B is in C
teach: C is in D

ask: is A in D?
→ YES. A→B→C→D (3-hop transitive derivation)
```

**Use case:** Geography (country→continent), organization (team→department→company), time (before→during→after).

### 3.2 The Functional Pattern

Teach facts that establish unique mappings, then verify the system detects contradictions.

```
relation: capital_of functional

teach: Paris is the capital of France
teach: Berlin is the capital of Germany

ask: are Paris and Berlin the same?
→ NO. capital_of is functional, Paris→France, Berlin→Germany
```

**Use case:** Identifiers, unique attributes, one-to-one relationships.

### 3.3 The Symmetric Pattern

Teach mutual relationships and verify the system can infer both directions.

```
relation: married_to symmetric

teach: Alice is married to Bob

ask: is Bob married to Alice?
→ YES (symmetric inference)
```

**Use case:** Social relationships, adjacency, equivalence.

### 3.4 The Composition Pattern

Teach facts across different relations and verify cross-relation inference.

```
teach: Paris is the capital of France
teach: France is in the EU

set_composition_rule: capital_of + in → in

ask: is Paris in the EU?
→ YES. capital_of(Paris,France) ∧ in(France,EU) → in(Paris,EU)
```

**Use case:** Cross-domain reasoning, multi-hop inference.

### 3.5 The WSD Pattern

Teach polysemous entities with `is_a` declarations, then verify sense routing.

```
teach: cell is_a organelle
teach: cell is_a room
teach: cell is_a device

teach: cell is part of organism    → biology sense
teach: cell is part of prison      → prison sense
teach: cell connects to network    → technology sense

ask: is cell part of organism?     → YES (biology sense)
ask: is cell part of prison?       → YES (prison sense)
```

**Use case:** Ambiguous terms, domain-specific vocabulary, real-world entities.

---

## 4. Anti-Patterns (What NOT to Do)

### 4.1 Don't Teach Facts Before Relations

```
# WRONG: teach facts first, relations later
teach: Paris is the capital of France
teach: France is in the EU
relation: in transitive              ← too late! in+in inference won't run

# RIGHT: declare relations first
relation: in transitive
relation: capital_of functional
teach: Paris is the capital of France
teach: France is in the EU           ← immediately derives Paris in EU
```

### 4.2 Don't Mix Relation Names

```
# WRONG: inconsistent relation names
teach: France is in the EU
teach: EU is located in Europe       ← "located_in" ≠ "in"

# RIGHT: consistent relation names
teach: France is in the EU
teach: EU is in Europe               ← same "in" relation
```

### 4.3 Don't Teach Polysemous Facts Without `is_a`

```
# WRONG: teach related facts before establishing senses
teach: cell is part of organism      ← which cell?
teach: cell is part of prison        ← same cell? different?

# RIGHT: establish senses first
teach: cell is_a organelle           ← biology sense
teach: cell is_a room                ← prison sense
teach: cell is part of organism      → routes to biology sense
teach: cell is part of prison        → routes to prison sense
```

### 4.4 Don't Teach Contradictory Facts Without Warning

```
# WRONG: silent contradiction
teach: Paris is the capital of France
teach: Paris is a country            ← LOTG warns: city ≠ country

# RIGHT: acknowledge the warning
teach: Paris is the capital of France
teach: Paris is a country            ← ⚠️ Contradiction: city vs country
                                       (stored anyway — user is authority)
```

---

## 5. Reproducibility Protocol

### 5.1 Record Every Teach Session

Every teach session should be recorded as a script that can be replayed.

```python
# teach_session_2026-07-08.py
from td.thinking import GenericThinkingDust

td = GenericThinkingDust(...)

# Phase 1: Relations
td.teach_relation("in", "transitive")
td.teach_relation("capital_of", "functional", "inverse:has_capital")

# Phase 2: Core facts
td.teach("Paris is the capital of France", "Paris")
td.teach("France is in the EU", "EU")

# Phase 3: WSD
td.teach("cell is_a organelle", "organelle")
td.teach("cell is_a room", "room")

# Phase 4: Domain vocabulary
td.teach("the cell membrane transports ions", "biology")

# Phase 5: Verify
result = td.think("is Paris in the EU?")
assert result.confidence > 0.8
```

### 5.2 Version Control Knowledge Graphs

Save the knowledge graph after each teaching session:

```python
td.kg.save("data/knowledge_graphs/session_2026-07-08.db")
```

### 5.3 Document the Corpus

The BEAGLE training corpus should be documented with:
- Exact prompt used for generation
- Model used (name, version, provider)
- Date of generation
- Post-processing steps (if any)
- Domain coverage verification

See `DEVELOPMENT.md` section "The 10K Synthetic Corpus" for the template.

---

## 6. Research Foundation

| Paper | Year | Relevance |
|-------|------|-----------|
| Jones & Mewhort, "BEAGLE" | 2007 | Context vectors for word meaning |
| McInnes et al., "BSC-WSD" | 2012 | HDC binary vectors for WSD (94.55%) |
| McInnes et al., "BSC-WSD clinical" | 2013 | Orientation/distance weighting |
| Sumanathilaka et al., "EAD Framework" | 2026 | Neighbour word analysis for WSD |
| Romanian WSD with attention vectors | 2025 | 5 sentences per sense sufficient |
| Navigli, "Is WSD Dead?" | 2026 | Contextualized embeddings needed |
| Mosolova et al., "WSI unsolved" | 2025 | Even LLMs struggle with WSI |
| OWL 2 PropertyChain | 2009 | Cross-relation composition |
| Allen, "Temporal Intervals" | 1983 | Temporal reasoning |
| Salton & Buckley, "TF-IDF" | 1988 | Term weighting |

---

## 7. Future: BSC-WSD for Teach-Time Context Disambiguation

**The breakthrough:** McInnes et al. (2012, 2013) showed that HDC binary vectors can achieve 94.55% WSD accuracy using the BSC-WSD algorithm. This is CPU-only, no GPU, and uses the SAME HDC infrastructure TD v2 already has.

**How it works:**
1. Assign a random elemental vector to each sense: `E(sense_i)`
2. For each word in the context, bind the ambiguous word's vector with the sense vector and bundle into the context word's semantic vector: `S(context_word) += E(ambiguous) ⊗ E(sense)`
3. For a new context, apply inverse binding: `S(context) ∅ E(ambiguous) ≈ E(sense)`

**Why this is feasible for TD v2:**
- Uses existing HDC operations (bind, bundle, permute)
- CPU-only, <1ms per disambiguation
- No pretraining needed — learns from teach() examples
- 5 example sentences per sense sufficient (Romanian WSD paper, 2025)

**What's needed:**
- Sense inventory (built from `is_a` declarations)
- Training loop: for each teach(), bind sense vectors into context words
- Query loop: for each ask(), unbind to recover the correct sense

**Reference:** McInnes, B.T. et al. "Hyperdimensional Computing Approach to Word Sense Disambiguation." *AMIA Annual Symposium*, 2012. PMC3540565.

---

*"Teach the dust how to think, and it will."*
