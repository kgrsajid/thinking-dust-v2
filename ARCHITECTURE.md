# TD v2 Architecture — Honest Documentation

_Last updated: 2026-07-01 04:17 GMT+5_

---

## What TD v2 Actually Is

A **neuro-symbolic reasoning engine** that derives facts it was never explicitly taught, using HDC algebra + Z3 constraint solving. No neural network. No GPU. No pretraining. <100K parameters.

**Slogan:** "Computer intelligence is just human beings teaching dust how to think."

---

## Architecture (4 Layers)

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 4: Knowledge Graph + Inference (THE THINKING)        │
│   Entity-pair path search over stored triples              │
│   General rule templates: transitive, symmetric, inverse,  │
│   functional. Cross-relation composition. Proof traces.    │
│   File: td/kg/__init__.py                                  │
├─────────────────────────────────────────────────────────────┤
│ Layer 3: BEAGLE Word Vectors (SEMANTIC MATCHING)           │
│   Environmental vectors (static identity) + context        │
│   vectors (accumulated co-occurrence). Trained on 10K      │
│   synthetic corpus in 1.4s. Position-independent encoding   │
│   for paraphrase matching. Online learning from teach().   │
│   File: td/perception/word_vectors.py                      │
├─────────────────────────────────────────────────────────────┤
│ Layer 2: NL Parser (STRUCTURE EXTRACTION)                  │
│   CA reservoir (Rule 90, 64-bit, 16 steps) for feature     │
│   extraction. Stop-word filtering. Single-token entity     │
│   spans. 14 innate relation prototypes (HDC centroid).     │
│   File: td/perception/nl_parser.py                         │
├─────────────────────────────────────────────────────────────┤
│ Layer 1: HDC + MHN + Z3 (STORAGE & SOLVING)               │
│   HDC: 10K-dim bipolar vectors. bind=element-wise multiply │
│   MHN: Modern Hopfield Network (Ramsauer 2020). Online     │
│   learning. Zero catastrophic forgetting.                  │
│   Z3: Microsoft SMT solver. 18 constraint primitives.      │
│   Files: td/perception/hdc.py, td/memory/mhn.py            │
└─────────────────────────────────────────────────────────────┘
```

---

## How Query Processing Works (Honest)

**IMPORTANT:** The system uses **entity-pair path search**, NOT relation-specific triple queries.

### What this means:

When a user asks "is Paris in the EU?":

1. **Parser** extracts entities: `[Paris, EU]`
2. **Parser** may find 0 relations in the query text (the query is ASKING about a relation, not stating one)
3. **Knowledge Graph** searches ALL paths between Paris and EU
4. If a path exists (Paris→capital_of→France→in→EU), the answer is YES
5. The relation in the query ("in") is matched against the LAST edge of the path for validation

### Why entity-pair path search (not triple query):

| Approach | Triple Query | Entity-Pair Path Search (what we do) |
|----------|-------------|--------------------------------------|
| Query parsing | Extract `(Paris, in, EU)` exactly | Extract `[Paris, EU]`, search all paths |
| Requires perfect relation extraction | Yes | No |
| Works with paraphrase | Only if relation matches | Yes — any path between entities |
| Ambiguity risk | Low (specific relation) | Higher (multiple paths) |
| Best for | Large KGs (millions of triples) | Small KGs (10-100 triples) |

**Trade-off:** Path search is more robust to query variation but less precise when multiple paths exist. For TD v2's scale (small KG), this is the right choice.

### Query flow:

```
User: "is Paris in the EU?"
  ↓
Parser: entities = [paris, eu], relations = []
  ↓
KG Query: check if KG has triples
  ↓ YES (5 facts stored)
KG: BFS paths(paris → eu, max_hops=4)
  ↓ Found: paris→capital_of→france→in→eu
KG: Validate path (does last edge match queried relation?)
  ↓ YES ("in" matches)
Return: YES, proof trace: "paris capital_of france → france in eu"
```

---

## How Teaching Works

```
teach: "Paris is the capital of France" | "Paris"
  ↓
1. MHN Store: encode_query("Paris is the capital of France") → semantic key
   Store (key → "Paris") in MHN for paraphrase retrieval
  ↓
2. Triple Extraction: structural patterns
   "X is the Y of Z" → (paris, capital_of, france)
   Store in Knowledge Graph
  ↓
3. Online BEAGLE: word vectors update from this sentence
   "paris", "capital", "france" vectors accumulate context
```

**Triple extraction patterns (structural, not word-specific):**
- "X is the Y of Z" → (X, Y_of, Z)
- "X is in Y" → (X, in, Y)
- "X is part of Y" → (X, part_of, Y)
- "X is before Y" → (X, before, Y)
- "X is after Y" → (X, after, Y)
- "X means Y" → (X, means, Y)

These apply to ANY words, not specific facts.

---

## Inference Engine (Rule Templates)

**GENERAL logical schemas, NOT hardcoded for specific words:**

| Template | Rule | Applies to |
|----------|------|-----------|
| transitive | R(X,Y) ∧ R(Y,Z) → R(X,Z) | in, part_of, before, after, etc. |
| symmetric | R(X,Y) → R(Y,X) | same_as, married_to, sibling_of, etc. |
| inverse | R1(X,Y) → R2(Y,X) | capital_of↔has_capital, parent_of↔child_of |
| functional | R(X,Y) ∧ R(X,Z) → Y=Z | capital_of, birth_date, etc. |

**Teaching relation properties:**
```python
td.teach_relation("north_of", "transitive")
td.teach_relation("married_to", "symmetric")
td.teach_relation("capital_of", "functional", "inverse:has_capital")
```

**Cross-relation composition:**
- If R1 is transitive and R2 is transitive
- And R1(X,Y) ∧ R2(Y,Z) exist
- Then R2(X,Z) is derivable
- Example: `capital_of(Paris, France) ∧ in(France, EU) → in(Paris, EU)`

**Functional contradiction:**
- "Are Berlin and Paris the same?" → NO
- Proof: `capital_of` is functional, Berlin→Germany, Paris→France, Germany≠France → Berlin≠Paris

**Proof traces show every hop:**
```
paris --capital_of--> france , france --in--> eu , eu --part_of--> europe
```

---

## What's Proven (July 1, 2026)

```
Teach 5 facts:
  Paris is the capital of France
  France is in the EU
  EU is part of Europe
  Berlin is the capital of Germany
  Germany is in the EU

Ask (never taught):
  Is Paris in the EU?      → YES (derived: Paris→France→EU)
  Is Paris in Europe?      → YES (derived: Paris→France→EU→Europe)
  Is Berlin in the EU?     → YES (derived: Berlin→Germany→EU)
  Is Berlin in Europe?     → YES (derived: Berlin→Germany→EU→Europe)

Ask contradiction:
  Is Paris capital of Germany? → NO (functional: Paris is capital of France)
```

**These answers were DERIVED via transitive composition, not retrieved from memory.**

---

## What's NOT Done Yet (Honest)

1. **Triple extraction is brittle** — only 6 structural patterns, no BEAGLE fallback yet
2. **Relation-specific path filtering** — when query says "in", prefers paths ending in "in" (implemented, needs broader testing)
3. **No multi-turn context** — "What about London?" doesn't resolve to previous conversation
4. **Scale testing** — KG tested with ~10 facts, needs testing at 50-100+

---

## Technical Stack

| Component | Technology | Parameters |
|-----------|-----------|-----------|
| HDC vectors | Custom (10K-dim bipolar) | 0 (random, fixed) |
| CA reservoir | Rule 90, 64-bit, 16 steps | 0 |
| MHN | Modern Hopfield Network | ~10K (patterns stored) |
| Word vectors | BEAGLE-style | 0 (random env + accumulated context) |
| Z3 solver | Microsoft Z3 (pip install z3-solver) | 0 |
| Knowledge Graph | Custom (adjacency list + BFS) | 0 |
| **Total trainable** | | **~0** (all vectors are random or accumulated) |

**Memory footprint:** ~20MB (word vectors) + ~1KB per KG triple + ~10KB per MHN pattern

**Speed:** <50ms per query on MacBook CPU

---

## Storage Architecture

**SQLite for structured knowledge. Pickle for dense vectors.**

| Data | Storage | File |
|------|---------|------|
| Triples (subject, relation, object) | SQLite | `data/td_knowledge.db` |
| Relation properties | SQLite | `data/td_knowledge.db` |
| BEAGLE word vectors | Pickle | `data/word_vectors_10k.pkl` |
| MHN patterns | Pickle | `data/td_knowledge_mhn.pkl` |

### SQLite Schema

```sql
CREATE TABLE triples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject TEXT NOT NULL,
    relation TEXT NOT NULL,
    object TEXT NOT NULL,
    source TEXT DEFAULT 'user',   -- 'user', 'derived', 'seed'
    proof TEXT DEFAULT '',        -- derivation chain
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE relation_properties (
    relation TEXT PRIMARY KEY,
    properties TEXT NOT NULL      -- 'transitive,functional'
);
```

### Querying

```python
kg.query_sql("SELECT * FROM triples WHERE subject = ?", ("paris",))
kg.query_sql("SELECT * FROM triples WHERE relation = ?", ("capital_of",))
```

```bash
sqlite3 data/td_knowledge.db "SELECT * FROM triples WHERE subject='paris';"
```

---

## Literature Foundation

| Component | Paper | Year |
|-----------|-------|------|
| HDC algebra | Kanerva, "Hyperdimensional Computing" | 2009 |
| CA + HDC reservoir | Yilmaz, arXiv:1503.00851 | 2015 |
| MHN (associative memory) | Ramsauer et al., "Hopfield Networks is All You Need" | 2020 |
| BEAGLE (word vectors) | Jones & Mewhort, Psychological Review | 2007 |
| IDP (MHN refinement) | Betteti et al., Science Advances | 2025 |
| BitNet (ternary weights) | Ma et al. | 2024 |
| Role-filler VSA composition | Lewis, arXiv:2401.06808 | 2024 |
| Syntax+vectors > transformers | Fodor et al., Computational Linguistics (STS3k) | 2025 |
| HDC graph reasoning | Liu et al. (PathHD), NeurIPS | 2025 |
| HDC/VSA survey | Kleyko et al., ACM Computing Surveys | 2022 |

---

## File Structure

```
td-v2/
├── td/
│   ├── perception/
│   │   ├── hdc.py            # HDC operations (bind, bundle, permute, similarity)
│   │   ├── nl_parser.py      # CA reservoir, entity spans, relation prototypes
│   │   └── word_vectors.py   # BEAGLE word vector model
│   ├── memory/
│   │   └── mhn.py            # Modern Hopfield Network
│   ├── kg/
│   │   └── __init__.py       # Knowledge Graph + inference engine
│   ├── thinking.py           # Main reasoning pipeline
│   └── minimal_seed.py       # Seed patterns (optional)
├── demos/
│   └── chat_flare.py         # Interactive demo with reasoning trace
├── data/
│   ├── synthetic_corpus_10k.txt  # 10K training sentences
│   └── word_vectors_10k.pkl      # Trained BEAGLE vectors
├── tests/
│   └── test_thinking.py      # 50 tests (all passing)
└── ARCHITECTURE.md           # This file
```
