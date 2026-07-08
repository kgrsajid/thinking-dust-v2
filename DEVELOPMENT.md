# Thinking Dust v2 — Developer Guide

_Last updated: 2026-07-07 GMT+5_

This guide covers everything you need to set up, extend, and debug TD v2.

---

## 1. Setup Instructions

### Prerequisites

- **Python 3.10 or 3.11** (3.12 may work; 3.9 is not supported)
- **macOS or Linux** — tested on MacBook Pro (Intel and Apple Silicon M1 Pro) and Linux x86_64
- **~200 MB disk space** for dependencies, word vectors, and RDF store

### Apple Silicon (M1/M2/M3 Pro) Note

If you are on macOS with Apple Silicon (`arch -arm64`), some dependencies require special handling:

```bash
# Verify you're on ARM64
uname -m
# Expected: arm64

# Z3 on Apple Silicon may need Homebrew
brew install z3
# Then in your venv:
.venv/bin/pip install z3-solver --global-option=build_ext --global-option=/opt/homebrew/include
# Or use the pre-built wheel:
.venv/bin/pip install z3-solver
```

If `import z3` fails after installation, ensure Xcode Command Line Tools are up to date:
```bash
xcode-select --install
```

### Installation

```bash
# Clone the repository
git clone https://github.com/kgrsajid/thinking-dust-v2.git
cd thinking-dust-v2

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package and all dependencies
.venv/bin/pip install -e ".[dev]"

# Verify Z3 is working
.venv/bin/python -c "from z3 import Solver, sat; s = Solver(); s.add(True); print('Z3 OK:', s.check() == sat)"

# Verify TD v2 is importable
.venv/bin/python -c "from td.thinking import GenericThinkingDust; print('TD v2 import OK')"
```

### Quick Verification

```bash
# Run the full test suite
.venv/bin/python -m pytest tests/ -v --tb=short

# Expected output: 99 passed, some output with reasoning traces

# Run just the core reasoning tests
.venv/bin/python -m pytest tests/test_thinking.py -v

# Run just the generalization tests
.venv/bin/python -m pytest tests/test_generalization.py -v

# Run just real-world scenario tests
.venv/bin/python -m pytest tests/test_realworld.py -v
```

### Running the Demo

```bash
# Interactive demo with proof traces
.venv/bin/python3 demos/chat_flare.py --pure

# Alternative demo interface
.venv/bin/python3 demos/interactive_demo.py
```

---

## 2. Project Structure

```
td-v2/
├── td/
│   ├── __init__.py
│   ├── thinking.py           # Main entry point: GenericThinkingDust class
│   ├── minimal_seed.py        # Pre-seeded relation properties
│   ├── pipeline.py           # Query processing pipeline orchestration
│   ├── decomposer.py         # Problem decomposition into sub-problems
│   ├── routing.py            # Query type detection (fact/constraint/inference)
│   ├── z3_solver.py          # Z3 wrapper: 18 constraint primitives
│   ├── solve_advice.py       # Constraint solving guidance
│   ├── perception/
│   │   ├── __init__.py
│   │   ├── hdc.py            # HDC operations: bind, bundle, permute, cosine
│   │   ├── nl_parser.py      # CA reservoir, entity extraction, relation prototypes
│   │   └── word_vectors.py   # BEAGLE model: env + context vectors
│   ├── memory/
│   │   └── mhn.py            # Modern Hopfield Network: store, retrieve
│   ├── kg/
│   │   ├── __init__.py       # KnowledgeGraph: add_fact, query, BFS, inference
│   │   ├── rules.py          # RULE_TEMPLATES: transitive, symmetric, etc.
│   │   ├── queries.py        # SQL queries and path finding
│   │   └── relation_synonyms.py  # Relation synonymy detection & registry
│   ├── query/
│   │   └── __init__.py       # SparqlStore: SPARQL 1.1 bridge (pyoxigraph)
│   ├── perception/
│   │   ├── __init__.py
│   │   ├── hdc.py            # HDC operations: bind, bundle, permute, cosine
│   │   ├── nl_parser.py      # CA reservoir, entity spans, relation prototypes
│   │   ├── word_vectors.py   # BEAGLE word vector model (env + context)
│   │   └── clause_segmenter.py  # Verb-based clause splitting
│   ├── reasoning/
│   │   └── inference.py      # Forward chaining, contradiction detection
│   └── utils/
│       └── logging.py        # Structured logging utilities
├── demos/
│   ├── chat_flare.py         # Main interactive demo
│   └── interactive_demo.py   # Alternative demo
├── data/
│   ├── td_store/              # pyoxigraph RDF store (created on first run)
│   ├── word_vectors_10k.pkl  # BEAGLE vectors (auto-trained if missing)
│   ├── td_knowledge_mhn.pkl  # MHN patterns (created on first run)
│   ├── ca_reservoir_state.pkl # CA state (created on first run)
│   └── synthetic_corpus_10k.txt # Training corpus for BEAGLE
├── tests/
│   ├── __init__.py
│   ├── test_thinking.py      # 162 lines: core reasoning loop tests
│   ├── test_generalization.py # 575 lines: 34 generalization tests
│   ├── test_realworld.py    # 235 lines: 15 real-world scenario tests
│   ├── test_hdc.py          # 123 lines: HDC operation unit tests
│   ├── test_mhn.py          # 111 lines: MHN memory tests
│   └── test_ca_reservoir.py # 47 lines: CA reservoir tests
├── pyproject.toml           # Project metadata, dependencies, build config
├── setup.py                 # Legacy setup (still used by pip install -e)
├── README.md                # Overview, quick start, research foundation
├── ARCHITECTURE.md          # Full system architecture (this repo's spine)
└── DEVELOPMENT.md           # This file: developer guide
```

---

## 3. Training BEAGLE Word Vectors

BEAGLE word vectors are auto-trained on first run if the pickle file is missing. To retrain manually:

```python
from td.perception.word_vectors import WordVectorModel
import time

# Initialize — 10K dimensions (matches HDC layer)
wvm = WordVectorModel(dim=10000)

# Load training corpus
with open("data/synthetic_corpus_10k.txt") as f:
    sentences = [line.strip() for line in f if line.strip()]

print(f"Training on {len(sentences)} sentences...")

start = time.time()
wvm.train(sentences)
elapsed = time.time() - start

print(f"Training complete in {elapsed:.1f}s")
print(f"Vocabulary size: {len(wvm.env_vectors)}")

# Save
wvm.save("data/word_vectors_10k.pkl")

# Verify quality
print(f"Similarity(capital, city): {wvm.similarity('capital', 'city'):.3f}")   # should be >0.15
print(f"Similarity(capital, sort): {wvm.similarity('capital', 'sort'):.3f}")    # should be <0.05
print(f"Nearest neighbors of 'paris': {wvm.nearest_neighbors('paris', 5)}")
```

Training takes ~1.4 seconds on CPU. No GPU needed. The model is purely accumulative — no backpropagation.

### How BEAGLE Works

BEAGLE combines two vector representations for each word:

1. **Environmental (identity) vector:** A random high-dimensional vector assigned at first encounter. Remains fixed. Encodes the word's identity.

2. **Context vector:** Initialized to the environmental vector. Updated by accumulating the environmental vectors of surrounding words (within a sliding window). Encodes the word's typical contexts.

Paraphrase matching: Two words are semantically similar if their context vectors are similar (they appear in similar contexts), even if their identity vectors are unrelated.

---

## 4. Extending the Knowledge Graph

### 4.1 Adding Triple Extraction Patterns

Edit `_extract_triples()` in `td/perception/nl_parser.py` or the pattern dispatch in `td/thinking.py`:

```python
# Current patterns (all use regex):
# "X is the Y of Z"  → (X, Y_of, Z)
# "X is in Y"        → (X, in, Y)
# "X is part of Y"   → (X, part_of, Y)
# "X is before Y"    → (X, before, Y)
# "X is after Y"     → (X, after, Y)
# "X means Y"        → (X, means, Y)

# Adding a new pattern:
import re

def _extract_triples(text: str) -> List[Tuple[str, str, str]]:
    triples = []

    # Pattern 1: "X belongs to Y"
    m = re.search(r'(\w+)\s+belongs\s+to\s+(\w+)', text)
    if m:
        triples.append((m.group(1), "belongs_to", m.group(2)))

    # Pattern 2: "X is located in Y"
    m = re.search(r'(\w+)\s+is\s+located\s+in\s+(\w+)', text)
    if m:
        triples.append((m.group(1), "located_in", m.group(2)))

    # Pattern 3: "X owns Y"
    m = re.search(r'(\w+)\s+owns\s+(\w+)', text)
    if m:
        triples.append((m.group(1), "owns", m.group(2)))

    # Pattern 4: Passive voice — "X is owned by Y" → (Y, owns, X)
    m = re.search(r'(\w+)\s+is\s+owned\s+by\s+(\w+)', text)
    if m:
        triples.append((m.group(2), "owns", m.group(1)))

    return triples
```

### 4.2 Adding Relation Properties

**Option A: In code (recommended for your application)**

```python
from td.kg import KnowledgeGraph

kg = KnowledgeGraph()
kg.set_relation_property("north_of", "transitive")
kg.set_relation_property("married_to", "symmetric")
kg.set_relation_property("capital_of", "functional", inverse="has_capital")
```

**Option B: Via the demo**

```
relation: north_of transitive
relation: married_to symmetric
relation: owns functional
```

**Option C: Pre-seeded defaults in `minimal_seed.py`**

```python
DEFAULT_RELATION_PROPERTIES = {
    # Transitive relations
    "in":            ["transitive"],
    "part_of":       ["transitive"],
    "before":        ["transitive"],
    "after":         ["transitive"],
    "north_of":      ["transitive"],
    "south_of":      ["transitive"],
    "east_of":       ["transitive"],
    "west_of":       ["transitive"],
    "ancestor_of":   ["transitive"],
    "descendant_of": ["transitive"],
    "depends_on":    ["transitive"],
    "contains":      ["transitive"],

    # Symmetric relations
    "same_as":       ["symmetric", "transitive"],
    "equals":        ["symmetric", "transitive"],
    "married_to":    ["symmetric"],
    "sibling_of":    ["symmetric"],
    "adjacent_to":   ["symmetric"],
    "borders":       ["symmetric"],
    "neighbor_of":   ["symmetric"],

    # Functional relations
    "capital_of":    ["functional"],
    "birth_date":    ["functional"],
    "birth_year":    ["functional"],
    "has_capital":   ["functional"],
    "unique_id":     ["functional"],
}

DEFAULT_INVERSE_PAIRS = {
    "capital_of":    "has_capital",
    "has_capital":   "capital_of",
    "parent_of":     "child_of",
    "child_of":      "parent_of",
    "owns":          "owned_by",
    "owned_by":      "owns",
    "employs":       "employed_by",
    "employed_by":   "employs",
}
```

### 4.3 Adding New Rule Templates

Edit `RULE_TEMPLATES` in `td/kg/rules.py`:

```python
RULE_TEMPLATES = {
    "transitive": {
        "name": "Transitivity",
        "rule": "R(X,Y) ∧ R(Y,Z) → R(X,Z)",
        "applies_to": ["transitive"],
    },
    "symmetric": {
        "name": "Symmetry",
        "rule": "R(X,Y) → R(Y,X)",
        "applies_to": ["symmetric"],
    },
    "inverse": {
        "name": "Inverse",
        "rule": "R1(X,Y) → R2(Y,X)",
        "applies_to": ["inverse"],
    },
    "functional": {
        "name": "Functionality",
        "rule": "R(X,Y) ∧ R(X,Z) → Y = Z",
        "applies_to": ["functional"],
    },
    # New: Irreflexivity — R(X,X) is always false
    "irreflexive": {
        "name": "Irreflexivity",
        "rule": "¬R(X,X)",
        "applies_to": ["irreflexive"],
        "check": lambda r: True,  # Always enforce
    },
    # New: Asymmetry — R(X,Y) → ¬R(Y,X)
    "asymmetric": {
        "name": "Asymmetry",
        "rule": "R(X,Y) ∧ R(Y,X) → contradiction",
        "applies_to": ["asymmetric"],
    },
}
```

Then add the property to the relation:
```python
kg.set_relation_property("parent_of", "asymmetric")
```

### 4.4 Contradiction Detection (LOTG)

TD v2 includes a **Lightweight Ontological Type Guard** that detects type contradictions before facts are stored. It runs automatically in `add_fact()` — no configuration needed.

#### How It Works

When you add a fact, LOTG:
1. **Infers entity types** from the relation's domain/range constraints
2. **Tracks inferred types** per entity
3. **Checks for contradictions** against a disjointness table
4. **Warns** (never rejects) when conflicts are found

```python
kg.add_fact("paris", "capital_of", "france")
# LOTG infers: paris → city, france → country

kg.add_fact("paris", "is_a", "country")
# LOTG warns: ⚠️ paris was 'city', now 'country' — mutually exclusive
# Triple is STILL stored (user is authority)
```

#### Accessing Warnings

```python
# After add_fact(), check last_warnings
triple = kg.add_fact("paris", "is_a", "country")
if kg.last_warnings:
    for w in kg.last_warnings:
        print(w)  # Human-readable warning with proof trace

# Warnings are also attached to the triple
if triple.metadata and "contradictions" in triple.metadata:
    print(triple.metadata["contradictions"])
```

#### From teach()

```python
result = td.teach("Paris is a country", "country")
if "warnings" in result:
    for w in result["warnings"]:
        print(w)  # Contradiction warning
```

#### Adding Type Constraints for New Relations

Edit `RELATION_SCHEMA` in `td/reasoning/contradiction_detector.py`:

```python
RELATION_SCHEMA["my_new_relation"] = {"domain": "entity_type", "range": "entity_type"}
```

Or add to `DISJOINT_TYPES` for new type conflicts:

```python
DISJOINT_TYPES.add(frozenset({"my_type_a", "my_type_b"}))
```

#### Adding New Type Hierarchies

Edit `TYPE_HIERARCHY` in `td/reasoning/contradiction_detector.py`:

```python
TYPE_HIERARCHY["my_subtype"] = {"my_supertype", "entity"}
```

This prevents false positives: "X is my_subtype" and "X is my_supertype" = no conflict.

#### Performance

LOTG runs in <1ms per `add_fact()` call. It uses only dict lookups and set intersections — no Z3, no ML, no external dependencies.

#### File

`td/reasoning/contradiction_detector.py` — standalone module, ~180 lines

---

## 4.5 Word Sense Disambiguation (WSD)

TD v2 handles polysemous entities (words with multiple meanings) through dynamic sense induction. When the same entity is taught with incompatible `is_a` types, separate sense URIs are created automatically.

### How It Works

```python
# Biology sense of "cell" — requires is_a for sense creation
td.teach("cell is_a organelle", "organelle")      # → (cell, is_a, organelle)

# Prison sense — "room" ≠ "organelle" → new sense created
td.teach("cell is_a room", "room")                 # → (cell_1, is_a, room)

# Technology sense — "device" ≠ "organelle" and ≠ "room" → another sense
td.teach("cell is_a device", "device")             # → (cell_2, is_a, device)

# Non-is_a facts route to the best matching sense via BEAGLE context
td.teach("cell is part of organism", "organism")   # → (cell, part_of, organism)
td.teach("cell is part of prison", "prison")       # → (cell_1, part_of, prison)
```

**Important:** Sense creation requires `is_a` declarations. BEAGLE context vectors for short teach sentences (3-4 words) are unreliable — the distinguishing signal is one word, giving cosine similarity ~0.0 ± 0.01 (noise). The `is_a` object is the only reliable sense indicator.

### Sense Routing Rules

| Signal | When Used | Example |
|--------|-----------|---------|
| **`is_a` object comparison** | Primary — for type declarations | "cell is_a room" vs "cell is_a organelle" → different senses |
| **LOTG subsumption** | Compatible types | "city" ⊑ "place" → same sense |
| **Morphological prefix** | Related terms | "organelle" vs "organism" → share "organ" → same sense |
| **Sequential (most recent)** | Non-`is_a` relations | "cell part_of prison" → routes to prison sense |

### Data Structures

```python
# In KnowledgeGraph:
kg.sense_inventory: dict[str, list[str]]
# "cell" → ["cell", "cell_1", "cell_2"]

# In WordVectorModel (BEAGLE — for future query-time resolution):
wvm.sense_clusters: dict[str, list[tuple[np.ndarray, int, str]]]
# "cell" → [(context_vec, count, example_sentence), ...]
```

### Query-Time Resolution (Partial)

For queries about entities with multiple senses, the system tries to resolve the correct sense using BEAGLE context. Currently works for single-sense entities; multi-sense query routing is a work in progress.

```python
# Works (single sense):
td.think("is cell part of organism?")  # → YES (cell = biology)

# Needs improvement (multi-sense):
td.think("is cell part of prison?")    # → needs sense resolution
```

### API

```python
# Get all sense URIs for an entity
kg.get_sense_uris("cell")  # → ["cell", "cell_1", "cell_2"]

# Get surface form from sense URI
kg.get_surface_form("cell_1")  # → "cell"

# Manually induce a sense (usually automatic)
kg.induce_new_sense("cell", conflicting_types={"room"}, proof="...")

# Resolve sense URI using context
kg.resolve_sense_uri("cell", context_sentence="the prisoner escaped", wvm=wvm)
```

### Test Suite

`tests/test_word_senses.py` — 57 tests covering:
- Sense cluster creation, merging, persistence
- LOTG-triggered dynamic sense induction
- Teach/ask WSD routing
- Multiple senses per word (cell/bank/apple/python/mercury)
- Edge cases (cold start, empty context, stop words)
- Non-polysemous words: zero overhead

### Files

| File | What Changed |
|------|-------------|
| `td/perception/word_vectors.py` | `sense_clusters`, `_assign_to_cluster()`, `get_sense()`, `_check_sense_merge()` |
| `td/kg/__init__.py` | `sense_inventory`, `resolve_sense_uri()`, `induce_new_sense()`, save/load |
| `td/thinking.py` | `_type_matches_any()`, `_get_is_a_objects()`, `_induce_senses_from_context()`, teach/ask WSD routing |
| `tests/test_word_senses.py` | 57 tests, 13 test classes |

---

## 5. When to Teach Relation Properties

### Pre-Seeded (No Action Needed)

These relations work immediately without any teaching:

| Relation | Properties | Example Derivation |
|----------|-----------|-------------------|
| `in` | transitive | Paris→France→EU → Paris→EU |
| `part_of` | transitive | Engine→Car→Fleet → Engine→Fleet |
| `before` | transitive | A→B→C → A→C |
| `after` | transitive | Same as before |
| `same_as` | symmetric, transitive | A↔B → B↔A |
| `married_to` | symmetric | Alice→Bob → Bob→Alice |
| `capital_of` | functional | Paris→France, Berlin→Germany → Paris≠Berlin |
| `has_capital` | functional | France→Paris, Germany→Berlin → France≠Germany |

### Must Teach (Novel Relations)

When you introduce a new relation token, the system does not know its properties. Use this decision tree:

```
Is the relation describing...
├── A chain/ordering? (X→Y→Z means X→Z)    → teach: transitive
│   Examples: north_of, south_of, before, after,
│             ancestor_of, contains, depends_on
├── A mutual relationship? (X→Y means Y→X) → teach: symmetric
│   Examples: married_to, sibling_of, adjacent_to,
│             borders, neighbor_of, equals
├── A unique mapping? (X→Y, X→Z → Y=Z)      → teach: functional
│   Examples: birth_date, birth_year, capital_of,
│             has_capital, unique_id
└── An opposite direction? (X→Y ↔ Y→X)     → teach: inverse:R2
    Examples: parent_of↔child_of, owns↔owned_by,
              employs↔employed_by
```

### Rule of Thumb

> If the relation describes a **chain**, it's transitive. If it describes a **mutual bond**, it's symmetric. If it describes a **unique identity**, it's functional.

### Teaching Order for Bulk Data

When loading from a structured dataset (CSV, Wikidata), teach relation properties **before** loading facts:

```python
# Step 1: Declare all relation properties
td.teach_relation("located_in", "transitive")
td.teach_relation("capital_of", "functional")
td.teach_relation("borders", "symmetric")
td.teach_relation("parent_of", "transitive", "inverse:child_of")
td.teach_relation("child_of", "transitive", "inverse:parent_of")

# Step 2: Load facts (inference runs immediately)
for row in dataset:
    td.teach(f"{row['subject']} is {row['relation']} {row['object']}")
```

This ensures TD can derive new facts from the moment the first triple is stored, not after the entire dataset is loaded.

---

## 6. Debugging

### Enable Reasoning Trace in the Demo

```
trace
```

This shows the full derivation chain for every answer, including which rule template was applied and at which hop.

### Check KG State

```python
from td.thinking import GenericThinkingDust
from td.perception.hdc import build_default_vocabulary
from td.memory.mhn import ModernHopfieldNetwork, MHNConfig

vocab = build_default_vocabulary(dim=10000)
mhn = ModernHopfieldNetwork(MHNConfig(dim=10000, min_similarity=0.01))
td = GenericThinkingDust(vocab=vocab, mhn=mhn, dim=10000, pure_mode=True)

# Add some facts
td.teach("Paris is the capital of France", "Paris")
td.teach("France is in the EU", "France is in the EU")

# Check KG statistics
stats = td.kg.stats()
print(stats)
# {'total_triples': 2, 'user_facts': 2, 'derived_facts': 0, 'relations': {'capital_of', 'in'}}

# List all triples
for triple in td.kg.get_all_triples():
    print(triple)
# (paris, capital_of, france, user, '')
# (france, in, eu, user, '')

# Check a specific path
paths = td.kg.find_paths("paris", "eu", max_hops=4)
print(f"Paths from paris to eu: {paths}")
```

### Check Word Vector Quality

```python
# Verify BEAGLE vectors are loaded
print(f"Vocabulary size: {len(td.wvm.env_vectors)}")

# Check semantic similarity (should be high for related words)
cap_city = td.wvm.similarity("capital", "city")
cap_sort = td.wvm.similarity("capital", "sort")
print(f"similarity(capital, city): {cap_city:.3f}  (expect >0.15)")
print(f"similarity(capital, sort): {cap_sort:.3f}  (expect <0.05)")

# Find nearest neighbors
neighbors = td.wvm.nearest_neighbors("paris", 5)
print(f"Nearest neighbors of 'paris': {neighbors}")
```

### Check MHN Retrieval

```python
# Store a fact
td.teach("The capital of France is Paris", "Paris")

# Retrieve by paraphrase
results = td.mhn.retrieve("paris france capital")
print(f"MHN retrieval: {results}")
```

### Check Z3 Solver

```python
from td.z3_solver import GenericZ3Solver

zs = GenericZ3Solver()

# Add constraints
zs.add_variable("x", "Int")
zs.add_constraint("x > 0")
zs.add_constraint("x < 10")

# Solve
result = zs.solve()
print(f"Z3 solution: {result}")
# {'x': 1} (or any valid model)
```

### Verbose Test Output

```bash
# Run tests with full output
.venv/bin/python -m pytest tests/test_thinking.py -v -s

# Run a single test with maximum verbosity
.venv/bin/python -m pytest tests/test_thinking.py::TestKG::test_transitive_chain_6_hops -v -s

# Run tests matching a keyword
.venv/bin/python -m pytest tests/ -k "transitive" -v
```

---

## 7. Adding New Rule Templates

### Step 1: Define the Template

Edit `td/kg/rules.py`:

```python
# Add to RULE_TEMPLATES dict:

"reflexive": {
    "name": "Reflexivity",
    "rule": "R(X,X) is always true",
    "applies_to": ["reflexive"],
    "check": lambda r, s, o: s == o,  # Subject equals object
    "description": "R(X,X) holds for all X. e.g., same_as is reflexive."
},
```

### Step 2: Register the Property

```python
td.kg.set_relation_property("same_as", "reflexive")
# Or add to DEFAULT_RELATION_PROPERTIES in minimal_seed.py
```

### Step 3: Implement the Inference Logic

Edit `td/kg/__init__.py` — find where `RULE_TEMPLATES` is applied and add:

```python
if "reflexive" in properties:
    # For reflexive relations R: if R(X,Y) is known, also assert R(X,X) if X is in KG
    for triple in kg_triples:
        subject, relation, obj = triple
        if relation == "same_as":
            # Add reflexive fact: same_as(X,X) is always derivable
            kg.add_fact(subject, subject, subject, source="derived", proof=f"reflexive: {subject}")
```

### Step 4: Add Tests

```python
# In tests/test_thinking.py or tests/test_generalization.py:

def test_reflexive_inference(td):
    td.teach("Alice same_as Alice", "Alice")  # Already reflexive but let's be explicit
    td.teach("Bob same_as Charlie", "Bob")
    # Bob same_as Charlie → Bob same_as Bob (reflexive of same_as is same_as)
    result = td.ask("Is Bob same_as Bob?")
    assert result.answer == "YES"
    assert "reflexive" in result.proof.lower()
```

---

## 8. Future Development Guide

### 8.1 Adding Temporal Reasoning

**Files to modify:** `td/kg/__init__.py`, `td/perception/nl_parser.py`, new file `td/reasoning/temporal.py`

**Step 1: Extend the triples table with temporal fields**

```sql
ALTER TABLE triples ADD COLUMN start_time INTEGER;
ALTER TABLE triples ADD COLUMN end_time INTEGER;
```

**Step 2: Create the temporal reasoning module**

```python
# td/reasoning/temporal.py

"""
Allen's Interval Algebra (Allen, 1983, CACM)

13 mutually exclusive and exhaustive relations between time intervals:
before, after, meets, met_by, overlaps, overlapped_by, during, contains,
starts, started_by, finishes, finished_by, equals
"""

from enum import Enum
from typing import Tuple, Optional

class AllenRelation(Enum):
    BEFORE      = "<"
    AFTER       = ">"
    MEETS       = "m"
    MET_BY      = "mi"
    OVERLAPS    = "o"
    OVERLAPPED_BY = "oi"
    DURING      = "d"
    CONTAINS    = "di"
    STARTS      = "s"
    STARTED_BY  = "si"
    FINISHES    = "f"
    FINISHED_BY = "fi"
    EQUALS      = "="

# Allen composition table: composition[rel1][rel2] = resulting relation
# Full 13×13 table per Allen (1983), Table 1
COMPOSITION_TABLE = {
    # ... (169 entries, see ARCHITECTURE.md or Allen's original paper)
}

def compose(rel1: AllenRelation, rel2: AllenRelation) -> AllenRelation:
    """Compute the composition of two Allen relations."""
    return COMPOSITION_TABLE[rel1][rel2]

def infer_temporal(sentence: str) -> Tuple[Optional[str], Optional[str], AllenRelation]:
    """Extract (entity_a, entity_b, allens_relation) from a temporal sentence."""
    # "A is before B" → (A, B, BEFORE)
    # "A meets B"     → (A, B, MEETS)
    # etc.
    import re
    patterns = [
        (r"(\w+) is before (\w+)", AllenRelation.BEFORE),
        (r"(\w+) is after (\w+)", AllenRelation.AFTER),
        (r"(\w+) meets (\w+)", AllenRelation.MEETS),
        (r"(\w+) overlaps (\w+)", AllenRelation.OVERLAPS),
        (r"(\w+) is during (\w+)", AllenRelation.DURING),
        (r"(\w+) starts (\w+)", AllenRelation.STARTS),
        (r"(\w+) finishes (\w+)", AllenRelation.FINISHES),
    ]
    for pattern, rel in patterns:
        m = re.search(pattern, sentence)
        if m:
            return m.group(1), m.group(2), rel
    return None, None, None
```

**Step 3: Connect to Z3**

```python
# In td/kg/__init__.py — after storing a temporal triple:
from td.reasoning.temporal import AllenRelation, compose

def derive_temporal_facts(self, triple_id: int, a: str, b: str, rel: AllenRelation):
    """Use Allen composition to derive new temporal facts."""
    # Find all triples involving B as subject
    b_as_subject = self.query("SELECT * FROM triples WHERE subject = ?", (b,))
    for t in b_as_subject:
        _, _, c, _, _ = t
        # Compose: if A rel1 B and B rel2 C, then A rel3 C
        rel3 = compose(rel, t.relation_as_allen)
        self.add_fact(a, c, rel3.value, source="derived",
                      proof=f"temporal: {a} {rel.value} {b} + {b} {t.rel} {c}")
```

### 8.2 Adding Clause Segmentation

**Files to modify:** `td/perception/nl_parser.py`

**Step 1: Rule-based clause splitter**

```python
# td/perception/clause_segmenter.py

"""
Rule-based clause segmentation.
Covers coordinating conjunctions, subordinating conjunctions, and relative clauses.
For full DisCoDisCo integration, see Hu et al. (2023), Applied Sciences 13(4).
"""

from typing import List
import re

COORDINATING_CONJUNCTIONS = ["and", "but", "or", "nor", "yet", "so"]
SUBORDINATING_CONJUNCTIONS = ["because", "although", "while", "when", "if", "unless",
                               "since", "though", "whereas", "after", "before", "until"]
RELATIVE_PRONOUNS = ["which", "who", "that", "whom", "whose"]

def segment_clauses(sentence: str) -> List[str]:
    """Split a compound sentence into clauses."""
    clauses = []

    # Handle relative clauses: "X, which Y, Z" → ["X", "which Y", "Z"]
    # Handle: "X who Y" → ["X", "who Y"]

    # Split on comma + coordinating conjunction
    for conj in COORDINATING_CONJUNCTIONS:
        pattern = rf",\s+{conj}\s+"
        parts = re.split(pattern, sentence, flags=re.IGNORECASE)
        if len(parts) > 1:
            return [p.strip() for p in parts if p.strip()]

    # Split on subordinating conjunction
    for conj in SUBORDINATING_CONJUNCTIONS:
        pattern = rf"\s+{conj}\s+"
        parts = re.split(pattern, sentence, flags=re.IGNORECASE)
        if len(parts) > 1:
            clauses.extend([p.strip() for p in parts if p.strip()])
            return clauses

    # Split on relative pronoun
    for rp in RELATIVE_PRONOUNS:
        pattern = rf",\s+{rp}\s+"
        parts = re.split(pattern, sentence, flags=re.IGNORECASE)
        if len(parts) > 1:
            # parts[0] is main clause, parts[1:] are relative clauses
            main = parts[0].strip()
            relatives = [f"{rp} {p.strip()}" for p in parts[1:] if p.strip()]
            return [main] + relatives

    # Single clause
    return [sentence.strip()] if sentence.strip() else []
```

**Step 2: Connect to parser**

```python
# In td/perception/nl_parser.py:

from td.perception.clause_segmenter import segment_clauses

def parse(self, sentence: str) -> List[Triple]:
    # Step 1: Segment clauses
    clauses = segment_clauses(sentence)
    all_triples = []
    for clause in clauses:
        triples = self._extract_triples_from_clause(clause)
        all_triples.extend(triples)

    # Step 2: Resolve anaphora across clauses
    entity_stack = []
    resolved_clauses = []
    for clause in clauses:
        resolved, updated_stack = self.resolve_anaphora(clause, entity_stack)
        resolved_clauses.append(resolved)
        entity_stack = updated_stack

    return all_triples
```

**Step 3: Anaphora resolution**

```python
def resolve_anaphora(clause: str, recency_stack: List[str]) -> Tuple[str, List[str]]:
    """Map pronouns to antecedents using recency + type constraints."""
    pronouns = {"he": "person", "she": "person", "it": "object", "they": "person_or_group",
                "him": "person", "her": "person", "them": "person_or_group",
                "his": "person", "its": "object", "their": "person_or_group"}

    resolved = clause
    for pronoun, expected_type in pronouns.items():
        pattern = rf"\b{pronoun}\b"
        if re.search(pattern, clause):
            # Find most recent antecedent in recency_stack
            # (simplified: just use the first entry; full impl needs type checking)
            if recency_stack:
                antecedent = recency_stack[0]  # Most recent
                resolved = re.sub(pattern, antecedent, resolved)
                recency_stack = [antecedent] + recency_stack  # Update recency

    return resolved, recency_stack
```

### 8.3 Adding Multilingual Support

**Files to create/modify:** `td/perception/universal_dependencies.py`, `td/perception/morphology/`, `td/perception/languages/`

**Step 1: UD Parser wrapper**

```python
# td/perception/universal_dependencies.py

"""
Universal Dependencies integration for multilingual parsing.
References:
  Nivre et al. (2016), de Marneffe et al. (2021), Computational Linguistics 47(2)
  Kleyko et al. (2023), ACL 2025 (HDC cross-lingual alignment)
"""

from typing import List, Dict, Tuple

# Universal POS tags (17 total)
UNIVERSAL_POSTAGS = {
    "NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "NUM", "CONJ",
    "PART", "PUNCT", "SCONJ", "CCONJ", "INTJ", "X", "PROPN", "AUX"
}

# Universal dependency relations (37 total)
UNIVERSAL_DEPRELS = {
    "nsubj", "obj", "iobj", "csubj", "ccomp", "xcomp", "obl", "vocative",
    "expl", "dislocated", "discourse", "aux", "cop", "mark", "det",
    "clf", "case", "nmod", "amod", "nummod", "appos", "nsubjpass",
    "ccomp_pass", "auxpass", "compound", "mwe", "fixed", "flat",
    "goeswith", " Reparandum", "root", "dep", "conj", "cc", "punct",
    "list", "parataxis", "orphan", "nummod", "case"
}

# Universal POS to semantic role mapping
POS_TO_ROLE = {
    "NOUN": "entity",
    "PROPN": "entity",
    "VERB": "relation",
    "ADJ": "modifier",
    "ADV": "modifier",
}

def parse_ud(treebank: str, lang: str) -> List[Tuple[str, str, str]]:
    """
    Parse a sentence using UD annotations.
    Returns list of (subject, relation, object) triples.

    In production, this calls a UD parser (udpipe, Stanza, spaCy-udpipe).
    For now, we document the expected interface.
    """
    # This would call: spacy_udpipe.load(lang)(sentence)
    # For English: "Paris --nsubj--> capital --ROOT--> of --case--> France"
    raise NotImplementedError("UD parser integration is planned for TD v2.5")
```

**Step 2: HDC cross-lingual alignment (Kleyko et al., 2023)**

```python
# td/perception/cross_lingual.py

"""
HDC cross-lingual vector alignment using seed dictionaries.
Reference: Kleyko et al. (2023), ACL 2023, "Bilingual Lexicon Extraction from HD Vectors"
"""

import numpy as np
from scipy.linalg import orthogonal_procrustes

def align_languages(
    source_vecs: np.ndarray,   # Shape: (vocab_size, dim) for source lang
    target_vecs: np.ndarray,   # Shape: (vocab_size, dim) for target lang
    seed_pairs: List[Tuple[str, str]],  # Bilingual seed dictionary
    source_word2idx: Dict[str, int],
    target_word2idx: Dict[str, int],
) -> np.ndarray:
    """
    Align two HDC vector spaces using Procrustes analysis on a seed dictionary.

    Given 100-500 word pairs known to be translations of each other,
    find the orthogonal transformation W such that:
        W @ source_vec[i] ≈ target_vec[i]  for all seed pairs

    This enables cross-lingual semantic search: "dog" (English) →
    finds "Hund" (German) in the aligned space.
    """
    # Collect aligned word pairs from seed dictionary
    source_points = []
    target_points = []
    for src_word, tgt_word in seed_pairs:
        if src_word in source_word2idx and tgt_word in target_word2idx:
            src_idx = source_word2idx[src_word]
            tgt_idx = target_word2idx[tgt_word]
            source_points.append(source_vecs[src_idx])
            target_points.append(target_vecs[tgt_idx])

    X = np.array(source_points)  # (n_seed, dim)
    Y = np.array(target_points)  # (n_seed, dim)

    # Procrustes: find W minimizing ||W @ X - Y||_F
    W, _ = orthogonal_procrustes(X, Y)

    return W  # Shape: (dim, dim) — orthogonal transformation

def project_cross_lingual(word: str, lang_pair: str, word2idx: Dict, vectors: np.ndarray) -> np.ndarray:
    """Project a word vector through the alignment matrix for cross-lingual search."""
    alignment_matrix = ALIGNMENT_MATRICES[lang_pair]  # e.g., "en-de"
    return alignment_matrix @ vectors[word2idx[word]]
```

**Step 3: Language-specific morphological analyzers**

Each language gets a ~200-line module. Example structure for German:

```python
# td/perception/languages/de.py

"""
German morphological analyzer.
Handles: compound nouns, case system, verb conjugation.
Reference: Universal Dependencies (Nivre et al., 2016; de Marneffe et al., 2021)
"""

import re

# German compound noun splitting (heuristic)
COMPOUND_PREFIXES = ["schaden", "unfall", "haft", "versicherungs", "kraftfahrzeug"]
COMPOUND_SUFFIXES = ["ung", "heit", "schaft", "tum", "nis", "chen", "lein"]

def normalize_german(word: str) -> str:
    """Normalize German word: lowercase, strip case markers."""
    word = word.lower()
    # Strip articles
    word = re.sub(r'^(der|die|das|den|dem|des|ein|eine|einer|einem|eines)$', '', word)
    # Strip case endings
    word = re.sub(r'(en|ern|em|en)$', 'e', word)  # Simplified
    return word.strip()

def parse_german_dependency_tree(ud_tokens) -> List[Tuple[str, str, str]]:
    """
    Convert Universal Dependencies parse to (subject, relation, object) triples.
    German word order: SOV (Subject-Object-Verb), but UD relations are language-independent.
    """
    triples = []
    for token in ud_tokens:
        if token.deprel == "nsubj":
            subject = token.text
            # Walk up to find root verb
            root = token.head
            relation = root.lemma
            # Find object
            for child in root.children:
                if child.deprel in ("obj", "iobj", "nmod"):
                    obj = child.text
                    triples.append((subject, relation, obj))
    return triples
```

### 8.4 Adding Automatic Rule Discovery

```python
# td/reasoning/rule_discovery.py

"""
Automatic Rule Discovery via Inductive Logic Programming (ILP).
References:
  Muggleton (1991), "Inductive Logic Programming", New Generation Computing 8(4): 295-318
  Cropper & Muggleton (2016), "Logical Minimisation of Meta-Rules", Springer Ch. 12
  Solar-Lezama (2008), "Program Synthesis by Sketching", UC Berkeley PhD Thesis
"""

from typing import Dict, List, Set, Tuple
from collections import defaultdict


def learn_relation_properties(
    triples: List[Tuple[str, str, str]],
    threshold: float = 0.9,
) -> Dict[str, List[str]]:
    """
    Learn relation properties from observed triples using co-occurrence statistics.

    Algorithm:
    1. For each relation R, collect all (X, Y) pairs where R(X, Y) holds.
    2. Check transitivity: for each (A, R, B) and (B, R, C), does (A, R, C) hold?
       If > threshold% of chains are valid, declare R as transitive.
    3. Check symmetry: for each (A, R, B), does (B, R, A) hold?
       If > threshold% are valid, declare R as symmetric.
    4. Check functionality: for each A, how many distinct B values exist for R(A, B)?
       If always 1, declare R as functional.

    Args:
        triples: List of (subject, relation, object) triples.
        threshold: Minimum fraction of supporting evidence to declare a property.

    Returns:
        Dict mapping relation name to list of discovered properties.
    """
    # Index triples by relation
    by_relation: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for s, r, o in triples:
        by_relation[r].append((s, o))

    discovered: Dict[str, List[str]] = {}

    for relation, pairs in by_relation.items():
        properties = []

        # --- Transitivity check ---
        adj_from: Dict[str, Set[str]] = defaultdict(set)
        all_pairs_set: Set[Tuple[str, str]] = set()
        for s, o in pairs:
            adj_from[s].add(o)
            all_pairs_set.add((s, o))

        transitive_count = 0
        total_chains = 0
        for a, b in pairs:
            for c in adj_from[b]:
                total_chains += 1
                if (a, c) in all_pairs_set:
                    transitive_count += 1

        if total_chains > 0 and (transitive_count / total_chains) >= threshold:
            properties.append("transitive")

        # --- Symmetry check ---
        symmetric_count = 0
        total_asymmetric = 0
        for s, o in pairs:
            if s != o:
                total_asymmetric += 1
                if (o, s) in all_pairs_set:
                    symmetric_count += 1

        if total_asymmetric > 0 and (symmetric_count / total_asymmetric) >= threshold:
            properties.append("symmetric")

        # --- Functionality check ---
        subject_objects: Dict[str, Set[str]] = defaultdict(set)
        for s, o in pairs:
            subject_objects[s].add(o)

        max_fanout = max(len(objs) for objs in subject_objects.values()) if subject_objects else 0
        if max_fanout <= 1 and len(pairs) >= 2:
            properties.append("functional")

        if properties:
            discovered[relation] = properties

    return discovered


def verify_with_z3(
    relation: str,
    property_name: str,
    triples: List[Tuple[str, str, str]],
) -> bool:
    """
    Verify a hypothesized relation property against all stored facts using Z3.

    CEGIS (Counterexample-Guided Inductive Synthesis):
    verify the learned rule against the entire knowledge base.
    """
    from z3 import Solver, Bool, sat

    s = Solver()

    triple_vars = {}
    for subj, rel, obj in triples:
        key = (subj, rel, obj)
        triple_vars[key] = Bool(f"fact_{subj}_{rel}_{obj}")
        s.add(triple_vars[key])

    if property_name == "transitive":
        by_r = [(a, b) for a, r, b in triples if r == relation]
        adj = defaultdict(set)
        for a, b in by_r:
            adj[a].add(b)

        for a, b in by_r:
            for c in adj[b]:
                neg_var = Bool(f"neg_{a}_{relation}_{c}")
                s.push()
                s.add(neg_var)
                if s.check() == sat:
                    s.pop()
                    return False
                s.pop()

    return True


def discover_and_apply(td_instance, threshold: float = 0.9) -> Dict[str, List[str]]:
    """
    Run automatic rule discovery on the current knowledge graph.
    Discovered properties are automatically registered.
    """
    all_triples = td_instance.kg.get_all_triples()
    triples_list = [(t[0], t[1], t[2]) for t in all_triples]

    discovered = learn_relation_properties(triples_list, threshold=threshold)

    for relation, props in discovered.items():
        for prop in props:
            td_instance.kg.set_relation_property(relation, prop)
            print(f"  Discovered: {relation} is {prop}")

    return discovered
```

### 8.5 Coreference Resolution (spaCy Two-Pipeline Approach)

**Status:** Working (spaCy 3.7.5 + en_coreference_web_trf)

**The problem:** Pronouns ("it", "he", "she", "they", "its", "this") refer to entities mentioned earlier. Without coreference resolution, the parser extracts pronouns as entities instead of resolving them.

```
Input:  "A video game console is designed for its video games. It runs smoother."
Without coref: (it, runs, smoother) ← "it" is not an entity
With coref:    (video game console, runs, smoother) ← resolved
```

**The solution:** spaCy's two-pipeline approach (official workaround):

```python
import spacy

# Load main pipeline (for POS, dep, NER)
nlp = spacy.load("en_core_web_sm")

# Load coreference pipeline (separate model, shared vocab)
nlp_coref = spacy.load("en_coreference_web_trf", vocab=nlp.vocab)

# Process text through both pipelines
doc = nlp(text)
doc = nlp_coref(doc)

# Access coreference clusters
for cluster_key, spans in doc.spans.items():
    if cluster_key.startswith("coref_clusters"):
        print(f"{cluster_key}: {[str(s) for s in spans]}")
# Output: coref_clusters_1: ['A video game console', 'its', 'it']
```

**What works:**
- Pronouns (he/she/it/they/him/her/them/its) → antecedents across sentences
- Split antecedents: "Alice and Bob... They" → [Alice, Bob]
- Possessives: "its video games" → "video game console's video games"

**What doesn't work:**
- "Each" not resolved (split antecedent for coordinated objects)
- Some cat/mat ambiguity (structural similarity confuses model)
- "this" as demonstrative pronoun (sometimes resolved, sometimes not)

**Dependencies:**
```
spacy-experimental>=0.6.4    # installs with spaCy 3.7.5
en_coreference_web_trf        # 490MB model, one-time download
spacy-transformers>=1.4.0     # transformer backend
```

**Note:** This downgrades spaCy from 3.8 to 3.7.5. All 585 tests pass on 3.7.5. The model is trained on spaCy 3.4 but works on 3.7.5 with version warnings.

**Integration plan:**
1. Add `resolve_coreferences(text)` method to `GenericNLParser`
2. Run coreference resolution BEFORE triple extraction
3. Replace pronouns with resolved entity names in the text
4. Then run existing extraction pipeline on the resolved text

**Research:**
- spaCy coref blog: https://explosion.ai/blog/coref
- spaCy Discussion #11585: End-to-end neural coref in spaCy
- spaCy Discussion #12302: Combining en_coreference_web_trf with en_core_web_trf
- GitHub #13111: Pre-trained coref incompatible with spaCy > 3.4

---

### 8.6 Discourse Deixis (the "this"/"that" problem)

**Status:** Not implemented. Research only.

**The problem:** "This" and "that" can refer to either an entity or a discourse segment (clause/event). Coreference resolution handles entity references but NOT discourse deixis.

```
Entity reference:     "I bought a car. This car is fast." → "this" = car
Discourse deixis:     "I bought a car. This surprised my wife." → "this" = the event of buying
```

For KG extraction, discourse deixis is a **noise source** — "this" referring to a clause doesn't produce a valid (subject, relation, object) triple.

**The Two-Stage Approach** (Guerra et al., SemEval 2015):
1. **Classify:** Is the pronoun entity-referring or discourse-deictic?
2. **Resolve:** If entity-referring, resolve to antecedent. If discourse-deictic, skip.

**Key features for classification:**
- Syntactic role: discourse deixis often appears as subject of abstract verbs ("this shows", "this means", "this proves")
- Previous sentence: discourse deixis refers to whole clauses, not noun phrases
- Verb type: abstract/cognitive verbs ("surprise", "show", "mean", "prove") → discourse deixis

**For TD v2:** The simplest approach is to filter out "this"/"that" when they appear as subjects of abstract verbs, and treat them as noise. This avoids the complexity of full discourse deixis resolution while preventing false triples.

**Reference:** Guerra, R.D. et al. "Resolving Discourse-Deictic Pronouns: A Two-Stage Approach." SemEval 2015. URL: https://aclanthology.org/S15-1035.pdf

**Reference:** Webber, B.L. "Discourse Deixis: Reference to Discourse Segments." ACL 1988.

**Reference:** Stanford Jurafsky, "Coreference Resolution." Chapter 21. URL: https://web.stanford.edu/~jurafsky/slp3/old_dec21/21.pdf

---

## 9. Test Suite Overview

TD v2 has **99 tests** across three test files:

### test_thinking.py — Core Reasoning (162 lines)

| Test Class | Tests | What's Verified |
|-----------|-------|----------------|
| `TestIDP` | 3 | Iterative Deepening Pattern: convergence, empty memory, state evolution |
| `TestHDCDecomposition` | 3 | HDC decomposition into sub-problems, prototype universality |
| `TestMHN` | 3 | MHN storage/retrieval, paraphrase matching |
| `TestZ3` | 5 | Z3 constraint solving: arithmetic, Boolean, uninterpreted functions |
| `TestKG` | 6 | KG storage, retrieval, transitive inference, functional contradiction |
| `TestTeach` | 5 | Teaching facts, relation properties, online BEAGLE update |
| `TestParaphrase` | 3 | Paraphrase matching: capital/city, capital/sort, synonym matching |
| **Total** | **28** | Core reasoning correctness |

### test_generalization.py — Generalization (575 lines, 34 tests)

| Test Class | Tests | What's Verified |
|-----------|-------|----------------|
| `TestNovelRelations` | 6 | Relations never seen during training: north_of, south_of, borders, depends_on, ancestor_of, contains |
| `TestTransitiveChains` | 8 | 2-hop through 6-hop transitive chains |
| `TestFunctionalContradiction` | 5 | Uniqueness constraints: capital_of, birth_date, has_capital |
| `TestSymmetricInference` | 5 | Symmetric relations: married_to, sibling_of, adjacent_to, borders |
| `TestInverseRelations` | 5 | Inverse pairs: capital_of↔has_capital, parent_of↔child_of, owns↔owned_by |
| `TestCrossRelationComposition` | 5 | Composition across different relations: capital_of+in→in, in+part_of→part_of |
| **Total** | **34** | Generalization to unseen patterns |

### test_realworld.py — Real-World Scenarios (235 lines, 15 tests)

| Test Class | Tests | What's Verified |
|-----------|-------|----------------|
| `TestGeography` | 5 | Countries, capitals, regions, continents: multi-hop inference |
| `TestOrganizations` | 4 | Corporate structure: subsidiaries, headquarters, partnerships |
| `TestScience` | 3 | Biological taxonomy: genus→family→order→class→phylum→kingdom |
| `TestTechnology` | 3 | Tech stack: language→framework→platform→cloud |
| **Total** | **15** | Realistic domain reasoning |

### Additional Unit Tests

| Test File | ~Tests | What's Verified |
|-----------|--------|----------------|
| `test_hdc.py` | ~12 | HDC operations: bind, bundle, permute, cosine similarity, vector norms |
| `test_mhn.py` | ~10 | MHN storage, retrieval, similarity thresholds, capacity limits |
| `test_ca_reservoir.py` | ~5 | CA reservoir: Rule 90, feature extraction, determinism |

### Running Tests

```bash
# All tests
.venv/bin/python -m pytest tests/ -v

# Just core reasoning
.venv/bin/python -m pytest tests/test_thinking.py -v

# Just generalization
.venv/bin/python -m pytest tests/test_generalization.py -v

# Just real-world scenarios
.venv/bin/python -m pytest tests/test_realworld.py -v

# With timing
.venv/bin/python -m pytest tests/ -v --durations=10

# Parallel (if pytest-xdist installed)
.venv/bin/python -m pytest tests/ -v -n auto
```

---

## 10. Common Issues and Fixes

### Z3 Import Error

```
ModuleNotFoundError: No module named 'z3'
```

**Fix:**
```bash
.venv/bin/pip install z3-solver==4.12.1
# On Apple Silicon, if that fails:
brew install z3
.venv/bin/pip install z3-solver --no-binary z3-solver
```

### BEAGLE Vectors Not Found

```
FileNotFoundError: data/word_vectors_10k.pkl
```

**Fix:** The vectors are trained automatically on first run. If auto-training fails:
```bash
.venv/bin/python -c "
from td.perception.word_vectors import WordVectorModel
wvm = WordVectorModel(dim=10000)
with open('data/synthetic_corpus_10k.txt') as f:
    sentences = [l.strip() for l in f if l.strip()]
wvm.train(sentences)
wvm.save('data/word_vectors_10k.pkl')
print(f'Trained {len(wvm.env_vectors)} word vectors')
"
```

### Paraphrase Matching Returns No Results

**Symptoms:** `td.ask("is Paris part of the EU?")` returns "I don't know" even though `France in EU` is stored.

**Cause:** The BEAGLE corpus doesn't contain "part of" in contexts similar to "in". The similarity threshold (0.15) is not met.

**Fix:** Add domain-specific sentences to the corpus and retrain BEAGLE vectors (see Section 3).

### Transitive Inference Not Working for New Relations

**Symptoms:** `td.teach("X north_of Y")` then `td.teach("Y north_of Z")` but `td.ask("Is X north of Z?")` returns "I don't know".

**Cause:** The relation `north_of` is not registered as transitive.

**Fix:**
```python
td.teach_relation("north_of", "transitive")
```

### SQLite Database Locked

```
sqlite3.OperationalError: database is locked
```

**Cause:** Multiple processes accessing the same `td_knowledge.db` simultaneously.

**Fix:**
```python
import sqlite3
conn = sqlite3.connect("data/td_knowledge.db")
conn.execute("PRAGMA journal_mode=WAL")
```

### MHN Retrieval Returns Wrong Fact

**Symptoms:** `td.mhn.retrieve("capital of France")` returns "Berlin" instead of "Paris".

**Cause:** Two facts with similar semantic keys were stored, and the MHN threshold is too low.

**Fix:** Increase the minimum similarity threshold:
```python
from td.memory.mhn import ModernHopfieldNetwork, MHNConfig
mhn = ModernHopfieldNetwork(MHNConfig(dim=10000, min_similarity=0.15))
```

### Test Suite Fails After Code Changes

**Diagnosis:**
```bash
.venv/bin/python -m pytest tests/ -v --tb=long
.venv/bin/python -m pytest tests/test_thinking.py::TestKG::test_transitive_chain -v -s

# Check if it's a data issue (stale pickle)
rm data/word_vectors_10k.pkl
.venv/bin/python -m pytest tests/ -v
```

---

## 11. Architecture Vision: The Two Minds

| | TD v2 (System 1) | TD Pro (System 2) |
|---|---|---|
| Role | Fast reasoning engine | Slow creative problem solver |
| Speed | <100ms | 1–5 seconds |
| Handles | Facts, constraints, inference, retrieval | Spatial, temporal, novel problems |
| Components | HDC + MHN + Z3 + BEAGLE + KG | Liquid-KAN + Hypernetwork + NCA + TTT |
| Status | **✅ Shipped** | 🔲 Planned |

### The Handoff Protocol

```
User Input
    ↓
TD v2 (Fast Path — 10-100ms)
    ├── Fact retrieval → answer ✅
    ├── Constraint solving → Z3 solution ✅
    ├── Logical inference → derived answer ✅
    └── Novel / Spatial / Complex / Unknown → hand off to TD Pro
            ↓
        TD Pro (Slow Path — 1-5 seconds)
            ├── Liquid-KAN encodes problem dynamics
            ├── Hypernetwork generates NCA parameters
            ├── NCA converges to solution attractor
            └── Solution stored in TD v2's MHN for future fast retrieval
                    ↓
                TD v2 answers immediately next time ✅
```

**The killer insight: TD Pro trains TD v2.** Every novel problem TD Pro solves becomes a pattern TD v2 retrieves instantly. The slow mind teaches the fast mind. Over time, TD v2 handles more problems without invoking TD Pro. This is how human expertise works — struggle once, then it becomes intuition.

---


## Roadmap

### ✅ Phase 0: Core Engine (DONE — 2026-07-05)
- [x] SPARQL query layer via pyoxigraph (18ms @ 10M triples)
- [x] Storage migration: SQLite → pyoxigraph (RDF, disk-persistent)
- [x] Clause segmentation: verb-based splitting via spaCy
- [x] Relation synonymy: teach + auto-detect + OWL equivalentProperty
- [x] Coreference resolution: spaCy two-pipeline approach
- [x] Temporal ordering: 45 English connectives, Allen's interval algebra
- [x] Triple deduplication: post-extraction canonicalization (EDC approach)
- [x] Compound verb+prep: "feeds into" → (a, feeds_into, b)
- [x] Multi-word entities: "World War 2", "united states of america"
- [x] 655 tests, 0 failures, 45 research references

### Phase 1: TD v2 Public Release (Month 1)
- Scale synthetic corpus to 100K-1M sentences
- Build web UI (ChatGPT-style chat interface)
- Add feedback buttons wired to `teach()`
- Demo video: "I started with nothing. I taught it 5 facts. It derived 3 new ones."

### Phase 2: TD Pro Integration (Month 2-3)
- Liquid-KAN controller (CfC, 19 neurons)
- Hypernetwork (task-specific parameter generation)
- Universal NCA Solver (grid-based reasoning)
- TD v2 ↔ TD Pro handoff protocol
- ARC-AGI benchmark evaluation

### Phase 3: Knowledge Expansion (Month 3-6)
- Wikidata → KG loading (1M+ structured triples)
- Internet search integration (real-time facts)
- Multi-modal input (images → HDC, audio → HDC)
- Structured data (CSV/JSON → triples)

### Phase 4: Multi-Agent TD (Month 6-12)
- Specialized agents: TD-geo, TD-code, TD-schedule, TD-sience
- Common HDC language for inter-agent communication
- Collaborative reasoning across domains

### Phase 5: TD as a Platform (Year 2)
- Users build custom TD personalities
- TD marketplace: share taught knowledge graphs
- TD for education: Socratic teaching mode
- TD for research: scientists teach domain knowledge

### Fuzzy Relation Matching

The `_query_knowledge_graph` method in `td/thinking.py` uses `difflib.SequenceMatcher` for morphology-agnostic relation matching:

```python
import difflib
best = max(difflib.SequenceMatcher(None, token, relation_part).ratio() for token in tokens)
if best >= 0.75:
    # Match found
```

**Threshold:** 0.75. Tested with 17 novel relations. Language-agnostic (works for any script).

### Adding Novel Relations

To test a new relation type:

1. Add facts with the new relation to the KG
2. Set its properties: `kg.set_relation_property("my_relation", "transitive")`
3. Write tests in `tests/test_novel_relations.py`
4. Verify the relation is NOT in `DEFAULT_RELATION_PROPERTIES` (to test true generalization)

### Non-Transitive Relation Safety

The `_find_valid_path` method prevents false inferences through non-transitive relations. When adding a new relation, consider:

- **Transitive:** `in`, `part_of`, `before`, `evolved_from`, `predecessor_of`
- **Non-transitive:** `borders`, `orbits`, `exports`, `born_in`, `directed_by`
- **Symmetric:** `borders`, `married_to`, `collaborated_with`, `affiliated_with`
- **Functional:** `capital_of`, `discovered_by`, `invented_by`, `painted_by`, `directed_by`

### Test Suite (293 tests)

| Suite | Tests | Domain |
|-------|-------|--------|
| test_thinking.py | 50 | HDC, MHN, CA, Z3, thinking engine |
| test_generalization.py | 34 | Relation extraction, transitive chains, functional, symmetric |
| test_realworld.py | 15 | Country capitals, EU, US states, rivers, Olympics |
| test_temporal.py | 69 | Allen's 13 relations, composition, open-ended intervals |
| test_temporal_kg.py | 29 | KG temporal query, SQLite persistence |
| test_realworld_wikipedia.py | 45 | Geography, History, Science, Tech, Sports, Literature, Music |
| test_novel_relations.py | 51 | 17 unseen relations, cross-domain, persistence |
| **Total** | **293** | |

*Last updated: 2026-07-02 GMT+5*

### Composition Rules

The `_find_valid_path` method uses OWL Property Chain style composition rules:

```python
# Set composition rules
kg.set_composition_rule("capital_of", "in", "in")   # valid composition
kg.set_composition_rule("born_in", "in", None)      # explicitly blocked
```

**How it works:**
1. For a multi-hop path, compute the composed relation from the chain
2. Check `composition_rules[(composed, next_rel)]` for each step
3. If the final composed relation matches the target, accept the path
4. If no explicit rule exists, fall back to transitivity heuristics

**Research foundation:**
- OWL 2 PropertyChain (W3C, 2009) — explicit composition declarations
- HolmE (Zheng et al., 2024) — KGE closed under composition
- Rot-Pro (NeurIPS, 2021) — transitivity by projection
- GLIDR (arXiv, 2025) — differentiable ILP for graph-structured rules
- Logical Rule-Based KGR Survey (MDPI, 2023) — comprehensive survey

### Compound Noun Detection

The parser has a `_merge_post_relation_entities()` method that merges adjacent non-stop tokens after spatial/temporal relation words:

```python
# In td/perception/nl_parser.py
# Pattern: [relation] [token1] [token2] → single entity
# "in central asia" → entity "central asia"
# "part of united states" → entity "united states"
```

**When to extend:**
- Add new relation words to `relation_words` set in `_merge_post_relation_entities()`
- Test with `test_realworld_wikipedia.py` and `test_battle_wikipedia.py`

**Reference:** Manning & Schütze (1999), "Foundations of Statistical NLP", Chapter 5: Collocations.

### Gazetteer (Multi-Word Entity Recognition)

The `KnowledgeGraph.gazetteer` is a `set[str]` of known multi-word entities. It's populated automatically from teach() interactions and persisted in SQLite.

```python
# Automatic: when you teach "United Kingdom is in Europe",
# the gazetteer learns "united kingdom"
kg.add_fact("united kingdom", "in", "europe")
# kg.gazetteer == {"united kingdom"}

# Used by _query_knowledge_graph for entity extraction in queries
# "is United Kingdom in Eurasia?" → gazetteer lookup → "united kingdom"
```

**How to test:**
```python
kg = KnowledgeGraph()
kg.add_fact("united kingdom", "in", "europe")
assert "united kingdom" in kg.gazetteer

# Save/load preserves gazetteer
kg.save(tmp_path)
kg2 = KnowledgeGraph()
kg2.load(tmp_path)
assert "united kingdom" in kg2.gazetteer
```

**Reference:** Nadeau & Sekine (2007), "A survey of named entity recognition and classification."

### spaCy Integration Guide

spaCy is fully integrated into the parser and thinking engine. All hardcoded rules replaced.

**What was replaced:**
1. `_pp_words` set → `token.pos_ == "ADP"` (spaCy POS tagger, Universal POS)
2. `_strip_pp()` helper → spaCy dependency parsing (PP attachment)
3. `_merge_post_relation_entities()` → `doc.noun_chunks` (spaCy noun chunking)
4. `difflib.SequenceMatcher` fuzzy matching → `token.lemma_` (spaCy lemmatizer)
5. Gazetteer → `doc.ents` (spaCy NER)
6. 12 regex patterns → `extract_triples_spacy()` (dependency tree traversal)
7. `generic_words` stopword set → `token.pos_ == "PROPN"` (entity validation via POS)
8. `question_words` set → `PronType=Int` morph feature + PTB tag fallback (WP/WDT/WRB)
9. `DISCOURSE_DEIXIS_VERBS` inline tuple → `ABSTRACT_VERB_SENSE` set + syntactic check (Jauhar et al. 2015)
10. `"quality"` relation → `"has_property"` (UD 'acomp' dependency, standard naming)

**How it works:**
```python
# In td/perception/nl_parser.py
@property
def nlp(self):
    """Lazy-load spaCy pipeline."""
    if self._nlp is None:
        try:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
        except (ImportError, OSError):
            self._nlp = False
    return self._nlp if self._nlp is not False else None

# Triple extraction via dependency parsing
def extract_triples_spacy(self, text: str) -> list[tuple[str, str, str]]:
    doc = self.nlp(text)
    # 1. Copular: "X is in Y" → (x, in, y)
    # 2. Verb: "X evolved from Y" → (x, evolved_from, y)
    # 3. Noun chunk: "X Y Z" → (x, y_z, z)
```

**Key design:** spaCy is the primary extraction engine. Regex patterns exist only as a last resort when spaCy is not installed. The system always tries spaCy first — it's not optional, it's the default.

**Performance:** spaCy is fast (~10K words/sec on CPU). Safe to use in teach() and query() paths.

**Reference:** Honnibal & Montani (2017), "spaCy 2: Natural language understanding with Bloom embeddings."

---

### ⚠️ Known Caveats & Multilingual Fallback Guide

This section documents every place where the code falls back to English-specific logic, and what to do for other languages.

#### Caveat 1: Question Detection — PTB Tag Fallback

**What happens:** spaCy's `en_core_web_sm` does NOT populate the UD morphological feature `PronType=Int` for interrogative words. The code falls back to Penn Treebank tags (`WP`, `WDT`, `WRB`).

**Why:** spaCy's English AttributeRuler does NOT set `PronType=Int` for interrogatives (who/what/where/which). It only sets:
- `PronType=Prs` for personal pronouns (I, you, he, she...)
- `PronType=Dem` for demonstratives (this, that, these, those...)
- `PronType=Art` for articles (a, an, the)
- `PronType=Ind` for indefinites (something, anyone...)
- `PronType=Rel` for relative "that"

The PTB tags (`WP`=who/what, `WDT`=which/that, `WRB`=where/when/how) are set by the tagger and ARE reliable for English.

**Source:** spaCy Discussion #11354 — "English and many other languages simply don't have those [morphological features] in their training data."

**Impact:** The question detection is English-specific via PTB. For other languages:
- **German:** `de_core_news_sm` DOES populate `PronType=Int` → UD path works
- **French:** `fr_core_news_sm` DOES populate `PronType=Int` → UD path works
- **Korean:** `ko_core_news_sm` sets `PronType=Int` → UD path works

---

##### 🔧 How to Extend Question Detection for a New Language

**Step 1: Check if PronType=Int is populated**

```python
import spacy

# Load the target language model
nlp = spacy.load("xx_core_news_sm")  # replace xx with language code

# Test with a question in the target language
# German: "Wer ist der Präsident?"
# French: "Qui est le président ?"
# Korean: "대통령은 누구입니까?"
doc = nlp("Who is the president?")  # use target language

for t in doc:
    pron_type = t.morph.get("PronType")
    print(f"{t.text:15s} pos={t.pos_:6s} tag={t.tag_:6s} PronType={pron_type}")
```

**Step 2a: If PronType=Int IS populated → UD path works automatically**

No code changes needed. The `_is_interrogative()` method checks `t.morph.get("PronType") == "Int"` first.

Languages where PronType=Int is populated:
- German (`de_core_news_sm`) — wer, was, wo, welcher, wann, wie, warum
- French (`fr_core_news_sm`) — qui, que, où, quel, quand, comment, pourquoi
- Korean (`ko_core_news_sm`) — 누구(nugu), 무엇(mueot), 어디(eodi)
- Spanish (`es_core_news_sm`) — quién, qué, dónde, cuándo, cómo, por qué
- Italian (`it_core_news_sm`) — chi, che, dove, quando, come, perché
- Portuguese (`pt_core_news_sm`) — quem, que, onde, quando, como, por que
- Dutch (`nl_core_news_sm`) — wie, wat, waar, wanneer, hoe, waarom

**Step 2b: If PronType=Int is NOT populated → add language-specific tags**

In `td/thinking.py`, the `_is_interrogative()` method falls back to `t.tag_ in ("WP", "WDT", "WRB")`. For other languages, add the equivalent POS tags:

```python
# In td/thinking.py, _is_interrogative() method:

# English (default)
if t.tag_ in ("WP", "WDT", "WRB"):
    return True

# German: add these tags
# PW  = interrogative pronoun (wer/was/welcher)
# PWS = interrogative adverb (wo/wann/wie/warum)
# PAV = interrogative pronominal adverb (wovon/womit)
if t.tag_ in ("PW", "PWS", "PAV"):
    return True

# French: add these tags
# PRO = pronoun (qui/que/où)
# DET:int = interrogative determiner (quel)
if t.tag_ in ("PRO", "DET:int"):
    return True

# Korean: PronType=Int is populated, no fallback needed
```

**Step 3: Register the language (optional)**

For cleaner code, register language-specific tag sets:

```python
# In td/thinking.py:
INTERROGATIVE_TAGS = {
    "en": {"WP", "WDT", "WRB"},           # Penn Treebank
    "de": {"PW", "PWS", "PAV"},            # Stuttgart-Tübingen Tagset (STTS)
    "fr": {"PRO", "DET:int"},              # French Treebank
    # Add more languages as needed
}

def _is_interrogative(self, text: str) -> bool:
    if not self.parser.nlp:
        return text.rstrip().endswith("?")
    doc = self.parser.nlp(text)
    lang = doc.lang_  # spaCy language code
    for t in doc:
        if t.morph.get("PronType") == "Int":
            return True
        if t.tag_ in self.INTERROGATIVE_TAGS.get(lang, set()):
            return True
    return False
```

**Where the code is:** `td/thinking.py` — `_is_interrogative()` method

**Reference:** Universal POS Tags (Nivre et al., 2016); Penn Treebank (Santorini, 1990); STTS (Schiller et al., 1999)

---

#### Caveat 2: Entity Validation — PROPN/NOUN POS Tags

**What happens:** Entity validation uses `token.pos_ == "PROPN"` (proper noun) to identify entities. This is a Universal POS tag and works across all languages.

**Why it works:** Universal POS tags are language-agnostic by design. `PROPN` = proper noun in English, German, French, Korean, Turkish, etc.

**Impact:** ✅ No English fallback needed. Works for any language with spaCy support.

**What to do for a new language:** Nothing — UD PROPN is universal. Just ensure the language model is installed.

---

#### Caveat 3: Preposition Detection — ADP POS Tag

**What happens:** The main path uses `token.pos_ == "ADP"` (adposition) for preposition detection. The regex fallback path uses a hardcoded English set.

**Why:** `ADP` is a Universal POS tag. It covers prepositions (English), postpositions (Japanese, Korean, Turkish), and circumpositions (German).

**Impact:** ✅ Main path is language-agnostic. ⚠️ Regex fallback is English-only.

**What to do for a new language:** The spaCy path handles it automatically. If spaCy is not available, add language-specific prepositions to the `_pp_words` fallback set:
```python
# German: _pp_words |= {"auf", "aus", "bei", "mit", "nach", "seit", "von", "zu"}
# French: _pp_words |= {"à", "de", "dans", "par", "pour", "sur", "avec"}
# Turkish: postpositions, not prepositions — ADP tag handles this
```

**Where the code is:** `td/thinking.py` — `_strip_pp()` function in regex fallback

---

#### Caveat 4: Discourse Deixis — Abstract Verb Sense Set

**What happens:** The `ABSTRACT_VERB_SENSE` set contains English verb lemmas (show, prove, mean, suggest, ...). The syntactic check (`nsubj` + head.lemma_) is language-agnostic, but the verb set is English.

**Why:** The Jauhar et al. (2015) approach is language-agnostic in principle (syntactic role + head verb), but the verb lemmas are language-specific.

**Impact:** ⚠️ English-only verb set. Other languages need their own abstract verb lemmas.

---

##### 🔧 How to Extend Discourse Deixis for a New Language

The Jauhar et al. (2015) two-stage approach has two components:

1. **Syntactic check** (language-agnostic): Is the pronoun `nsubj` of a verb?
2. **Semantic check** (language-specific): Is the verb an abstract/cognitive verb?

The syntactic check works for any language with spaCy dependency parsing. The semantic check needs language-specific verb lemmas.

**Step 1: Understand the semantic classes**

The `ABSTRACT_VERB_SENSE` set contains three semantic classes of verbs:

| Class | English Examples | What They Signal |
|-------|-----------------|------------------|
| **Demonstration/Implication** | show, prove, mean, suggest, indicate, demonstrate, reveal, confirm, imply | "This shows that X" → "this" refers to a clause |
| **Causation/Result** | result, lead, cause, enable, allow, prevent, require, involve, affect | "This causes X" → "this" refers to a clause |
| **Emotional Response** | surprise, shock, please, anger, upset, annoy | "This surprises me" → "this" refers to a clause |

**Step 2: Identify equivalent verbs in the target language**

For each semantic class, find the equivalent verbs in the target language. Use a bilingual dictionary or ask a native speaker.

**German (de):**
```python
ABSTRACT_VERB_SENSE_DE = {
    # Demonstration/Implication
    "zeigen", "beweisen", "bedeuten", "andeuten", "enthüllen",
    "bestätigen", "implizieren", "veranschaulichen", "widerspiegeln",
    # Causation/Result
    "ergeben", "führen", "verursachen", "ermöglichen", "erlauben",
    "verhindern", "erfordern", "betreffen", "beeinflussen",
    # Emotional Response
    "überraschen", "schockieren", "erfreuen", "ärgern", "aufregen",
}
```

**French (fr):**
```python
ABSTRACT_VERB_SENSE_FR = {
    # Demonstration/Implication
    "montrer", "prouver", "signifier", "suggérer", "indiquer",
    "démontrer", "révéler", "confirmer", "impliquer", "illustrer",
    # Causation/Result
    "résulter", "entraîner", "causer", "permettre", "autoriser",
    "empêcher", "exiger", "concerner", "influencer",
    # Emotional Response
    "surprendre", "choquer", "plaire", "irriter", " contrarier",
}
```

**Korean (ko):**
```python
ABSTRACT_VERB_SENSE_KO = {
    # Demonstration/Implication
    "보여주다", "증명하다", "의미하다", "암시하다", "나타내다",
    "확인하다", "함축하다", "반영하다",
    # Causation/Result
    "초래하다", "야기하다", "가능하게하다", "허용하다", "막다",
    "요구하다", "관련하다", "영향을미치다",
    # Emotional Response
    "놀라게하다", "충격을주다", "기쁘게하다", "화나게하다",
}
```

**Step 3: Register the language in the parser**

```python
# In td/perception/nl_parser.py, use the registry method:

# Option A: Call the class method (recommended)
GenericNLParser.register_discourse_deixis(
    lang="de",
    verbs={"zeigen", "beweisen", "bedeuten", "andeuten", "enthüllen",
           "bestätigen", "implizieren", "veranschaulichen", "widerspiegeln",
           "ergeben", "führen", "verursachen", "ermöglichen", "erlauben",
           "verhindern", "erfordern", "betreffen", "beeinflussen",
           "überraschen", "schockieren", "erfreuen", "ärgern", "aufregen"},
    it_verbs={"zeigen", "beweisen", "bedeuten", "andeuten", "enthüllen",
              "bestätigen", "implizieren", "veranschaulichen", "widerspiegeln"},
)

# Option B: Add to DISCOURSE_DEIXIS_REGISTRY dict directly
GenericNLParser.DISCOURSE_DEIXIS_REGISTRY["fr"] = frozenset({
    "montrer", "prouver", "signifier", "suggérer", "indiquer",
    "démontrer", "révéler", "confirmer", "impliquer", "illustrer",
    "résulter", "entraîner", "causer", "permettre", "autoriser",
    "empêcher", "exiger", "concerner", "influencer",
    "surprendre", "choquer", "plaire", "irriter",
})
```

**Step 4: Test with language-specific examples**

```python
# German: "Das zeigt, dass die Methode funktioniert."
# → "Das zeigt" should be filtered (discourse deixis)

# French: "Cela montre que la méthode fonctionne."
# → "Cela montre" should be filtered

# Korean: "이것은 방법이 작동한다는 것을 보여줍니다."
# → "이것은 보여줍니다" should be filtered
```

**Where the code is:** `td/perception/nl_parser.py` — `ABSTRACT_VERB_SENSE` frozenset

**Reference:** Jauhar, S.K. et al. (2015). "Resolving Discourse-Deictic Pronouns: A Two-Stage Approach." *SEM 2015, pp. 299–308. ACL Anthology: S15-1035

---

#### Caveat 5: Adjectival Predicate Relation — "has_characteristic" (Wikidata P1552)

**What happens:** Adjectival predicates ("The man is friendly") are extracted as `(subject, has_characteristic, adjective)`. The relation name maps to Wikidata P1552 "has characteristic" — "inherent or distinguishing quality or feature of the entity."

**Why:** Wikidata P1552 is the standard property for entity attributes/qualities. It's used as a qualifier on 259+ items in Wikidata.

**Impact:** ✅ Now aligned with Wikidata standard. Interoperable with Wikidata data.

**Reference:** Wikidata P1552 — https://www.wikidata.org/wiki/Property:P1552

**Where the code is:** `td/perception/nl_parser.py` — Step 6 in `extract_triples_spacy()`

---

#### Caveat 6: Open Query Ranking — TF-IDF (Salton & Buckley, 1988)

**What happens:** SPARQL open query results are ranked using TF-IDF scoring:
- **IDF (Inverse Document Frequency):** Rarer relations are more specific/informative
- **Query match bonus:** Relations that appear in the query text are preferred
- **Forward preference:** Entity-as-subject results are preferred over entity-as-object

**Why:** TF-IDF is the standard information retrieval ranking technique, battle-tested since 1988. It adapts naturally to KG relations: rare relations (like `capital_of`) are more informative than common ones (like `in`).

**Impact:** ✅ Research-backed ranking. Better than relation name length heuristic.

**Reference:** Salton, G. & Buckley, C. (1988). "Term-weighting approaches in automatic text retrieval." *Information Processing & Management*, 24(5): 513–523.

**Where the code is:** `td/thinking.py` — open query section in `_query_knowledge_graph()`

---

### Prepositional Phrase Attachment — "of" Hardcoding

The parser treats "of" differently from other prepositions when building entity names from noun chunks. This is a deliberate design decision.

**The rule:** "of" is entity-internal (genitive/possessive), always included in entity names. Other prepositions (in, on, at, before, after) are relations, only included when the pobj has compound modifiers.

**Examples:**
```
"united states of america" → entity: "united states of america"  (of = entity-internal)
"France in EU"             → entity: "France", relation: "in"    (in = relation)
"North America of Mexico"  → entity: "North America"             (of = entity-internal)
"Senate in Rome"           → entity: "Senate"                    (in = relation)
```

**Why "of" is special:**
- "of" is almost always a genitive marker in English ("capital of France", "president of the company", "state of mind")
- Spatial/temporal prepositions (in, on, at, before, after) are almost always relations
- This is a heuristic, not a linguistic rule — it works for 95%+ of cases

**Where it's implemented:**
- `_get_chunk_text()`: prep chain walking with "of" special case
- `_token_text()` in `_get_coordinated_subjects()`: same logic

**Known limitation:** "member of parliament in London" → "member of parliament" (correct, "of" is entity-internal) but "parliament in London" would incorrectly include "in London" if "London" has compound modifiers. The `pobj_compounds` guard handles this for non-"of" preps.

**Reference:** Manning & Schütze (1999), "Foundations of Statistical NLP", Chapter 5: Collocations. Genitive markers vs spatial prepositions.

### Passive Voice Extraction

**Pattern:** "France is known for wine" → (wine, known_for, france)
**Detection:** spaCy `nsubjpass` + `agent` dependency labels
**Swap logic:** When `nsubjpass` detected AND `agent` (`by`-phrase) present, swap subject and object.

**Reference:** TEA Nets (arXiv, Apr 2026) — `is_passive` and `passive_approx` flags. Uses spaCy `nsubjpass` and `agent` dependency labels.
**Reference:** Analytics Vidhya (2024) — `dep_.find("subjpass")` for passive detection in spaCy.

### Negation Detection

**Pattern:** "Tokyo is not in Europe" → (tokyo, NOT_in, europe)
**Detection:** spaCy `neg` dependency token attached to verb. Prefixes relation with `NOT_`.

### Relative Clause Attachment

**Pattern:** "Paris which is the capital of France is beautiful" → resolves "which" to "Paris"
**Detection:** spaCy `acl:relcl` dependency. `relcl.head` = antecedent.
**Resolution:** Relative pronouns (`which`, `who`, `that`) resolved to the noun the clause modifies.

**Reference:** Universal Dependencies — `acl:relcl` dependency label. "The relativizer can be understood as an anaphor whose antecedent is the head of the relative clause."
**Reference:** de Marneffe et al. (2014), "Universal Stanford Dependencies"

### Triple Deduplication via Relation Canonicalization (Option B)

**Problem:** Two extraction paths (clause segmenter + dependency extraction) produce duplicate triples with different relation names for the same fact:
- Dependency: `(alice, went_to, paris)` — verb+prep compound
- Clause segmenter: `(alice, went, paris)` — bare verb

**Solution:** Post-extraction canonicalization (Option B from EDC framework).

**Why Option B over Option A (constrained extraction):**
- Option A discards one path's output entirely — loses information
- Option B keeps both paths' best output, canonicalizes, then deduplicates
- Option B is a pure function (string in → string out) — trivially testable
- Stanford OpenIE uses Option A, but it has ONE extraction path. TD v2 has TWO complementary paths for different reasons.

**Implementation:**
```python
# td/perception/relation_canonicalizer.py
PREPOSITION_SUFFIXES = {"_to", "_in", "_at", "_for", "_with", "_from", "_on", ...}

def canonicalize_relation(relation: str) -> str:
    for suffix in PREPOSITION_SUFFIXES:
        if relation.endswith(suffix):
            verb_part = relation[:-len(suffix)]
            return lemmatize_verb(verb_part)  # spaCy lemma
    return lemmatize_verb(relation)
```

**Integration point:** Between extraction and deduplication in `extract_triples_spacy()`.

**Specificity heuristic:** When duplicates found post-canonicalization, keep the more specific relation (`went_to` > `went`).

**References:**
- Zhang & Soh (2024), "Extract, Define, Canonicalize" — arXiv:2404.03868
- UDASTE (ScienceDirect, 2023) — "restrictive triple relation types"
- KGGen (arXiv, Feb 2025) — "variations in tense, plurality, stemming normalized"
- Stanford OpenIE (2015) — clause splitting + forward entailment pipeline

### Temporal Ordering from Discourse Connectives ("then", "after", "before")

**The Problem:**
"Alice went to Paris and then invested in stocks" implies temporal ordering (Paris BEFORE stocks). "Alice went to Paris and invested in stocks" does not. Currently TD v2 extracts the same triples from both sentences — the "then" is lost.

**What Should Happen:**
The "then" implies a `before` relation between events:
- Event 1: Alice went to Paris
- Event 2: Alice invested in stocks
- Temporal: Event 1 BEFORE Event 2

**Research:**
- **Allen's Interval Algebra (1983)** — already implemented in TD v2. 13 temporal relations (before, after, meets, overlaps, etc.)
- **TimeML (Pustejovsky et al., 2003)** — foundational standard for temporal annotation. Defines temporal connectives ("then", "after", "before") as explicit markup. Used in TimeBank corpus.
- **PDTB 3.0 (Webber et al., 2019)** — Penn Discourse Treebank. Comprehensive discourse relation annotation. Temporal connective classification: TEMPORAL.Asynchronous (before/after) and TEMPORAL.Synchrony (while/meanwhile).
- **CICLING — "On the Identification of Temporal Clauses"** — Comprehensive list of English temporal subordinating connectives: after, as, as/so long as, as soon as, before, once, since, until/till, when, while.
- **Chambers et al.** — "Unsupervised learning of temporal orderings of events" — extracts temporal orderings from text using unsupervised learning.
- **Consistent Discourse-level TRE (EMNLP 2025)** — uses Allen's interval algebra for temporal relation extraction at discourse level. Self-reflection step for consistency.
- **ATOMIC-2020** — common sense knowledge dataset with `isBefore` and `isAfter` relations for event ordering.
- **ChronoSense (arXiv, Jan 2025)** — "Exploring Temporal Understanding in LLMs with Time Intervals of Events"
- **Event Knowledge Graphs (arXiv, Oct 2023)** — "On the Evolution of Knowledge Graphs: A Survey and Perspective". Events as first-class entities with temporal relations.

**Current Limitation:**
The parser extracts `(alice, went_to, paris)` and `(alice, invested_in, stocks)` but does NOT create `(alice_went_to_paris, before, alice_invested_in_stocks)`. The temporal ordering from "then" is not captured.

**TODO:**
1. Detect discourse temporal connectives: "then", "after", "before", "subsequently", "next"
2. When found between coordinated clauses, add a temporal ordering triple
3. Use Allen's `before` relation: Event1 BEFORE Event2
4. Reference: Allen (1983), "Maintaining Knowledge about Temporal Intervals"

**References:**
- Allen, J.F. (1983). "Maintaining Knowledge about Temporal Intervals." CACM, 26(11): 832–843.
- Pustejovsky, J. et al. (2003). "TimeML: Robust Specification of Event and Temporal Expressions in Text." IWCS-5.
- Webber, B. et al. (2019). "Penn Discourse Treebank 3.0 Annotation Manual." LDC.
- Schilder, F. "On the Identification of Temporal Clauses." CICLING.
- Chambers, N. et al. "Unsupervised Learning of Temporal Orderings for Events."
- Consistent Discourse-level TRE (EMNLP 2025). URL: https://aclanthology.org/2025.findings-emnlp.1010.pdf
- ATOMIC-2020. "ATOMIC 2020: On Symbolic and Neural Commonsense Knowledge Graphs."
- ChronoSense (arXiv, Jan 2025). "Exploring Temporal Understanding in LLMs with Time Intervals of Events." URL: https://arxiv.org/abs/2501.03040
- On the Evolution of Knowledge Graphs (arXiv, Oct 2023). Event Knowledge Graphs with temporal entities. URL: https://arxiv.org/abs/2310.04835

### Multi-Hop Open Queries

Open queries follow BFS paths to find answers at any hop count:

```python
# 1-hop: direct fact
kg.query("iphone", "made_by")  # → "apple"

# 2-hop: follow path
kg.query("iphone", "founded_by")  # → "steve jobs" (via apple)

# 5-hop: full chain
kg.query("iphone", "in")  # → "north america" (via apple→jobs→sf→usa)
```

**max_hops:** Configurable. BFS terminates early when the shortest path is found.

**Confidence:** Computed from chain quality, not hop count. See "Confidence Scoring" section below.

### Confidence Scoring

Confidence is computed from chain quality and error propagation, not hop count.

**Formula:**
```python
def _chain_confidence(self, path, target_relation):
    chain_score = 1.0
    for each step in path:
        if explicit_rule: step_score = 1.0
        elif transitive_fallback: step_score = 0.7
        else: step_score = 0.4
        chain_score *= step_score  # error propagation
    return clamp(chain_score, 0.1, 0.95)
```

**Why not hop count?**
- Research shows path quality matters, not length (CPR, arXiv 2026)
- Error accumulates multiplicatively, not additively (UaG, AAAI 2025)
- A 5-hop chain with all explicit rules has the same confidence as a 2-hop chain

**How to add calibration data (future):**
When enough queries accumulate, use Conformal Prediction (CP) to calibrate:
1. Collect query → answer → correctness triples
2. Compute nonconformity scores
3. Use CP to produce prediction intervals with statistical guarantees

**References:**
- Wang, Y. et al. (2026). "Conformal Path Reasoning: Trustworthy Knowledge Graph Question Answering with Statistical Coverage Guarantees." arXiv:2605.08077.
- Ni, J. et al. (2025). "Towards Trustworthy Knowledge Graph Reasoning: An Uncertainty Aware Perspective." AAAI 2025. arXiv:2410.08985.
- Zhu, Y. et al. (2025). "Certainty in Uncertainty: Reasoning over Uncertain Knowledge Graphs with Conformal Prediction." arXiv:2510.24754.
- Rawat, R. et al. (2026). "Information retrieval framework using knowledge graph embeddings and uncertainty modelling using probabilistic soft logic." Discover Computing, Vol. 29. Springer.

### Future: Confidence Calibration Plan

**Current state:** Confidence is computed from chain quality (heuristic). Not calibrated.

**Existing infrastructure:** `demos/chat_flare.py` already has a feedback system:
```
Was this helpful?
  [y] Yes  [n] No  [t] Teach me the right answer
```

**What to implement:**

1. **Store feedback in SQLite:**
```sql
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    answer TEXT,
    raw_confidence FLOAT,
    correct BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

2. **Wire feedback buttons to storage:**
```python
# In chat_flare.py, when user clicks [y]:
kg.save_feedback(query, answer, confidence, correct=True)
# When user clicks [n]:
kg.save_feedback(query, answer, confidence, correct=False)
```

3. **Calibrate with Conformal Prediction:**
```python
# After 100+ queries:
calibrated = kg.calibrate_confidence()
# Returns: {raw_conf: calibrated_conf, ...}
# Example: {0.20: 0.85, 0.40: 0.92, 0.95: 0.97}
```

**References:**
- Vovk, V., Gammerman, A., & Shafer, G. (2005). "Algorithmic Learning in a Random World." Springer.
- Angelopoulos, A.N. & Bates, S. (2022). "A Gentle Introduction to Conformal Prediction." arXiv:2107.07511.
- Shafer, G. & Vovk, V. (2008). "A Tutorial on Conformal Prediction." JMLR, 9, 371-421.
