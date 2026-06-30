# Thinking Dust v2 — Developer Guide

## Setup

### Prerequisites
- **Python 3.10+**
- **macOS or Linux** (tested on MacBook Pro x86_64 and arm64)

### Install

```bash
git clone https://github.com/kgrsajid/thinking-dust-v2.git
cd thinking-dust-v2

python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# Verify Z3 works
.venv/bin/python -c "from z3 import Solver; print('Z3 OK')"
```

### Run Tests

```bash
.venv/bin/python -m pytest tests/ -v
# Expected: 50 passed
```

### Run Demo

```bash
.venv/bin/python3 demos/chat_flare.py --pure
```

## Project Structure

```
td-v2/
├── td/
│   ├── perception/
│   │   ├── hdc.py            # HDC operations (bind, bundle, permute, similarity)
│   │   ├── nl_parser.py      # CA reservoir, entity extraction, relation prototypes
│   │   └── word_vectors.py   # BEAGLE word vector model
│   ├── memory/
│   │   └── mhn.py            # Modern Hopfield Network
│   ├── kg/
│   │   └── __init__.py       # Knowledge Graph + inference engine
│   ├── thinking.py           # Main reasoning pipeline (think, teach, route)
│   └── minimal_seed.py       # Optional seed patterns
├── demos/
│   └── chat_flare.py         # Interactive demo with reasoning trace
├── data/
│   ├── synthetic_corpus_10k.txt  # 10K training sentences for BEAGLE
│   └── word_vectors_10k.pkl      # Trained BEAGLE vectors (19MB)
├── tests/
│   └── test_thinking.py      # 50 unit tests
├── ARCHITECTURE.md           # Full system architecture documentation
└── README.md
```

## Training BEAGLE Word Vectors

Word vectors are pre-trained and included in `data/word_vectors_10k.pkl`.
To retrain from scratch:

```python
from td.perception.word_vectors import WordVectorModel

wvm = WordVectorModel(dim=10000)

with open("data/synthetic_corpus_10k.txt") as f:
    sentences = [line.strip() for line in f if line.strip()]

wvm.train(sentences)
wvm.save("data/word_vectors_10k.pkl")
```

Training takes ~1.4 seconds on CPU. No GPU needed.

## Extending the Knowledge Graph

### Adding new triple extraction patterns

Edit `_extract_triples()` in `td/thinking.py`:

```python
# Add new structural pattern
m = re.search(r'(\w+)\s+belongs\s+to\s+(\w+)', text)
if m:
    triples.append((m.group(1), "belongs_to", m.group(2)))
```

### Adding new relation properties

```python
# In your code:
td.kg.set_relation_property("north_of", "transitive")
td.kg.set_relation_property("married_to", "symmetric")
td.kg.set_relation_property("capital_of", "functional", "inverse:has_capital")
```

### Adding new rule templates

Edit `RULE_TEMPLATES` in `td/kg/__init__.py`:

```python
RULE_TEMPLATES["irreflexive"] = "→ ¬{R}(X,X)"
```

## Debugging

### Enable reasoning trace in demo:
```
trace
```

### Check KG state:
```python
print(td.kg.stats())
# {'total_triples': 7, 'user_facts': 5, 'derived_facts': 2, ...}
```

### Check word vector quality:
```python
print(td.wvm.similarity("capital", "city"))    # should be >0.15
print(td.wvm.similarity("capital", "sort"))    # should be <0.05
print(td.wvm.nearest_neighbors("paris", 5))
```

## Common Issues

**Z3 not found:** Run `.venv/bin/pip install z3-solver==4.12.1`

**Word vectors not loaded:** Ensure `data/word_vectors_10k.pkl` exists. Demo auto-loads if found.

**Paraphrase matching fails:** The BEAGLE corpus may not cover your vocabulary. Add domain sentences to the corpus and retrain.
