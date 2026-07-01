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

**Option A: Pre-seeded defaults (already handled)**

Common relations work out of the box:
- `in`, `part_of`, `before`, `after` → transitive
- `capital_of` → functional
- `same_as`, `equals` → symmetric + transitive
- `married_to`, `sibling_of` → symmetric

No user action needed. TD knows these from `DEFAULT_RELATION_PROPERTIES`.

**Option B: Interactive prompt (demo only)**

When you teach a fact with an unknown relation in `chat_flare.py`:
```
teach: Kazakhstan is north of Uzbekistan
→ TD asks: Is 'north_of' [1] Transitive [2] Symmetric [3] Functional [4] Skip
→ User picks 1
→ TD: Got it. 'north_of' is now transitive.
```

**Option C: Manual via demo command**
```
relation: north_of transitive
relation: married_to symmetric
relation: capital_of functional inverse:has_capital
```

**Option D: In code**
```python
td.teach_relation("north_of", "transitive")
td.teach_relation("married_to", "symmetric")
td.kg.set_relation_property("capital_of", "functional", "inverse:has_capital")
```

### Available relation properties

| Property | Rule | Example |
|----------|------|---------|
| `transitive` | R(X,Y) ∧ R(Y,Z) → R(X,Z) | in, before, north_of |
| `symmetric` | R(X,Y) → R(Y,X) | same_as, married_to |
| `functional` | R(X,Y) ∧ R(X,Z) → Y=Z | capital_of, birth_date |
| `inverse:R2` | R1(X,Y) → R2(Y,X) | capital_of ↔ has_capital |

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

---

## Architecture Vision

### The Two Minds

TD has two minds, like humans do:

| | TD v2 (System 1) | TD Pro (System 2) |
|---|---|---|
| Role | Fast reasoning engine | Slow creative problem solver |
| Speed | <100ms | 1-5 seconds |
| Handles | Facts, constraints, inference, retrieval | Spatial, temporal, novel problems |
| Components | HDC + MHN + Z3 + BEAGLE + KG | Liquid-KAN + Hypernetwork + NCA + TTT |
| Status | **Shipped** | Planned |

### The Killer Insight: TD Pro Trains TD v2

```
User poses novel problem
    ↓
TD v2: "I don't know how to solve this."
    ↓
TD Pro: Liquid-KAN + NCA → solves it (1-5 seconds)
    ↓
Solution stored in TD v2's MHN
    ↓
Next time: TD v2 retrieves solution instantly (<100ms)
```

**The slow mind teaches the fast mind.** TD Pro is the "research department" — it handles edge cases, novel problems, spatial reasoning. Every solution it finds gets stored in TD v2's memory. Over time, TD v2 handles more problems without invoking TD Pro.

This is how human expertise works: you struggle through a problem once, then it becomes intuition.

### Teaching Channels

All teaching flows into the same knowledge graph. The architecture doesn't change — the input channel does.

| Channel | Source | How | Status |
|---------|--------|-----|--------|
| **Text** | User types facts | `teach()` → triples | ✅ Shipped |
| **Bulk** | Wikidata dump | Load structured triples into KG | Planned |
| **Real-time** | Internet search | Fetch → store → expire | Planned |
| **Visual** | Images | Perception → HDC vector → concept | Planned (TD Pro) |
| **Audio** | Speech | Phonetic → HDC → text → teach | Planned (TD Pro) |
| **Structured** | CSV/JSON | Table rows → triples | Planned |

**The teaching pipeline is the same for all channels:**
```
Input (any format)
    ↓
Parse → extract (subject, relation, object)
    ↓
KG.add_fact() → SQLite
    ↓
BEAGLE vectors updated (if text)
    ↓
Derive new facts (transitive, symmetric, etc.)
```

### Handoff Protocol

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

### What TD Does Better Than ChatGPT

1. **Exact constraint solving** — Z3 proves answers, doesn't guess
2. **Real-time teachability** — teach a fact, use it immediately
3. **Full interpretability** — proof traces for every answer
4. **Privacy** — runs locally, no data leaves your machine
5. **Cost** — free after setup (no API keys, no subscriptions)
6. **Honesty** — calibrated confidence, says "I don't know"

### What ChatGPT Does Better Than TD

1. **Broad world knowledge** — trained on entire internet
2. **Natural language generation** — writes essays, poems, stories
3. **Conversational fluency** — multi-turn context, emotional intelligence
4. **Handling ambiguity** — tolerates vague questions
5. **Creative tasks** — brainstorming, writing, design

**TD is not a ChatGPT replacement. It is a different tool for a different job.**

---

## Roadmap

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
- Specialized agents: TD-geo, TD-code, TD-schedule, TD-science
- Common HDC language for inter-agent communication
- Collaborative reasoning across domains

### Phase 5: TD as a Platform (Year 2)
- Users build custom TD personalities
- TD marketplace: share taught knowledge graphs
- TD for education: Socratic teaching mode
- TD for research: scientists teach domain knowledge
