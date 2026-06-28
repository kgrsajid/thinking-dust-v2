# Thinking Dust v2 — Developer Guide

## Quick Start

### Prerequisites

- **Mac with Apple Silicon (M1/M2/M3)**
- **Python 3.10+** (universal binary from python.org)

### Setup

```bash
git clone git@github.com:kgrsajid/thinking-dust-v2.git
cd thinking-dust-v2

# Create venv (on M1, make sure Terminal is NOT Rosetta)
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# If your terminal is Rosetta:
/usr/bin/arch -arm64 /Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10 -m venv --copies .venv
lipo -remove x86_64 .venv/bin/python3.10 -output .venv/bin/python3.10

# Verify
.venv/bin/python -c "import platform; print(platform.machine())"  # → arm64
```

### Run Tests

```bash
.venv/bin/python -m pytest tests/ -v
# Rosetta shell: /usr/bin/arch -arm64 .venv/bin/python -m pytest tests/ -v
```

### Run Demos

**Reasoning demos** (showcase intelligence):
```bash
# Formal logic proofs via Z3
.venv/bin/python demos/demo_reasoning.py

# Memory capacity, noise robustness, IDP, online learning
.venv/bin/python demos/demo_memory_intelligence.py
```

**Application demos** (TD as agentic controller):
```bash
.venv/bin/python demos/demo_form_automation.py
.venv/bin/python demos/demo_monitor.py
```

---

## What Thinking Dust Actually Is

Thinking Dust is **not an agent**. It is a **reasoning engine** — a system that:

1. **Perceives** input (NL, DOM, API data, metrics) and encodes it as 10K-dim HDC vectors
2. **Remembers** past experiences via Modern Hopfield Networks with exponential capacity
3. **Classifies** what type of reasoning is needed via a ternary router cascade
4. **Validates** decisions formally using Z3 SMT solver (proofs, not approximations)
5. **Learns** from every interaction — corrections stored as new attractors in O(dim)

It **never executes side effects**. It thinks, proves, and decides. Execution is OpenClaw's job.

### Two Modes of Intelligence

| Mode | Mechanism | Speed | Example |
|------|-----------|-------|---------|
| **Pattern matching** (System 1) | MHN retrieval + router | <5ms | "I've seen this — do X" |
| **Formal reasoning** (System 2) | Z3 constraint solving | 1-100ms | "Proving this is valid" |

---

## Demonstrating Intelligence

### 1. Formal Logic Proof (Z3)

```python
from td.reasoning.z3_bridge import Z3Bridge

bridge = Z3Bridge()
result = bridge.validate_action(
    action_plan=[{"action": "click", "target": "submit"}],
    constraints={"submit_visible": True, "captcha_present": False},
)
# → Z3Result(status="sat") — formally valid
```

### 2. HDC Algebraic Inference

```python
from td.perception.hdc import *

vocab = build_default_vocabulary(dim=10000)

# Encode knowledge: bind(paris, bind(capital_of, france))
fact = bind(vocab.get("name"), bind(vocab.get("condition"), vocab.get("valid")))

# Query algebraically: fact ⊗ condition ≈ name
query = bind(fact, vocab.get("condition"))
# similarity(query, bind(name, valid)) > 0
```

### 3. Memory-Based Pattern Retrieval

```python
from td.memory.mhn import ModernHopfieldNetwork, MHNConfig

mhn = ModernHopfieldNetwork(MHNConfig(dim=10000))

# Store reasoning patterns as attractors
mhn.store(situation_vec, action_vec, {"pattern": "modus_ponens"})

# Retrieve relevant pattern with noise tolerance
results = mhn.retrieve(query_vec, top_k=3)  # Works even with 20% noise
```

### 4. Online Learning from Corrections

```python
# TD makes a decision, user corrects it
pipeline.learn_correction(
    situation_text="Extract product prices",
    wrong_actions=[{"action": "click", "target": "wrong"}],
    correct_actions=[{"action": "extract", "target": "price_table"}],
)
# → Old pattern deactivated, new stored. Zero forgetting. O(dim) time.
```

---

## Project Structure

```
td-v2/
├── td/
│   ├── perception/          # How TD sees the world
│   │   ├── hdc.py           # HDC algebra (bind, bundle, similarity)
│   │   ├── ca_reservoir.py  # Rule 90 feature extraction
│   │   ├── nl_parser.py     # NL → HDC encoding
│   │   ├── dom_encoder.py   # DOM → HDC
│   │   ├── api_encoder.py   # JSON → HDC
│   │   └── metrics_encoder.py
│   ├── memory/              # How TD remembers
│   │   ├── mhn.py           # Modern Hopfield Network + IDP
│   │   └── attractor_store.py
│   ├── routing/             # How TD classifies
│   │   ├── ternary_linear.py  # BitNet b1.58 weights
│   │   ├── router_a.py      # Domain (5 classes)
│   │   ├── router_b.py      # Task type (4 per domain)
│   │   ├── router_c.py      # Strategy (3 classes)
│   │   └── hierarchical_router.py
│   ├── reasoning/           # How TD thinks
│   │   ├── z3_bridge.py     # Formal proofs
│   │   ├── constraint_schemas.py
│   │   └── confidence.py    # Multi-factor confidence
│   ├── learning/            # How TD learns
│   │   └── online.py        # O(dim) correction learning
│   └── pipeline.py          # Full orchestration
├── tests/                   # 98 tests
├── demos/                   # Reasoning + application demos
└── docs/
```

---

## Key Numbers

| Metric | Value |
|--------|-------|
| Total trainable params | ~5,000 (router only) |
| HDC vector dimension | 10,000 |
| MHN capacity | ~0.14 × D × e^D patterns |
| Router inference | <0.1ms |
| MHN retrieval (1K patterns) | <2ms |
| Z3 solve (simple constraint) | 1-10ms |
| Full pipeline | <5ms (no Z3), <50ms (with Z3) |
| Memory footprint | ~7.3MB |
| Learning time per correction | O(dim) ≈ instant |

---

## Common Issues

**"libz3.dylib not found"** — Shell is Rosetta. Run with `arch -arm64`.

**Router gives 1.0 confidence** — Router is untrained. Run `pipeline.train_routers()`.

**MHN returns empty** — Lower `min_similarity` in MHNConfig (default 0.3).

---

*"Build the dust. Let it think."*
