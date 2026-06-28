# Thinking Dust v2 — Developer Guide

## Quick Start

### Prerequisites

- **Mac with Apple Silicon (M1/M2/M3)** — runs natively on ARM
- **macOS Terminal** — make sure "Open using Rosetta" is **unchecked** (Right-click Terminal app → Get Info → uncheck Rosetta)
- **Python 3.10+** (universal binary from python.org) or 3.12+ (homebrew ARM)

### Setup

```bash
# Clone
git clone git@github.com:kgrsajid/thinking-dust-v2.git
cd thinking-dust-v2

# Create ARM venv (IMPORTANT for M1 Macs migrated from Intel)
# If your Terminal is NOT Rosetta, standard venv works:
python3 -m venv .venv

# If your Terminal IS Rosetta (check with `arch` → if says i386):
/usr/bin/arch -arm64 /Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10 -m venv --copies .venv
lipo -remove x86_64 .venv/bin/python3.10 -output .venv/bin/python3.10
# (This strips x86_64 so it can only run ARM)

# Install deps
.venv/bin/pip install -e ".[dev]"

# Verify ARM
.venv/bin/python -c "import platform; print(platform.machine())"
# Should print: arm64
```

### Running Tests

```bash
# If your Terminal runs natively (not Rosetta):
.venv/bin/python -m pytest tests/ -v

# If your Terminal is Rosetta:
/usr/bin/arch -arm64 .venv/bin/python -m pytest tests/ -v

# Expected: 70 passed, 0 failed
```

### Running Demos

```bash
# Form automation demo
.venv/bin/python demos/demo_form_automation.py

# System monitoring demo
.venv/bin/python demos/demo_monitor.py

# API workflow demo
.venv/bin/python demos/demo_api_workflow.py

# File processing demo
.venv/bin/python demos/demo_file_parse.py
```

---

## Project Structure

```
thinking-dust-v2/
├── td/                         # Main package
│   ├── __init__.py
│   ├── pipeline.py             # Main orchestrator — TDPipeline
│   │
│   ├── perception/             # Input encoding layer
│   │   ├── hdc.py              # HDC core: bind, bundle, permute, similarity
│   │   ├── ca_reservoir.py     # Rule 90 cellular automata
│   │   ├── nl_parser.py        # Natural language → HDC (keyword-based)
│   │   ├── dom_encoder.py      # Web DOM → HDC
│   │   ├── api_encoder.py      # JSON/API → HDC
│   │   └── metrics_encoder.py  # System metrics → HDC
│   │
│   ├── memory/                 # Associative memory
│   │   ├── mhn.py              # Modern Hopfield Network with IDP
│   │   └── attractor_store.py  # Pattern lifecycle management
│   │
│   ├── routing/                # Task classification
│   │   ├── ternary_linear.py   # BitNet b1.58 ternary layer
│   │   ├── router_a.py         # Domain detector (5 classes)
│   │   ├── router_b.py         # Task type detector (4 per domain)
│   │   ├── router_c.py         # Strategy selector (3 classes)
│   │   ├── hierarchical_router.py  # Full 3-level cascade
│   │   └── router_train.py     # Training utilities + synthetic data
│   │
│   ├── reasoning/              # Formal reasoning
│   │   ├── z3_bridge.py        # HDC ↔ Z3 constraint validation
│   │   ├── constraint_schemas.py  # Predefined constraints per domain
│   │   └── confidence.py       # Multi-factor confidence scoring
│   │
│   ├── learning/               # Online learning
│   │   └── online.py           # Outcome + correction learning
│   │
│   └── utils/                  # Utilities
│       ├── io.py               # Save/load state
│       └── viz.py              # Energy landscape visualization
│
├── tests/                      # Test suite (70 tests)
├── demos/                      # Example applications
├── data/                       # Training data, Z3 templates
├── pyproject.toml
├── requirements.txt
├── CODE_REVIEW.md
└── README.md
```

---

## Architecture Overview

```
User Input (NL, DOM, API, Metrics)
         │
         ▼
┌─────────────────────────────────┐
│  PERCEPTION LAYER               │
│  ┌─────────┐  ┌──────────────┐  │
│  │ CA Res  │  │ HDC Encoder  │  │
│  │(Rule 90)│  │(10K bipolar) │  │
│  └────┬────┘  └──────┬───────┘  │
│       └──────┬──────┘          │
│              ▼                  │
│     [Unified HDC Vector]        │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  MEMORY LAYER                   │
│  MHN.retrieve(hdc_vector)       │
│  → past patterns + similarities │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  ROUTING LAYER (3-level cascade)│
│  RouterA: Domain (5 classes)    │
│  RouterB: Task Type (4 classes) │
│  RouterC: Strategy (3 classes)  │
│  Total: ~5K ternary params      │
└──────────────┬──────────────────┘
               │
       ┌───────┼───────┐
       ▼       ▼       ▼
   MEMORY   VALIDATE  ESCALATE
   _ONLY    _THEN_     (to TD Pro)
            VALIDATE
       │       │
       │       ▼
       │  Z3 SMT Solver
       │  (formal proof)
       │       │
       └───────┘
               │
               ▼
┌─────────────────────────────────┐
│  CONFIDENCE SCORING             │
│  router × MHN × Z3 → score     │
│  >0.9: execute                  │
│  0.7-0.9: confirm               │
│  <0.7: escalate                 │
└──────────────┬──────────────────┘
               │
               ▼
        Action Plan
        (to OpenClaw)
```

---

## Using the Pipeline

### Basic Usage

```python
from td.pipeline import TDPipeline

# Initialize (builds default vocabulary, empty memory, untrained router)
pipeline = TDPipeline()

# Train routers on synthetic data (~60 examples, 30 seconds)
pipeline.train_routers(epochs=50)

# Save trained state
pipeline.save_state("checkpoints/v1/")

# Make a decision
decision = pipeline.decide("Click the submit button on the login form")

# Check confidence
if decision.should_execute:
    print("Auto-executing:", decision.action_plan)
elif decision.needs_confirmation:
    print("Needs user confirmation")
    print("Confidence:", decision.confidence.combined)
elif decision.should_escalate:
    print("Escalating to TD Pro")
```

### Learning from Outcomes

```python
# After executing an action plan
pipeline.learn(
    situation_text="Fill out contact form with name and email",
    action_plan=[
        {"action": "click", "target": "name_field"},
        {"action": "type", "target": "name_field", "value": "Alice"},
        {"action": "click", "target": "submit_button"},
    ],
    outcome="success",
    metadata={"domain": "Web", "task_type": "Form"},
)

# Learn from a user correction
pipeline.learn_correction(
    situation_text="Extract product prices",
    wrong_actions=[{"action": "click", "target": "wrong_button"}],
    correct_actions=[{"action": "extract", "target": "price_table"}],
    metadata={"domain": "Web", "task_type": "Extraction"},
)
```

### Full Decision Trace

```python
decision = pipeline.decide("Fetch user profile then fetch orders")
print(decision.full_trace())
```

Output:
```
=== Thinking Dust v2 Decision ===

Domain: API (conf=0.890)
Task Type: Sequential (conf=0.850)
Strategy: MEMORY_THEN_VALIDATE (conf=0.720)
Combined Router Confidence: 0.544
MHN Similarity: 0.000
Z3 Validation: unknown
Combined Confidence: 0.381
Decision: ESCALATE

Action Plan:
  1. {'action': 'escalate', 'reason': 'low_confidence'}

Trace:
  → Encoded natural_language input → HDC vector
  → MHN: no matching patterns
  → Router: API/Sequential/MEMORY_THEN_VALIDATE (conf=0.544)
  → Strategy: MEMORY_THEN_VALIDATE — using retrieved pattern
  → Z3: no constraints to validate (skipped)
  → Confidence: 0.381 → escalate
```

---

## Component APIs

### HDC Operations (`td.perception.hdc`)

```python
from td.perception.hdc import *

# Generate random hypervectors
v = generate_hypervector(dim=10000, seed=42)  # {-1,+1}^10000

# Bind (association, self-inverse)
bound = bind(v1, v2)      # v1 * v2 element-wise
recovered = bind(bound, v2)  # == v1

# Bundle (superposition, majority vote)
combined = bundle(v1, v2, v3)  # sign(v1+v2+v3)

# Permute (sequence encoding)
shifted = permute(v, shift=1)  # np.roll(v, 1)

# Similarity (cosine for bipolar)
sim = similarity(v1, v2)  # dot(v1,v2)/dim, range [-1,1]

# Concept vocabulary
vocab = build_default_vocabulary(dim=10000)  # ~182 concepts
vec = vocab.encode_record(action="click", target="button")
```

### MHN (`td.memory.mhn`)

```python
from td.memory.mhn import ModernHopfieldNetwork, MHNConfig

mhn = ModernHopfieldNetwork(MHNConfig(dim=10000))

# Store
idx = mhn.store(key=situation_vec, value=action_vec,
                metadata={"domain": "Web"})

# Retrieve
results = mhn.retrieve(query_vec, top_k=3)
for value, similarity, metadata in results:
    print(f"sim={similarity:.3f}: {metadata}")

# Correct (no catastrophic forgetting)
mhn.update_pattern(idx, new_key, new_value, new_metadata)
```

### Router (`td.routing`)

```python
from td.routing import HierarchicalRouter

router = HierarchicalRouter(input_dim=10000)
result = router.route(hdc_vector)
# → RoutingResult(domain="Web", task_type="Form",
#   strategy="MEMORY_THEN_VALIDATE", combined_confidence=0.72)
```

### Z3 Bridge (`td.reasoning.z3_bridge`)

```python
from td.reasoning.z3_bridge import Z3Bridge

bridge = Z3Bridge()
result = bridge.validate_action(
    action_plan=[{"action": "click", "target": "submit"}],
    constraints={"submit_visible": True, "captcha_present": False},
)
# → Z3Result(status="sat") = action is valid
```

---

## Training the Router

```python
from td.routing.router_train import train_router

# Train all 3 routers on ~60 synthetic examples
metrics = train_router(epochs=50, lr=1e-3, verbose=True)

# Output:
#   Router A accuracy: 0.950
#   Router B (Web) accuracy: 0.900
#   Router B (API) accuracy: 0.850
#   Router C accuracy: 0.880
```

To add new training examples, edit `TRAINING_EXAMPLES` in `td/routing/router_train.py`.

---

## Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| HDC type | Bipolar {-1,+1} | Self-inverse binding, simplest math |
| Binding | Element-wise multiply | O(dim), self-inverse |
| Bundling | Sign of sum (deterministic ties→+1) | Commutative, reproducible |
| CA Rule | Rule 90 (XOR neighbors) | Additive, chaotic, zero training |
| MHN | Key-based retrieval (not composite) | Query similarity must match key |
| Router | Ternary weights {-1,0,+1} | 16× compression, integer ops |
| Z3 | Lazy import + graceful degradation | Won't crash if Z3 unavailable |

---

## Common Issues

### "libz3.dylib not found" or architecture mismatch

Your shell is running under Rosetta. Fix:
```bash
# Check
arch  # if says "i386" you're Rosetta

# Run python explicitly as ARM:
/usr/bin/arch -arm64 .venv/bin/python your_script.py

# Or permanently fix: Terminal app → Get Info → uncheck "Open using Rosetta"
```

### "bundle requires at least 2 vectors"

This was fixed — bundle now accepts 1+ vectors. Update your code.

### Router gives 1.0 confidence on everything

The router is untrained. Run `pipeline.train_routers()` first.

### MHN retrieval returns empty

Check `min_similarity` in MHNConfig. Default is 0.3. Lower it for fuzzy matching:
```python
mhn = ModernHopfieldNetwork(MHNConfig(min_similarity=0.1))
```

---

## Performance Benchmarks

| Component | Time (M1) | Memory |
|-----------|-----------|--------|
| HDC encode | <0.1ms | 10KB/vector |
| CA Reservoir (T=50) | <1ms | 10KB |
| MHN retrieve (1K patterns) | <2ms | ~30MB |
| Router cascade (3 levels) | <0.1ms | ~5KB weights |
| Z3 solve (simple) | 1-10ms | ~5MB binary |
| Full pipeline (no Z3) | <5ms | ~7MB total |
| Full pipeline (with Z3) | 5-50ms | same |

Run benchmarks:
```bash
.venv/bin/python -m pytest tests/ --durations=10
```

---

## Contributing

1. Write tests for new features
2. Run `pytest tests/ -v` before committing
3. Follow existing code style (type hints, docstrings)
4. Update this guide if you change setup or architecture

---

*"Build the dust. Let it act."*
