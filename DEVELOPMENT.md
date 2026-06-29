# Thinking Dust v2 — Developer Guide

## Setup

### Prerequisites
- **Mac with Apple Silicon (M1/M2/M3)** — ARM Python required for torch/numpy/z3
- **Python 3.10+**

### Install

```bash
git clone git@github.com:kgrsajid/thinking-dust-v2.git
cd thinking-dust-v2

# Create venv (Terminal must NOT be Rosetta)
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# Verify ARM (not x86_64 Rosetta)
.venv/bin/python -c "import platform; print(platform.machine())"  # → arm64
```

### Run Tests

```bash
.venv/bin/python -m pytest tests/ -v
# Expected: 75 passed in ~3s
```

---

## The Two Engines

### Engine 1: `thinking.py` (Working, Tested)

**File:** `td/thinking.py` → class `ThinkingDust`

Entity-driven Z3. The parser extracts structured entities (who, how_many, goals, dollars, propositions), classifies goals via HDC similarity, then routes to one of four Z3 model builders based on entity structure.

```
Problem → NL Parser (HDC goal classification)
        → IDP Refinement (Betteti 2025)
        → HDC Decomposition (Kanerva 2009)
        → Z3 (entity-driven: assignment / budget / logic / CSP)
        → Auto-Store (Ramsauer 2020)
```

**When to use:** Demos, production, anything that needs to work today.

**Limitations:**
- Goal prototypes are hardcoded HDC strings (not learned)
- Z3 model builders still have domain assumptions (what variables to create)
- Parser uses regex for numbers, capitalized words for entities

### Engine 2: `thinking_generic.py` (Experimental)

**File:** `td/thinking_generic.py` → class `GenericThinkingDust`

Universal constraint primitives. The parser discovers anonymous entities via CA reservoir + MHN similarity (no hardcoded types). The solver applies 6 universal mathematical primitives instead of domain models.

**Six Universal Primitives:**

| Primitive | What It Does | Covers |
|-----------|-------------|--------|
| `all_different` | Variables take distinct values | Scheduling, graph coloring, Sudoku |
| `ordered` | Variables satisfy ordering | Sequencing, sorting |
| `bounded` | Variables have domain [min,max] | Scheduling, budget, CSP |
| `excluded` | Value combinations forbidden | Logic, mutual exclusion |
| `grouped` | Partitioned with per-group constraints | Budget, knapsack |
| `optimized` | Maximize/minimize objective | Budget fairness, knapsack value |

**When to use:** Research, extending the system, proving the thesis.

**Limitations:**
- Entity discovery via MHN similarity needs populated memory to work well
- In pure mode (0 patterns), defaults to `bounded` primitive (not very useful)
- Relation discovery is basic — needs richer CA reservoir features
- Z3 output is generic (`e0: 0, e1: 0`) without domain formatting

---

## Key Files

### Active Code (use these)

| File | What |
|------|------|
| `td/thinking.py` | `ThinkingDust` — the working engine |
| `td/thinking_generic.py` | `GenericThinkingDust` — the research engine |
| `td/perception/hdc.py` | HDC operations: `bind`, `bundle`, `similarity`, `permute`, `normalize_hdc`, `ConceptVocabulary` with `get_char_vector` |
| `td/perception/nl_parser.py` | `NLParser` — entity extraction with HDC goal classification |
| `td/perception/nl_parser_generic.py` | `GenericNLParser` — CA reservoir + anonymous entity graph |
| `td/memory/mhn.py` | `ModernHopfieldNetwork` with IDP, retrieval, storage |
| `td/minimal_seed.py` | 50 seed patterns (10 prototypes + 10 constraints + 10 strategies + 20 concepts) |
| `demos/chat.py` | Interactive chat (main demo) |
| `demos/chat_generic.py` | Generic chat (experimental) |
| `demos/demo_thinking.py` | Thinking loop with visual trace |
| `demos/demo_teaching.py` | Pure mode teaching narrative |

### Legacy Code (still imported but not actively used)

| File | What | Status |
|------|------|--------|
| `td/pipeline.py` | Old orchestrator | Import-fixed, not used by thinking.py |
| `td/z3_solver.py` | Old Z3 solver | Superseded by thinking.py Z3 methods |
| `td/decomposer.py` | Old keyword decomposer | Not imported by active code |
| `td/solve_advice.py` | Old advice solver | Superseded by thinking.py |
| `td/routing/` | BitNet b1.58 routers | Trained, working, but not in the thinking loop |
| `demos/td_cli.py` | Old CLI | Don't use — use `chat.py` instead |

### Deleted (in this session)

| File | Why Deleted |
|------|------------|
| `td/training_data.py` | 225 patterns — replaced by `minimal_seed.py` (50) |
| `td/reasoning_decomposer.py` | Keyword-based — replaced by HDC decomposition |
| `td/reasoning/z3_bridge.py` | Keyword-routed Z3 — replaced by entity-driven Z3 |

---

## Architecture: How the Thinking Loop Works

```
┌─────────────────────────────────────────────────────────────┐
│                    PROBLEM TEXT                              │
│  "Schedule meetings with Alice Bob and Carol"               │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP 1: NL PARSER (Kanerva 2009, Kleyko 2022)              │
│  • Tokenize → encode each word as HDC via char n-grams      │
│  • Extract entities: who=[Alice,Bob,Carol], goals=[schedule]│
│  • Goals classified by HDC similarity to prototypes          │
│  • Output: problem_hdc (10K-dim vector) + entities dict     │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP 2: IDP ITERATIVE REFINEMENT (Betteti et al. 2025)     │
│  • Retrieve nearest pattern from MHN                         │
│  • Blend: state = 0.7×current + 0.3×retrieved               │
│  • Repeat until convergence (sim > 0.98)                    │
│  • Output: evolved_state (changed by memory)                │
│  • Stats: 100% convergence, avg 2.5 iterations              │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP 3: HDC ALGEBRAIC DECOMPOSITION (Kanerva 2009)         │
│  • For each universal role (entities, constraints, solution,│
│    validation):                                              │
│    component = bind(evolved_state, role_prototype)          │
│  • Retrieve from MHN using each component                   │
│  • Output: sub_problems list with retrieved metadata        │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP 4: Z3 CONSTRAINT SOLVING                               │
│  • Entity-driven (thinking.py):                              │
│    - People + schedule goal → assignment model              │
│    - Dollars → budget model                                  │
│    - Propositions → logic model                              │
│    - Optimize + count → CSP model                            │
│  • Universal (thinking_generic.py):                          │
│    - Infer primitives from entity relations                  │
│    - Apply: all_different, bounded, grouped, optimized, ... │
│  • Output: real Z3 solution (times, amounts, proof)          │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP 5: AUTO-STORE (Ramsauer et al. 2020)                  │
│  • mhn.store(problem_hdc, composed_solution_hdc, metadata)  │
│  • Memory grows by 1. No retraining. Zero forgetting.       │
│  • Seed ratio tracked: seed_count / total * 100             │
│  • Target: <5% after 1000 interactions                       │
└──────────────────────────────────────────────────────────────┘
```

---

## Pure Mode vs Seed Mode

```bash
# Pure: 0 seed patterns. System starts ignorant, learns everything.
python demos/chat.py --pure

# Seed: 50 seed patterns ("innate reflexes"). Useful from day 1.
python demos/chat.py
```

| | Pure Mode | Seed Mode |
|---|---|---|
| Start memory | 0 patterns | 50 patterns |
| Z3 scheduling | ✅ Works (entity-driven, no memory needed) | ✅ Works |
| Z3 budget | ✅ Works | ✅ Works |
| Z3 proof | ✅ Works | ✅ Works |
| Advice | ❌ "I don't know yet" | ✅ Retrieves strategies |
| Teaching | ✅ Core feature | ✅ Also works |
| Seed ratio | Always 0% | Starts 100%, drops with use |

---

## Teaching

```
teach: <problem> | <solution>
```

Teaching stores the problem-solution pair as an MHN attractor. Next time a similar question is asked, the system retrieves it at sim ≈ 1.00.

**In code:**
```python
td.teach("What is the capital of France", "Paris")
result = td.think("What is the capital of France")
# → "Paris" at 85% confidence, sim=1.00
```

---

## What's Not Hardcoded (and What Still Is)

### ✅ Removed (this session)
- `KNOWN_NAMES`, `GOAL_KEYWORDS`, `ACTION_KEYWORDS`, `TARGET_KEYWORDS`, `TIME_KEYWORDS`
- Keyword routing in Z3: `if "schedule" in text.lower()`
- Hardcoded department names: `["Operations", "Marketing", "Development"]`
- Hardcoded knapsack weights/values
- Hardcoded word-to-number dict
- 225 training patterns

### ⚠️ Still present (acceptable)
- Day/time name mappings (`1→Monday`, `1→9:00 AM`) — universal constants
- Convergence/blend parameters (`0.98`, `0.3`) — documented tunable params
- Goal prototype strings in parser — semantic descriptions, not keywords
- Entity extraction heuristics (capitalized words = names) — reasonable default

### 🎯 Generic engine eliminates
- All domain-specific entity typing
- All Z3 domain models
- All hardcoded formatting
- Uses 6 universal mathematical primitives instead

---

## Adding New Z3 Problem Types

### In `thinking.py` (entity-driven approach):

1. Add entity extraction in `nl_parser.py` (what signals this problem type?)
2. Add entity-pattern check in `_try_all_z3_models()`
3. Build Z3 constraints in a new `_z3_<type>_entities()` method

### In `thinking_generic.py` (universal approach):

1. Store constraint templates in MHN via `teach()` with template metadata
2. System retrieves template on similar problems automatically
3. No code changes needed — primitives are universal

---

## Common Issues

| Issue | Fix |
|-------|-----|
| `libz3.dylib not found` | Shell is Rosetta. Run with `arch -arm64` |
| Tests fail on import | Run `.venv/bin/pip install -e ".[dev]"` to reinstall |
| `word2number not found` | `.venv/bin/pip install word2number` |
| MHN returns empty | Lower `min_similarity` in `MHNConfig` (try 0.01) |
| Z3 gives all zeros | Problem type not recognized — check entity extraction |
| chat.py says "I don't know" | Pure mode with empty memory. Teach it first. |

---

*"Build the dust. Let it think."*
