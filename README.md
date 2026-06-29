# Thinking Dust v2 — The Thinking Loop

**"Computer intelligence is just human beings teaching dust how to think."**

A neuro-symbolic reasoning engine built on four research mechanisms. <100K parameters. CPU-only. No GPU, no internet-scale pretraining, no attention mechanism.

## How It Works

Every problem goes through a four-mechanism loop:

```
Problem → [1. IDP Refinement] → [2. HDC Decomposition] → [3. Z3 Solving] → [4. Auto-Store]
              ↑                                                              |
              └──────────── Memory grows from every interaction ────────────┘
```

### The Four Mechanisms

| # | Mechanism | Paper | What It Does |
|---|-----------|-------|-------------|
| 1 | **IDP Iterative Refinement** | Betteti et al. 2025 (*Science Advances*) | Query state evolves through iterative MHN retrieval. Energy landscape reshapes. Converges. |
| 2 | **HDC Algebraic Decomposition** | Kanerva 2009, Kleyko et al. 2022 | Sub-problems extracted via `bind(state, prototype)`. Mathematical projection, not keyword matching. |
| 3 | **Z3 Constraint Solving** | Microsoft Z3 | Real integer/boolean variables. Universal primitives: `all_different`, `bounded`, `ordered`, `excluded`, `grouped`, `optimized`. |
| 4 | **Automatic Attractor Storage** | Ramsauer et al. 2020 | Every `think()` stores a new pattern. No retraining. Zero forgetting. Memory grows monotonically. |

### Two Engines

| Engine | File | Status | Description |
|--------|------|--------|-------------|
| **ThinkingDust** | `td/thinking.py` | ✅ Working, tested | Entity-driven Z3 (goals classified via HDC similarity) |
| **GenericThinkingDust** | `td/thinking_generic.py` | ⚠️ Experimental | Full generic: CA reservoir + 6 universal Z3 primitives + anonymous entity graph |

Both share the same MHN, HDC operations, and teaching interface. The generic engine is the research target; the current engine is the demo-ready version.

## Quick Start

```bash
git clone git@github.com:kgrsajid/thinking-dust-v2.git
cd thinking-dust-v2

# Create venv (M1 Mac — use ARM python, NOT Rosetta)
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# Run tests (75 tests, ~3 seconds)
.venv/bin/python -m pytest tests/ -v
```

## Demos

```bash
# ─── Interactive chat (ChatGPT-style, teaches + thinks) ──
.venv/bin/python demos/chat.py --pure     # Start from 0 knowledge
.venv/bin/python demos/chat.py            # With 50 seed reflexes

# ─── Generic chat (universal primitives, experimental) ──
.venv/bin/python demos/chat_generic.py --pure

# ─── Thinking loop demo (shows IDP iterations) ──
.venv/bin/python demos/demo_thinking.py "Schedule meetings with Alice Bob and Carol"
.venv/bin/python demos/demo_thinking.py --stress 20

# ─── Teaching demo (pure mode narrative) ──
.venv/bin/python demos/demo_teaching.py
```

### Try These in Chat

```
you › Schedule meetings with Alice Bob and Carol
td  › Monday 9:00 AM: alice, Monday 10:00 AM: bob, Monday 11:00 AM: carol
     (90% confidence, Z3 verified)

you › Allocate 5000 to seven departments
td  › Group_1: $714, Group_2: $714, ... Group_7: $716
     Total: $5,000 (90% confidence, Z3 verified)

you › Prove that if A implies B and B implies C then A implies C
td  › Verified by Z3: No counterexample exists (95% confidence)

you › Optimize the knapsack problem with 8 items
td  › Item 1: weight=3, value=4, ... Total: weight=18/19, value=35

you › teach: What is the capital of France | Paris
td  › ✅ Learned!

you › What is the capital of France
td  › Paris (85% confidence, sim=1.00)

you › stats
td  › Memory: 9 patterns (0 seed, 9 learned), Seed ratio: 0.0%
```

## Architecture

```
td-v2/
├── td/
│   ├── perception/
│   │   ├── hdc.py                 # HDC core: bind, bundle, similarity, permute
│   │   ├── nl_parser.py           # Entity extraction + HDC goal classification
│   │   ├── nl_parser_generic.py   # Generic: CA reservoir + anonymous entity graph
│   │   ├── dom_encoder.py         # DOM → HDC (legacy)
│   │   ├── api_encoder.py         # JSON → HDC (legacy)
│   │   └── metrics_encoder.py
│   ├── memory/
│   │   └── mhn.py                 # Modern Hopfield Network with IDP
│   ├── routing/                   # BitNet b1.58 ternary routers (legacy)
│   │   ├── ternary_linear.py      # absmean RoundClip quantization
│   │   ├── router_a.py / b.py / c.py
│   │   └── hierarchical_router.py
│   ├── thinking.py                # ✅ ThinkingDust (4 mechanisms, entity-driven)
│   ├── thinking_generic.py        # ⚠️ GenericThinkingDust (6 universal primitives)
│   ├── minimal_seed.py            # 50 seed patterns ("innate reflexes")
│   ├── pipeline.py                # Legacy orchestrator
│   └── z3_solver.py               # Legacy Z3 solver
├── tests/
│   └── test_thinking.py           # 21 tests: IDP, HDC, Z3, storage, teaching
├── demos/
│   ├── chat.py                    # ✅ Interactive chat (main demo)
│   ├── chat_generic.py            # Generic chat (experimental)
│   ├── demo_thinking.py           # Thinking loop with trace
│   ├── demo_teaching.py           # Pure mode teaching narrative
│   └── td_cli.py                  # Legacy CLI
├── data/
│   └── behavioral_strategies.json # 13 seed strategies
└── THINKING_LOOP_DESIGN.md        # Architecture document
```

## Key Design Decisions

### No Pretraining (Minimal Seed)
- Start with 50 seed patterns max (like innate reflexes)
- Or start with 0 (`--pure` mode) and teach everything
- After 1000 interactions, seed should be <5% of total memory
- Tracked via `stats()['seed_ratio_pct']`

### No Keyword Routing
- Z3 models triggered by **entity structure** (people + schedule goals → scheduling)
- Goals classified via **HDC similarity** to prototypes (Kanerva 2009)
- Not `if "schedule" in text.lower()`

### No Hardcoded Domain Data
- Department names: `Group_1, Group_2, ...` (generic)
- Knapsack weights: deterministic from index, not hardcoded
- Number parsing: `word2number` library (handles "seven", "forty", "million")

### Honest Confidence
- Z3 SAT → 90% (proven)
- Z3 proof → 95% (refutation)
- Advice → 40-65% (heuristic, based on retrieval quality)
- Unknown → 20-30% (honest ignorance)

## Research Citations

| Paper | Mechanism |
|-------|-----------|
| Betteti et al. (2025) *Science Advances* | IDP iterative refinement |
| Kanerva (2009) *Cognitive Computation* | HDC algebra (bind, bundle, permute) |
| Kleyko et al. (2022) *ACM Computing Surveys* | HDC/VSA survey, n-gram encoding |
| Kleyko et al. (2025) *Nature Communications* | CA reservoir computing |
| Ramsauer et al. (2020) *ICLR 2021* | Modern Hopfield Networks |
| Ma et al. (2024) | BitNet b1.58 ternary quantization |
| Yilmaz (2015) *arXiv:1503.00851* | CA + HDC reservoir |

## License

MIT

## Authors

- **Kazi Rabbany** — architecture, implementation
- **Kimi K2.6** — co-author, design partner, code reviewer

---

*"Build the dust. Let it think."*
