# Thinking Dust v2 — Neuro-Symbolic Reasoning Engine

**"Computer intelligence is just human beings teaching dust how to think."**

A reasoning engine that **derives facts it was never taught**. No neural network. No GPU. No pretraining. Under 100,000 parameters. Runs on CPU in milliseconds.

---

## What It Does

```
teach: Paris is the capital of France        | Paris
teach: France is in the EU                  | France is in the EU
teach: EU is part of Europe                  | EU is part of Europe

ask:   is Paris in Europe?
→ YES. paris --capital_of--> france --in--> eu --part_of--> europe
  Derived via transitive composition (not retrieved from memory)
```

**TD v2 was never taught "Paris is in Europe."** It derived this using two inference rules: (1) `capital_of + in → in` and (2) `in + part_of → in`.

The proof trace shows every hop of the derivation — no black box.

---

## How It Works

Four layers, each with a specific role:

```
┌──────────────────────────────────────────────────────────────┐
│ Layer 4: Knowledge Graph + Inference Engine                 │
│  BFS path search between entity pairs. Rule templates       │
│  apply transitive, symmetric, inverse, and functional       │
│  inference. Proof traces for every answer.                  │
├──────────────────────────────────────────────────────────────┤
│ Layer 3: BEAGLE Word Vectors                                 │
│  Environmental (identity) vectors + context (co-occurrence) │
│  vectors. Paraphrase matching without neural networks.       │
├──────────────────────────────────────────────────────────────┤
│ Layer 2: Natural Language Parser                            │
│  CA reservoir (Rule 90) for feature extraction.             │
│  Rule-based triple extraction from sentence structure.      │
│  14 innate relation prototypes as HDC centroids.           │
├──────────────────────────────────────────────────────────────┤
│ Layer 1: HDC + MHN + Z3                                     │
│  10,000-dim bipolar vectors. Modern Hopfield Network for    │
│  associative memory. Z3 SMT solver for constraints.         │
└──────────────────────────────────────────────────────────────┘
```

**Query path:** Parser extracts entities → KG BFS finds all paths → Rule templates derive new facts → MHN retrieves paraphrase matches → Answer + proof trace returned.

**Teach path:** MHN stores semantic key → Parser extracts triples → KG stores facts → BEAGLE updates context vectors → Inference immediately derives new facts.

---

## Key Features

- **Derives new facts** via transitive composition (`capital_of + in → in`)
- **Functional contradiction** — "Are Berlin and Paris the same?" → NO (each city has exactly one capital, different countries)
- **Symmetric inference** — "Alice married Bob" → TD knows "Bob married Alice"
- **Paraphrase matching** via BEAGLE word vectors (teach "capital of France", ask "france capital")
- **Constraint solving** via Z3 with 18 mathematical primitives
- **Honest uncertainty** — says "I don't know" instead of hallucinating
- **Proof traces** — shows the reasoning chain for every derived answer
- **Online learning** — word vectors evolve from every teach() interaction
- **Zero GPU** — pure HDC algebra + Z3, runs on any CPU
- **<100K parameters** total
- **SQLite persistence** — knowledge survives restarts
- **Interpretable** — every inference step is traceable

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/kgrsajid/thinking-dust-v2.git
cd thinking-dust-v2
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# Run the interactive demo
.venv/bin/python3 demos/chat_flare.py --pure
```

Then type:
```
teach: Paris is the capital of France
teach: France is in the EU
teach: EU is part of Europe
is Paris in Europe?
```

Enable proof traces:
```
trace
```

Teach relation properties (for novel relations):
```
relation: north_of transitive
relation: married_to symmetric
relation: capital_of functional inverse:has_capital
```

---

## Current Capabilities

| Capability | Status | Details |
|------------|--------|---------|
| 2-hop transitive inference | ✅ Working | Paris→France→EU |
| 3-hop transitive inference | ✅ Working | Paris→France→EU→Europe |
| 6-hop transitive inference | ✅ Working | Tested up to 6 hops |
| Functional contradiction | ✅ Working | Berlin ≠ Paris |
| Symmetric inference | ✅ Working | married_to, same_as |
| Inverse relations | ✅ Working | capital_of ↔ has_capital |
| Paraphrase matching | ✅ Working | "capital of" ↔ "france capital" |
| BEAGLE online learning | ✅ Working | Context vectors accumulate |
| SQLite persistence | ✅ Working | Knowledge survives restarts |
| Proof traces | ✅ Working | Full derivation chain shown |
| Z3 constraint solving | ✅ Working | 18 mathematical primitives |
| Test suite | ✅ 99 passing | 50 original + 34 generalization + 15 realworld |

### What TD v2 Can Answer

```
Teach: Paris capital_of France, France in EU, EU part_of Europe
  → Is Paris in the EU?          YES (2 hops: France→EU)
  → Is Paris in Europe?          YES (3 hops: France→EU→Europe)
  → Is Berlin in Europe?         YES (via Germany→EU→Europe)
  → Are Paris and Berlin same?   NO (functional contradiction)

Teach: Kazakhstan north_of Uzbekistan, Uzbekistan north_of Tajikistan
  → Is Kazakhstan north of Tajikistan?  YES (novel transitive relation)
```

### What TD v2 Cannot (Yet) Answer

```
→ "What about London?"            (no multi-turn context)
→ "Alice called Bob because..."   (no clause segmentation)
→ "Did A happen before B?"        (no temporal reasoning)
→ "How do you say X in Spanish?"  (no multilingual support)
```

---

## Research Foundation

Built on verified research from the HDC/VSA literature and symbolic AI:

| Paper | Year | Venue | Role in TD v2 |
|-------|------|-------|---------------|
| Kanerva, P. "Hyperdimensional Computing" | 2009 | *IEEE CIM*, 4(2): 12–29 | HDC algebra (bind, bundle, permute) |
| Jones & Mewhort, "BEAGLE Word Vectors" | 2007 | *Psychological Review*, 114(1): 1–37 | Paraphrase matching via word vectors |
| Ramsauer et al., "Hopfield Networks is All You Need" | 2020 | arXiv:2008.02217 | MHN associative memory |
| de Moura & Bjørner, "Z3: An Efficient SMT Solver" | 2008 | *TACAS*, LNCS 4963, 337–340 | Z3 constraint solving |
| Yilmaz, K. "CA + HDC Reservoir" | 2015 | arXiv:1503.00851 | CA reservoir feature extraction |
| Betteti et al., "IDP Semantic Memory" | 2025 | *Science Advances*, 11(5): eadn8648 | Iterative deepening pattern retrieval |
| Lewis, M. "Role-Filler Binding in VSA" | 2024 | arXiv:2401.06808 | Role-filler structured representations |
| Fodor et al., "Syntax+Semantics with Transformers" | 2025 | *Computational Linguistics*, 51(1): 1–45 | Neuro-symbolic hybrid motivation |
| Liu et al., "PathHD: KG Reasoning with HD Vectors" | 2025 | *NeurIPS* | HDC multi-hop KG reasoning |

**Future layers:**

| Paper | Year | Venue | Planned for |
|-------|------|-------|-------------|
| Allen, J.F. "Maintaining Knowledge about Temporal Intervals" | 1983 | *CACM*, 26(11): 832–843 | Temporal reasoning (TD v2.5) |
| Shervashidze et al., "Weisfeiler-Lehman Graph Kernels" | 2011 | *JMLR*, 12: 2539–2561 | Multi-path disambiguation |
| Borgwardt & Kriegel, "Shortest-Path Kernels" | 2005 | *ICDM*, 74–81 | Structural KG similarity |
| Mann & Thompson, "Rhetorical Structure Theory" | 1988 | *Text*, 8(3): 243–281 | Clause segmentation |
| Hu et al., "Hierarchical Clause Annotation" | 2023 | *Applied Sciences*, 13(4): 2341 | DisCoDisCo clause parser |
| Nivre et al., "Universal Dependencies 2.18" | 2024 | *LREC* | Multilingual parsing |
| Kleyko et al., "HD Cross-Lingual Alignment" | 2023 | *ACL*, 3128–3140 | Cross-lingual vector spaces |
| Muggleton, S. "Inductive Logic Programming" | 1991 | *New Generation Computing*, 8(4): 295–318 | Automatic rule discovery |
| Cropper & Muggleton, "Logical Minimisation of Meta-Rules" | 2016 | Springer, Ch. 12 | Learning relation properties |

---

## How to Use

### Setup

```bash
cd ~/Documents/Thinking\ Dust/td-v2
arch -arm64 .venv-arm64/bin/python demos/chat_flare.py
```

### Commands

| Command | Example | What it does |
|---------|---------|-------------|
| `teach: <fact>` | `teach: Paris is the capital of France` | Stores a fact |
| `teach: <fact> \| <answer>` | `teach: What is the capital of France? \| Paris` | Stores fact with answer |
| `relation: <name> <property>` | `relation: capital_of functional` | Teaches how a relation behaves |
| `ask: <question>` | `ask: is Paris in the EU?` | Asks a question |
| `stats` | | Shows memory state |
| `save` | | Saves to SQLite |
| `quit` | | Exit |

### Question Types

TD v2 supports **7 types of questions**:

#### 1. Yes/No — "Is X in Y?"
```
teach: Paris is the capital of France
teach: France is in the EU
ask: is Paris in the EU?
→ YES. Paris → France → EU (2-hop derivation)
```

#### 2. Open Query — "What/Who/Where is X?"
```
teach: iPhone is made by Apple
teach: Apple was founded by Steve Jobs
ask: who founded the company that makes iPhone?
→ Steve Jobs (2-hop derivation)
```

#### 3. Functional Contradiction — "Are X and Y the same?"
```
teach: Paris is the capital of France
teach: Berlin is the capital of Germany
relation: capital_of functional
ask: are Paris and Berlin the same?
→ NO. Paris has capital_of=France, Berlin has capital_of=Germany.
  Since capital_of is functional, they are different.
```

#### 4. Temporal — "Was X before Y?"
```
teach: Obama was president 2009-2017
teach: Trump was president 2017-2021
ask: did Obama's term meet Trump's?
→ YES. [2009,2017) meets [2017,2021) — Allen's MEETS relation
```

#### 5. Multi-hop
```
teach: My room is in the apartment
teach: The apartment is in the building
teach: The building is on the street
... (20 facts)
ask: is my room in the Observable Universe?
→ YES. 20-hop chain with proof trace
```

#### 6. Proof Trace — "Why is X in Y?"
```
ask: why is DNA part of the organism?
→ DNA → genes → chromosome → nucleus → cell → organ → organism
  (6-hop chain with full reasoning trace)
```

#### 7. Confidence Scoring
Confidence is based on **chain quality**, not hop count:
- All explicit rules → 0.95 (highest)
- Mixed rules + heuristic → 0.50-0.70
- All heuristic → 0.10 (lowest)

---

## What It Can and Can't Do

### ✅ Can do
- Answer questions about facts it's been taught
- Derive new facts through logical inference
- Explain its reasoning (proof traces)
- Handle temporal reasoning (before, after, meets, during)
- Detect contradictions (functional relations)
- Persist knowledge to SQLite

### ❌ Can't do
- Answer general knowledge questions (only knows what you teach it)
- Handle conversational questions ("How are you?")
- Access real-time information ("What's the weather?")
- Reason about opinions ("Is Python better than Java?")
- Handle negative reasoning ("Is Tokyo NOT in Europe?")

---

## Architecture

```
User Input
    ↓
Parser: entities + relations extracted
    ↓
KG: BFS paths(entity_a → entity_b)
    ↓ Path found
Rule Templates: apply inference (transitive/symmetric/functional/inverse)
    ↓ New facts derived
MHN: store semantic keys for paraphrase retrieval
    ↓
Answer + proof trace returned (<50ms)
```

---

## Roadmap Summary

| Phase | Timeline | What's Coming |
|-------|---------|---------------|
| **v2.1** | This week | BEAGLE→KG integration, passive voice fixes, 6-hop validation |
| **v2.5** | Month 2 | Clause segmentation, anaphora resolution, temporal reasoning |
| **v3.0** | Month 3 | Multilingual (UD), automatic rule discovery via ILP, graph kernels |
| **Pro** | Month 3–6 | TD Pro integration: Liquid-KAN + Hypernetwork + NCA for novel problems |

---

## License

MIT License

Copyright (c) 2026 Kazi Rabbany

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

*"Build the dust. Let it think."*
