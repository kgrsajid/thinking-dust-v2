# TD Architecture Analysis — What We Got Wrong, What To Fix

**Date:** 2026-06-29
**Based on:** Re-reading both original plans (v1.0 and v2.0) by Kazi & Kimi K2.6

---

## The Core Problem

**We built a domain-specific agent controller. The plan says to build a generic problem solver.**

The original plan (v1.0) explicitly says:

> "Make Thinking Dust usable by **random people** for **everyday structured problems** without requiring AI expertise."

> Domains: Mathematics/Logic, Scheduling/Planning, Structured Data Query, Spatial/Visual, Unknown/Ambiguous

We hardcoded: Web, API, File, Monitor, Unknown. **Wrong domains entirely.** The plan envisions a system that solves math problems, schedules meetings, finds bugs in code, plans trips — not one that classifies web automation tasks.

---

## What The Plan Actually Says

### TD v2 (System 1) — "The Agentic Controller" / "Everyday Problem Solver"

**Two plan versions exist, with different framings:**

| Aspect | v1.0 (Problem Solver) | v2.0 (Agentic Controller) |
|--------|----------------------|--------------------------|
| Domains | Math/Logic, Scheduling, Data Query, Spatial, Unknown | Web, API, File, Monitor, Unknown |
| Use cases | Smart scheduling, budget planning, code bugs, travel, logic puzzles, data analysis | Web forms, API workflows, file processing, system monitoring |
| LLM fallback | Yes (explicitly mentioned) | No (removed) |
| Focus | **Generic** problem solving for everyday people | **Specific** agent control for OpenClaw |

**We implemented the v2.0 framing** (Web/API/File/Monitor), which is the narrower of the two. The v1.0 framing is more generic and more aligned with the project's thesis.

### What Both Plans Agree On

1. **TD is a pure function** — `f(perception, memory) → decision`. No side effects.
2. **<100K params total** — CPU-only, no GPU, no internet-scale pretraining.
3. **HDC is the unified representation** — everything becomes 10K-bit vectors.
4. **MHN is the memory** — associative recall, zero catastrophic forgetting.
5. **Z3 is the reasoner** — exact proofs, not approximations.
6. **Router is the skill selector** — picks strategy based on input type.
7. **Confidence determines autonomy** — >0.9 auto, 0.7-0.9 confirm, <0.7 escalate.
8. **Online learning** — user corrections become MHN attractors instantly.

---

## What We Built vs What The Plan Says

| Plan Requirement | What We Built | Gap |
|-----------------|--------------|-----|
| **Generic problem solver** | Domain-specific agent controller (Web/API/File/Monitor) | 🔴 Wrong framing |
| **Math/Logic domain** | Not implemented | 🔴 Missing |
| **Scheduling/Planning domain** | Not implemented | 🔴 Missing |
| **Data Query domain** | Not implemented | 🔴 Missing |
| **Spatial/Visual domain** | Not implemented (TD Pro) | ⚠️ Deferred |
| **LLM fallback for low confidence** | Not implemented | 🔴 Missing |
| **Code/AST input** | Not implemented | 🔴 Missing |
| **Image/Grid input** | Not implemented (TD Pro) | ⚠️ Deferred |
| **HDC algebra** | ✅ Working | ✅ Done |
| **CA Reservoir** | ✅ Working (simplified) | ✅ Done |
| **MHN with IDP** | ✅ Working | ✅ Done |
| **Hierarchical Router** | ✅ Working (wrong domains, poor accuracy) | 🔴 Needs redesign |
| **Z3 validation** | ✅ Working | ✅ Done |
| **Online learning** | ✅ Working | ✅ Done |
| **Confidence check** | ✅ Working | ✅ Done |
| **<5ms latency** | ✅ ~10-25ms | ✅ Done |
| **<100K params** | ✅ ~16K params | ✅ Done |

---

## How To Make It Generic

### Step 1: Drop Hardcoded Domains

The current router classifies into Web/API/File/Monitor. The plan says the domains should be:

**Plan v1.0 domains:**
- Mathematics / Logic
- Scheduling / Planning  
- Structured Data Query
- Spatial / Visual
- Unknown / Ambiguous

**But even these are too rigid.** A truly generic system should not have hardcoded domains at all. The MHN naturally clusters inputs by similarity. The router should discover structure, not impose it.

**Proposed fix:** Replace the domain router with **HDC prototype matching**.

```
Instead of:  input → RouterA → "Web" → RouterB(Web) → "Form" → ...
Do:          input → HDC → compare to domain prototypes → closest match
             (prototypes are HDC vectors, updated via online learning)
```

Domain prototypes are just the **average (bundle)** of all HDC vectors that belong to that domain. No neural network needed. No training needed. Domains emerge from usage.

### Step 2: Generic Input Encoding

The plan specifies 4 input types:

| Input Type | Encoder | Status |
|-----------|---------|--------|
| Natural language | Bag-of-concepts → HDC | ✅ Done |
| Structured data (JSON/CSV) | Schema-aware HDC binding | ✅ Done |
| Image / Grid | CA Reservoir | ⚠️ Simplified |
| **Code / Logic** | **AST → HDC tree encoding** | 🔴 **Missing** |

**Code input is critical** for the educational platform use case. A student submits code, TD encodes the AST as HDC, matches against known bug patterns in MHN, and Z3 proves the bug exists.

**Implementation:** Parse Python/JS AST, encode each node as `bind(node_type, encode(children))`, bundle all nodes. This gives a structural fingerprint of the code.

### Step 3: Generic Strategy Selection

Instead of hardcoded strategies (MEMORY_ONLY, MEMORY_THEN_VALIDATE, ESCALATE), use:

```
1. Always try MHN first (what does this remind me of?)
2. If high similarity: retrieve and return
3. If medium similarity: retrieve + Z3 validate
4. If low similarity: 
   a. Try Z3 from scratch (can I reason about this?)
   b. If Z3 succeeds: return proof + plan
   c. If Z3 fails: LLM fallback (best-effort answer, flagged as approximate)
5. Store every outcome as MHN attractor (online learning)
```

### Step 4: LLM Fallback

The v1.0 plan explicitly includes an LLM fallback for cases where Z3 and MHN both fail. This is important — it means TD always gives *some* answer, even for novel inputs. The key is transparency: the user knows when they're getting a proven answer vs a probabilistic guess.

```
Confidence > 0.9: "Here's the answer. [Proof trace]"
Confidence 0.7-0.9: "I'm 80% sure. Here's my reasoning..."
Confidence < 0.7: "I'm not sure. Here's my best guess (from LLM, unverified): ...
                   Want me to learn this for next time?"
```

---

## Dataset Strategy

### What We Need Datasets For

| Component | Data Purpose | Current | Needed |
|-----------|-------------|---------|--------|
| **MHN attractors** | Example problem→solution pairs | 0 (empty at startup) | 1K-10K |
| **Router training** | Labeled (input, domain) pairs | 117 hardcoded | 1K+ |
| **Z3 templates** | Constraint patterns per domain | 4 hardcoded schemas | 100+ |
| **Concept vocabulary** | Semantic HDC concepts | ~200 concepts | 1K-10K |
| **Correction data** | (wrong, right) pairs for online learning | 0 | Ongoing |

### How To Gather Datasets (Generic, Not Just OpenClaw)

#### 1. Synthetic Generation (Largest Source)

**Problem-Solution Pairs:**
- Use existing LLMs (GPT-4, Claude, Kimi) to generate diverse problem descriptions
- Categories: math, scheduling, logic, data analysis, code debugging, planning
- For each problem: generate the Z3 constraints and expected solution
- Target: 10,000 pairs across all categories
- Filter: validate with Z3 (only keep pairs where Z3 proves the solution)

```
Prompt: "Generate 100 scheduling problems with constraints.
         For each, provide: problem description, variables, constraints, optimal solution.
         Vary: number of tasks, resources, conflict types, optimization goals."
```

**Code Bug Patterns:**
- Collect from: Stack Overflow dump, GitHub issue trackers, CodexGlue, CodeSearchNet
- Extract: bug description → buggy code → fixed code → bug category
- Encode: AST → HDC for structural matching
- Target: 5,000 bug patterns covering 50+ bug categories

**Logic Puzzles:**
- Use existing puzzle datasets: ARC-AGI (for TD Pro), Zebra puzzles, SAT competition problems
- Encode: puzzle → HDC → Z3 formulation
- Target: 1,000 puzzles with formal solutions

#### 2. Educational Datasets (Public)

**Mathematical Problems:**
- MATH dataset (Hendricks et al.) — 12,500 competition math problems with solutions
- GSM8K — 8,500 grade school math word problems
- Encode: problem text → HDC, solution steps → action plan, verify with Z3

**Code Datasets:**
- Codeforces/AtCoder problems — algorithmic challenges with test cases
- HumanEval — 164 hand-written programming problems
- Encode: problem → HDC, solution AST → HDC, test cases → Z3 constraints

**Scheduling/Optimization:**
- Vehicle Routing Problem (VRP) benchmarks — Solomon dataset, Gehring-Homberger
- Job Shop Scheduling benchmarks — Taillard, Demirkol
- Encode: problem → HDC, optimal solution → action plan, constraints → Z3

#### 3. Community / Crowdsourcing

**User Interaction Logging:**
- Every TD user interaction generates training data:
  - Input description
  - TD's response and confidence
  - User's confirmation/correction
  - Outcome (did it work?)
- Privacy: sanitize all data, store only HDC vectors (not raw text)

**Open Source Contribution:**
- Release TD as open source with a "teach mode"
- Users submit problem→solution pairs
- Community validates quality (upvote/downvote)
- Best pairs merged into core dataset

**Domain Expert Contribution:**
- Mathematicians: contribute proof patterns
- Developers: contribute code bug patterns
- Project managers: contribute scheduling scenarios
- Each contribution encoded as HDC attractor, shared across all TD instances

#### 4. Automated Discovery

**LLM-Guided Generation:**
```
Loop:
1. LLM generates a novel problem in a target domain
2. TD attempts to solve it
3. Z3 verifies the solution
4. If correct: store as MHN attractor (new training data)
5. If incorrect: LLM generates correction, store as negative attractor
6. Repeat
```

This creates a **self-improving loop** where TD generates its own training data via LLM + Z3 verification. The LLM provides creativity, Z3 provides correctness, TD provides the memory.

**Adversarial Generation:**
```
1. Find TD's weak spots (low confidence regions)
2. Generate problems specifically in those regions
3. Solve with LLM, verify with Z3
4. Store solutions as new attractors
5. Re-test: did confidence improve?
```

### Dataset Format

All data stored as HDC tuples for privacy and efficiency:

```json
{
  "problem_hdc": "[-1, 1, 1, -1, ...]",     // 10K-dim HDC of problem
  "solution_hdc": "[-1, -1, 1, 1, ...]",     // 10K-dim HDC of solution
  "domain": "scheduling",                      // domain label (optional)
  "z3_constraints": "...",                     // SMT-LIB constraints
  "z3_solution": "...",                        // SMT-LIB solution
  "confidence": 0.95,                         // TD's confidence
  "source": "synthetic|educational|user|adversarial"
}
```

No raw text stored — everything is HDC vectors. This makes the dataset:
- **Privacy-preserving** (can't reverse HDC to text)
- **Compact** (10K int8 per problem = 10KB)
- **Universal** (HDC vectors are language-agnostic)
- **Fast** (no encoding needed at runtime)

---

## Resource Estimate: Scaling TD

### Does TD Need 3B Params?

**No.** The BitNet 3B threshold is for language modeling (next-token prediction over vocab ~128K). TD does something completely different:

| Task | What Determines Capacity | TD's Approach |
|------|------------------------|---------------|
| Language modeling | Vocabulary size, context length, world knowledge | TD doesn't do this |
| Classification | Number of classes, feature complexity | TD has ~5-10 domains, not 128K tokens |
| Memory | Number of stored patterns | MHN handles this (capacity = exponential in dim) |
| Reasoning | Problem complexity, constraint depth | Z3 handles this (no params needed) |

**TD's "intelligence" comes from:**
1. HDC algebra (zero params — it's math)
2. MHN memory (zero trainable params — patterns stored at runtime)
3. Z3 reasoning (zero params — it's a solver)
4. Router (only trainable component — ~5K params)

The router is the ONLY component that needs training. And it's a simple classifier, not a language model.

### What If We Scaled to 3B Anyway?

If we replaced the router with a 3B-param transformer:

| Resource | Estimate | MacBook M1 (8GB) | MacBook M1 Pro (16GB) |
|----------|----------|-------------------|----------------------|
| Weight memory (1.58-bit) | ~594 MB | ✅ | ✅ |
| Training (16-bit shadow + Adam) | ~18 GB | ❌ (swap death) | ⚠️ (tight) |
| Inference | ~1 GB | ✅ (via bitnet.cpp) | ✅ |
| Training speed | — | ~100 tokens/sec | ~500 tok/sec |
| Inference speed | — | ~60 tokens/sec | ~150 tok/sec |

**But this defeats the purpose.** The whole thesis of TD is: architecture > scale. If we use 3B params, we've just built a worse LLM. The innovation is doing more with less.

### Recommended Scaling Path

| Stage | Params | RAM | What Changes |
|-------|--------|-----|--------------|
| **Current (v2 MVP)** | 16K | 16MB | Works, router accuracy limited |
| **v2 Hardened** | 45K | 20MB | Better router (prototype matching), more concepts |
| **v2 + Pro** | 100K | 30MB | Full reasoning engine with Liquid-KAN + hypernetworks |
| **v2 + Pro + 10K MHN patterns** | 100K | 140MB | Rich memory from real usage |
| **v2 + Pro + 100K MHN patterns** | 100K | 1.2GB | Expert-level memory |

**All stages run on any MacBook M1 with 8GB RAM.** No GPU needed. Ever.

The bottleneck is never compute — it's **data quality**. More good MHN patterns > more parameters.

---

## What To Do Next (Priority Order)

1. **Redesign the router** — replace neural classifier with HDC prototype matching (zero params, domain-agnostic, domains emerge from usage)
2. **Add Code/AST input encoding** — critical for educational use case
3. **Add generic Z3 template generation** — not hardcoded schemas, but LLM-assisted or MHN-retrieved templates
4. **Add LLM fallback** — when Z3 and MHN both fail, use external LLM with explicit "unverified" label
5. **Start dataset collection** — begin with synthetic generation (LLM + Z3 verification loop)
6. **Build a CLI** — `td "schedule 5 meetings next week avoiding conflicts"` → outputs plan + proof
7. **Write the generic demos** — not web automation, but everyday problems (scheduling, budgeting, logic puzzles, code bugs)

---

*"The worm has 302 neurons and it navigates. The agent has 5,000 neurons and it acts. Scale is not intelligence. Architecture is."*

We have the architecture right. We have the components working. What we got wrong was the **framing** — we built an agent controller when the plan says to build a problem solver. Fix the framing, and the rest follows.
