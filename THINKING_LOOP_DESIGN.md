# Thinking Dust v2 — The Thinking Loop

## What This Is

Four mechanisms from the cited papers, connected into one reasoning loop.
This replaces ALL keyword matching, string decomposition, and static lookup.

## The Loop

```
Problem → HDC Encode
    ↓
┌───────────────────────────────────────┐
│  IDP ITERATIVE REFINEMENT             │
│  (Betteti et al. 2025)                │
│                                       │
│  state = problem_hdc                  │
│  for i in range(max_iters):           │
│      retrieved = mhn.retrieve(state)   │
│      state = bundle(state, retrieved)  │ ← energy landscape reshapes
│      if converged: break              │
│                                       │
│  → evolved_state (richer than input)  │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│  HDC ALGEBRAIC DECOMPOSITION          │
│  (Kanerva 2009, Kleyko 2022)          │
│                                       │
│  for prototype in sub_prototypes:     │
│      component = bind(state, proto)    │ ← algebraic extraction
│      sub_solution = mhn.retrieve(comp) │
│                                       │
│  full_solution = bundle(solutions)    │ ← algebraic composition
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│  Z3 CONSTRAINT SOLVING                │
│  (Microsoft Z3, neuro-symbolic lit)   │
│                                       │
│  vars = extract_integer_vars(state)   │
│  constraints = generate_constraints() │
│  result = Optimize(vars, constraints) │
│  → concrete values (Alice=Mon 9am)    │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│  AUTOMATIC ATTRACTOR STORAGE          │
│  (Ramsauer et al. 2020)              │
│                                       │
│  mhn.store(problem_hdc, solution_hdc) │ ← append, no retrain
└───────────────────────────────────────┘
    ↓
Answer + Proof Trace
```

## Implementation Order

1. `td/thinking.py` — IDP iterative refinement loop
2. `td/algebra.py` — HDC algebraic decomposition/composition
3. `td/constraints.py` — Real Z3 constraint generation from HDC state
4. Auto-storage wired into the end of every solve()

## What Gets Deleted

- String-based decomposition patterns
- Keyword-based prototype matching in solve path
- Boolean Z3 placeholder constraints
- All hardcoded Python dicts

## What Stays

- HDC operations (bind, bundle, similarity) — proven correct
- MHN storage/retrieval — proven correct
- NL parser (entity extraction only, not classification)
- Z3 bridge (lazy import)
- Router (fallback only)
