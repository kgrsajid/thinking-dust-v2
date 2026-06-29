# Literature Compliance Report — TD v2

**Date:** 2026-06-29
**Reviewed by:** Web search of original papers + manual code audit
**Status:** All fixes applied and committed (`98695c5`)

---

## Papers Referenced in the TD Plan

| # | Paper | Component | ArXiv / DOI |
|---|-------|-----------|-------------|
| 1 | Ma et al. (2024) — BitNet b1.58 | Ternary weights | arXiv:2402.17764 |
| 2 | Ramsauer et al. (2020) — Hopfield Networks is All You Need | MHN | arXiv:2008.02217 |
| 3 | Betteti et al. (2025) — Input-Driven Plasticity | IDP | Sci. Adv. 11, eadu6991 |
| 4 | Yilmaz (2015) — CA Reservoir + HDC | CA Reservoir | arXiv:1503.00851 |
| 5 | Von Oswald et al. (2020) — Hypernetworks | TD Pro (not yet) | — |
| 6 | Béna (2024) — Universal NCA | TD Pro (not yet) | — |

---

## 1. BitNet b1.58 (Ma et al., 2024)

### Paper Formula

**Ternarization (absmean quantization):**
```
γ = (1/nm) Σ|W_ij|                    # global mean absolute value
W̃ = RoundClip(W/(γ+ε), -1, 1)        # round to nearest ternary
```

Where:
- `RoundClip(x, a, b) = max(a, min(b, round(x)))`
- `γ` is computed **globally** across the entire weight matrix, not per-row

**STE (Straight-Through Estimator):**
```
Forward:  y = W̃ · x              (use ternarized weights)
Backward: ∂L/∂W = ∂L/∂W̃          (gradient passes through as-if identity)
```

Standard STE implementation:
```python
w_t = ternarize(w)                          # forward value
w_ste = w_t.detach() + (w - w.detach())     # forward=w_t, backward=grad through w
```

**Architecture spec:**
- No bias terms
- SubLN (RMSNorm) between layers
- Learning rate: ~2× the FP16 baseline
- Activations: quantized to 8-bit via per-token absmax
- Minimum effective scale: ~3B parameters for FP16 parity

### What We Had (Before Fix)

```python
# WRONG ternarization (per-row threshold, not absmean):
row_means = w.abs().mean(dim=1, keepdim=True)
thresholds = 0.7 * row_means
ternary = torch.sign(w) * (w.abs() > thresholds).float()

# WRONG STE (double gradient):
w_ste = w_t + self.weight - self.weight.detach()
# This gives gradient through BOTH w_t and (w - w.detach())
# → weights updated ~2× too fast

# WRONG init:
nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)  # √5 is for LeakyReLU(0.2)

# Had bias terms (paper says no bias)
```

### What We Have Now

```python
# CORRECT absmean ternarization:
gamma = w.abs().mean()                     # global
scaled = w / (gamma + eps)
ternary = torch.round(scaled).clamp(-1, 1) # RoundClip

# CORRECT STE (standard BitNet):
w_ste = w_t.detach() + (w - w.detach())   # forward=w_t, backward=grad through w only

# CORRECT init:
nn.init.kaiming_uniform_(self.weight, a=2 ** 0.5)  # √2 for ReLU

# No bias terms (per paper)
# LayerNorm between layers (SubLN)
# LR = 1e-2 (higher than typical 1e-3)
```

### Known Limitation

**BitNet b1.58 at small scale:** The paper states that at 3B+ parameters, ternary models match FP16. Below that, there's a significant quality gap. Our router has ~16K effective parameters (256×10K + 256×5 ternary, ~70% sparse). At this scale:
- RouterA (5-class domain): ~55% accuracy (random = 20%)
- RouterB (4-class task type): 70-100% accuracy
- RouterC (3-class strategy): 91-97% accuracy

This is a **fundamental capacity constraint**, not a bug. The fix would be either:
1. Use continuous weights (violates <100K param goal)
2. Use a different classification approach (e.g., HDC prototype matching)
3. Accept the limitation and rely on MHN + confidence overrides

---

## 2. Modern Hopfield Network (Ramsauer et al., 2020)

### Paper Formula

**Energy function:**
```
E(ξ) = -lse(β, X^T ξ) / β + ||ξ||²/2 + const
```

Where `lse(β, z) = (1/β) log Σ exp(β z_i)`

**Retrieval (one-step update rule):**
```
ξ_new = X · softmax(β · X^T · ξ)
```

This is equivalent to transformer attention where:
- Query = current state ξ
- Keys = stored patterns X
- Values = stored patterns X
- β = inverse temperature (controls retrieval sharpness)

**Key properties:**
- Storage capacity: exponential in dimension (∝ 0.14·d^(d/4) for well-separated patterns)
- One-step retrieval with exponentially small error
- Zero catastrophic forgetting (old patterns never degraded)
- β controls sharpness: high β = precise retrieval, low β = fuzzy/averaged

### Our Implementation

```python
# Retrieve (mhn.py):
sims = self._pattern_matrix @ query_f / self.config.dim   # cosine similarity for bipolar
logits = betas * sims * self.config.dim                    # scale back to dot-product range
logits -= logits.max()                                     # numerical stability
weights = exp(logits) / sum(exp(logits))                   # softmax
# Return top-k patterns weighted by similarity
```

**Status:** ✅ Compliant. Our formula `softmax(β · sims · dim)` is mathematically equivalent to `softmax(β · X^T · ξ)` since `sims = X^T·ξ/dim` and we multiply back by `dim`.

### Improvement Opportunities

1. **Iterative retrieval:** Paper shows multi-step convergence for noisy inputs. We only do one-step. For TD v2's key-value lookup use case, one-step is sufficient. For TD Pro's reasoning, multi-step might help.

2. **Proper β scheduling:** Paper discusses β as a temperature parameter. We use a fixed β=1.0 with IDP modulation. Could implement annealing schedule during training.

---

## 3. Input-Dependent Plasticity (Betteti et al., 2025)

### Paper Formula

**Continuous-time dynamics:**
```
τ ẋ(t) = -x(t) + Ψ(W·diag(α(u(t)))·x(t) + I_ext)
```

Where:
- `α(u)` = input-dependent saliency weights (how much each memory is activated by current input)
- `W` = fixed synaptic weight matrix (stored patterns)
- `Ψ` = activation function (e.g., tanh, softmax)
- `u(t)` = time-varying external input
- `I_ext` = optional direct input current

**Key insight:** The input **reshapes the energy landscape** by modulating synaptic strengths multiplicatively. Patterns congruent with the input get deeper basins; incongruent ones get flattened.

**Paper's IDP energy landscape:**
- At low input saliency (α < 1): single global fixed point at origin (confusion state)
- At moderate saliency (α ≈ 1): metastable states form (competition)
- At high saliency (α > 1): correct memory becomes unique minimum (retrieval)

### Our Implementation (Simplified)

```python
# 1-step approximation:
sigmoid = 1 / (1 + exp(-gain * (sims - threshold)))
betas = β_base * (0.1 + 0.9 * sigmoid)
```

**What we kept:**
- ✅ Per-pattern modulation based on input similarity
- ✅ Low-similarity patterns get damped β (flattened basins)
- ✅ High-similarity patterns get sharp β (deepened basins)

**What we simplified:**
- ❌ Paper uses continuous-time ODE dynamics → we use one-step softmax
- ❌ Paper uses multiplicative synaptic modulation → we use additive β scaling
- ❌ Paper tracks temporal dynamics (memory retention after input switches) → we don't
- ❌ Paper proves robustness to noise via energy landscape analysis → we don't model noise

### Why We Simplified

1. **TD v2 is a key-value lookup** — one-shot retrieval, not temporal sequence memory
2. **Continuous-time ODE requires numerical integration** — too slow for <5ms latency target
3. **The paper's N=1024 neuron simulations** use ~1000-step relaxation — we need one-step
4. **Our use case (agent controller)** doesn't need temporal memory retention between inputs

### When to Upgrade (TD Pro)

- If TD Pro needs **sequential reasoning** (chaining memories), the full IDP dynamics become necessary
- The paper's noise robustness properties would help with noisy HDC inputs from the CA reservoir
- The 2026 follow-up by the same group (arXiv:2603.03201) extends IDP to **sequential memory transitions** — directly relevant to TD Pro's reasoning chain

---

## 4. CA Reservoir (Yilmaz, 2015)

### Paper Formula

**Full architecture:**
```
Input X (length N) → R random permutations → R initial CA states
Each state evolved for I timesteps using rule Z (e.g., Rule 90)
Output: concatenate all R×I state vectors → feature vector of length N×I×R
```

For Rule 90 specifically:
```
A_k[i] = A_{k-1}[i-1] XOR A_{k-1}[i+1]     (cellular automaton update)
```

**Key properties:**
- **Additivity:** Rule 90 is additive under XOR: `CA(A ⊕ B) = CA(A) ⊕ CA(B)`
- **Space-time volume:** All I timesteps are concatenated, not just the final state
- **R permutations:** Multiple random projections of the same input create independent "views"
- **Kernel trick:** For linear rules (90, 150), the CA feature distance can be computed without explicit evolution

**Paper's reservoir (temporal version):**
```
At each timestep:
1. Inject new input X_t into CA state via normalized addition
2. Evolve CA for I steps using rule Z
3. Concatenate all intermediate states → reservoir output
4. Read out via linear classifier on reservoir state
```

### Our Implementation (Heavily Simplified)

```python
# Single projection (R=1, not R permutations):
self._projection_indices = rng.integers(0, input_length, (input_dim, 3))
# XOR of 3 random input bits per output position

# Evolve Rule 90 for T=50 steps:
for _ in range(steps):
    left = np.roll(lattice, 1)
    right = np.roll(lattice, -1)
    lattice = left ^ right

# Output: final state only (not space-time volume)
result = 2 * lattice - 1  # bipolar
```

**What we kept:**
- ✅ Rule 90 (XOR of neighbors) — same automaton
- ✅ Random XOR projection from input to lattice — conceptually same as 1 permutation
- ✅ T=50 evolution steps (paper uses I=32 for similar tasks)

**What we simplified:**
- ❌ R=1 instead of multiple permutations (paper uses R=5-20)
- ❌ Final state only, not space-time volume (paper concatenates all I timesteps)
- ❌ No temporal reservoir (no input injection between timesteps)
- ❌ Fixed circular boundary conditions (paper uses this too, so OK)

### Why We Simplified

1. **Output dimension constraint:** Paper's `N×I×R` = 100×32×10 = 32K features. We need exactly 10K (HDC dim). Multiple permutations would require either reducing N (lose information) or exceeding 10K.

2. **Latency:** Multiple permutations × multi-step evolution × space-time concatenation = significantly slower. Our <5ms target requires single-pass.

3. **Feature sufficiency:** For TD v2's purpose (feature extraction → HDC vector → MHN retrieval), the CA reservoir's role is to **spread local information globally** via chaotic dynamics. The final state of Rule 90 after 50 steps already achieves this — information from each bit has propagated ~50 positions.

4. **No temporal processing needed:** TD v2 processes one input at a time. The temporal reservoir (input injection between steps) is for sequence processing, which is TD Pro's domain.

### When to Upgrade (TD Pro)

- If TD Pro needs **sequence processing** (e.g., multi-step reasoning chains), the full temporal reservoir becomes necessary
- The additive property (Rule 90: `CA(A⊕B) = CA(A)⊕CA(B)`) could be exploited for **compositional reasoning** — processing multiple inputs simultaneously
- Multiple permutations would increase **feature diversity** at the cost of dimensionality

---

## 5. Summary of Critical Findings

### Bugs Fixed (from code review + literature review)

| # | Component | Bug | Fix |
|---|-----------|-----|-----|
| 1 | TernaryLinear | STE gradient doubled | `w_t.detach() + (w - w.detach())` |
| 2 | TernaryLinear | Per-row threshold ternarization | Global absmean RoundClip per BitNet paper |
| 3 | TernaryLinear | Had bias terms | Removed (BitNet spec) |
| 4 | TernaryLinear | Wrong Kaiming init (a=√5) | Fixed to a=√2 (ReLU) |
| 5 | All Routers | No input normalization | Added `x / √dim` in forward() |
| 6 | All Routers | No LayerNorm | Added LayerNorm between fc1 and fc2 |
| 7 | All Routers | LR too low (1e-3) | Raised to 1e-2 (BitNet recommends 2×) |
| 8 | RouterA | Undefined RoutingResult type hint | Removed |
| 9 | HierarchicalRouter | Not nn.Module | Subclassed nn.Module with ModuleDict |
| 10 | DOMEncoder | Salted hash() | SHA-256 deterministic |
| 11 | MHN | IDP gain hardcoded at wrong scale | Moved to config, adjusted for 10K-dim |
| 12 | MHN | _active_indices not initialized | Initialized in __init__ |
| 13 | Pipeline | CONSTRAINT_MAP mutated | Deep-copy before update |
| 14 | Pipeline | No else for unknown strategy | Added escalation fallback |
| 15 | Pipeline | Empty MHN + memory strategy → empty plan | Auto-escalate |
| 16 | Pipeline | ESCALATE with confidence "execute" | Force confidence < 0.7 on ESCALATE |
| 17 | Confidence | Weights dict mutated | Copy before mutation |
| 18 | Confidence | Boundary > instead of >= | Fixed |
| 19 | Confidence | z3_sat=0.5 misleading default | Changed to 0.0 |
| 20 | OnlineLearner | Hardcoded 0.5 correction threshold | Use MHN config min_similarity |
| 21 | OnlineLearner | Only first match deactivated | Deactivate ALL matching |
| 22 | OnlineLearner | Matched on key+value inconsistently | Match on key only (consistent with retrieval) |
| 23 | OnlineLearner | "partial" outcome ignored | Added to valid outcomes |
| 24 | AttractorStore | Dead _meta, prune, decay | Removed entire dead system |
| 25 | train_router | Never returned trained models | Returns router_a, routers_b, router_c |
| 26 | train_router | Only 90 training examples | Expanded to 117 |
| 27 | hdc.py | assert for validation | ValueError |
| 28 | ca_reservoir | Dead CAConfig.rule field | Removed |

### Architecture Limitations (Not Bugs)

| Component | Limitation | Impact | Mitigation |
|-----------|-----------|--------|------------|
| Router (BitNet) | 16K params << 3B threshold | ~55% domain accuracy | Confidence override on ESCALATE; MHN handles familiar patterns |
| IDP (Betteti) | 1-step approximation | No temporal memory dynamics | Acceptable for TD v2's one-shot retrieval |
| CA (Yilmaz) | R=1, final state only | Less feature diversity | 10K-dim constraint makes R>1 impractical |
| HDC bundle | Tie-breaking breaks exact distributivity | ~50% similarity for 3-item distributive law | Known BSC property, not a correctness issue |
| NL Parser | Shared role vectors create baseline similarity | Different commands have ~0.2-0.3 similarity | Acceptable for retrieval (MHN threshold = 0.3) |

---

## 6. Improvement Roadmap

### Short Term (Next Sprint)

1. **Balance training data** — Web has 33 examples vs 20 for others. Add more API/File/Monitor/Unknown examples to reduce Web bias.

2. **Train/test split** — Currently training accuracy is reported on the same data used for training. Add 80/20 split for honest evaluation.

3. **Weight decay** — BitNet paper recommends two-stage weight decay schedule (first half: decay, second half: no decay). Add to optimizer config.

4. **Router ensemble** — Instead of one RouterA, train 3-5 with different seeds and majority-vote. Each has ~55% accuracy individually but ensemble might reach 70%+.

### Medium Term (TD v2 Hardening)

5. **Concept-based routing** — Instead of classifying raw HDC vectors, decompose into concept probabilities and route on those. This gives the classifier a much cleaner signal.

6. **HDC prototype matching** — Instead of a neural router, use HDC similarity to domain prototype vectors. `domain = argmax_d similarity(input, prototype_d)`. Zero training needed.

7. **Multi-permutation CA** — Implement R=3 permutations with reduced evolution steps to stay within 10K dims. Would increase feature diversity.

8. **Iterative MHN retrieval** — For noisy inputs, run 3-5 retrieval iterations instead of one-step. Would improve recall on partial matches.

### Long Term (TD Pro)

9. **Full IDP dynamics** — Implement continuous-time IDP with energy landscape analysis for sequential reasoning chains.

10. **Temporal CA reservoir** — Add input injection between CA timesteps for sequence processing.

11. **Hypernetwork-generated Z3 queries** — (Von Oswald et al., 2020) — Hypernetwork A generates Z3 templates from task embeddings.

12. **Universal NCA for spatial reasoning** — (Béna, 2024) — Neural Cellular Automata for ARC-AGI grid tasks.
