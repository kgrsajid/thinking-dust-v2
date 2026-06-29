# External Code Review v2 — Complete Findings

**Reviewer:** GLM-5-Turbo (8 parallel review agents)
**Scope:** All TD v2 source files (21 files)
**Date:** 2026-06-29
**Baseline:** 98 tests passing, 0 failures (after v1 fixes)

## Files Reviewed

| File | Findings | CRITICAL | HIGH | MEDIUM | LOW | Status |
|------|----------|----------|------|--------|-----|--------|
| hdc.py | 8 | 0 | 0 | 3 | 5 | 🔧 |
| ca_reservoir.py | 10 | 2 | 2 | 3 | 3 | 🔧 |
| mhn.py | 10 | 1 | 0 | 4 | 3 | 🔧 |
| ternary_linear.py | 4 | 1 | 0 | 1 | 2 | 🔧 |
| router_a.py | 4 | 1 | 0 | 1 | 2 | 🔧 |
| router_b.py | 1 | 0 | 0 | 0 | 1 | ✅ clean |
| router_c.py | 1 | 0 | 0 | 0 | 1 | ✅ clean |
| hierarchical_router.py | 6 | 1 | 0 | 3 | 2 | 🔧 |
| z3_bridge.py | 5 | 0 | 0 | 4 | 1 | 🔧 |
| confidence.py | 3 | 0 | 0 | 2 | 1 | 🔧 |
| constraint_schemas.py | 1 | 0 | 0 | 1 | 0 | 🔧 |
| pipeline.py | 5 | 0 | 1 | 2 | 2 | 🔧 |
| online.py | 6 | 1 | 2 | 3 | 1 | 🔧 |
| attractor_store.py | 7 | 1 | 2 | 3 | 1 | 🔧 |
| nl_parser.py | 1 | 0 | 0 | 0 | 1 | ✅ clean |
| dom_encoder.py | 4 | 1 | 0 | 2 | 1 | 🔧 |
| api_encoder.py | 2 | 0 | 0 | 1 | 1 | 🔧 |
| metrics_encoder.py | 2 | 0 | 0 | 1 | 1 | 🔧 |
| router_train.py | 4 | 0 | 1 | 2 | 1 | 🔧 |
| io.py | 3 | 0 | 0 | 1 | 2 | 🔧 |
| viz.py | 1 | 0 | 0 | 0 | 1 | ✅ clean |
| **TOTAL** | **88** | **10** | **8** | **35** | **32** | |

---

## CRITICAL Findings (10)

### 1. STE gradient is doubled (ternary_linear.py)
**Location:** `forward()`, training path
**Bug:** `w_ste = w_t + self.weight - self.weight.detach()` gives gradients through both `w_t` and `self.weight`. Standard BitNet STE is `w_t.detach() + (self.weight - self.weight.detach())`.
**Impact:** Weight updates ~2× intended magnitude, degrading training stability.
**Fix:** `w_ste = w_t.detach() + (self.weight - self.weight.detach())`

### 2. Salted hash() in dom_encoder.py
**Location:** `_features_to_binary()`
**Bug:** `pos = hash(name) % 200` — Python's hash() is randomized per process. Same HTML → different CA output every run.
**Impact:** Non-reproducible DOM encodings, cached models invalid on restart.
**Fix:** Replace with `int(hashlib.sha256(name.encode()).hexdigest(), 16) % 200`

### 3. CAConfig.rule is dead code (ca_reservoir.py)
**Location:** `CAConfig.rule: int = 90`
**Bug:** Rule number stored but `evolve()` hardcodes Rule 90. Setting rule=30 does nothing.
**Fix:** Remove `rule` from config (we only support Rule 90).

### 4. Projection rebuild breaks reproducibility (ca_reservoir.py)
**Location:** `_build_projection()` called conditionally in `evolve()`
**Bug:** Variable-length inputs cause projection matrix rebuild with new RNG draw, destroying previous projection. Order-dependent results.
**Fix:** Pre-build for max expected input, or fail loudly if input exceeds capacity.

### 5. AttractorStore retrieve never updates access tracking (attractor_store.py)
**Location:** `retrieve()` method
**Bug:** Docstring says "Retrieve and update access tracking" but method does nothing with `_meta`. `access_count` always 0, `last_accessed` always creation time.
**Fix:** Wire up tracking or remove dead `_meta` system.

### 6. AttractorStore._meta desyncs from MHN (attractor_store.py)
**Location:** `store()` and parallel `_meta` dict
**Bug:** If patterns are stored directly through MHN (bypassing AttractorStore), `_meta` indices desync. `_active_indices` filters make this worse.
**Fix:** Move metadata into StoredPattern, eliminate parallel dict.

### 7. IDP gain/threshold misconfigured (mhn.py)
**Location:** `_compute_idp_betas()`, hardcoded `gain = 10.0`
**Bug:** For 10K-dim bipolar vectors, typical random similarity ≈ ±0.01. Sigmoid with gain=10 at threshold=0.3 means essentially ALL patterns get β ≈ β_base × 0.1 (damped). IDP is effectively neutering retrieval sharpness.
**Fix:** Move gain to MHNConfig, adjust for 10K-dim similarity distribution.

### 8. Router_a.py references undefined RoutingResult + np import at bottom
**Location:** `classify()` return type hint, module-level imports
**Bug:** `RoutingResult` defined in `hierarchical_router.py`, not imported. `np` imported at bottom with noqa hack.
**Fix:** Remove type hint, move np import to top.

### 9. HierarchicalRouter not an nn.Module (hierarchical_router.py)
**Location:** Class definition
**Bug:** Plain class with manual train/eval/parameters. Can't use with torch.jit, torch.compile, standard trainers. `model.to(device)` won't recurse.
**Fix:** Subclass nn.Module, use ModuleDict for router instances.

### 10. learn_from_correction hardcoded threshold (online.py)
**Location:** `learn_from_correction()`, `key_sim > 0.5 and val_sim > 0.5`
**Bug:** Hardcoded 0.5 disconnected from MHN's `min_similarity` (0.3). If dim or normalization changes, silently breaks.
**Fix:** Use `self.mhn.config.min_similarity` or make configurable.

---

## HIGH Findings (8)

### 11. evolve_batch doesn't use additive property it claims (ca_reservoir.py)
**Location:** `evolve_batch()` docstring + implementation
**Fix:** Remove false additive property claim from docstring.

### 12. Modulo projection is lossy (ca_reservoir.py)
**Location:** `evolve()`, `idx % binary_input.size`
**Fix:** Pre-build projection for max input length, reject larger inputs.

### 13. Empty action_plan passed to Z3 (pipeline.py)
**Location:** `decide()`, MEMORY_THEN_VALIDATE when mhn_results empty
**Fix:** Guard `if action_plan and constraints:` before Z3 call.

### 14. Correction only deactivates first match (online.py)
**Location:** `learn_from_correction()`, `break` after first match
**Fix:** Remove `break`, deactivate all matching patterns.

### 15. Correction similarity uses value vectors inconsistently (online.py)
**Location:** `learn_from_correction()`, matches on both key and value
**Fix:** Match only on key (situation) similarity, consistent with retrieval.

### 16. prune_inactive is no-op, decay_half_life unused (attractor_store.py)
**Fix:** Remove both or implement.

### 17. train_router() never returns trained models (router_train.py)
**Location:** `train_router()` function
**Fix:** Return trained RouterA, RouterB dict, RouterC alongside metrics.

### 18. load_state doesn't restore RouterB hierarchy (io.py)
**Location:** `load_state()`
**Fix:** Already handled by `pipeline.router.load()` which saves all RouterB instances.

---

## MEDIUM Findings (selected — 35 total, key ones below)

- `decompose()` similarity is dot/len not true cosine (z3_bridge.py)
- Template system is dead code relative to validate_action (z3_bridge.py)
- encode_record/encode_sequence silently auto-create concepts for typos (hdc.py)
- assert → ValueError in load()/add_concept() (hdc.py)
- Product confidence too pessimistic in hierarchical_router.py
- Memory strategies yield empty plans when no MHN hits (pipeline.py)
- learn_from_outcome doesn't handle "partial" outcome (online.py)
- encode_action_plan bundles lossily for 5+ actions (online.py)
- Feature collision from hash() % 200 in dom_encoder.py
- No train/test split in router_train.py
- Kaiming init a=√5 doesn't match ReLU usage (ternary_linear.py)
