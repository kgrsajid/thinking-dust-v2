# Gemini/Grok Code Review — External Tool Findings

## Tool Used: Grok Code-Fast-1 (bugs mode)
## Target: `td/perception/hdc.py`
## Date: 2026-06-29

---

## Bugs Found by External Reviewer (10 total)

### HIGH Priority

**1. `hash(name)` breaks reproducibility across Python invocations**
- Location: `ConceptVocabulary.add_concept`
- Python salts `hash()` per-process since 3.3. Same concept name → different vector each run.
- Fix: Use `hashlib.md5(name.encode()).hexdigest()` as seed

**2. No dimension validation on `add_concept` and `load`**
- Vectors of wrong dimension can be silently stored
- Fix: Add `assert len(vector) == self.dim`

**3. Missing shape validation in core operations**
- `bind()`, `similarity()` can silently broadcast wrong shapes
- Fix: Add shape checks

### MEDIUM Priority

**4. Inconsistent seeding between `build_default_vocabulary` and `add_concept`**
- Different RNG mechanisms produce incompatible vectors
- Fix: Use same seeding mechanism

**5. `load()` doesn't clear existing state**
- Old concepts persist, can cause dimension mismatches
- Fix: Clear dict before loading

**6. `len(a)` in similarity — fragile for >1D arrays**
- Fix: Use `a.shape[0]` with dimension check

### LOW Priority

**7. Unused `HDC_CONFIG` global**
**8. No empty string validation in encode_record/encode_sequence**
**9. `_counter` not updated in `load()`**
**10. Direct dict mutation bypasses `add_concept` encapsulation**

---

## Status: Bugs #1-6 to be fixed, #7-10 are minor/code-quality
