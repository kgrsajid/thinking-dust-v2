# TD v2 — TODO List

_Last updated: 2026-07-09_

---

## 🚨 URGENT (Next Session)

### 1. Fix teach() Gloss Quality
**Problem:** `teach()` extracts triples and loses context words from the original sentence. "bats use echolocation to navigate in the dark" → Lesk gloss only gets triple-form words, not "navigate" or "dark". This makes Lesk WSD weak for facts taught through `teach()`.

**Evidence:**
- Standalone Lesk with full Wikipedia sentences: 100% accuracy (66/66)
- Lesk integrated with `teach()`: sparse glosses, higher fallback
- Manual enrichment via `add_sense_example()` restores accuracy to 90-100%

**Fix:**
- Store the FULL original teach sentence in the Lesk gloss (not triple-form)
- Reduce frame-word exclusion — keep domain-relevant words
- Add both raw and lemmatized forms to the gloss
- `_rebuild_lesk_glosses()` should use original sentences, not reconstructed triples

**Files:** `td/thinking.py` (line ~1016), `td/perception/lesk_wsd.py`

---

## 📋 HIGH Priority

### 2. Automated WSD Testing Framework
Run the TESTING_FRAMEWORK.md pipeline on 10+ random Wikipedia words. Automate:
- Random word selection from Wikipedia
- Dynamic gloss gathering from Wikipedia articles
- Novel test sentence generation
- Tabular results + overall benchmark score

**File:** `TESTING_FRAMEWORK.md` (spec exists, implementation needed)

### 3. Persistent Teaching Pipeline
Each `teach()` should auto-save to:
- RDF store (pyoxigraph) — triples
- SQLite — KG relations, sense inventory
- Pickle — BEAGLE vectors, Lesk glosses

`chat_flare` and web demo should auto-load on startup.

### 4. Web Demo "Quick Train" Button
Upon clicking, search random topics from Wikipedia and teach like the benchmark pipeline. Same training framework as TESTING_FRAMEWORK.md.

---

## 📝 MEDIUM Priority

### 5. Clean Up Dead Methods (Gemini Review)
Flagged but not all removed:
- `_move_triple_to_sense` — should be a KG method
- `_infer_type_from_fact` — unused
- `_check_type_conflict` — unused
- `_induce_senses_on_conflict` — unused

### 6. God Object Decomposition
`GenericThinkingDust` handles orchestration, WSD, NLP, Z3, SPARQL. Split into:
- `thinking/orchestrator.py` — main dispatch
- `thinking/wsd.py` — sense disambiguation
- `thinking/triple_extractor.py` — NLP extraction

`KnowledgeGraph` handles storage, inference, temporal, WSD, SPARQL. Split into:
- `kg/graph.py` — core graph
- `kg/inference.py` — BFS, rules
- `kg/wsd.py` — sense inventory

### 7. Dimensionality Reduction for Contextual Embeddings
Random project 10K→50 before NG-RC or AERC. Reduces quadratic features from 100M to 10K. Enables the 4 contextualized embedding approaches from CONTEXTUALIZED_EMBEDDINGS_SPEC.md.

### 8. BSC-WSD Implementation
McInnes et al. (2012): HDC binary vectors achieve 94.55% WSD accuracy. Uses existing TD v2 HDC infrastructure. CPU-only, <1ms per disambiguation.

---

## 💡 LOW Priority

### 9. Factory Pattern for Contextualizers
Don't hardcode the winner. Use:
```python
self.contextualizer = create_contextualizer(config.context_model_type, wvm)
```

### 10. Ensemble Voting
Don't pick one WSD winner — ensemble the top 2-3 for robustness.

### 11. Cross-Lingual WSD
Current WSD is English-only. Need multilingual glosses and tokenization.

### 12. Rare Sense Testing
Common senses are overrepresented in benchmarks. Need tests for rare/domain-specific senses.

### 13. German Language Support
`td/languages/de.py` has 6 TODO items:
- More German stop words
- More German prepositions
- Accusative/dative forms
- More German pronouns
- Verify with native speaker

---

## ✅ DONE (Recent)

| Date | Item | Commit |
|------|------|--------|
| 2026-07-09 | Lesk WSD + benchmark (66 instances, 100%) | 9308874 |
| 2026-07-09 | Wikipedia-sourced benchmark | 9807e91 |
| 2026-07-09 | WSD routing fix (3 sense creation) | 8f982d7 |
| 2026-07-09 | 4 parser bugs fixed (GLM 5.2 review) | b229ce9 |
| 2026-07-09 | TESTING_FRAMEWORK.md created | 4aaffaf |
| 2026-07-09 | Proper citations in ARCHITECTURE.md | 3dbdcc9 |
| 2026-07-09 | Unknown word benchmarks (bat 90%, crane 100%, rock 91.7%) | 17acd7a |
| 2026-07-08 | Contextualized Embeddings Spec (5 approaches) | 5e89d9a |
| 2026-07-08 | 7 dead methods removed (-204 lines) | 265c87d |
| 2026-07-08 | SpaCy dependency-based WSD routing | 7ab5a07 |
| 2026-07-08 | Tiered WSD (BEAGLE + KG + LOTG) | 510f643 |

---

_"Fix the gloss, fix the world."_
