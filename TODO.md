# TD v2 — TODO List

_Last updated: 2026-07-10_

---

## 🚨 URGENT (RIGHT NOW)

### 1. Fix think() query pipeline — THE bottleneck

**Problem:** `think()` returns "unknown" for most questions even when facts are in KG.

**Benchmark scores:**
- "match": 25.0% (3/12) — was 8.3% before copular fix
- "spring": 41.7% (5/12)

**Root cause:** `_query_knowledge_graph` can't match question words to stored relations.

Example:
- KG has: `(match, tool_for, fire)`
- Question: "what is a match used for?"
- Tokens: `['what', 'is', 'a', 'match', 'used', 'for']`
- Relation matching: "used" not in kg_relations, "for" not in kg_relations
- Compound matching: "used" + "for" → "used_for" not in kg_relations (KG has "tool_for")
- Result: "unknown" despite conf 0.85

**BEAGLE similarity code exists** at ~line 1690 in thinking.py — should match "used" to "tool_for" via word vector similarity. But:
1. BEAGLE vectors never loaded (self.wvm was None) — FIXED in f7ac116
2. BEAGLE vocabulary too small (1992 words) — "used", "for", "match" not in vocab
3. Added spaCy similarity fallback — but en_core_web_sm has NO real word vectors (context tensors only, unreliable)

**What's still broken:**
- **`direct` method returns WRONG fact** — "what is a match used for?" → "is(match) → competitive game" instead of "tool_for(match) → fire". The SPARQL open query returns the first fact about "match" without using similarity.
- **spaCy vectors unreliable** — en_core_web_sm has "no word vectors loaded". Similarities are context tensors from tagger/parser, NOT real word vectors. Need en_core_web_lg (685MB) for real vectors.
- **Context-enriched questions WORK** — "what is a match used for in starting a fire?" → correct answer because "fire" is a KG entity and the path is found. The pipeline needs KG entities in the question to find the right path.
- **Declarative sentences** like "the team won the match" still return "unknown"
- **Answer text is a dict** — keyword matching fails on dict str() representation

**Files:** `td/thinking.py` — `_query_knowledge_graph` (line 1570)

**Tomorrow's fix plan:**
1. Upgrade to en_core_web_lg for real word vectors (or use BEAGLE with larger corpus)
2. When `direct` finds a result, also try similarity and compare confidences
3. For single-entity queries, try ALL facts about that entity ranked by relevance
4. Extract formatted proof trace for keyword matching (not dict str())

---

## 📋 HIGH Priority

### 2. Sentence Simplification (REVERTED — needs proper fix)
The clause segmenter's `source_text` strips articles and loses subjects.
"a spring is a flexible elastic device" → "spring is flexible elastic device"
which the parser can't handle. Need to either:
- Fix clause segmenter source_text to preserve original text
- Or use a different approach (LLM simplification externally)
- Or detect when simplification would produce low-quality output and skip

### 3. Automated WSD Testing Framework
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
| 2026-07-10 | Auto-load BEAGLE + spaCy similarity fallback | f7ac116 |
| 2026-07-10 | Query pipeline — allow declarative sentences | 5e1ca59 |
| 2026-07-10 | Copular handler — pcomp + coordination + clean relations | 107e9da |
| 2026-07-10 | Testing framework table format | 9ed6165 |
| 2026-07-10 | Revert: remove broken simplification layer | a08d50e |
| 2026-07-09 | Self-review: lemmatize bug + hardcoded articles | 1aefdda |
| 2026-07-09 | teach() gloss quality — spaCy lemmatization | a8e0e3e |
| 2026-07-09 | Duplicate triple extraction fix | e78eea3 |
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
