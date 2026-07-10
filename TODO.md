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

**Research-backed solution (3-stage retrieval):**
1. **Stage 1: Entity Matching** — exact + fuzzy (gazetteer + BEAGLE)
2. **Stage 2: Relation Matching** — BEAGLE query expansion → fuzzy match against KG relations
3. **Stage 3: Ranking** — BEAGLE_sim × TF-IDF × source_weight

**Key references:**
- Aneja et al. (2025), arXiv:2510.19181 — KG-only QA, 71.9% on CRAG, three-stage retrieval
- Esposito et al. (2019), *Information Sciences* — MultiWordNet + Word2Vec QE blueprint
- Li et al. (2025), arXiv:2509.07794 — 42-page QE survey, KGQE approach
- Perna (2025) — KGQE: inject entity type + alias into query

**Fix plan:**
1. **Scale BEAGLE corpus** 10K→100K+ (see #1a below) — fixes vocab coverage
2. **Fix `direct` method ranking** — don't return first hit, rank by BEAGLE_sim × TF-IDF × source
3. **BEAGLE query expansion** — expand query tokens via nearest neighbors, match against KG relations
4. **Three-stage retrieval** — entity match → relation match → ranking

**Files:** `td/thinking.py` — `_query_knowledge_graph` (line 1570)

---

### 1a. Scale BEAGLE Corpus (10K → 100K+) — DONE ✅

**Problem:** BEAGLE vocab = 1992 words. "used", "for", "match" not in vocab → queries fail.

**Research-backed approach:**
- **Jones et al. (2015):** Corpus quality > raw size. Domain-specific > generic web text.
- **Switch to Random Permutations** (RP scales, convolution doesn't — Jones et al., 2015)
- **Domain coverage:** biology, prison, tech, finance, geography, food, programming, zoology, astronomy, chemistry, tools, general knowledge

**Results (2026-07-10):**
- Vocab: 1992 → 2890 words
- Training: 1.4s → 116s (118K sentences, 10K dims)
- Model: `data/word_vectors_110k.pkl` (548MB)
- Key similarities: match/tool=0.903, match/fire=0.548, capital/city=0.199
- **Still missing:** "for" (stop word) — need to handle in query encoding
- **Next:** Implement query expansion + ranking fix in thinking.py

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

### 4. Persistent Teaching Pipeline
Each `teach()` should auto-save to:
- RDF store (pyoxigraph) — triples
- SQLite — KG relations, sense inventory
- Pickle — BEAGLE vectors, Lesk glosses

`chat_flare` and web demo should auto-load on startup.

### 5. Web Demo "Quick Train" Button
Upon clicking, search random topics from Wikipedia and teach like the benchmark pipeline. Same training framework as TESTING_FRAMEWORK.md.

---

## 📝 MEDIUM Priority

### 6. Clean Up Dead Methods (Gemini Review)
Flagged but not all removed:
- `_move_triple_to_sense` — should be a KG method
- `_infer_type_from_fact` — unused
- `_check_type_conflict` — unused
- `_induce_senses_on_conflict` — unused

### 7. God Object Decomposition
`GenericThinkingDust` handles orchestration, WSD, NLP, Z3, SPARQL. Split into:
- `thinking/orchestrator.py` — main dispatch
- `thinking/wsd.py` — sense disambiguation
- `thinking/triple_extractor.py` — NLP extraction

`KnowledgeGraph` handles storage, inference, temporal, WSD, SPARQL. Split into:
- `kg/graph.py` — core graph
- `kg/inference.py` — BFS, rules
- `kg/wsd.py` — sense inventory

### 8. Dimensionality Reduction for Contextual Embeddings
Random project 10K→50 before NG-RC or AERC. Reduces quadratic features from 100M to 10K. Enables the 4 contextualized embedding approaches from CONTEXTUALIZED_EMBEDDINGS_SPEC.md.

### 9. BSC-WSD Implementation
McInnes et al. (2012): HDC binary vectors achieve 94.55% WSD accuracy. Uses existing TD v2 HDC infrastructure. CPU-only, <1ms per disambiguation.

### 10. GPRF: Iterative Query Refinement (Future)
Generalized Pseudo-Relevance Feedback. First pass retrieves top-k facts, uses as "pseudo-relevant" feedback to expand query, second pass retrieves with higher precision.
- Reference: Li et al. (2025), arXiv:2510.25488

---

## 💡 LOW Priority

### 11. Factory Pattern for Contextualizers
Don't hardcode the winner. Use:
```python
self.contextualizer = create_contextualizer(config.context_model_type, wvm)
```

### 12. Ensemble Voting
Don't pick one WSD winner — ensemble the top 2-3 for robustness.

### 13. Cross-Lingual WSD
Current WSD is English-only. Need multilingual glosses and tokenization.

### 14. Rare Sense Testing
Common senses are overrepresented in benchmarks. Need tests for rare/domain-specific senses.

### 15. German Language Support
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
| 2026-07-10 | BEAGLE corpus scaling 10K→118K (2890 vocab, 12 domains) | — |
| 2026-07-10 | ARCHITECTURE/DEVELOPMENT/TODO updated with research findings | — |
| 2026-07-10 | Auto-load BEAGLE + spaCy similarity fallback | f7ac116 |
| 2026-07-10 | Query pipeline — allow declarative sentences | 5e1ca59 |
| 2026-07-10 | Copular handler — pcomp + coordination + clean relations | 107e9da |
| 2026-07-10 | Testing framework table format | 9ed6165 |
| 2026-07-10 | Revert: remove broken simplification layer | a08d50e |
| 2026-07-10 | Research review: QE survey, HDC scaling, KG QA | — |
| 2026-07-10 | BEAGLE corpus scaling 10K→118K (2890 vocab, 12 domains) | — |
| 2026-07-10 | ARCHITECTURE/DEVELOPMENT/TODO updated with research findings | — |
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

_"Fix the gloss, fix the world. Scale the corpus, scale the mind."_
