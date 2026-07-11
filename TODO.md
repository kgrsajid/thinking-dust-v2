# TD v2 — TODO List

_Last updated: 2026-07-10_

---

## ✅ MAJOR MILESTONE: Query Pipeline Fixed (2026-07-10)

**Before:** 25% accuracy on benchmark queries
**After:** 92.9% on fresh Wikipedia topics (crane, spring, port, court, seal)

**What was done:**
- BEAGLE corpus scaling: 10K → 118K sentences (2890 vocab, 12 domains)
- Query expansion via BEAGLE nearest neighbors
- Collect-all-then-rank (no more short-circuit on first match)
- YES formatting for yes/no questions
- O(n²) fix: pre-compute BEAGLE similarities
- Language isolation: no hardcoded English in core logic
- 4 parser bugs fixed (copular attr, advmod guard, acl prep, passive agent)
- is→is_a canonicalization via language config
- Regex fallback: underscore-only guard (no magic numbers)
- equality_signals from lang_config.relation_prototypes
- BEAGLE weight 2.0 documented as calibration constant
- Skipped sentence logging (data/skipped_sentences.log)
- 16 new research references (#56-71)
- Layer 0 preprocessing architecture documented

**Caveat:** Benchmark uses clean, crafted queries. Real-world performance unknown.

---

## 🚨 URGENT (RIGHT NOW)

### 1. Implement Preprocessing Layer with Gemini — NEXT SESSION

**Prompts:**
- `PREPROCESSING_PROMPT.md` — v1 for TEACH (fact ingestion), confirmed as final
- `QUERY_PREPROCESSING_PROMPT.md` — v1 for QUERY (user questions), NEW (2026-07-12)
- Why two prompts? v1 Rule 7 ("rewrite as declarative") causes hallucinations on queries.
  Gemini review confirmed: queries need `?` syntax for unknown variables.

**Model:** Gemini (selected over Kimi K2.7 Code and K2.6 — cleanest output, fastest)
**Files:** `PREPROCESSING_PROMPT.md`, `QUERY_PREPROCESSING_PROMPT.md`, `td/preprocessing/__init__.py`

**What to do:**
- Wire Gemini API call into `td/preprocessing/__init__.py`
- Add Gemini API key to environment
- Test with the two benchmark sentences (seals, Python)
- Integrate into `demos/chat_flare.py` for live demo
- Test with real messy queries

**Test cases:**
- "So I was curious, what's the deal with matches and fire?" → `["what is match used for"]`
- "The cell in biology, what's it made of vs the one in prison?" → `["what is cell in biology made of", "what is cell in prison"]`
- "seals are marine mammals that live in cold waters along the Atlantic coast and they haul out on rocks to rest" → 5 atomic sentences

### 1a. Preprocessing Layer — MAY NOT BE NEEDED (2026-07-12)

**Finding:** TD v2's vocabulary-matching approach handles messy queries natively.
No preprocessing needed for QUERIES. May still be needed for complex TEACH sentences.

**Evidence:**
- "So like, what do seals actually eat?" → `eat(seals) → fish` ✅ (591ms, no preprocessing)
- "I was wondering, whats a match used for?" → `used_for(match) → fire` ✅ (299ms, no preprocessing)
- How it works: `re.findall(r'\w+', text)` extracts all words → matches against KG vocabulary → filler words ignored (not in KG)

**Research backing:**
- GraphRAG (Min et al., 2025): "SpaCy noun phrase extractor to pinpoint key concepts within the query"
- Aneja et al. (2025): "Fuzzy Entity Matching against graph nodes, edit distance up to 3"
- Standard KGQA pattern: tokenize → match against KG → ignore non-matching tokens

**When preprocessing IS needed:**
- Complex TEACH sentences: "seals are marine mammals that live in cold waters along the Atlantic coast and they haul out on rocks to rest" → parser extracts 2/5 facts, needs splitting
- The parser already handles messy QUERIES via vocabulary filtering

**What's done (infrastructure exists if needed):**
- ✅ Coreference resolution (he/she/it/they + discourse deixis for this/that)
- ✅ Clause segmentation (spaCy conj dependency)
- ✅ Preprocessing prompt v1 for teach (`PREPROCESSING_PROMPT.md`)
- ✅ Query prompt v1 (`QUERY_PREPROCESSING_PROMPT.md`)
- ✅ PREPROCESSING_PLAN.md written (three-layer architecture)
- ✅ PREPROCESSING_PLAN.md written (three-layer architecture, research-backed)

**Status:** DEFERRED. Focus on scaling the KG (millions of facts) instead of preprocessing.
Preprocessing for teach can be revisited when loading bulk data (Wikipedia, Wikidata).

### 1c. Scaling to Millions of Facts — NEXT PRIORITY (2026-07-12)

**Goal:** Load millions of structured facts into TD v2's KG.

**Data Sources:**
- **Wikidata** — 116M items, 16B triples, free API, CC0 license
- **Wikidata5m** — 5M entities, 20M triples, aligned with Wikipedia, ready to download
  - `https://deepgraphlearning.github.io/project/wikidata5m`
- **Domain subsets** — filter Wikidata to geography, biology, tech, etc.

**Why preprocessing may NOT be needed:**
- Wikidata triples are already clean `(subject, predicate, object)` — no sentence parsing needed
- The vocabulary-matching trick handles messy queries automatically
- Focus: data LOADING, not data CLEANING

**Research backing:**
| Paper | Year | What | Scale | Result |
|-------|------|------|-------|--------|
| PathHD (Liu et al.) | Dec 2025 | HDC path retrieval for KGQA | WebQSP/CWQ/GrailQA | Matches neural baselines, 40-60% less latency |
| HDReason (Chen et al.) | 2024 | HDC for KG completion | YAGO3-10 (120K entities) | 10.6x speedup over GPU |
| VS-Graph | Dec 2025 | HDC graph classification | Multiple benchmarks | 250x faster than GNNs, robust at D=128 |
| pyoxigraph | 2026 | SPARQL store | 10M triples | 18ms per query |
| Wikidata5m | 2020 | KG dataset | 5M entities, 20M triples | Standard benchmark |

**What to do:**
- [ ] Download Wikidata5m dataset
- [ ] Write bulk loader: Wikidata5m → pyoxigraph
- [ ] Benchmark query speed at 1M, 5M, 10M triples
- [ ] Scale BEAGLE corpus to 1M sentences (aligned Wikipedia text)
- [ ] Test TD v2 reasoning on million-scale KG

**Key insight:** "We just need to have millions of records" — the records already exist. Wikidata has 16 billion of them. The trick is loading them efficiently, not preprocessing them.

### 1b. Real-World Query Handling — IN PROGRESS

**Problem:** We tested with clean queries like "what is a match used for". Real users say:
- "I was wondering, what do you use a match for when starting a fire?"
- "So like, that thing you strike to make fire, what's it called?"
- "The cell in biology, what's it made of vs the one in prison?"

**Architecture:** TD v2 is a reasoning engine, NOT an NLP engine. A separate **preprocessing layer** handles messy human input before it reaches TD v2. Same pattern as ChatGPT o1/o3/R1: preprocess → reason → answer.

**What's done:**
- ✅ Prompt designed (v1, tested against v2 via Gemini API)
- ✅ Module skeleton created (`td/preprocessing/__init__.py`)
- ✅ LLM model selected (Gemini — cleanest output, fastest)
- ✅ `PREPROCESSING_PROMPT.md` documented with comparison results
- ✅ `QUERY_PREPROCESSING_PROMPT.md` created (v1 for queries, Gemini review confirmed v1 broken for queries)
- ✅ Coreference resolution implemented (spaCy two-pipeline, he/she/it/they + this/that)
- ✅ PREPROCESSING_PLAN.md written (three-layer architecture, research-backed)

**What's NOT done:**
- 🔲 Wire Gemini API call into `td/preprocessing/__init__.py`
- 🔲 Add Gemini API key to environment
- 🔲 Test with messy queries end-to-end
- 🔲 Integrate into `demos/chat_flare.py`
- 🔲 Multi-turn context ("what about prison?" → "cell in prison") — needs SessionContext

---

### 2. Sentence Simplification (REVERTED — needs proper fix)
The clause segmenter's `source_text` strips articles and loses subjects.
"a spring is a flexible elastic device" → "spring is flexible elastic device"
which the parser can't handle. Need to either:
- Fix clause segmenter source_text to preserve original text
- Or use a different approach (LLM simplification externally)
- Or detect when simplification would produce low-quality output and skip

### 2a. Parser Bugs Found (2026-07-10) — DONE ✅

**Bug 1: Missing copular attribute triple** ✅ Fixed
```
Input:    "a cell is a small room in a prison"
Extracted: (cell, room_in, prison)
Missing:   (cell, is_a, small room)
```
spaCy: cell=nsubj, is=ROOT, room=attr, small=amod(room)
Root cause: parser doesn't extract `attr` dependency as (subject, is_a, attr).
Fix: Emit `(subj, is_a, attr)` before `continue` in attr_preps branch
Reference: UD `attr` — nominal predicate of copular construction

**Bug 2: Adverb leaking into extraction** ✅ Fixed
```
Input:    "the prisoner was locked in a cell overnight"
Extracted: (prisoner, locked_in, cell) ← correct
Issue: "overnight" (advmod) shouldn't be in training data
```
Root cause: advmod tokens aren't filtered from triple extraction.
Fix: POS guard: only accept NOUN/PROPN advmod as subject
Reference: UD `advmod` — adverbs are NOT entities

**Bug 3: No triple from copular + reduced relative clause** ✅ Fixed
```
Input:    "mitochondria are organelles found in cells"
Extracted: (none)
Expected: (mitochondria, is_a, organelles)
```
spaCy: mitochondria=nsubj, are=ROOT, organelles=attr, found=acl(organelles)
Root cause: parser doesn't handle copular + `acl` (reduced relative clause).
Fix: Added prep chain handling for acl verbs + fallback is_a
Reference: TEA Nets (arXiv, Apr 2026) — cascading extraction

**Bug 4: No triple from passive voice with agent** ✅ Fixed
```
Input:    "the river bank was eroded by flooding"
Extracted: (none)
Expected: (flooding, eroded, river bank)
```
spaCy: bank=nsubjpass, eroded=ROOT, by=agent(eroded), flooding=pcomp(by)
Root cause: parser doesn't handle `agent` (by-phrase) in passive voice.
Fix: Accept `pcomp` alongside `pobj` in agent dependency
Reference: TEA Nets (2026) — nsubjpass + agent dep swap

**Additional fixes:**
- `is` → `is_a` canonicalization via `lang_config.copula_to_isa`
- Regex fallback: underscore-only guard (no magic numbers)
- `equality_signals` from `lang_config.relation_prototypes`
- BEAGLE weight 2.0 documented as calibration constant

**GLM 5.2 code review:** 2 actionable findings fixed (equality_signals, BEAGLE weight docs).

**Commits:** 4f5e98a, b0cce9c, 33c8e65, d1d1527, bd690bf, f4109b3, 4e3add7, c9f63bd

### 2b. Skipped Sentence Logging — DONE ✅
When spaCy can't extract triples, log the sentence for future parser improvement.
File: `data/skipped_sentences.log` (append-only)

---

## 📋 HIGH Priority

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

## ✅ DONE (2026-07-10)

| Item | Commit |
|------|--------|
| Query expansion + ranking fix | 57694fc |
| BEAGLE corpus scaling 10K→118K (2890 vocab, 12 domains) | efdbed4 |
| Language isolation (no hardcoded English in primary paths) | efdbed4, d1d1527 |
| YES formatting for yes/no questions | 128f87b |
| O(n²) pre-compute BEAGLE similarities | 490dc04 |
| BEAGLE weight research-backed (2.0) + documented | a88960c, c9f63bd |
| Parser bugs 1-4 fixed (copular, advmod, acl, passive) | 4f5e98a, b0cce9c |
| is→is_a canonicalization via lang_config.copula_to_isa | d1d1527 |
| Regex fallback: underscore-only guard (no magic numbers) | 4e3add7 |
| equality_signals from lang_config.relation_prototypes | c9f63bd |
| Skipped sentence logging (data/skipped_sentences.log) | f5bbdb2 |
| Layer 0 preprocessing architecture documented | 727321c, a78ea86 |
| Research review (16 new refs #56-71) | efdbed4 |
| ARCHITECTURE/DEVELOPMENT/TODO updated | efdbed4 |
| GLM 5.2 code review findings addressed | 90455e6, c9f63bd |
| All tests passing (54 passed, 0 failed) | 4e3add7 |

---

## ✅ DONE (2026-07-09 and earlier)

| Item | Commit |
|------|--------|
| Auto-load BEAGLE + spaCy similarity fallback | f7ac116 |
| Query pipeline — allow declarative sentences | 5e1ca59 |
| Copular handler — pcomp + coordination | 107e9da |
| Testing framework table format | 9ed6165 |
| Revert: remove broken simplification layer | a08d50e |
| Self-review: lemmatize bug + hardcoded articles | 1aefdda |
| teach() gloss quality — spaCy lemmatization | a8e0e3e |
| Duplicate triple extraction fix | e78eea3 |
| Wikipedia-sourced benchmark | 9807e91 |
| WSD routing fix (3 sense creation) | 8f982d7 |
| 4 parser bugs fixed (GLM 5.2 review) | b229ce9 |
| TESTING_FRAMEWORK.md created | 4aaffaf |
| Proper citations in ARCHITECTURE.md | 3dbdcc9 |
| Unknown word benchmarks (bat 90%, crane 100%, rock 91.7%) | 17acd7a |
| Contextualized Embeddings Spec (5 approaches) | 5e89d9a |
| 7 dead methods removed (-204 lines) | 265c87d |
| SpaCy dependency-based WSD routing | 7ab5a07 |
| Tiered WSD (BEAGLE + KG + LOTG) | 510f643 |

---

_"Fix the gloss, fix the world. Scale the corpus, scale the mind. Test with real-world, not just crafted queries."_
