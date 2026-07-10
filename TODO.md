# TD v2 — TODO List

_Last updated: 2026-07-10_

---

## ✅ MAJOR MILESTONE: Query Pipeline Fixed (2026-07-10)

**Before:** 25% accuracy on benchmark queries
**After:** 100% accuracy on benchmark queries

**What was done:**
- BEAGLE corpus scaling: 10K → 118K sentences (2890 vocab, 12 domains)
- Query expansion via BEAGLE nearest neighbors
- Collect-all-then-rank (no more short-circuit on first match)
- YES formatting for yes/no questions
- O(n²) fix: pre-compute BEAGLE similarities
- Language isolation: no hardcoded English in core logic
- 16 new research references (query expansion, HDC scaling, KG QA)

**Caveat:** Benchmark uses clean, crafted queries. Real-world performance unknown.

---

## 🚨 URGENT (RIGHT NOW)

### 1. Real-World Query Handling — THE next bottleneck

**Problem:** We tested with clean queries like "what is a match used for". Real users say:
- "I was wondering, what do you use a match for when starting a fire?"
- "So like, that thing you strike to make fire, what's it called?"
- "The cell in biology, what's it made of vs the one in prison?"

**Architecture:** TD v2 is a reasoning engine, NOT an NLP engine. A separate **preprocessing layer** handles messy human input before it reaches TD v2. Same pattern as ChatGPT o1/o3/R1: preprocess → reason → answer.

**Bootstrap approach:** Use an LLM for preprocessing during demo/development:
```
LLM prompt: "Simplify this query for a knowledge graph engine. 
Remove filler, resolve anaphora, extract intent. 
Query: '{user_input}'
Output: clean, structured query."
```
This lets us focus on the core reasoning engine. Production can use rule-based + spaCy.

**What's missing:**
- **Preprocessing layer** — LLM-based (bootstrap) or rule-based + spaCy (production)
- **Anaphora resolution** — "that thing" → "match" (needs context)
- **Multi-turn context** — "what about prison?" → resolve to "cell in prison"

**Research backing:**
- GraphRAG (Min et al., 2025) — sentence simplification improves KG extraction
- UDASTE (2023) — complex sentences are primary source of extraction errors

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
- Emit `(subj, is_a, attr)` before `continue` in attr_preps branch
- Reference: UD `attr` — nominal predicate of copular construction

**Bug 2: Adverb leaking into extraction** ✅ Fixed
- POS guard: only accept NOUN/PROPN advmod as subject
- Reference: UD `advmod` — adverbs are NOT entities

**Bug 3: No triple from copular + reduced relative clause** ✅ Fixed
- Added prep chain handling for acl verbs + fallback is_a
- Reference: TEA Nets (arXiv, Apr 2026) — cascading extraction

**Bug 4: No triple from passive voice with agent** ✅ Fixed
- Accept `pcomp` alongside `pobj` in agent dependency
- Reference: TEA Nets (2026) — nsubjpass + agent dep swap

**Additional fixes:**
- `is` → `is_a` canonicalization via `lang_config.copula_to_isa`
- Regex fallback: underscore-only guard (no magic numbers)
- `equality_signals` from `lang_config.relation_prototypes`
- BEAGLE weight 2.0 documented as calibration constant

**GLM 5.2 code review:** 2 actionable findings fixed (equality_signals, BEAGLE weight docs). `_extract_triples_regex()` hardcoded English documented as "English-only fallback".

**Commits:** 4f5e98a, b0cce9c, 33c8e65, d1d1527, bd690bf, f4109b3, 4e3add7, c9f63bd

### 2b. Skipped Sentence Logging
When spaCy can't extract triples, log the sentence for future parser improvement.
Track: sentence, reason (no_triple, complex_structure, passive_voice, etc.)
File: `data/skipped_sentences.log` (append-only)

**Bug 1: Missing copular attribute triple**
```
Input:    "a cell is a small room in a prison"
Extracted: (cell, room_in, prison)
Missing:   (cell, is_a, small room)
```
spaCy: cell=nsubj, is=ROOT, room=attr, small=amod(room)
Root cause: parser doesn't extract `attr` dependency as (subject, is_a, attr).

**Bug 2: Adverb leaking into extraction**
```
Input:    "the prisoner was locked in a cell overnight"
Extracted: (prisoner, locked_in, cell) ← correct
Issue: "overnight" (advmod) shouldn't be in training data
```
Root cause: advmod tokens aren't filtered from triple extraction.

**Bug 3: No triple from copular + reduced relative clause**
```
Input:    "mitochondria are organelles found in cells"
Extracted: (none)
Expected: (mitochondria, is_a, organelles)
```
spaCy: mitochondria=nsubj, are=ROOT, organelles=attr, found=acl(organelles)
Root cause: parser doesn't handle copular + `acl` (reduced relative clause).

**Bug 4: No triple from passive voice with agent**
```
Input:    "the river bank was eroded by flooding"
Extracted: (none)
Expected: (flooding, eroded, river bank)
```
spaCy: bank=nsubjpass, eroded=ROOT, by=agent(eroded), flooding=pcomp(by)
Root cause: parser doesn't handle `agent` (by-phrase) in passive voice.
Reference: TEA Nets (arXiv, Apr 2026) — nsubjpass + agent dep swap.

**Status:** GLM 5.2 subagent investigating. Fixes needed in `td/perception/nl_parser.py`.

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
| BEAGLE corpus scaling 10K→118K | efdbed4 |
| Language isolation (no hardcoded English) | efdbed4 |
| YES formatting for yes/no questions | 128f87b |
| O(n²) pre-compute BEAGLE similarities | 490dc04 |
| BEAGLE weight research-backed (2.0) | a88960c |
| All GLM 5.2 review findings fixed | 90455e6 |
| Research review (16 new refs #56-71) | efdbed4 |
| ARCHITECTURE/DEVELOPMENT/TODO updated | efdbed4 |

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
