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

**BEAGLE similarity code exists** at ~line 1690 in thinking.py — should match "used" to "tool_for" via word vector similarity. But either:
1. The BEAGLE threshold is too high (0.3)
2. The BEAGLE vectors don't have "used" or "tool" in vocabulary
3. The code path isn't reached

**Proposed fix:**
1. Debug BEAGLE similarity — check if "used" and "tool" have vectors, check similarity score
2. Lower threshold from 0.3 to 0.15 if needed
3. Also try matching query tokens against KG entity names (not just relations)
4. For single-entity + question mark queries, try ALL facts about that entity

**Files:** `td/thinking.py` — `_query_knowledge_graph` (line 1570)

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
Run the TESTING_FRAMEWORK.md pipeline on 10+ random Wikipedia words.

### 4. Persistent Teaching Pipeline
Each `teach()` should auto-save to RDF store + SQLite + Pickle.

---

## 📝 MEDIUM Priority

### 5. Clean Up Dead Methods
- `_move_triple_to_sense`, `_infer_type_from_fact`, `_check_type_conflict`, `_induce_senses_on_conflict`

### 6. God Object Decomposition
Split `GenericThinkingDust` and `KnowledgeGraph` into focused modules.

### 7. Dimensionality Reduction for Contextual Embeddings
Random project 10K→50 for NG-RC/AERC.

### 8. BSC-WSD Implementation
McInnes et al. (2012): HDC binary vectors, 94.55% WSD accuracy.

---

## ✅ DONE (Recent)

| Date | Item | Commit |
|------|------|--------|
| 2026-07-10 | Query pipeline — allow declarative sentences | 5e1ca59 |
| 2026-07-10 | Copular handler — pcomp + coordination + clean relations | 107e9da |
| 2026-07-10 | Testing framework table format | 9ed6165 |
| 2026-07-10 | Revert: remove broken simplification layer | a08d50e |
| 2026-07-09 | Self-review: lemmatize bug + hardcoded articles | 1aefdda |
| 2026-07-09 | teach() gloss quality — spaCy lemmatization | a8e0e3e |
| 2026-07-09 | Duplicate triple extraction fix | e78eea3 |

---

_"Fix the query, fix the score."_
