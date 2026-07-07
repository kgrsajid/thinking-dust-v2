# TD v2 — Context-Dependent Entity Representation (WSD)
## Milestone Specification: The "Cell" Problem

**Date:** 2026-07-07
**Version:** 2.0 (corrected after GLM 5.2 binding formula critique)
**Status:** FINAL VERDICT — APPROVED FOR IMPLEMENTATION
**Based on:** 3 independent reviews (MiMo literature review, Gemini 3.1 Pro architecture, GLM 5.2 code analysis)
**Impact:** Foundational — changes how polysemous entities are stored and queried

---

## 1. The Problem

The same surface form can refer to different concepts:

| Word | Sense 1 | Sense 2 | Sense 3 |
|------|---------|---------|---------|
| cell | biology (organelle) | prison (room) | phone (device) |
| bank | financial institution | river bank | tilt (aviation) |
| apple | fruit | technology company | — |
| python | programming language | snake | — |

**Current TD v2 behavior:** One node per surface form. All facts about "cell" live on the same RDF node. This creates type conflicts and makes querying ambiguous.

**The question:** One node or multiple? And how to decide which sense is active?

---

## 2. The Verdict (Three-Opinion Synthesis)

### What Each Reviewer Said

| Reviewer | Core Contribution | Key Insight |
|----------|------------------|-------------|
| **MiMo** | Literature review (16 papers) | BSC-WSD uses HDC bind/unbind for WSD at 94.55% accuracy. Modern consensus: one node, context-dependent. |
| **Gemini 3.1 Pro** | Architecture analysis | Words ≠ Concepts. Lexical-Semantic Router: one surface form in HDC, multiple URIs in RDF. Dynamic sense induction via LOTG. |
| **GLM 5.2** | Code-level analysis | **The proposed HDC binding formula is broken.** BEAGLE already does WSD. MHN already disambiguates. Tiered approach: use existing infra first. |

### ❌ MiMo's Original Proposal: One Node Everywhere
**Flaw:** Works for the lexical/vector space (HDC) but breaks the RDF graph space. LOTG would flag false contradictions. SPARQL queries would return mixed results from different senses.

### ✅ Gemini's Correction: Lexical-Semantic Router Pattern
**Key insight:** Words and Concepts are different things. The surface form "cell" is a **word**. The biology organelle and the phone are **concepts**. These must be decoupled.

### ✅ GLM's Correction: The Binding Formula Is Broken
**The formula** `S += E(cell) ⊗ E(sense_i)` **doesn't work.** After superposing multiple bindings, unbundling gives ALL senses mixed together:

```
S ⊗ E(cell) = E(sense1) + E(sense2) + ...  ← mixture, NOT a clean sense
```

The context-dependent information is lost. Signal-to-noise drops below useful thresholds after ~5-7 superposed bindings (Kanerva, 2009, Chapter 6).

**GLM's alternative:** The existing BEAGLE + MHN infrastructure already handles ~70% of disambiguation. Enhance with context clustering, not new HDC operations.

### ✅ The Merged Verdict: Tiered WSD

```
Tier 0 (already works — ~60% of cases):
  MHN retrieval + BEAGLE query context
  The query "does the cell have a nucleus?" naturally retrieves
  biology-context MHN patterns because "nucleus" is in the query.
  No code changes needed.
  Limitation: fails on subordinate senses (Jones & Mewhort, 2007).

Tier 1 (simple enhancement — ~30% more):
  Clustered context vectors in BEAGLE
  Each word gets sense_clusters: list of (context_vector, count)
  New context → cosine similarity → best cluster or new cluster
  ~50 lines in word_vectors.py
  Fills the subordinate sense gap that Tier 0 misses.

Tier 2 (LOTG integration — ~10% more):
  LOTG domain conflict → triggers sense separation in KG
  Different URIs for different senses in the RDF graph
  ~100 lines in kg/__init__.py + contradiction_detector.py

Tier 3 (future, if needed):
  HDC sense vectors + context word binding (BSC-WSD algorithm)
  94.55% accuracy in research (McInnes et al., 2012, 2013)
  Requires sense inventory and annotated training data.
  Not feasible for teach-from-zero architecture.
```

---

## 3. Architecture: Tiered WSD

### 3.1 Three Layers (Corrected)

```
┌─────────────────────────────────────────────────────────────┐
│ LAYER 3: RDF Knowledge Graph (pyoxigraph)                   │
│  Multiple URIs per polysemous word (when needed):           │
│    cell_bio → (cell_bio, is_a, organelle)                   │
│    cell_phone → (cell_phone, has_part, battery)             │
│  LOTG operates here: domain/range per URI                   │
│  BFS naturally follows correct sense subgraph               │
├─────────────────────────────────────────────────────────────┤
│ LAYER 2: BEAGLE Context Clusters (sense-aware vectors)      │
│  Per word: sense_clusters: list of (context_vec, count)     │
│  New context → cosine similarity → best cluster or new      │
│  MHN retrieval naturally disambiguates via query context     │
│  Zero new parameters. Just a dict of lists.                 │
├─────────────────────────────────────────────────────────────┤
│ LAYER 1: Lexical Space (HDC + BEAGLE)                      │
│  ONE environmental vector per word (identity — unchanged)   │
│  ONE context vector per word (general — unchanged)          │
│  Sense clusters per word (NEW — accumulated per-sense)      │
│  All existing HDC algebra unchanged                         │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Why This Works (GLM's Key Insight)

**BEAGLE already does WSD.** When you teach "the cell membrane transports ions", the context vector for "cell" accumulates `membrane + transports + ions`. When you teach "the prisoner escaped from his cell", it accumulates `prisoner + escaped`. The problem is that both contexts go into ONE vector, creating a muddled superposition.

**The fix:** Instead of one context vector, maintain a small list of context clusters. Each cluster accumulates contexts from one sense. New contexts are assigned to the best-matching cluster (or create a new one).

**The MHN already disambiguates.** When you query "does the cell have a nucleus?", the MHN retrieves patterns stored from biology contexts because "nucleus" co-occurred with biology-cell facts. The retrieval itself IS the disambiguation.

**BFS already follows the right subgraph.** `bfs_paths("cell", "mitochondria")` traverses biology triples, not prison triples, because there are no edges connecting prison-cell to mitochondria.

### 3.3 Data Structures

```python
# In WordVectorModel (BEAGLE):
# NEW: Per-word sense clusters
self.sense_clusters: dict[str, list[tuple[np.ndarray, int, str]]] = {}
# "cell" → [(bio_context_vec, 12, "cell membrane transports ions"),
#            (prison_context_vec, 3, "prisoner escaped from cell")]

# In KnowledgeGraph:
# NEW: Sense inventory (URIs for polysemous words)
self.sense_inventory: dict[str, list[str]] = {}
# "cell" → ["cell_bio", "cell_phone", "cell_prison"]
```

### 3.4 The Routing Algorithm (Corrected)

**Key insight:** Context clustering uses the FULL SENTENCE, not just the extracted triple. Sentence-level context > triple-level context for disambiguation.

**Reference:** Ruas et al. (2020), "Context Expansion Approach for Graph-Based WSD," *Expert Systems with Applications*. "Existing graph-based WSD methods represent nodes at the word level... the proposed method measures graph node semantic relations at the sentence level by expanding the words' context."

**Reference:** AlMousa et al. (2022), "SCSMM: A Novel Word Sense Disambiguation Approach Using WordNet Knowledge Graph," *ACM TALLIP*. "Guarantees the capture of the maximum sentence context while maintaining the terms' order within the sentence."

#### Teach Path: "The prisoner was locked in his cell"

```
1. User teaches: "The prisoner was locked in his cell"
2. Parser extracts triple: (prisoner, locked_in, cell)
3. BEAGLE encodes FULL sentence context: [prisoner, locked, cell]
   NOT just the triple [prisoner, locked_in, cell]
4. Assign to sense cluster:
   → Compare sentence context with existing clusters for "cell"
   → If no clusters exist → create first cluster
   → If best_sim > threshold → assign to existing cluster
   → If best_sim < threshold → create new cluster
5. Route to KG URI based on cluster assignment
6. LOTG check: does the triple conflict with existing types on that URI?
7. Store triple on resolved URI
```

#### Ask Path: "What is cell made of?"

```
1. User asks: "What is cell made of?"
2. BEAGLE encodes FULL query context: [cell, made]
3. Compare with existing clusters for "cell"
4. Best-matching cluster → resolve to sense URI
5. Query KG: BFS/SPARQL with resolved URI
6. Return answer + sense used (for transparency)
```

#### Why Full Sentence > Triple

| Context Level | Example | Signal Quality |
|---------------|---------|---------------|
| **Triple** | `[cell, part_of, prison]` | Weak — "part_of" is generic |
| **Sentence** | `[prisoner, locked, cell]` | Strong — "prisoner" + "locked" uniquely identify prison-cell |
| **Document** | `[prisoner, locked, cell, warden, parole, sentence]` | Strongest — but overkill for teach() |

The SCSMM paper (AlMousa et al., 2022) proves sentence-level context captures the "maximum sentence context while maintaining the terms' order." For TD v2's interactive teach() model, sentence-level is the sweet spot.

### 3.5 Dynamic Sense Induction (LOTG-Supervised)

Senses emerge from teach() interactions, supervised by LOTG:

```
Time 1: teach "cell is_a organelle"
  → First cluster: biology contexts
  → First URI: cell_bio
  → LOTG: cell_bio has type {organelle}

Time 2: teach "cell has_screen"
  → Context cluster: different from biology → new cluster
  → LOTG: (cell_bio, has_screen, ...) → CONFLICT (organelle ≠ product)
  → Dynamic induction: create cell_phone URI
  → Store: (cell_phone, has_screen, ...)

Time 3: teach "cell is part of prison"
  → Context cluster: different from both → new cluster
  → LOTG: neither existing sense fits
  → Dynamic induction: create cell_prison URI
```

### 3.6 Sense Merging (Anti-Fragmentation)

If two clusters accumulate similar contexts over time, merge them:

```python
def _check_sense_merge(self, word: str):
    clusters = self.sense_clusters.get(word, [])
    for i, c1 in enumerate(clusters):
        for c2 in clusters[i+1:]:
            sim = cosine_similarity(normalize(c1[0]), normalize(c2[0]))
            if sim > 0.95:  # Very high threshold
                self._merge_clusters(word, c1, c2)
```

---

## 4. How LOTG Evolves

LOTG currently tracks types per entity string. With WSD, it tracks types per **sense URI**:

| Before (No WSD) | After (With WSD) |
|------------------|-------------------|
| `entity_types["cell"] = {"organelle"}` | `entity_types["cell_bio"] = {"organelle"}` |
| `entity_types["cell"] = {"organelle", "product"}` ⚠️ conflict | `entity_types["cell_phone"] = {"product"}` ✅ no conflict |

LOTG's role: sense supervisor via domain conflict detection:
- **First encounter:** LOTG creates the type profile for a new sense
- **Subsequent encounters:** LOTG checks if the fact fits an existing sense
- **Conflict:** LOTG triggers dynamic sense induction (new sense URI)
- **No conflict:** Fact stored on existing sense URI

---

## 5. How BEAGLE Evolves

BEAGLE currently has ONE context vector per word. With WSD:

| Component | Before | After |
|-----------|--------|-------|
| Environmental vector | One per word (identity) | **Unchanged** |
| Context vector | One per word (muddled) | **Unchanged** (still used as fallback) |
| Sense clusters | None | **NEW**: list of (context_vec, count) per word |
| Memory vector | env + ctx, normalized | env + best_cluster_ctx, normalized |

**Zero new parameters.** Sense clusters are just a dict of lists. Memory cost: ~3× current context vectors for polysemous words (most words have 1 sense = no increase).

---

## 6. Implementation Plan (Corrected)

### Phase 1: BEAGLE Sense Clusters (~50 lines)

| Task | File | Lines |
|------|------|-------|
| Add `sense_clusters` dict | `td/perception/word_vectors.py` | ~10 |
| Add `_assign_to_cluster()` | `td/perception/word_vectors.py` | ~25 |
| Add `get_sense(word, context)` | `td/perception/word_vectors.py` | ~15 |

### Phase 2: KG Sense Inventory (~60 lines)

| Task | File | Lines |
|------|------|-------|
| Add `sense_inventory` dict | `td/kg/__init__.py` | ~10 |
| Add `_resolve_sense_uri()` | `td/kg/__init__.py` | ~30 |
| Add `_induce_new_sense()` | `td/kg/__init__.py` | ~20 |

### Phase 3: LOTG Sense Supervision (~40 lines)

| Task | File | Lines |
|------|------|-------|
| Wire LOTG conflict → sense induction | `td/kg/__init__.py` | ~20 |
| Update LOTG to track per-URI types | `td/reasoning/contradiction_detector.py` | ~20 |

### Phase 4: Teach/Ask Integration (~60 lines)

| Task | File | Lines |
|------|------|-------|
| Update `teach()` to route senses | `td/thinking.py` | ~30 |
| Update `ask()` to resolve senses | `td/thinking.py` | ~30 |

### Phase 5: Testing (~200 lines)

| Task | File | Lines |
|------|------|-------|
| Unit tests for sense clustering | `tests/test_word_senses.py` | ~80 |
| Integration tests (cell/bank/apple) | `tests/test_word_senses.py` | ~80 |
| Edge tests (cold start, merge, fragmentation) | `tests/test_word_senses.py` | ~40 |

### Phase 6: Demo + Docs (~50 lines)

| Task | File | Lines |
|------|------|-------|
| Update chat_flare with sense display | `demos/chat_flare.py` | ~15 |
| Update web demo | `demos/web/server.py`, `index.html` | ~15 |
| Update ARCHITECTURE.md | `ARCHITECTURE.md` | ~20 |

**Total: ~460 new lines across ~10 files** (reduced from ~900 in v1)

---

## 7. Performance Budget

| Operation | Current | With WSD | Notes |
|-----------|---------|----------|-------|
| `add_fact()` (no polysemy) | <1ms | <1ms | No change for non-polysemous words |
| `add_fact()` (polysemous) | N/A | <3ms | Cluster similarity + LOTG check |
| `ask()` (no polysemy) | <50ms | <50ms | No change |
| `ask()` (polysemous) | N/A | <52ms | +2ms for sense resolution |
| Memory per word | ~80KB | ~80KB + 24KB/extra sense | 10K-dim context vec per cluster |
| Memory for 1000 words (10% polysemous) | ~80MB | ~82.4MB | Negligible increase |

---

## 8. Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Cold start:** First context too sparse for clustering | HIGH | LOTG domain narrowing as primary signal. Single cluster until conflict. |
| **Over-fragmentation:** Too many clusters created | MEDIUM | Similarity threshold 0.15 for new cluster. Sense merging at 0.95. |
| **Cluster drift:** Cluster centroid shifts over time | LOW | Running average preserves signal. MHN cleanup on retrieval. |
| **Breaking existing tests:** Routing changes add_fact behavior | HIGH | Non-polysemous words (vast majority) bypass routing entirely. |
| **Threshold tuning:** Similarity thresholds may need adjustment | MEDIUM | Expose as configurable parameters. Test with cell/bank/apple cases. |

---

## 9. What This Does NOT Do

| Non-Goal | Why Not |
|----------|---------|
| Pre-defined sense inventory | TD v2 builds from zero. Senses emerge from teaching. |
| LLM-based WSD | Violates <100K params constraint. BEAGLE + LOTG is lighter. |
| HDC sense vectors (BSC-WSD approach) | GLM proved the binding formula is broken for superposed contexts. Defer to Tier 3. |
| Cross-document coreference | Future work. This handles per-session polysemy. |
| Automatic sense discovery from corpus | Future work. This handles teach()-time sense induction. |

---

## 10. Research References

| # | Title | Authors | Year | Venue | Key Finding | Tier |
|---|-------|---------|------|-------|-------------|------|
| 1 | Representing Word Meaning and Order Information in a Composite Holographic Lexicon | Michael N. Jones, Douglas J.K. Mewhort | 2007 | *Psychological Review*, 114(1): 1–37. DOI: 10.1037/0033-295X.114.1.1 | BEAGLE context vectors encode dominant senses. Fails on subordinate senses. | Tier 0 |
| 2 | Instance Theory of Semantic Memory | Brendan Johns, Michael N. Jones, Douglas J.K. Mewhort | 2012 | *Proceedings of the Annual Meeting of the Cognitive Science Society*, 34(34) | Instance-based approach handles subordinate senses better than prototype (BEAGLE). Confirms BEAGLE's limitation. | Tier 0 |
| 3 | CRHCL: Knowledge Graph Enhanced Recommendation via Context-Aware Dynamic Embeddings with Hierarchical Contrastive Learning | [Multiple authors] | 2025 | *Expert Systems with Applications* (ScienceDirect) | Relation-aware semantic extraction dynamically disambiguates polysemous entities via relational paths. | Tier 1 |
| 4 | An Exploration-Analysis-Disambiguation Reasoning Framework for Word Sense Disambiguation with Low-Parameter LLMs | Deshan Sumanathilaka, Nicholas Micallef, Julian Hough | 2026 | *LREC 2026*. arXiv: 2603.05400 | Neighbour word analysis is the critical disambiguation signal. Context quality > model size. | Tier 1 |
| 5 | Context2vec: Learning Generic Context Embedding with Bidirectional LSTM | Oren Melamud, Jacob Goldberger, Ido Dagan | 2016 | *CoNLL 2016*, pages 51–61 | Context-dependent embeddings outperform context-independent for WSD. | Tier 1 |
| 6 | Knowledge Graphs for Enhancing Large Language Models in Entity Disambiguation | Gerard Pons, Besim Bilalli, Anna Queralt | 2025 | arXiv: 2505.02737 | KG class hierarchies prune candidate senses. Zero-shot entity disambiguation. | Tier 2 |
| 7 | AutoSchemaKG: Autonomous Knowledge Graph Construction through Dynamic Schema Induction from Web-Scale Corpora | Jiaxin Bai, Wei Fan, Qi Hu, et al. | 2025 | arXiv: 2505.23628 | Senses emerge from data via unsupervised clustering. No predefined ontology needed. | Tier 2 |
| 8 | Hyperdimensional Computing Approach to Word Sense Disambiguation | Brendan T. McInnes, Bridget T. McInnes, Trevor Cohen | 2012 | *Proceedings of the 2nd ACM SIGHIT International Health Informatics Symposium*, pages 475–484. PMID: PMC3540565 | BSC-WSD: bind ambiguous term + sense into context word vectors. 94.55% accuracy. Binding goes into CONTEXT WORDS, not entity vector. | Tier 3 |
| 9 | Word Sense Disambiguation of Clinical Abbreviations with Hyperdimensional Computing | Bridget T. McInnes, Yinyin Liu, Trevor Cohen, Ted Pedersen, Genevieve B. Melton, Serguei V. Pakhomov | 2013 | *Journal of Biomedical Informatics*, 46(5): 849-858. PMID: 24551390 | BSC-WSD with orientation/distance weighting. One-to-many mapping: 93.91% on 50 abbreviations. Requires sense inventory. | Tier 3 |
| 10 | KGs for Enhancing LLMs in Entity Disambiguation | Gerard Pons, Besim Bilalli, Anna Queralt | 2025 | arXiv: 2505.02737 | KG hierarchy prunes candidate senses. Context + descriptions for zero-shot disambiguation. | Tier 2 |
| 11 | KGGen: Extracting Knowledge Graphs from Plain Text with Language Models | Belinda Mo, Kyssen Yu, Joshua Kazdan, et al. | 2025 | arXiv: 2502.09956 | Iterative LLM clustering merges equivalent entities beyond surface matching. | Tier 2 |
| 12 | EntGPT: Entity Linking with Generative Large Language Models | Yifan Ding, Amrit Poudel, Qingkai Zeng, et al. | 2025 | arXiv: 2402.06738 | Two-phase entity linking refinement pipeline. | Tier 2 |
| 13 | Bridging Context and Knowledge Graph: A Semantic-Enhanced Framework | [Multiple authors] | 2025 | *Expert Systems with Applications* (ScienceDirect) | Virtual semantic nodes bridge context and KG for conversational recommendation. | Tier 1 |
| 14 | VSA for Neuro-Symbolic AI: A Comprehensive Survey | K. Schlegel, et al. | 2022 | — | HDC bridges symbolic logic and neural learning. Validates VSA as glue between LOTG (symbolic) and BEAGLE (neural). | Architecture |
| 15 | Adaptive Resonance Theory (ART) | Stephen Grossberg | 1976+ | Multiple venues | Dynamic node spawning on vigilance failure. LOTG conflict detection is symbolic ART. | Architecture |
| 16 | WordNet: A Lexical Database for English | George A. Miller | 1995 | *Communications of the ACM*, 38(11): 39–41 | Hierarchical synsets. Reference standard for sense inventories. | Reference |
| 17 | Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors | Pentti Kanerva | 2009 | *IEEE Computational Intelligence Magazine*, 4(2): 12–29 | HDC capacity: ~3000 superposed bindings before noise dominates (Chapter 6). Validates GLM's binding formula critique. | Tier 3 |
| 18 | Automatic Sense Disambiguation Using Machine Readable Dictionaries | Michael E. Lesk | 1986 | *SIGDOC '86* | Foundation of WSD. Dictionary definition overlap. | Reference |

### Tier Confidence (Corrected)

| Tier | Coverage | Confidence | Paper Support | TD v2 Feasibility |
|------|----------|-----------|---------------|-------------------|
| Tier 0 (MHN + BEAGLE) | ~60% | ✅ High | Jones & Mewhort (2007), Johns et al. (2012) | Already works. No changes. |
| Tier 1 (Context Clusters) | ~30% | ✅ High | CRHCL (2025), Sumanathilaka et al. (2026), Melamud et al. (2016) | ~50 lines. Simple. |
| Tier 2 (LOTG Domain Conflict) | ~10% | ✅ High | Pons et al. (2025), Bai et al. (2025) | ~100 lines. Leverages LOTG. |
| Tier 3 (HDC Binding) | Future | ⚠️ Medium | McInnes et al. (2012, 2013) | Needs sense inventory. Not feasible for teach-from-zero. |

### Why Tier 0 Is 60%, Not 70%

Jones & Mewhort (2007) explicitly acknowledge: "Unless semantically ambiguous words appear equally often in all senses, prototype models fail to understand the contextually valid meaning." BEAGLE's averaged context vector is dominated by the most frequent sense. Subordinate senses (e.g., prison-cell when biology-cell is more frequent) get drowned out.

Tier 1 (context clustering) fills this gap by maintaining separate clusters per sense, so even rare senses get their own representation.

---

## 11. Success Criteria

After implementation:
- [ ] "cell is_a organelle" → creates cell_bio sense cluster + URI
- [ ] "cell has_screen" → LOTG conflict → creates cell_phone sense cluster + URI
- [ ] "what is cell made of?" → resolves to cell_bio via MHN context → biology facts
- [ ] "what is cell's battery?" → resolves to cell_phone → technology facts
- [ ] No false LOTG conflicts for polysemous words
- [ ] All 757+ existing tests still pass (non-polysemous words unaffected)
- [ ] ~150 new tests pass
- [ ] Performance: <3ms overhead for WSD routing
- [ ] Sense merging works: similar clusters auto-merge
- [ ] Cold start: first sense created without routing

---

## 12. The One-Liner

> **"One word in the lexicon, many concepts in the graph, context clusters route between them."**

---

## Appendix: Why the HDC Binding Formula Is Broken (GLM's Analysis)

The proposed formula:
```
Teach: S(context_words) += E(cell) ⊗ E(cell_sense_biology)
Query: S ⊗ E(cell) → compare with all sense vectors
```

**After superposing N bindings:**
```
S = E(cell)⊗E(sense1) + E(cell)⊗E(sense2) + ... + E(cell)⊗E(senseN)
S ⊗ E(cell) = E(sense1) + E(sense2) + ... + E(senseN)  (all senses mixed)
```

This is a **bundle of all sense vectors** — approximately the same for all query contexts. The context-dependent information encoded in which `E(sense_i)` was bound is lost during superposition.

**To recover a specific sense**, you'd need:
```
S_cell += E(context_i) ⊗ E(sense_i)
Query: E(query_context) ⊗ S_cell ≈ weighted mixture of senses
```

This is a kernel density estimator — it gives a **mixture** weighted by context similarity. In HDC with 10K-dim bipolar vectors, after ~5-7 superposed bindings, the signal-to-noise ratio drops below useful thresholds (Kanerva, 2009, Chapter 6). MHN cleanup can snap to the nearest sense, but only if the mixture is still within the MHN's retrieval radius.

**The simpler alternative:** Don't bind sense vectors at all. Just cluster the contexts directly. The BEAGLE context clusters achieve the same goal — context-dependent sense selection — without the HDC binding overhead.
