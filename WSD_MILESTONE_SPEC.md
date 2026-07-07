# TD v2 — Context-Dependent Entity Representation (WSD)
## Milestone Specification: The "Cell" Problem

**Date:** 2026-07-07
**Status:** FINAL VERDICT — APPROVED FOR IMPLEMENTATION
**Based on:** 3 independent reviews (MiMo literature review, Gemini 3.1 Pro architecture, GLM 5.2 partial)
**Impact:** Foundational — changes how every entity is stored and queried

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

### ❌ MiMo's Original Proposal: One Node Everywhere
**Flaw:** Works for the lexical/vector space (HDC) but breaks the RDF graph space. LOTG would flag false contradictions. SPARQL queries would return mixed results from different senses.

### ✅ Gemini's Correction: Lexical-Semantic Router Pattern
**The key insight:** Words and Concepts are different things. The surface form "cell" is a **word**. The biology organelle and the phone are **concepts**. These must be decoupled.

```
Lexical Space (HDC/BEAGLE):  ONE surface form → "cell"
Semantic Space (RDF/SPARQL): MULTIPLE URIs → cell_bio, cell_phone, cell_prison
Router (HDC Context Binding): Maps surface form → correct URI based on context
```

### ✅ MiMo's Literature Review Confirms
Every major 2025-2026 paper converges on the same pattern:
- **CRHCL (2025):** Dynamic entity embeddings based on relational paths
- **BSC-WSD (2012):** HDC bind/unbind for sense recovery (94.55% accuracy)
- **Pons et al. (2025):** KG class hierarchies prune candidate senses
- **EAD Framework (2026):** Neighbor word analysis as disambiguation signal
- **AutoSchemaKG (2025):** Senses emerge from data, not predefined

---

## 3. Architecture: The Lexical-Semantic Router

### 3.1 Three Layers

```
┌─────────────────────────────────────────────────────────────┐
│ LAYER 3: RDF Knowledge Graph (pyoxigraph)                   │
│  Multiple URIs per polysemous word:                         │
│    cell_bio → (cell_bio, is_a, organelle)                   │
│    cell_phone → (cell_phone, has_part, battery)             │
│  LOTG operates here: domain/range per URI                   │
├─────────────────────────────────────────────────────────────┤
│ LAYER 2: Sense Router (HDC Context Binding)                 │
│  BEAGLE context vector ↔ Sense vector matching              │
│  MHN cleanup: snap noisy vector to nearest sense            │
│  LOTG domain narrowing: only consider domain-consistent     │
│  Dynamic sense induction: new sense auto-created on         │
│  LOTG conflict                                              │
├─────────────────────────────────────────────────────────────┤
│ LAYER 1: Lexical Space (HDC + BEAGLE)                      │
│  ONE surface form vector per word: E("cell")                │
│  ONE context vector per word: C("cell") accumulates         │
│  Sense elemental vectors: E_sense_1, E_sense_2, ...         │
│  All HDC algebra lives here                                 │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Data Structures

```python
# NEW: Sense Inventory
# Maps surface form → list of (sense_id, sense_vector, kg_uri, domain)
@dataclass
class WordSense:
    sense_id: str              # "cell_bio", "cell_phone"
    sense_vector: np.ndarray   # 10K-dim random elemental vector
    kg_uri: str                # URI in the RDF graph
    domain: str                # "biology", "technology", etc.
    context_signature: np.ndarray  # Accumulated context vector for this sense

# On the KnowledgeGraph class:
self.sense_inventory: dict[str, list[WordSense]] = {}  # "cell" → [WordSense, ...]
self._sense_memory: dict[str, np.ndarray] = {}          # "cell" → bundled sense-context HDC
```

### 3.3 The Routing Algorithm

#### Teach Path: "cell is_a organelle"

```
1. Parse: entities = ["cell"], relation = "is_a", object = "organelle"
2. Generate context vector: C = BEAGLE(["cell", "is_a", "organelle"])
3. Route to sense:
   a. If "cell" has no senses yet:
      → Create first sense: cell_bio, random vector S_1
      → Create RDF URI: td:entity/cell_bio
      → Store: (cell_bio, is_a, organelle) in KG
      → Bind: M_cell += C ⊗ S_1
   b. If "cell" has existing senses:
      → Unbind: V_retrieved = M_cell ⊗ C
      → MHN cleanup: snap to nearest S_i
      → Check LOTG: does (cell_X, is_a, organelle) conflict?
      → If no conflict: store on existing URI
      → If conflict: CREATE NEW SENSE (dynamic induction)
        → New vector S_j, new URI cell_new
        → Bind: M_cell += C ⊗ S_j
        → Store on new URI
4. Update BEAGLE context vectors (existing online learning)
```

#### Ask Path: "what is cell made of?"

```
1. Parse: entities = ["cell"], relation = "made_of"
2. Generate query context: C_query = BEAGLE(["cell", "made_of"])
3. Route to sense:
   → Unbind: V_retrieved = M_cell ⊗ C_query
   → MHN cleanup: snap to nearest S_i
   → Resolve: S_i → kg_uri (e.g., cell_bio)
4. Query KG: SPARQL with resolved URI
5. Return answer + sense used (for transparency)
```

### 3.4 Dynamic Sense Induction

Senses are NOT predefined. They emerge from teach() interactions, supervised by LOTG:

```
Time 1: teach "cell is_a organelle"
  → First sense: cell_bio (domain: biology)

Time 2: teach "cell has_screen"
  → Route: context similar to cell_bio? NO
  → LOTG check: (cell_bio, has_screen, ...) → domain conflict!
  → Dynamic induction: create cell_phone (domain: technology)
  → New URI: td:entity/cell_phone
  → Store: (cell_phone, has_screen, ...)

Time 3: teach "cell is part of prison"
  → Route: context similar to cell_bio? NO. cell_phone? NO.
  → LOTG check: neither existing sense fits.
  → Dynamic induction: create cell_prison (domain: location)
  → New URI: td:entity/cell_prison
```

### 3.5 Sense Merging (Anti-Fragmentation)

If two senses accumulate similar context signatures over time, they should merge:

```python
def _check_sense_merge(self, word: str):
    """If two senses of the same word have very similar context vectors, merge them."""
    senses = self.sense_inventory.get(word, [])
    for i, s1 in enumerate(senses):
        for s2 in senses[i+1:]:
            similarity = cosine_similarity(s1.context_signature, s2.context_signature)
            if similarity > 0.95:  # Very high threshold
                self._merge_senses(word, s1, s2)
```

---

## 4. How LOTG Evolves

LOTG currently tracks types per entity string. With WSD, it tracks types per **sense URI**:

| Before (No WSD) | After (With WSD) |
|------------------|-------------------|
| `entity_types["cell"] = {"organelle"}` | `entity_types["cell_bio"] = {"organelle"}` |
| `entity_types["cell"] = {"organelle", "product"}` ⚠️ conflict | `entity_types["cell_phone"] = {"product"}` ✅ no conflict |

LOTG's role changes from "contradiction detector" to "sense supervisor":
- **First encounter:** LOTG creates the type profile for a new sense
- **Subsequent encounters:** LOTG checks if the fact fits an existing sense
- **Conflict:** LOTG triggers dynamic sense induction (new sense creation)
- **No conflict:** LOTG routes to the existing sense

---

## 5. How BEAGLE Evolves

BEAGLE currently has ONE context vector per word. With WSD, it has:
- **One environmental vector per word** (unchanged — identity)
- **One context vector per word** (unchanged — general context)
- **One sense-specific context vector per sense** (NEW — accumulates context for each sense)

The sense-specific context vectors enable fine-grained similarity matching:
```
C_cell_bio = context of "cell" when used in biology
C_cell_phone = context of "cell" when used in technology
```

---

## 6. Implementation Plan

### Phase 1: Sense Infrastructure (Week 1)

| Task | File | Lines |
|------|------|-------|
| Add `WordSense` dataclass | `td/perception/word_senses.py` | ~50 |
| Add sense inventory to KG | `td/kg/__init__.py` | ~30 |
| Add sense vectors (HDC) | `td/perception/hdc.py` | ~20 |
| Add `_sense_memory` dict | `td/kg/__init__.py` | ~10 |

### Phase 2: Sense Router (Week 1-2)

| Task | File | Lines |
|------|------|-------|
| Implement `route_to_sense()` | `td/perception/word_senses.py` | ~80 |
| Implement `induce_new_sense()` | `td/perception/word_senses.py` | ~40 |
| Implement `cleanup_sense_vector()` via MHN | `td/memory/mhn.py` | ~20 |
| Wire router into `add_fact()` | `td/kg/__init__.py` | ~30 |

### Phase 3: Teach/Ask Integration (Week 2)

| Task | File | Lines |
|------|------|-------|
| Update `teach()` to route senses | `td/thinking.py` | ~30 |
| Update `ask()` to resolve senses | `td/thinking.py` | ~30 |
| Update SPARQL queries to use resolved URIs | `td/query/__init__.py` | ~20 |
| Add sense info to proof traces | `td/kg/__init__.py` | ~10 |

### Phase 4: Sense Management (Week 2-3)

| Task | File | Lines |
|------|------|-------|
| Implement sense merging | `td/perception/word_senses.py` | ~40 |
| Add sense statistics to `stats()` | `td/thinking.py` | ~10 |
| Add `list_senses(word)` API | `td/perception/word_senses.py` | ~20 |

### Phase 5: Testing (Week 3)

| Task | File | Lines |
|------|------|-------|
| Unit tests for sense router | `tests/test_word_senses.py` | ~150 |
| Integration tests (cell/bank/apple) | `tests/test_word_senses.py` | ~100 |
| Edge tests (cold start, merge, fragmentation) | `tests/test_word_senses.py` | ~80 |

### Phase 6: Demo + Docs (Week 3)

| Task | File | Lines |
|------|------|-------|
| Update chat_flare with sense display | `demos/chat_flare.py` | ~20 |
| Update web demo | `demos/web/server.py`, `index.html` | ~30 |
| Update ARCHITECTURE.md | `ARCHITECTURE.md` | ~200 |

**Total: ~900 new lines across ~15 files**

---

## 7. Performance Budget

| Operation | Current | With WSD | Notes |
|-----------|---------|----------|-------|
| `add_fact()` (no WSD) | <1ms | <1ms | No change for non-polysemous words |
| `add_fact()` (WSD active) | N/A | <5ms | HDC bind + MHN cleanup + LOTG check |
| `ask()` (no WSD) | <50ms | <50ms | No change |
| `ask()` (WSD active) | N/A | <55ms | +5ms for sense resolution |
| Memory per word | ~80KB | ~80KB + 40KB/sense | 10K-dim vector per sense |
| Memory for 1000 words (10% polysemous) | ~80MB | ~84MB | Negligible increase |

---

## 8. Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Cold start:** Early contexts too sparse for disambiguation | HIGH | LOTG domain narrowing as primary signal. HDC similarity as secondary. |
| **Over-fragmentation:** Too many senses created | MEDIUM | Sense merging at 0.95 similarity threshold. Minimum context count before inducing new sense. |
| **HDC noise floor:** Bundling too many contexts degrades retrieval | LOW | 10K dims can safely bundle ~3000 vectors. Monitor and warn at 80% capacity. |
| **Breaking existing tests:** New routing changes add_fact behavior | HIGH | Non-polysemous words (vast majority) bypass the router entirely. Only activates when LOTG detects a conflict. |
| **URI proliferation:** Too many URIs for the same word | LOW | Sense merging + garbage collection for unused senses. |

---

## 9. What This Does NOT Do

| Non-Goal | Why Not |
|----------|---------|
| Pre-defined sense inventory | TD v2 builds from zero. Senses emerge from teaching. |
| LLM-based WSD | Violates <100K params constraint. HDC+BEAGLE is lighter. |
| Cross-document coreference | Future work. This handles per-session polysemy. |
| Sense inheritance across sessions | Future work. Sense inventory persisted in RDF. |
| Automatic sense discovery from corpus | Future work. This handles teach()-time sense induction. |

---

## 10. Research References

| # | Paper | Year | Venue | Key Contribution |
|---|-------|------|-------|-----------------|
| 1 | **CRHCL** — Context-aware Relation Modeling + Hierarchical CL | 2025 | ScienceDirect | Dynamic entity disambiguation via relational paths |
| 2 | **BSC-WSD** — HDC Approach to WSD | 2012 | PMC/Biomedinformatics | HDC bind/unbind for sense recovery (94.55% acc) |
| 3 | **Pons et al.** — KGs for Entity Disambiguation | 2025 | arXiv:2505.02737 | KG hierarchy prunes candidate senses |
| 4 | **EAD Framework** — Exploration-Analysis-Disambiguation | 2026 | LREC (arXiv:2603.05400) | Neighbor word analysis > model size for WSD |
| 5 | **AutoSchemaKG** — Dynamic Schema Induction | 2025 | arXiv:2505.23628 | Senses emerge from data, not predefined |
| 6 | **KGGen** — Iterative LLM Clustering | 2025 | arXiv:2502.09956 | Semantic grouping beyond surface matching |
| 7 | **EntGPT** — Entity Linking with LLMs | 2025 | arXiv:2402.06738 | Two-phase entity linking refinement |
| 8 | **BRICR** — Semantic-Enhanced KG | 2025 | ScienceDirect | Virtual semantic nodes for context bridging |
| 9 | **Context2Vec** — Context-Dependent Embeddings | 2016 | ACL | Bidirectional LSTM for context modeling |
| 10 | **GrapHD** — Graph Structures in HDC | 2024 | — | Encoding RDF directly into HDC vectors |
| 11 | **VSA for Neuro-Symbolic AI** (Schlegel et al.) | 2022 | — | Survey: HDC bridges symbolic logic and neural learning |
| 12 | **Adaptive Resonance Theory** (Grossberg) | 1976+ | — | Dynamic node spawning on vigilance failure |
| 13 | **WordNet** — Sense Inventory | 1995 | CACM | Hierarchical synsets (reference standard) |
| 14 | **Wikidata QIDs** — Unique Entity IDs | 2012+ | Wikidata | Production solution: QID per sense |
| 15 | **Lesk** — Automatic Sense Disambiguation | 1986 | SIGDOC | Foundation of WSD — dictionary overlap |
| 16 | **OWL 2** — disjointWith, subClassOf | 2009 | W3C | Type hierarchy and disjointness axioms |

---

## 11. Success Criteria

After implementation:
- [ ] "cell is_a organelle" → creates cell_bio sense, stores on cell_bio URI
- [ ] "cell has_screen" → LOTG conflict detected → creates cell_phone sense
- [ ] "what is cell made of?" → resolves to cell_bio based on context → returns biology facts
- [ ] "what is cell's battery?" → resolves to cell_phone → returns technology facts
- [ ] No false LOTG conflicts for polysemous words
- [ ] All 757+ existing tests still pass (non-polysemous words unaffected)
- [ ] ~250 new tests pass
- [ ] Performance: <5ms overhead for WSD routing
- [ ] Sense merging works: similar senses auto-merge
- [ ] Cold start: first sense created without routing

---

## 12. The One-Liner

> **"One word in the lexicon, many concepts in the graph, HDC routes between them."**
