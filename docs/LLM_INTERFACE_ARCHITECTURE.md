# TD v2 — LLM Interface Layer Architecture

**Date:** 2026-07-07
**Status:** PLANNED
**Reviewed by:** GLM 5.2 + Gemini (architecture review)

---

## Overview

Use an LLM as an interface layer to teach TD v2 from complex books/texts. The LLM handles language understanding; TD v2 handles deterministic reasoning.

```
Teaching:  Book → LLM (simplifier) → Simple triples → TD v2
Querying:  Question → LLM (reformulator) → TD v2 → LLM (explainer) → Answer
```

**Key insight:** TD v2 never sees raw text. The LLM breaks down complex sentences into simple (subject, relation, object) triples that TD v2 can store and reason over.

---

## Architecture

### Teaching Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Book/Text   │───▶│  Chunker     │───▶│  LLM         │───▶│  TD v2       │
│              │    │  (sections)  │    │  Simplifier   │    │  teach()     │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                            │
                                            ▼
                                    ┌──────────────┐
                                    │  Canonicalize │
                                    │  (EDC pattern)│
                                    └──────────────┘
```

**Step 1: Chunker** — Split book into sections (by headings, paragraphs)

**Step 2: LLM Simplifier** — For each chunk, extract simple triples:
```
Input:  "SQL Injection is a code injection technique that exploits 
         vulnerabilities in web applications. Use parameterized 
         queries to prevent it."

Output: SQL Injection is_a code injection attack
        SQL Injection exploits SQL vulnerabilities
        parameterized queries prevents SQL Injection
        SQL vulnerabilities is_a input validation weakness
```

**Step 3: Canonicalize (EDC pattern)** — Merge entity variants:
```
"SQL Injection" = "SQLi" = "SQL injection attacks" → SQL Injection
```

**Step 4: TD v2 teach()** — Store triples + derive new facts

### Query Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  User        │───▶│  LLM         │───▶│  TD v2       │───▶│  LLM         │
│  Question    │    │  Reformulator│    │  query()     │    │  Explainer   │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                            │
                                            ▼
                                    ┌──────────────┐
                                    │  Proof Trace  │
                                    └──────────────┘
```

**Step 1: LLM Reformulator** — Convert natural language to TD v2 query
```
"What attacks exploit input validation weaknesses?"
→ ask: what exploits input validation weakness?
```

**Step 2: TD v2 query()** — Execute query + return proof trace
```
SQL Injection → exploits → SQL vulnerabilities → is_a → input validation weakness
```

**Step 3: LLM Explainer** — Convert proof trace to human-readable answer
```
"SQL Injection exploits SQL vulnerabilities, which are a type of 
input validation weakness. So SQL Injection is one attack that 
exploits input validation weaknesses."
```

---

## Critical Risks (GLM 5.2 Review)

| # | Risk | Severity | Mitigation |
|---|------|----------|------------|
| R1 | **Ontological drift** — LLM invents inconsistent predicates ("is_a" vs "type_of") | 🔴 High | Constrained relation vocabulary + EDC canonicalization |
| R2 | **Entity resolution** — "SQL Injection" ≠ "SQLi" across chapters | 🔴 High | Canonical entity registry + post-extraction clustering |
| R3 | **Context loss in chunking** — coreference breaks across chunks | 🟡 Med | Chunk-aware coreference resolution |
| R4 | **Nuance degradation** — conditional knowledge doesn't fit flat triples | 🟡 Med | Reification for qualified facts |
| R5 | **Query syntax hallucination** — LLM generates invalid TD v2 queries | 🟡 Med | Schema-guided query generation |
| R6 | **Explainer hallucination** — LLM embellishes TD v2's factual output | 🔴 High | Strict grounding: LLM only rephrases proof trace, never adds facts |
| R7 | **Contradiction ingestion** — conflicting facts from different chapters | 🟡 Med | Source tracking + conflict detection |

---

## Research Papers — Sorted by Implementation Priority

### 🔴 Implement Now

#### 1. EDC: Extract, Define, Canonicalize (Zhang & Soh, 2024)
**arXiv:2404.03868 | EMNLP 2024**

**What it does:** Three-phase KG construction from text:
1. **Extract** — Open IE, LLM freely extracts S-P-O triples via few-shot prompting
2. **Define** — LLM generates natural-language definitions for each relation type it extracted
3. **Canonicalize** — Vector similarity finds semantically equivalent relations, LLM verifies before merging

**Key results:** Outperforms SOTA on WebNLG, REBEL, Wiki-NRE. Works without predefined schema (self-canonicalization) OR with a target schema (target alignment). No parameter tuning needed.

**TD v2 relevance: ★★★★★ CRITICAL**

**What to steal:**
- The `Define` phase — creates a self-documenting schema. We didn't have this. LLM generates definitions for each predicate, making the schema auditable.
- Vector similarity + LLM verification pattern for canonicalization (not just clustering)
- `EDC+R` refinement loop — re-run extraction with previous triples as context
- **Code:** github.com/clear-nus/edc — study their prompts

**How to implement:**
```python
# Phase 1: Extract (our existing simplifier)
triples = llm_extract(chunk)

# Phase 2: Define (NEW — from EDC)
definitions = llm_define(triples)  # "capital_of means X is the capital city of Y"

# Phase 3: Canonicalize (NEW — from EDC)
canonical_triples = canonicalize(triples, definitions)  # merge "SQLi" ↔ "SQL Injection"
```

---

#### 2. KGGen: Extracting KGs from Plain Text (Mo et al., 2025)
**arXiv:2502.09956**

**What it does:** Two-step extraction (entities first, then relations) + iterative LM-based entity clustering. Decomposes extraction into two sequential LLM calls to reduce cognitive load and error propagation.

**Key results:** Produces far denser, less redundant KGs than single-pass extraction. Entity clustering normalizes variations in tense, plurality, stemming, capitalization.

**TD v2 relevance: ★★★★☆ HIGH**

**What to steal:**
- **Two-step extraction pattern:** First extract all entities, then extract relations between those specific entities. Forces the LLM to be consistent about entity names.
- **Iterative clustering:** After extraction, LLM examines all nodes and edges to merge variants

**How to implement:**
```python
# Step 1: Extract entities only
entities = llm_extract_entities(chunk)
# → ["SQL Injection", "parameterized queries", "input validation"]

# Step 2: Extract relations between known entities
relations = llm_extract_relations(chunk, entities)
# → [("SQL Injection", "exploits", "input validation")]

# Step 3: Cluster entity variants
canonical_entities = llm_cluster_entities(entities)
# → {"SQL Injection": ["SQL Injection", "SQLi", "SQL injection attacks"]}
```

---

#### 3. MRKL Systems (Karpas et al., 2022)
**arXiv:2205.00445 | AI21 Labs**

**What it does:** Modular neuro-symbolic architecture: LLM + external knowledge sources + discrete reasoning modules. LLM = language interface. External modules = knowledge + reasoning.

**Key insight:** "LMs are inherently limited in a number of ways. We discuss these limitations and how they can be avoided by adopting a systems approach."

**TD v2 relevance: ★★★★★ CRITICAL (architecture precedent)**

**What to steal:**
- The architectural pattern: LLM routes to specialized modules, never reasons itself
- The "systems approach" philosophy — don't try to make one model do everything
- Our architecture IS a MRKL system. TD v2 = the discrete reasoning module

---

#### 4. Unifying LLMs and KGs: A Roadmap (Pan et al., 2023)
**arXiv:2306.08302 | IEEE TKDE**

**What it does:** Three frameworks: KG-enhanced LLMs, LLM-augmented KGs, Synergized LLMs+KGs.

**Key insight:** "KGs are difficult to construct and evolving by nature. It is complementary to unify LLMs and KGs together."

**TD v2 relevance: ★★★★★ CRITICAL (theoretical foundation)**

**What to steal:**
- We're in the "LLM-augmented KGs" category — LLMs help construct KGs
- The roadmap validates our approach: LLM for extraction, KG for reasoning
- The "Synergized" framework is our future: bidirectional reasoning

---

#### 5. LLM-empowered KG Construction Survey (Bian et al., 2025)
**arXiv:2510.20345**

**What it does:** Comprehensive survey of how LLMs reshape KG construction. Schema-based vs schema-free paradigms.

**Key insight:** Schema-based = constrained, consistent, but rigid. Schema-free = flexible, but needs post-hoc canonicalization. Our approach = hybrid (constrained relation vocabulary + EDC canonicalization).

**TD v2 relevance: ★★★★☆ HIGH (architecture validation)**

**What to steal:**
- The hybrid approach: constrained relations (schema-based) + entity flexibility (schema-free)
- The "dynamic knowledge memory for agentic systems" future direction

---

### 🟡 Next Feature

#### 6. Think-on-Graph 2.0 (Ma et al., 2024)
**arXiv:2407.10805**

**What it does:** Hybrid RAG: tight-couples KG-based and text-based retrieval. Iterative: graph retrieval → context retrieval → LLM reasoning.

**Key results:** SOTA on 6/7 knowledge-intensive datasets with GPT-3.5. Elevates Llama-2-13B to GPT-3.5 level. Training-free, plug-and-play.

**TD v2 relevance: ★★★★☆ HIGH (query pipeline enhancement)**

**What to steal:**
- **Tight-coupling pattern:** KG triples guide text retrieval, text enriches KG context
- **Iterative retrieval:** Don't stop at first result — keep retrieving until sufficient
- **Entity context:** Store rich descriptions alongside triples, not just S-P-O

**How to implement (Phase 3):**
```python
# Current: single-pass KG query
answer = td.query(question)

# ToG-2 style: iterative KG + text retrieval
for iteration in range(max_iterations):
    kg_triples = td.query(question)  # graph retrieval
    text_context = retrieve_docs(kg_triples)  # context retrieval
    answer = llm_reason(question, kg_triples, text_context)
    if answer.confident:
        break
```

---

#### 7. GraphRAG: From Local to Global (Edge et al., Microsoft Research, 2024)
**arXiv:2404.16130**

**What it does:** LLM builds entity KG from documents, partitions into communities via graph algorithms, generates hierarchical community summaries, answers queries via map-reduce over summaries.

**Key results:** Substantial improvements over vector RAG for "global sensemaking" questions. At 1M token scale, community summaries used 26-33% fewer tokens with <7% quality drop.

**TD v2 relevance: ★★★★★ CRITICAL (future feature)**

**What to steal:**
- **Community detection** over the knowledge graph — group related entities into clusters
- **Hierarchical summaries** — generate summaries at multiple levels of abstraction
- **Map-reduce query answering** — each community generates a partial answer, then synthesize

**How to implement (Phase 4):**
```python
# After teaching TD v2 from a book:
communities = detect_communities(td.kg)  # graph algorithm
for community in communities:
    community.summary = llm_summarize(community.entities)

# Query: "What are the main security vulnerabilities?"
answers = [c.summary for c in communities if c.relevant_to(query)]
final_answer = llm_synthesize(answers)
```

---

#### 8. QA-GNN (Yasunaga et al., 2021)
**arXiv:2104.06378 | NAACL 2021**

**What it does:** Joint reasoning: LLM scores KG node relevance, GNN updates representations. Connects QA context and KG into a joint graph.

**Key results:** Outperforms LM and LM+KG models on CommonsenseQA, OpenBookQA. Handles negation in questions.

**TD v2 relevance: ★★★☆☆ MEDIUM (relevance scoring)**

**What to steal:**
- **Relevance scoring:** Use LLM to estimate importance of KG nodes before querying. Could improve our open query ranking (currently TF-IDF).
- **Joint graph approach:** Connect query entities with KG subgraph, propagate relevance

---

### 🟢 Future

#### 9. AutoKG (Zhu et al., 2023)
**arXiv:2305.13168 | WWW Journal**

**What it does:** Multi-agent LLM + external sources for KG construction. VINE dataset for virtual knowledge extraction.

**Key finding:** "LLMs are more suited as inference assistants rather than few-shot information extractors." GPT-4 excels at reasoning tasks, surpassing fine-tuned models in certain cases.

**TD v2 relevance: ★★★☆☆ MEDIUM (benchmarking)**

**What to steal:**
- **VINE dataset** — benchmark for virtual knowledge extraction
- **AutoKG multi-agent pattern** — multiple LLMs collaborating on KG construction
- The finding validates our architecture: LLM extracts, TD v2 reasons

---

#### 10. Neuro-Symbolic AI: The State of the Art (Hitzler et al., 2022)

**What it does:** Taxonomy of neuro-symbolic approaches.

**TD v2 relevance: ★★☆☆☆ LOW (theoretical classification)**

**What to steal:**
- Our system = "decoupled neuro-symbolic with structured interface"
- Use this classification in papers/documentation

---

## Summary: What to Steal

| Paper | Key Steal | Implementation Phase |
|-------|-----------|---------------------|
| EDC | Define phase + EDC+R refinement | **Now** (teaching pipeline) |
| KGGen | Two-step extraction (entities → relations) | **Now** (teaching pipeline) |
| MRKL | Architecture precedent | **Now** (documentation) |
| Pan et al. | Theoretical foundation | **Now** (documentation) |
| ToG-2 | Iterative KG + text retrieval | **Next** (query pipeline) |
| GraphRAG | Community detection + map-reduce | **Next** (global queries) |
| QA-GNN | Relevance scoring for KG nodes | **Next** (ranking) |
| AutoKG | VINE benchmark + multi-agent | **Future** (benchmarking) |

---

## Implementation Plan

### Phase 1: Simplifier LLM (Week 1)

**Goal:** LLM breaks down book text into simple triples.

**Key design decisions (GLM 5.2 review):**
- **JSON schema enforcement** for LLM output (structured outputs / function calling)
- **Chunk overlap** (2+ sentences) to handle coreference at boundaries
- **Constrained relation vocabulary** (20-30 allowed relations max)
- **3-8 triples per paragraph** (not per sentence)
- **Log every triple** with source (page/chunk) for provenance tracking

```python
SIMPLIFIER_PROMPT = """Break down the following text into simple facts.
Each fact must be: (subject, relation, object)

Rules:
- Use ONLY these relations: is_a, exploits, prevents, mitigates, enables, requires, depends_on, contains, part_of, has_property, causes, produces, derives_from, contradicts
- Each fact must be a single sentence
- No pronouns — use full entity names
- No conjunctions — split into separate facts
- Output as JSON: [{"subject": "...", "relation": "...", "object": "..."}]

Text: {chunk}

Facts:"""
```

**Model:** Use a small local model (SLM) for cost:
- **Qwen 2.5 7B** — good at structured extraction, runs on MacBook
- **Phi-3 Mini** — fast, good at following instructions
- **Llama 3.1 8B** — strong at instruction following

### Phase 2: Canonicalization (Week 2)

**Goal:** Merge entity variants across chunks.

```python
CANONICALIZE_PROMPT = """Given these entity names from a knowledge graph:
{entity_list}

Group them by canonical name. Output JSON:
{{
  "SQL Injection": ["SQL Injection", "SQLi", "SQL injection attacks"],
  "XSS": ["Cross-Site Scripting", "XSS", "cross-site scripting"]
}}"""
```

### Phase 3: Query Pipeline (Week 3)

**Goal:** LLM reformulates questions + explains answers.

**Key design decisions (GLM 5.2 review):**
- **Retry loop** — if TD v2 returns parse error, feed back to Reformulator with error message (max 3 retries)
- **Grounded explainer** — "Answer using ONLY the provided data. If insufficient, say 'I don't know based on what I've been taught.'"
- **Provenance** — include source text chunk in explainer context (RAG-style enrichment)

```python
REFORMULATOR_PROMPT = """Convert this question into a TD v2 query.
Available relations: is_a, exploits, prevents, mitigates, enables, requires, depends_on, contains, part_of, has_property, causes, produces, derives_from, contradicts
Format: ask: what [relation] [entity]?
Or: ask: is [entity1] [relation] [entity2]?

Question: {question}
Query:"""

EXPLAINER_PROMPT = """Explain this reasoning chain in plain English.
You MUST use ONLY the information provided. Do NOT add any facts, 
examples, or context not explicitly stated in the chain.
If the chain is insufficient to fully answer, say "Based on what 
I've been taught, I can only tell you..."

Reasoning chain: {proof_trace}

Answer:"""
```

### Phase 4: Teaching UI (Week 4)

**Goal:** Interactive teaching session.

```
> teach-book Web_Application_Security_Guide.md

Processing chapter 1/12: SQL Injection...
  ✓ 15 facts taught, 3 derived
Processing chapter 2/12: Cross-Site Scripting...
  ✓ 12 facts taught, 5 derived
...

Total: 180 facts taught, 45 derived

> ask: what attacks exploit input validation weaknesses?
> SQL Injection, XSS, Command Injection (derived from transitive chains)

> explain: how does parameterized queries protect against SQL Injection?
> Parameterized queries prevent SQL Injection, which exploits SQL 
  vulnerabilities — a type of input validation weakness. So parameterized 
  queries indirectly protect against all input validation weaknesses.
```

---

## Key Design Decisions

### 1. Why LLM for simplification, not regex?

Regex patterns are brittle and English-only. LLMs handle:
- Complex sentence structures
- Implicit relations ("X prevents Y" from "to prevent X, use Y")
- Coreference resolution
- Multi-clause sentences

### 2. Why not LLM for reasoning?

LLMs hallucinate on multi-hop reasoning. TD v2:
- Derives facts it was never taught (transitive inference)
- Provides proof traces (interpretable)
- Never hallucinates (deterministic logic)
- <5ms latency on CPU

### 3. Why constrained relation vocabulary?

Without constraints, the LLM invents inconsistent predicates:
- "is_a" vs "type_of" vs "instance_of" vs "kind_of"
- "prevents" vs "stops" vs "blocks" vs "mitigates"

TD v2's inference rules work on **relation properties** (transitive, symmetric, functional). If the LLM uses inconsistent relations, the inference engine can't compose them.

**Solution:** Provide a fixed relation vocabulary in the prompt:
```
Available relations: is_a, exploits, prevents, mitigates, enables, 
requires, depends_on, contains, part_of, has_property
```

### 4. Why EDC canonicalization?

The EDC (Extract-Define-Canonicalize) pattern from Zhang & Soh (2024) is the research-backed approach:

1. **Extract:** Open IE — LLM freely extracts triples
2. **Define:** LLM generates definitions for each entity/relation
3. **Canonicalize:** Vector similarity + LLM verification to merge variants

This solves the entity resolution problem (R2) without hardcoding rules.

---

## Example: Teaching the Security Checklist

### Input (from Wikibooks)

```
## SQL Injection

SQL injection is a code injection technique used to attack data-driven 
applications, in which malicious SQL statements are inserted into an 
entry field for execution.

### Prevention
- Use parameterized queries (also known as prepared statements)
- Use stored procedures
- Validate all user-supplied input
- Escape all user-supplied input
- Limit database permissions
```

### LLM Simplifier Output

```
SQL Injection is_a code injection attack
SQL Injection targets data-driven applications
SQL Injection uses malicious SQL statements
parameterized queries prevents SQL Injection
stored procedures prevents SQL Injection
input validation prevents SQL Injection
input escaping prevents SQL Injection
least privilege mitigates SQL Injection
```

### TD v2 Derivations (never taught)

```
ask: what prevents data-driven applications from being attacked?
→ parameterized queries, stored procedures, input validation, input escaping
  (derived: these prevent SQL Injection, which targets data-driven applications)

ask: what attacks are prevented by input validation?
→ SQL Injection (derived: input_validation → prevents → SQL_injection)
```

---

## File Structure

```
td-v2/
├── td/
│   ├── llm/                          # NEW: LLM interface layer
│   │   ├── __init__.py
│   │   ├── simplifier.py             # Book → triples
│   │   ├── reformulator.py           # Question → TD v2 query
│   │   ├── explainer.py              # Proof trace → human answer
│   │   ├── canonicalizer.py          # EDC entity resolution
│   │   └── prompts.py                # Prompt templates
│   └── ...
├── demos/
│   ├── teach_book.py                 # NEW: Book teaching demo
│   └── ...
└── docs/
    └── LLM_INTERFACE_ARCHITECTURE.md # This file
```

---

## References

1. Zhang & Soh (2024). "Extract-Define-Canonicalize: An LLM-based Framework for Knowledge Graph Construction." arXiv:2404.03868
2. KGGen (Mo et al., 2025). "Extracting Knowledge Graphs from Plain Text with Language Models." arXiv:2502.09956
3. Min et al. (2025). "Towards Practical GraphRAG: Efficient Knowledge Graph Construction via Dependency Parsing." arXiv:2507.03226
4. Lewis et al. (2023). "Toolformer: Language Models Can Teach Themselves to Use Tools." arXiv:2302.04761
5. Khot et al. (2022). "Decomposed Prompting: A Modular Approach for Solving Complex Tasks." arXiv:2110.02160
6. Press et al. (2023). "Measuring and Narrowing the Compositionality Gap in Language Models." arXiv:2210.03350

---

_"The LLM speaks. The dust thinks. Together, they teach."_
