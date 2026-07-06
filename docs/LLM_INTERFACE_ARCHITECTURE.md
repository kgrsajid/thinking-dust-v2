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

## Research Papers to Cite

### Core Architecture

| # | Paper | Year | Relevance |
|---|-------|------|-----------|
| 1 | Pan et al., "Unifying Large Language Models and Knowledge Graphs: A Roadmap" | 2023 | Foundational survey. Maps our architecture: LLMs for KG construction, KGs for LLM reasoning |
| 2 | Zhang & Soh, "Extract-Define-Canonicalize" | 2024 | EDC pattern for entity/relation canonicalization |
| 3 | KGGen (Mo et al.) | 2025 | Iterative LM-guided clustering for KG refinement |
| 4 | Min et al., "Towards Practical GraphRAG" | 2025 | Dependency parsing = 94% of LLM KG extraction |
| 5 | Karpas et al., "MRKL Systems" | 2022 | LLM as router/interface for symbolic reasoning engines |
| 6 | Microsoft Research, "GraphRAG: Local and Global Understanding" | 2024 | LLMs extracting entity KGs from complex documents |

### Neuro-Symbolic Reasoning

| # | Paper | Year | Relevance |
|---|-------|------|-----------|
| 7 | Hitzler et al., "Neuro-Symbolic AI: The State of the Art" | 2022 | Taxonomy. Our system = "decoupled neuro-symbolic with structured interface" |
| 8 | Sun et al., "Think-on-Graph 2.0" | 2024 | LLM as agent iteratively reasoning over KG |
| 9 | Lewis et al., "Toolformer" | 2023 | LLM learns to use tools (symbolic reasoning) |
| 10 | Khot et al., "Decomposed Prompting" | 2022 | Breaking complex tasks into subtasks |

### Entity Resolution

| # | Paper | Year | Relevance |
|---|-------|------|-----------|
| 11 | Ding et al., "EntGPT" | 2025 | Two-phase entity alignment |
| 12 | Wang et al., "COMEM" | 2024 | Cascaded small+large LLM pipeline for entity fusion |
| 13 | arXiv:2510.20345, "LLM-empowered KG Construction: A Survey" | 2025 | Most recent comprehensive survey |

### KG Reasoning

| # | Paper | Year | Relevance |
|---|-------|------|-----------|
| 14 | Press et al., "Measuring and Narrowing the Compositionality Gap" | 2023 | Self-Ask: decomposing multi-hop questions |
| 15 | Yasunaga et al., "QA-GNN" | 2021 | Joint LLM+GNN reasoning |
| 16 | arXiv:2510.21425, "Advancing Symbolic Integration in LLMs" | 2025 | Survey of symbolic-integrated LLMs |

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
