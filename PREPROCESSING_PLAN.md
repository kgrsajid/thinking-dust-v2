# Preprocessing Layer — Three-Layer Architecture Plan

_Last updated: 2026-07-12_

---

## Overview

TD v2 is a **reasoning engine**, not an NLP engine. A preprocessing layer handles messy human input before it reaches `think()` or `teach()`. This document defines a three-layer preprocessing architecture that minimizes LLM dependency while maximizing coverage.

```
User Input (messy)
    ↓
Layer A: Rule-Based (spaCy) — ~0ms, handles ~70% of cases
    ↓
Layer B: Small Model (T5-small, 60M params) — ~50ms, handles ~20% more
    ↓
Layer C: Confidence Gate — decides if output is clean enough for TD v2
    ↓
Clean sentences → TD v2 think() / teach()
```

**Design principle:** Each layer is a pure function. Input: messy string. Output: list of clean sentences. If Layer A's output passes the confidence check, Layers B and C are never invoked.

---

## Current State

### What's Already Built

| Component | Status | File |
|-----------|--------|------|
| spaCy clause segmentation | ✅ Production | `td/perception/clause_segmenter.py` |
| spaCy dependency extraction | ✅ Production | `td/perception/nl_parser.py` |
| Coreference resolution (he/she/it/they) | ✅ Production | `td/perception/nl_parser.py` (enable_coreference) |
| Discourse deixis filtering (this/that + abstract verbs) | ✅ Production | `td/perception/nl_parser.py` (Jauhar et al. 2015) |
| Relative clause attachment (acl:relcl) | ✅ Production | `td/perception/clause_segmenter.py` |
| Passive voice extraction (nsubjpass + agent) | ✅ Production | `td/perception/nl_parser.py` |
| Coordination expansion (conj chains) | ✅ Production | `td/perception/clause_segmenter.py` |
| Preprocessing prompt v1 (Gemini) | ✅ Designed | `PREPROCESSING_PROMPT.md` |
| Preprocessing module skeleton | ✅ Created | `td/preprocessing/__init__.py` |

### What's NOT Built

| Component | Status | Priority |
|-----------|--------|----------|
| Filler word removal (rule-based) | 🔲 Not built | P0 |
| Rule-based query normalization | 🔲 Not built | P0 |
| T5-small sentence splitting model | 🔲 Not built | P1 |
| Confidence gate (is output clean SVO?) | 🔲 Not built | P1 |
| Multi-turn context resolution | 🔲 Not built | P2 |
| Gemini API integration | 🔲 Not wired | P0 |

---

## Layer A: Rule-Based Preprocessing (spaCy)

### What It Does

Transforms messy input into simpler sentences using deterministic rules. No model inference. Pure spaCy dependency parsing + regex.

### Capabilities

| Operation | How | Research Backing |
|-----------|-----|-----------------|
| Filler removal | Regex + lang_config | Standard NLP preprocessing |
| Coordination splitting | spaCy `conj` dependency | Sahaj Software (2023), Manning & Schütze (1999) Ch. 5 |
| Subject propagation | spaCy `nsubj` walking across conj chains | Kamana et al. (arXiv:2601.00506, Jan 2026) |
| Relative clause detection | spaCy `acl:relcl` dependency | Universal Dependencies (Nivre et al. 2016) |
| Passive to active | spaCy `nsubjpass` + `agent` | TEA Nets (arXiv:2604.27673, Apr 2026) |
| Prepositional phrase attachment | spaCy `prep` + `pobj` dependency | Stanford Dependencies (de Marneffe et al. 2014) |
| Demonstrative filtering | spaCy POS + lang_config | Jauhar et al. (*SEM 2015) |

### Research Foundation

**Primary reference:** Kamana et al. (January 2026). "Rule-Based Approaches to Atomic Sentence Extraction." arXiv:2601.00506.

- Used spaCy `en_core_web_sm` dependency parser for rule-based sentence decomposition
- Tested on WikiSplit dataset with 100 gold-standard atomic sentence sets
- Results: ROUGE-1 F1=0.6714, ROUGE-2 F1=0.478, ROUGE-L F1=0.650, BERTScore F1=0.5898
- **Key finding:** "Rule-based extraction is reasonably accurate but sensitive to syntactic complexity"
- **Handles well:** Simple SVO, basic coordination, subject propagation
- **Struggles with:** Relative clauses, appositions, coordinated predicates, adverbial clauses

**What this means for TD v2:** Layer A doesn't need to be perfect. TD v2's parser already handles the hard cases (relative clauses, passive voice, coordination). Layer A just needs to reduce complexity enough that the parser can extract triples.

**Supporting references:**
- Sahaj Software (2023): "Knowledge graphs from complex text" — verb-based sentence splitting via spaCy dependency tree
- Manning & Schütze (1999): "Foundations of Statistical NLP", Chapter 5 — coordinated noun phrase extraction
- Min et al. (2025): "Towards Practical GraphRAG" — spaCy dependency parsing achieves 94% of LLM-based KG extraction

### Implementation

```python
# td/preprocessing/rule_based.py

import re
from typing import list

# Filler patterns — loaded from lang_config in production
FILLER_PATTERNS = [
    r'\b(so|like|you know|I mean|basically|actually|well|right)\b,?\s*',
    r'^(I was wondering|can you tell me|tell me|what\'s the deal with)\s*',
    r'^(do you know|I\'m curious about|I want to know)\s*',
    r'^(hey|hi|hello|um|uh|ah),?\s*',
]

# Question intent patterns
QUERY_PATTERNS = [
    (r'what(?:\'s| is) the deal with (.+)', r'what is \1'),
    (r'can you tell me (?:about |what )?(.+)\??', r'what is \1'),
    (r'I was wondering (?:if |whether )?(.+)\??', r'\1'),
    (r'tell me about (.+)', r'what is \1'),
    (r'how does (.+?) work\??', r'how does \1 work'),
    (r'what about (.+)\??', r'what about \1'),
]


def remove_fillers(text: str) -> str:
    """Remove conversational filler words using regex patterns."""
    for pattern in FILLER_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    return text.strip()


def normalize_query_intent(text: str) -> str:
    """Normalize conversational queries to cleaner intent forms."""
    for pattern, replacement in QUERY_PATTERNS:
        m = re.match(pattern, text, re.IGNORECASE)
        if m:
            return replacement.format(*m.groups())
    return text


def split_coordination_spacy(doc) -> list[str]:
    """Split coordinated sentences using spaCy conj dependency.

    "Alice and Bob went to Paris" → ["Alice went to Paris", "Bob went to Paris"]
    "Python is used for data science and web development" →
        ["Python is used for data science", "Python is used for web development"]

    Reference: Kamana et al. (2026), arXiv:2601.00506
    Reference: Sahaj Software (2023)
    """
    clauses = []
    root = None
    for sent in doc.sents:
        root = sent.root
        # Find coordinated verbs
        conj_verbs = [root]
        for child in root.children:
            if child.dep_ == "conj":
                conj_verbs.append(child)

        if len(conj_verbs) == 1:
            # No coordination — return as-is
            clauses.append(sent.text)
            continue

        # For each verb, reconstruct its clause
        for verb in conj_verbs:
            subject = _find_subject(verb, root)
            objects = _find_objects(verb)
            if subject and objects:
                clause = f"{subject} {verb.text} {' '.join(objects)}"
                clauses.append(clause)
            elif subject:
                clause = f"{subject} {verb.text}"
                clauses.append(clause)

    return clauses if clauses else [doc.text]


def _find_subject(verb, root) -> str | None:
    """Find subject of a verb, propagating from root if needed."""
    for child in verb.children:
        if child.dep_ in ("nsubj", "nsubjpass"):
            return child.text
    # Propagate from root
    for child in root.children:
        if child.dep_ in ("nsubj", "nsubjpass"):
            return child.text
    return None


def _find_objects(verb) -> list[str]:
    """Find all objects of a verb with their prepositions."""
    objects = []
    for child in verb.children:
        if child.dep_ in ("dobj", "pobj", "attr"):
            objects.append(child.text)
        elif child.dep_ == "prep":
            for pobj in child.children:
                if pobj.dep_ == "pobj":
                    objects.append(f"{child.text} {pobj.text}")
    return objects


def preprocess_rules(text: str, nlp=None) -> list[str]:
    """Full rule-based preprocessing pipeline.

    Returns list of simplified sentences. If no transformation
    was needed, returns [original_text].
    """
    # Step 1: Filler removal
    text = remove_fillers(text)

    # Step 2: Query intent normalization
    text = normalize_query_intent(text)

    # Step 3: spaCy coordination splitting (if available)
    if nlp is not None:
        doc = nlp(text)
        clauses = split_coordination_spacy(doc)
        return clauses

    return [text]
```

### Expected Coverage

Based on Kamana et al. (2026) findings and TD v2's existing parser capabilities:

| Input Type | Layer A Alone | Layer A + TD v2 Parser |
|-----------|--------------|----------------------|
| Simple SVO | ✅ 95% | ✅ 99% |
| Coordination ("X and Y verb Z") | ✅ 85% | ✅ 95% |
| Relative clauses ("X that verb Y") | ⚠️ 60% | ✅ 90% |
| Passive voice ("X was verbed by Y") | ✅ 80% | ✅ 95% |
| Conversational filler | ✅ 99% | ✅ 99% |
| Multi-clause complex | ⚠️ 50% | ✅ 80% |
| **Overall** | **~70%** | **~90%** |

---

## Layer B: Small Model Sentence Splitting (T5-small)

### What It Does

A 60M-parameter encoder-decoder model fine-tuned on WikiSplit++ for sentence splitting. Runs on CPU. No API key needed. Handles the ~20% of cases where rules fail.

### Research Foundation

**Primary reference:** Tsukagoshi et al. (2024). "WikiSplit++: Easy Data Refinement for Split and Rephrase." arXiv:2404.09002.

- T5-small fine-tuned on WikiSplit++ achieves **99.13% entailment rate**
- Fewer hallucinations than GPT-3 (zero-shot and 3-shot)
- More splits than vanilla T5 trained on WikiSplit
- Training data: ~1M sentence pairs from Wikipedia edit history
- WikiSplit++ refines WikiSplit by: (1) removing contradiction pairs via NLI, (2) reversing sentence order

**Key results:**

| Model | Entailment % | # Splits | Hallucinations |
|-------|-------------|----------|----------------|
| T5-small (WikiSplit) | 99.01 | 2.00 | Low |
| T5-small (WikiSplit++) | 99.13 | 2.00 | **Lower** |
| GPT-3 (zero-shot) | 96.77 | 2.14 | Medium |
| GPT-3 (3-shot) | 98.51 | 1.86 | Medium |

**What this means:** T5-small is **better than GPT-3** at sentence splitting when fine-tuned on WikiSplit++. It's 60M params vs 175B params. Runs on CPU in ~50ms.

**Supporting references:**
- Botha et al. (2018): "Learning To Split and Rephrase From Wikipedia Edit History" — WikiSplit dataset (1M pairs)
- Niklaus et al. (2019): "MinWikiSplit" — minimal proposition decomposition (203K pairs)
- Narayan et al. (2017): "Split and Rephrase" — formalized the task, introduced BLEU/SARI metrics
- Kim et al. (2021): "BiSECT" — cross-lingual sentence compression via bisection

### Implementation Plan

```python
# td/preprocessing/t5_splitter.py

from transformers import T5Tokenizer, T5ForConditionalGeneration
import re

class T5SentenceSplitter:
    """Sentence splitter using T5-small fine-tuned on WikiSplit++.

    60M params, runs on CPU, ~50ms per sentence.
    No API key needed. No external dependencies beyond transformers.

    Reference: Tsukagoshi et al. (2024), arXiv:2404.09002
    """

    def __init__(self, model_path: str = "data/models/t5-wikisplit-pp"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)

    def split(self, text: str, max_splits: int = 5) -> list[str]:
        """Split a complex sentence into simpler atomic sentences.

        Args:
            text: Complex input sentence
            max_splits: Maximum number of output sentences

        Returns:
            List of simplified sentences
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", max_length=512, truncation=True
        )
        outputs = self.model.generate(
            **inputs,
            max_length=256,
            num_beams=4,
            early_stopping=True,
        )
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', result)
        return [s.strip() for s in sentences if s.strip()][:max_splits]
```

### Training Plan

```python
# scripts/train_t5_splitter.py

# 1. Download WikiSplit++ from HuggingFace
#    https://huggingface.co/datasets/cl-nagoya/wikisplit-pp

# 2. Fine-tune T5-small
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer

model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# WikiSplit++ format: (complex_sentence, [simple_sentence_1, simple_sentence_2])
# Training input: "split: {complex_sentence}"
# Training output: "{simple_sentence_1} . {simple_sentence_2}"

# 3. Evaluate on WikiSplitBench-100 (from CREDENCE paper)
# Expected: ~99% entailment, ~2.0 splits per sentence
```

### When Layer B Activates

Layer B is invoked ONLY when Layer A's output fails the confidence check (Layer C). This means:
- Simple sentences → Layer A handles, Layer B never runs
- Complex sentences → Layer A simplifies, Layer C checks, Layer B fixes if needed
- API cost: $0 (runs locally on CPU)

---

## Layer C: Confidence Gate

### What It Does

Checks whether the output from Layers A/B is "clean enough" for TD v2's parser to extract triples. A simple heuristic — no model inference.

### Implementation

```python
# td/preprocessing/confidence.py

def is_clean_svo(sentence: str, nlp=None) -> bool:
    """Check if a sentence is simple enough for TD v2's parser.

    A "clean" sentence has:
    - At least one noun (subject)
    - At least one verb (relation)
    - No nested clauses (no relcl, advcl, ccomp, xcomp)
    - No coordination (no conj)
    - Length < 30 words

    Returns True if the sentence is likely parseable.
    """
    if nlp is None:
        # Fallback: length check only
        return len(sentence.split()) < 25

    doc = nlp(sentence)

    has_noun = any(t.pos_ in ("NOUN", "PROPN") for t in doc)
    has_verb = any(t.pos_ == "VERB" for t in doc)
    short_enough = len(doc) < 30

    # Check for complex structures
    has_nested = any(
        t.dep_ in ("relcl", "advcl", "ccomp", "xcomp", "conj")
        for t in doc
    )

    return has_noun and has_verb and short_enough and not has_nested


def confidence_score(sentences: list[str], nlp=None) -> float:
    """Compute overall confidence for a list of preprocessed sentences.

    Returns 0.0-1.0. If any sentence fails the clean check, score drops.
    """
    if not sentences:
        return 0.0

    clean_count = sum(1 for s in sentences if is_clean_svo(s, nlp))
    return clean_count / len(sentences)
```

### Decision Logic

```python
def preprocess(text: str, nlp=None, t5_splitter=None) -> list[str]:
    """Three-layer preprocessing pipeline."""

    # Layer A: Rule-based
    sentences = preprocess_rules(text, nlp)

    # Layer C: Confidence check
    score = confidence_score(sentences, nlp)

    if score >= 0.8:
        return sentences  # Layer A output is good enough

    # Layer B: T5 splitting (only if available and needed)
    if t5_splitter is not None:
        improved = []
        for s in sentences:
            if not is_clean_svo(s, nlp):
                split = t5_splitter.split(s)
                improved.extend(split)
            else:
                improved.append(s)
        return improved

    return sentences  # Fallback to Layer A output
```

---

## Multi-Turn Context (Layer D — Future)

### The Problem

Multi-turn queries require resolving anaphora from previous turns:

```
Turn 1: "Tell me about seals"
Turn 2: "What do they eat?"        → "they" = seals
Turn 3: "And in the Arctic?"       → "seals in the Arctic"
Turn 4: "What about that migration thing?" → "seal migration"
```

### Current State

- Coreference resolution is implemented (spaCy two-pipeline approach)
- Handles he/she/it/they/him/her/them/its within a single text
- Discourse deixis filters "this shows", "that means"
- **Does NOT handle cross-turn resolution** (session state needed)

### Implementation Plan

```python
# In td/preprocessing/context.py

class SessionContext:
    """Maintains entity context across conversation turns.

    Stores recent entities mentioned. When a new query contains
    pronouns or anaphoric references, resolves them to recent entities.
    """

    def __init__(self, max_history: int = 5):
        self.entity_stack: list[str] = []
        self.max_history = max_history

    def update(self, entities: list[str]):
        """Push new entities to the recency stack."""
        for e in entities:
            if e not in self.entity_stack:
                self.entity_stack.insert(0, e)
        self.entity_stack = self.entity_stack[:self.max_history]

    def resolve_anaphora(self, text: str, nlp=None) -> str:
        """Resolve cross-turn pronouns using entity stack.

        "What do they eat?" + stack=["seals"]
        → "What do seals eat?"
        """
        if not self.entity_stack:
            return text

        # Simple pronoun resolution using recency
        pronouns = {"they": 0, "them": 0, "it": 0, "he": 0, "she": 0}
        tokens = text.lower().split()
        for i, token in enumerate(tokens):
            if token in pronouns:
                # Replace with most recent entity
                text = text[:text.lower().index(token)] + \
                       self.entity_stack[0] + \
                       text[text.lower().index(token) + len(token):]
                break

        return text
```

---

## LLM Layer (Optional Enhancement)

### When to Use

The LLM layer is **optional**. It's invoked only when:
1. Layer A + B output fails confidence check
2. The user explicitly asks for complex decomposition
3. Domain-specific jargon needs normalization

### Gemini Integration

```python
# td/preprocessing/llm_preprocessor.py

import os
import json
import requests

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"

def preprocess_with_gemini(text: str, api_key: str = None) -> list[str]:
    """Call Gemini API for complex sentence decomposition.

    Falls back to rule-based if API unavailable.
    """
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return [text]

    prompt = open("PREPROCESSING_PROMPT.md").read()
    # ... API call implementation ...
```

### Prompt Variants

The current v1 prompt handles **both teach and query** inputs. It already includes:

| Input Type | v1 Prompt Example | Status |
|-----------|------------------|--------|
| Filler removal | "So I was curious..." → stripped | ✅ Handled |
| Indirect questions | "I was wondering if..." → clean query | ✅ Handled |
| Conversational queries | "what's the deal with..." → "what is..." | ✅ Handled |
| Anaphora (within turn) | "the thing that you strike" → resolved | ✅ Handled |
| Teach decomposition | "Paris is capital of France and..." → 2 sentences | ✅ Handled |
| Subject repetition | "Alice and Bob went" → 2 sentences | ✅ Handled |
| Relative clause | "birds with long legs that migrate" → 3 sentences | ✅ Handled |

**Conclusion: A separate QUERY_PREPROCESSING_PROMPT is NOT needed.** The v1 prompt already handles both teach and query inputs. The gap is **multi-turn context** (resolving "it"/"they" from previous turns), which is a session-state problem, not a prompt problem.

What IS needed:
1. Add 5-10 more query examples to v1 prompt (edge cases)
2. Implement SessionContext for multi-turn resolution
3. Wire SessionContext into the preprocessing pipeline before the LLM call

---

## Performance Budget

| Layer | Latency | Params | Dependencies | Cost |
|-------|---------|--------|-------------|------|
| A: Rules (spaCy) | ~0ms | 0 | spaCy en_core_web_sm | Free |
| B: T5-small | ~50ms | 60M | transformers, torch | Free (local) |
| C: Confidence | ~0ms | 0 | spaCy (reuses parse) | Free |
| D: Context | ~0ms | 0 | Session state | Free |
| LLM: Gemini | ~2-5s | 1T+ | API key | ~$0.001/call |

**Total without LLM:** ~50ms, 60M params, $0/call
**Total with LLM (fallback):** ~5s, 1T+ params, ~$0.001/call

---

## TODO

### P0 — Next Session

- [ ] Implement `td/preprocessing/rule_based.py` (filler removal + query normalization)
- [ ] Wire rule-based preprocessing into `demos/chat_flare.py`
- [ ] Test with 10 messy queries (conversational, filler-heavy, anaphoric)
- [ ] Add 5-10 more query examples to `PREPROCESSING_PROMPT.md` v1 prompt

### P1 — This Week

- [ ] Implement `td/preprocessing/confidence.py` (is_clean_svo check)
- [ ] Wire confidence gate into preprocessing pipeline
- [ ] Download WikiSplit++ dataset for T5 training
- [ ] Fine-tune T5-small on WikiSplit++ (expected: 99% entailment)

### P2 — Month 1

- [ ] Implement `td/preprocessing/t5_splitter.py` (T5 inference wrapper)
- [ ] Integrate T5 splitter as Layer B in preprocessing pipeline
- [ ] Implement `td/preprocessing/context.py` (SessionContext for multi-turn)
- [ ] Wire SessionContext into chat_flare.py
- [ ] Wire Gemini API as optional Layer D (fallback only)

### P3 — Month 2

- [ ] Evaluate T5-small on WikiSplitBench-100 (from CREDENCE paper)
- [ ] Benchmark Layer A coverage on 100 real user queries
- [ ] Benchmark Layer A+B coverage on same 100 queries
- [ ] Document accuracy vs latency tradeoff
- [ ] Consider fine-tuning T5 on TD-specific domain data (biology, tech, geography)

---

## References

| # | Paper | Year | Venue | Relevance |
|---|-------|------|-------|-----------|
| 1 | Kamana et al. "Rule-Based Approaches to Atomic Sentence Extraction" | 2026 | arXiv:2601.00506 | spaCy rule-based sentence decomposition, ROUGE-1=0.67 |
| 2 | Tsukagoshi et al. "WikiSplit++: Easy Data Refinement for Split and Rephrase" | 2024 | arXiv:2404.09002 | T5-small fine-tuning, 99.13% entailment |
| 3 | Botha et al. "Learning To Split and Rephrase From Wikipedia Edit History" | 2018 | EMNLP | WikiSplit dataset (1M pairs) |
| 4 | Niklaus et al. "MinWikiSplit: A Sentence Splitting Corpus with Minimal Propositions" | 2019 | EMNLP | 203K minimal proposition pairs |
| 5 | Narayan et al. "Split and Rephrase" | 2017 | EMNLP | Formalized the task, BLEU/SARI metrics |
| 6 | Min et al. "Towards Practical GraphRAG" | 2025 | arXiv:2507.03226 | spaCy = 94% of LLM for KG extraction |
| 7 | Sahaj Software "Knowledge graphs from complex text" | 2023 | Technical Report | Verb-based sentence splitting via spaCy |
| 8 | Manning & Schütze "Foundations of Statistical NLP" | 1999 | MIT Press Ch. 5 | Coordinated NP extraction |
| 9 | CREDENCE "Claim Reduction for Decomposition" | 2026 | arXiv:2606.19819 | Rule-repair + verifier pipeline, WikiSplitBench |
| 10 | ATOM "AdapTive and OptiMized dynamic TKG" | 2025 | arXiv:2510.22590 | Atomic fact decomposition, parallel merging |
| 11 | CoDe-KG "Combining Coreference Resolution with Syntactic Sentence Decomposition" | 2025 | EMNLP | Coref + decomposition, 65.8% macro-F1 on REBEL |
| 12 | Kim et al. "BiSECT: Cross-lingual Sentence Compression" | 2021 | — | Bisection-based splitting |
| 13 | Jauhar et al. "Resolving Discourse-Deictic Pronouns" | 2015 | *SEM | Two-stage classify+resolve for this/that |

---

_"The preprocessing layer doesn't need to be perfect. It needs to be good enough that the parser can handle the rest."_
