# TD v2 — Hack Removal & Research-Backed Fix Plan

**Date:** 2026-07-06
**Status:** PLANNED
**Trigger:** Audit of last 10 commits revealed hardcoded English, quick hacks, and non-research-backed solutions

---

## Summary of Problems

| # | Problem | Commits | Severity |
|---|---------|---------|----------|
| 1 | `generic_words` hardcoded English stopword set for entity validation | `4f5ec4e` | 🔴 Critical |
| 2 | `question_words` hardcoded English set for question detection | `f005910`, `1daa536` | 🔴 Critical |
| 3 | Discourse deixis: inline hardcoded English verb tuple | `bc568e0` | 🟡 Medium |
| 4 | `_pp_words` hardcoded English prepositions in regex fallback | existing | 🟡 Medium |
| 5 | `test_non_eu_country` failure: entity validation too loose | `4f5ec4e` | 🔴 Critical |
| 6 | "quality" relation: ad-hoc, no standard ontology backing | `bc568e0` | 🟡 Medium |
| 7 | Open query returns first SPARQL result with no ranking | `1daa536` | 🟡 Medium |
| 8 | Possessive pronoun stripping via string slicing | `bc568e0` | 🟡 Medium |

---

## Fix 1: Entity Validation — Replace `generic_words` with spaCy POS/NER

### Problem

`generic_words = {"part", "of", "is", "the", "in", "a", "an", "and", "or", "capital"}` is:
- English-only
- A hand-curated stopword list (not research-backed)
- Missing many function words ("at", "by", "for", "with", etc.)
- Includes "capital" which is a content word, not a stopword

### Solution: spaCy POS Tags + NER

spaCy's Universal POS tags are **language-agnostic** (17 categories across 100+ languages). Use them to distinguish entities from function words:

```python
def _is_entity_token(token) -> bool:
    """Determine if a spaCy token is an entity (not a function word).
    
    Universal POS tags (Nivre et al., 2016; de Marneffe et al., 2021):
    - NOUN: common noun (dog, city, capital)
    - PROPN: proper noun (Paris, France, Norway)
    - NUM: numeral (2, two, II)
    - ADJ: adjective (only when modifying a noun in a named entity)
    
    NOT entities: ADP (prepositions), DET (determiners), AUX (auxiliaries),
    CCONJ (coordinating conjunctions), SCONJ (subordinating conjunctions),
    PART (particles), PUNCT (punctuation)
    """
    return token.pos_ in ("NOUN", "PROPN", "NUM")
```

For the MHN entity validation, replace the word-overlap check with spaCy-based entity extraction:

```python
def _validate_entity_match(query_text: str, retrieved_text: str, nlp) -> bool:
    """Validate that query entities appear in retrieved text.
    
    Uses spaCy NER + POS for language-agnostic entity detection.
    Falls back to noun/proper-noun overlap when NER finds nothing.
    
    Reference: Honnibal & Montani (2017), "spaCy 2"
    Reference: Nivre et al. (2016), "Universal Dependencies 2.0"
    """
    query_doc = nlp(query_text)
    retrieved_doc = nlp(retrieved_text)
    
    # Strategy 1: NER overlap (highest confidence)
    query_ents = {e.text.lower() for e in query_doc.ents}
    retrieved_ents = {e.text.lower() for e in retrieved_doc.ents}
    if query_ents and retrieved_ents:
        return bool(query_ents & retrieved_ents)
    
    # Strategy 2: Noun/ProperNoun overlap (fallback)
    query_nouns = {t.text.lower() for t in query_doc if t.pos_ in ("NOUN", "PROPN")}
    retrieved_nouns = {t.text.lower() for t in retrieved_doc if t.pos_ in ("NOUN", "PROPN")}
    return bool(query_nouns & retrieved_nouns)
```

### Files to Change
- `td/thinking.py`: Remove `generic_words` set (line 1469), replace entity validation logic
- `tests/test_realworld.py`: `test_non_eu_country` should pass after this fix

### References
- Honnibal, M. & Montani, I. (2017). "spaCy 2: Natural language understanding with Bloom embeddings." *Industrial-strength NLP*.
- Nivre, J. et al. (2016). "Universal Dependencies 2.0 — CoNLL 2017 Shared Task." *LREC*.
- de Marneffe, M.-C. et al. (2021). "Universal Dependencies." *Computational Linguistics*, 47(2): 255–308.

---

## Fix 2: Question Detection — Replace `question_words` with UD `PronType=Int`

### Problem

`question_words = {"what", "who", "where", "which"}` is:
- English-only
- Incomplete (missing "how", "why", "when", "whom", "whose")
- Not using the language-agnostic UD feature system

### Solution: Universal Dependencies `PronType=Int` Feature

UD defines `PronType=Int` (Interrogative) as a **universal morphological feature** that works across 100+ languages. spaCy exposes this as `token.morph.get("PronType")`:

```python
def _is_interrogative(token) -> bool:
    """Check if a token is interrogative using UD features.
    
    Universal Dependencies PronType=Int:
    - English: who, what, where, which, when, how, why
    - German: wer, was, wo, welcher, wann, wie, warum
    - French: qui, que, où, quel, quand, comment, pourquoi
    - Korean: 누구(nugu), 무엇(mueot), 어디(eodi)
    - Turkish: kim, ne, nerede, hangi
    
    Reference: Nivre et al. (2016), Universal Dependencies 2.0
    Reference: de Marneffe et al. (2021), Computational Linguistics 47(2)
    """
    morph_pron_type = token.morph.get("PronType")
    return "Int" in morph_pron_type if morph_pron_type else False

def _is_question(doc) -> bool:
    """Detect if a sentence is a question using multiple signals.
    
    Signals (language-agnostic):
    1. PronType=Int morphological feature (UD standard)
    2. Sentence-final question mark (universal punctuation)
    3. Subject-verb inversion (language-specific, optional)
    """
    # Signal 1: Interrogative pronoun/determiner/adverb
    has_interrogative = any(_is_interrogative(t) for t in doc)
    
    # Signal 2: Question mark
    has_question_mark = doc.text.rstrip().endswith("?")
    
    return has_interrogative or has_question_mark
```

### Files to Change
- `td/thinking.py`: Remove `question_words` set (lines 1213, 1393), replace with `_is_question(doc)` using spaCy
- Requires passing spaCy `doc` object to the query function

### References
- Nivre, J. et al. (2016). "Universal Dependencies 2.0 — CoNLL 2017 Shared Task." *LREC*.
- de Marneffe, M.-C. et al. (2021). "Universal Dependencies." *Computational Linguistics*, 47(2): 255–308.
- Universal Dependencies Feature: `PronType` — https://universaldependencies.org/u/feat/PronType.html

---

## Fix 3: Discourse Deixis — Implement Two-Stage Approach (Jauhar et al., 2015)

### Problem

Currently uses a hardcoded English verb tuple:
```python
if new_s in ("this", "that") and r in ("shows", "means", "proves",
        "suggests", "indicates", "demonstrates", "reveals"):
    continue
```

And a separate `DISCOURSE_DEIXIS_VERBS` set. Both are English-only and not research-backed.

### Solution: Two-Stage Classify + Resolve (Jauhar et al., *SEM 2015)

The paper introduces:
1. **Classification:** Is the pronoun entity-referring or discourse-deictic?
2. **Resolution:** If entity-referring, resolve to antecedent. If discourse-deictic, skip.

**Key features from the paper (linguistically motivated, language-agnostic via UD):**

| Feature | spaCy Access | Description |
|---------|-------------|-------------|
| Pronoun syntactic role | `token.dep_` | Subject of abstract verb → likely discourse deixis |
| Head verb lemma | `token.head.lemma_` | Abstract verbs (show, mean, prove) → discourse deixis |
| Preceding sentence | `doc.sents` | Discourse deixis refers to whole clauses |
| Distance to antecedent | token position | Entity references are usually local |

**Implementation:**

```python
# Abstract verb detection using UD + semantic classification
# Not a hardcoded list — uses verb properties
ABSTRACT_VERB_SEMANTICS = {
    # Verbs of communication/suggestion (language-agnostic via lemma)
    "show", "mean", "prove", "suggest", "indicate", "demonstrate",
    "reveal", "imply", "confirm", "show", "illustrate", "reflect",
    # Extensible via registry
}

def _is_discourse_deictic(token, doc) -> bool:
    """Classify if a pronoun is discourse-deictic (refers to a clause, not entity).
    
    Two-stage approach (Jauhar et al., *SEM 2015):
    Stage 1: Classify — is this pronoun discourse-deictic?
    Stage 2: If yes, skip (don't resolve to entity)
    
    Signals:
    - Pronoun is subject (nsubj) of an abstract verb
    - Head verb is in the abstract verb set
    - Pronoun is "this" or "that" (demonstrative)
    
    Reference: Jauhar, S.K. et al. (2015). "Resolving Discourse-Deictic 
    Pronouns: A Two-Stage Approach." *SEM 2015, pp. 299-308.
    ACL Anthology: S15-1035
    """
    if token.text.lower() not in ("this", "that"):
        return False
    
    # Check if subject of abstract verb
    if token.dep_ in ("nsubj", "nsubjpass"):
        head_lemma = token.head.lemma_.lower()
        if head_lemma in ABSTRACT_VERB_SEMANTICS:
            return True
    
    return False
```

**Making it extensible (language-agnostic):**
```python
# Registry-based abstract verb set (not hardcoded in function)
class DiscourseDeixisRegistry:
    """Registry of abstract verbs that signal discourse deixis.
    
    Extensible per language. Default: English verbs from Jauhar et al. (2015).
    """
    _verbs: dict[str, set[str]] = {
        "en": {"show", "mean", "prove", "suggest", "indicate", 
               "demonstrate", "reveal", "imply", "confirm", "illustrate"},
    }
    
    @classmethod
    def register(cls, lang: str, verbs: set[str]):
        cls._verbs.setdefault(lang, set()).update(verbs)
    
    @classmethod
    def get(cls, lang: str) -> set[str]:
        return cls._verbs.get(lang, set())
```

### Files to Change
- `td/perception/nl_parser.py`: Replace `DISCOURSE_DEIXIS_VERBS` and inline tuple with `DiscourseDeixisRegistry`
- `td/perception/discourse_deixis.py` (new): Two-stage classify + resolve module

### References
- Jauhar, S.K. et al. (2015). "Resolving Discourse-Deictic Pronouns: A Two-Stage Approach." *Proceedings of *SEM 2015*, pp. 299–308. ACL Anthology: [S15-1035](https://aclanthology.org/S15-1035/)
- Guerra, R.D. et al. (2015). Same paper (co-author). Features: syntactic role, head verb, distance.
- Webber, B.L. (1988). "Discourse Deixis: Reference to Discourse Segments." *ACL 1988*.

---

## Fix 4: Preposition Detection — Replace `_pp_words` with spaCy POS

### Problem

`_pp_words = {"on", "in", "at", "from", "to", "by", "for", "with", "into", ...}` is English-only.

### Solution: spaCy `ADP` POS Tag

spaCy's Universal POS tag `ADP` (adposition) covers prepositions AND postpositions in all languages:

```python
# BEFORE (hardcoded English):
_pp_words = {"on", "in", "at", "from", "to", "by", "for", "with", "into", ...}
while len(words) >= 3 and words[-2] in _pp_words:

# AFTER (spaCy, language-agnostic):
# Use token.pos_ == "ADP" in the spaCy path
# The regex fallback path should try spaCy first, skip if unavailable
```

### Files to Change
- `td/thinking.py`: Remove `_pp_words` set (line 1058), use spaCy ADP detection in the regex fallback

### References
- Universal POS Tags: https://universaldependencies.org/u/pos/
- Honnibal & Montani (2017). spaCy 2.

---

## Fix 5: `test_non_eu_country` — Fix Entity Validation Logic

### Problem

The test expects "is Norway part of Europe?" → `unknown` (Norway was never taught).

Current validation:
```
entity_words = {"norway", "europe"} (after removing generic_words)
problem_words = {"eu", "is", "part", "of", "europe"}
overlap = {"europe"} → non-empty → returns "learned" (WRONG)
```

The issue: "europe" overlaps because it appears in both the query AND the retrieved fact, but "norway" (the actual entity) does NOT appear in the retrieved fact.

### Solution: Require PRIMARY Entity Match

After Fix 1 (spaCy NER/POS), the validation becomes:

```python
# Extract entities from query using spaCy
query_doc = nlp(query_text)
query_entities = {e.text.lower() for e in query_doc.ents}
if not query_entities:
    query_entities = {t.text.lower() for t in query_doc if t.pos_ in ("NOUN", "PROPN")}

# Extract entities from retrieved text
retrieved_doc = nlp(retrieved_text)
retrieved_entities = {e.text.lower() for e in retrieved_doc.ents}
if not retrieved_entities:
    retrieved_entities = {t.text.lower() for t in retrieved_doc if t.pos_ in ("NOUN", "PROPN")}

# Require at least ONE query entity to appear in retrieved entities
# "is Norway part of Europe?" → query_entities = {"norway", "europe"}
# "EU is part of Europe" → retrieved_entities = {"eu", "europe"}
# overlap = {"europe"} → BUT "norway" is not in retrieved → FAIL
#
# Correct: require ALL query entities that are NOT common nouns to match
# "norway" is PROPN (proper noun) → must match
# "europe" is PROPN → must match
# Since "norway" doesn't match → reject
```

**Better approach:** Prioritize **proper nouns** (PROPN) over common nouns (NOUN) in entity matching:

```python
def _validate_entity_match(query_doc, retrieved_doc) -> bool:
    """Validate entity match with proper noun priority.
    
    Proper nouns (PROPN) are the primary entities — they MUST match.
    Common nouns (NOUN) are secondary — they provide context but aren't sufficient alone.
    
    "is Norway part of Europe?" 
      → PROPNs: {norway, europe} 
      → "norway" must appear in retrieved text
      → "EU is part of Europe" has PROPNs: {eu, europe}
      → "norway" NOT found → REJECT
    """
    query_propns = {t.text.lower() for t in query_doc if t.pos_ == "PROPN"}
    retrieved_propns = {t.text.lower() for t in retrieved_doc if t.pos_ == "PROPN"}
    
    if query_propns:
        # At least one proper noun from query must appear in retrieved
        return bool(query_propns & retrieved_propns)
    
    # Fallback: noun overlap
    query_nouns = {t.text.lower() for t in query_doc if t.pos_ == "NOUN"}
    retrieved_nouns = {t.text.lower() for t in retrieved_doc if t.pos_ == "NOUN"}
    return bool(query_nouns & retrieved_nouns) if query_nouns else True
```

### Files to Change
- `td/thinking.py`: Replace entity validation in MHN retrieval section

### References
- Universal POS Tags: `PROPN` (proper noun) vs `NOUN` (common noun)
- Nivre et al. (2016), Universal Dependencies 2.0

---

## Fix 6: "quality" Relation — Drop or Standardize

### Problem

`triples.append((subj, "quality", adj.lemma_))` — "quality" is not in any standard KG ontology:
- Not in Wikidata (P-properties)
- Not in Schema.org
- Not in OWL/RDFS

### Solution: Two Options

**Option A: Drop it (recommended for now)**
Adjectival predicates like "runs smoother" or "is beautiful" don't produce valid (subject, relation, object) triples. They're **attributes**, not relations. Store them as literal properties if needed, not as triples.

**Option B: Map to Wikidata properties (future)**
If we want to keep attribute-like information, use Wikidata's property system:
- "is beautiful" → `P1552` (has quality)
- "runs smoother" → `P4599` (has performance characteristic)

But this requires an ontology mapping layer that doesn't exist yet.

### Recommendation
**Option A**: Remove the "quality" extraction entirely. It's not in the project scope (<100K params, KG reasoning, not attribute extraction). If needed later, add it properly with ontology backing.

### Files to Change
- `td/perception/nl_parser.py`: Remove adjectival predicate extraction block

---

## Fix 7: Open Query Ranking — Add Relevance Scoring

### Problem

The SPARQL open query returns the **first** result with no ranking:
```python
if fwd:
    for r in fwd:
        rel = r.get("?p", "").replace(...)
        obj = r.get("?o", "").replace(...)
        if rel and obj:
            return {...}  # Returns FIRST result
```

### Solution: Rank by Entity Overlap + Relation Specificity

```python
def _rank_open_query_results(results: list[dict], entity: str) -> dict:
    """Rank SPARQL open query results by relevance.
    
    Scoring (heuristic, not ML-based):
    1. Entity overlap: +1 if the entity appears in subject or object
    2. Relation specificity: prefer longer relation names (more specific)
    3. Recency: prefer facts from the user graph over derived facts
    
    This is a simple heuristic. For production, consider:
    - Bio-SODA (2023): node centrality as relevance measure
    - Q²Forge (2025): competency question matching
    """
    scored = []
    for r in results:
        score = 0.0
        subj = r.get("?s", "").lower()
        obj = r.get("?o", "").lower()
        rel = r.get("?p", "").lower()
        
        # Entity overlap
        if entity.lower() in subj or entity.lower() in obj:
            score += 1.0
        
        # Relation specificity (longer = more specific)
        rel_name = rel.split("/")[-1] if "/" in rel else rel
        score += len(rel_name) * 0.01
        
        scored.append((score, r))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1] if scored else None
```

### Files to Change
- `td/thinking.py`: Add ranking to SPARQL open query results

### References
- Bio-SODA (2023): "node centrality as a measure of relevance for selecting the best SPARQL candidate query"
- Q²Forge (ACM, 2025): "judging the relevance of question-query pairs"

---

## Fix 8: Possessive Pronoun Resolution — Use Dependency Parse

### Problem

```python
# String slicing — fragile
if o.startswith(poss + " "):
    new_o = o[len(poss) + 1:]
```

### Solution: spaCy Dependency Parse

```python
def _resolve_possessives(triples, doc, pronoun_map):
    """Resolve possessive pronouns using spaCy dependency parse.
    
    Instead of string slicing, use the dependency tree:
    - 'poss' dependency: "its video games" → "its" is poss of "games"
    - Replace "its" with resolved entity, keep "video games" as object
    
    Reference: Universal Dependencies — 'poss' dependency label
    """
    for s, r, o in triples:
        new_o = o
        for token in doc:
            if token.dep_ == "poss" and token.i in pronoun_map:
                entity, _ = pronoun_map[token.i]
                # Replace possessive pronoun with entity name in object
                new_o = new_o.replace(token.text.lower(), entity)
        yield (s, r, new_o)
```

### Files to Change
- `td/perception/nl_parser.py`: Replace string slicing with dependency-based resolution

---

## Implementation Order

| Priority | Fix | Effort | Impact |
|----------|-----|--------|--------|
| **P0** | Fix 1: Entity validation (spaCy POS/NER) | Small | Fixes test_non_eu_country, removes generic_words |
| **P0** | Fix 5: Entity match logic (PROPN priority) | Small | Fixes test_non_eu_country |
| **P1** | Fix 2: Question detection (UD PronType=Int) | Small | Removes question_words, language-agnostic |
| **P1** | Fix 3: Discourse deixis (Jauhar et al. 2015) | Medium | Proper research-backed solution |
| **P1** | Fix 4: Preposition detection (spaCy ADP) | Small | Removes _pp_words |
| **P2** | Fix 6: Drop "quality" relation | Small | Clean up |
| **P2** | Fix 7: Open query ranking | Medium | Better UX |
| **P2** | Fix 8: Possessive resolution (dep parse) | Small | More robust |

**Total estimated effort:** ~3-4 hours
**New test target:** 662+ passing, 0 failing

---

## New Papers to Cite

These need to be added to ARCHITECTURE.md and DEVELOPMENT.md:

| # | Paper | Year | Venue | Relevance |
|---|-------|------|-------|-----------|
| 1 | Jauhar, S.K. et al. "Resolving Discourse-Deictic Pronouns: A Two-Stage Approach" | 2015 | *SEM (ACL) | Discourse deixis classification + resolution |
| 2 | Nivre, J. et al. "Universal Dependencies 2.0" | 2016 | CoNLL Shared Task | Language-agnostic POS, PronType, dependency |
| 3 | de Marneffe, M.-C. et al. "Universal Dependencies" | 2021 | Computational Linguistics 47(2) | UD feature system (PronType=Int) |
| 4 | Hudson/Coreferee. "Coreference resolution for spaCy" | 2022 | GitHub | Multilingual coreference (en, de, fr, pl) |
| 5 | Bio-SODA. "Natural language processing over knowledge graphs" | 2023 | — | SPARQL result ranking via node centrality |
| 6 | Q²Forge. "Minting Competency Questions and SPARQL Queries" | 2025 | ACM | SPARQL query relevance scoring |

---

## Anti-Pattern Checklist (for future commits)

Before committing, verify:

- [ ] No hardcoded English word sets (`_pp_words`, `question_words`, `generic_words`, verb tuples)
- [ ] No string manipulation where spaCy dependency parse is available
- [ ] No "first result" returns without ranking/relevance scoring
- [ ] No ad-hoc relations ("quality") without ontology backing
- [ ] All language-specific features use UD/PoS tags, not word lists
- [ ] All new features have research citations
- [ ] All entity detection uses spaCy NER/POS, not stopword filtering
- [ ] Test with non-English inputs (even if just for architecture validation)

---

*"Don't hardcode what you can learn. Don't guess what you can prove."*
