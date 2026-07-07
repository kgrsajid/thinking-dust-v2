# TD v2 — Contradiction Detection Implementation Plan

**Based on:** 3 independent reviews (MiMo, Gemini 3.1 Pro, GLM 5.2)
**Date:** 2026-07-07
**Status:** APPROVED FOR IMPLEMENTATION

---

## 1. Problem Statement

TD v2's `add_fact()` stores any triple without consistency checking. A user can teach "Paris is a country" after teaching "Paris is the capital of France" — the system accepts both without flagging the contradiction.

**Root cause:** `add_fact()` is a pure storage function with zero semantic validation.

---

## 2. Design: Lightweight Ontological Type Guard (LOTG)

**Core idea:** A pre-commit hook in `add_fact()` that infers entity types from relations, checks for type conflicts against a disjointness table, and warns (not rejects) when contradictions are found.

### 2.1 Three Data Structures

#### A. Relation Schema Registry

Extends the existing `DEFAULT_RELATION_PROPERTIES` with domain/range type constraints.

```python
RELATION_SCHEMA = {
    "capital_of":   {"domain": "city",          "range": "country"},
    "born_in":      {"domain": "person",        "range": "place"},
    "made_by":      {"domain": "product",       "range": "organization"},
    "founded_by":   {"domain": "organization",  "range": "person"},
    "married_to":   {"domain": "person",        "range": "person"},
    "located_in":   {"domain": "place",         "range": "place"},
    "is_a":         {"domain": None,            "range": None},  # type declaration
    "in":           {"domain": None,            "range": None},  # too broad
    "part_of":      {"domain": None,            "range": None},  # too broad
}
```

**Research backing:**
- OWL 2 `rdfs:domain` and `rdfs:range` (W3C, 2009) — standard ontology property constraints
- Every production KG (Wikidata, DBpedia, Google KG) uses domain/range
- Wikidata's "constraint violations" system: warnings not blocks

#### B. Type Disjointness Table

Sets of mutually exclusive types. An entity cannot belong to two types in the same set.

```python
DISJOINT_TYPES = {
    frozenset({"city", "country", "continent"}),
    frozenset({"person", "organization", "product"}),
    frozenset({"animal", "plant"}),
    frozenset({"living", "non_living"}),
}
```

**Research backing:**
- OWL 2 `owl:disjointWith` axiom (W3C, 2009)
- Description Logic (DL) — disjointness is a fundamental axiom in all DL reasoners
- Protégé, HermiT, Pellet — all implement disjointness checking

#### C. Type Hierarchy (Minimal Subsumption)

Subtype → supertype mapping. Enables: "Paris is a city" AND "Paris is a place" = no conflict (city ⊑ place).

```python
TYPE_HIERARCHY = {
    "city":          {"settlement", "place"},
    "country":       {"place"},
    "continent":     {"place"},
    "state":         {"place"},
    "river":         {"place", "geographical_feature"},
    "person":        {"living"},
    "organization":  {"entity"},
    "product":       {"entity"},
}
```

**Research backing:**
- OWL 2 `rdfs:subClassOf` (W3C, 2009)
- RDFS semantics (W3C, 2004) — subsumption is the foundation of ontological reasoning
- WordNet, Wikidata taxonomies — battle-tested type hierarchies

---

### 2.2 Entity Type Tracking

A dict on `KnowledgeGraph` that records inferred types per entity, with proof of inference.

```python
self._entity_types: dict[str, set[str]] = defaultdict(set)
# {"paris": {"city"}, "france": {"country"}, "einstein": {"person"}}
```

Types are inferred from three sources:
1. **Explicit:** `X is_a T` → T recorded directly
2. **Domain inference:** `capital_of(X, Y)` → X is inferred as `city`
3. **Range inference:** `capital_of(X, Y)` → Y is inferred as `country`

**Research backing:**
- Wikidata entity typing — every entity has `instance_of` / `subclass_of`
- DBpedia ontology — type inference from properties is standard practice
- NELL (Never-Ending Language Learner) — learns entity types from relation patterns

---

### 2.3 Consistency Check Algorithm

```
check_consistency(subject, relation, object) → list[Warning]

1. If relation == "is_a":
     new_type = object
     Check: for each existing_type in entity_types[subject]:
       If disjoint(new_type, existing_type) AND NOT subsumes(new_type, existing_type):
         → WARNING: "{subject} was {existing_type}, now {new_type}"
     Record type: entity_types[subject].add(new_type)

2. If relation has schema:
     For subject: inferred_type = schema.domain
       Check against entity_types[subject] (same disjointness logic)
       Record type
     For object: inferred_type = schema.range
       Check against entity_types[object] (same disjointness logic)
       Record type

3. Return warnings (empty list = no conflicts)
```

**Performance:** dict lookup + set intersection = O(1) amortized. <1ms guaranteed.

---

### 2.4 Conflict Policy

| Action | Behavior |
|--------|----------|
| **Store** | ✅ Always store the triple. User is the authority. |
| **Warn** | ⚠️ Return warning with proof trace |
| **Annotate** | 📝 Triple gets `metadata: {"contradiction": warning_text}` |
| **Reject** | ❌ Never. This is a reasoning engine, not a database constraint. |

**Research backing:**
- Wikidata constraint violations: "suggestions, not hard constraints" (Wikidata Help:Constraint)
- Open World Assumption (OWL): absence of a fact ≠ negation of a fact
- Collaborative KG systems (YAGO, Freebase): soft constraints with human override

---

## 3. Files to Create/Modify

### 3.1 NEW: `td/reasoning/contradiction_detector.py` (~180 lines)

```
Contents:
- RELATION_SCHEMA dict (pre-seeded)
- DISJOINT_TYPES set (pre-seeded)
- TYPE_HIERARCHY dict (pre-seeded)
- ContradictionDetector class:
    - __init__() — empty entity type registry
    - check(subject, relation, obj) → list[Warning]
    - _infer_from_relation(entity, type, proof) → None
    - _check_disjoint(entity, new_type, existing_type) → bool
    - _is_subsumed(subtype, supertype) → bool
    - get_entity_types(entity) → set[str]
    - reset() — clear all inferred types
```

### 3.2 MODIFY: `td/kg/__init__.py` (~25 lines changed)

```
Changes:
- Import ContradictionDetector
- In KnowledgeGraph.__init__(): create self.detector = ContradictionDetector()
- In add_fact(): call self.detector.check(s, r, o) BEFORE storing
- add_fact() return type: tuple[Triple, list[str]] (triple + warnings)
- Add metadata field to Triple for contradiction annotations
```

### 3.3 MODIFY: `td/thinking.py` (~15 lines changed)

```
Changes:
- In teach(): capture warnings from kg.add_fact()
- Return warnings alongside the normal response
- Format warnings as human-readable proof traces
```

### 3.4 NEW: `tests/test_contradiction_detector.py` (~120 lines)

```
Test cases:
- test_capital_of_infers_city_type()
- test_is_a_country_contradicts_city()
- test_born_in_infers_person_type()
- test_person_is_place_contradiction()
- test_city_and_country_disjoint()
- test_subsumption_no_false_positive()  — "Paris is a place" after "Paris is a city" = OK
- test_no_schema_relation_passes()     — "in" has no domain/range, no warnings
- test_explicit_is_a_before_relation() — type set first, then relation = check works
- test_relation_before_explicit()      — relation first, then is_a = check works
- test_warnings_include_proof_trace()
- test_triple_still_stored_despite_warning()
- test_entity_types_persist_across_adds()
```

---

## 4. Implementation Order

```
Step 1: td/reasoning/contradiction_detector.py
  - Define data structures (RELATION_SCHEMA, DISJOINT_TYPES, TYPE_HIERARCHY)
  - Implement ContradictionDetector class
  - Unit test in isolation

Step 2: tests/test_contradiction_detector.py
  - Write all 12 test cases
  - Verify ContradictionDetector works standalone

Step 3: td/kg/__init__.py
  - Import and wire ContradictionDetector into add_fact()
  - Update return type
  - Run existing test suite (655 tests should still pass)

Step 4: td/thinking.py
  - Surface warnings in teach()
  - Manual demo test with chat_flare.py

Step 5: Full test suite
  - Run all 655 existing tests (no regressions)
  - Run 12 new contradiction tests
  - Total: ~667 tests
```

---

## 5. What This Does NOT Do (Explicit Non-Goals)

| Non-Goal | Why Not |
|----------|---------|
| Z3-based consistency checking on teach() | Too slow (>10ms per call). Save Z3 for ask(). |
| Full OWL reasoning | Overkill. TD v2 needs lightweight guards, not a DL reasoner. |
| Automatic type inference from text | spaCy NER could help but adds 500MB dependency. Out of scope. |
| Reject/rollback on contradiction | Violates "user is authority" principle. |
| Multi-turn contradiction detection | Future work. This catches single-teach contradictions. |
| Temporal contradiction | "Paris was a city in 1900, now it's a country" — future work with temporal context. |

---

## 6. Research References

| # | Source | Year | Relevance |
|---|--------|------|-----------|
| 1 | OWL 2 Web Ontology Language (W3C) | 2009 | `rdfs:domain`, `rdfs:range`, `owl:disjointWith`, `rdfs:subClassOf` |
| 2 | RDFS Semantics (W3C) | 2004 | Type hierarchy, subsumption reasoning |
| 3 | Wikidata Constraint Violations | ongoing | Soft constraints as warnings, not hard blocks |
| 4 | Description Logic (Baader et al.) | 2003 | Disjointness as fundamental DL axiom |
| 5 | Protégé + HermiT/Pellet | 2000s+ | Battle-tested OWL reasoners with disjointness checking |
| 6 | NELL (CMU) | 2010+ | Entity type inference from relation patterns |
| 7 | Open World Assumption (OWL) | 2004 | Absence of fact ≠ negation of fact |
| 8 | YAGO / Freebase | 2007+ | Soft constraints in collaborative KG systems |
| 9 | Wikidata `instance_of` / `subclass_of` | 2012+ | Production entity typing system |
| 10 | Chain of Responsibility pattern (GoF) | 1994 | Pre-commit validation hook design pattern |

---

## 7. Success Criteria

After implementation:
- [ ] "Paris is a country" generates a warning when "Paris capital_of France" exists
- [ ] "Einstein is a place" generates a warning when "Einstein born_in Germany" exists
- [ ] "Paris is a place" does NOT generate a warning (city ⊑ place)
- [ ] All 655 existing tests still pass (no regressions)
- [ ] 12 new contradiction tests pass
- [ ] `add_fact()` runs in <1ms with the hook (performance budget)
- [ ] chat_flare.py shows warnings to the user during teach
