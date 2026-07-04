# TD v2 — SPARQL Query Layer Design

_Using pyoxigraph (Rust-backed, SPARQL 1.1, 18ms @ 10M triples)_

---

## 1. Why This Exists

TD v2 currently uses SQLite + custom BFS for knowledge graph queries. This works for <1K triples, but:

1. **Inverse queries** are heuristics, not standards-compliant
2. **No SPARQL** — can't query with standard patterns, property paths, FILTER, OPTIONAL
3. **Custom BFS** doesn't scale past ~10K triples efficiently
4. **No interoperability** — can't load/export standard RDF data

pyoxigraph gives us a production-grade SPARQL engine that:
- Runs on CPU, pip-installable, Apple Silicon compatible
- Handles 10M+ triples with 18ms query time
- Supports full SPARQL 1.1 (property paths, subqueries, aggregates)
- Persists to disk, survives restarts
- Loads/exports standard RDF formats (Turtle, N-Triples, JSON-LD)

---

## 2. TD v2 → RDF Mapping

### 2.1 Namespace

```python
TD = "http://thinking-dust.org/"       # Base namespace
TD_ENT = "http://thinking-dust.org/entity/"     # Entities
TD_REL = "http://thinking-dust.org/relation/"   # Relations
TD_META = "http://thinking-dust.org/meta/"      # Metadata predicates
TD_GRAPH = "http://thinking-dust.org/graph/"    # Named graphs
```

### 2.2 Triple Mapping

Every TD v2 triple `(subject, relation, object, source, proof)` maps to RDF:

```
TD v2: (paris, capital_of, france, "user", "")

RDF (default graph):
  td_ent:paris  td_rel:capital_of  td_ent:france .

Named graph td_graph:user_facts:
  td_ent:paris  td_rel:capital_of  td_ent:france .
```

**Entities** become `NamedNode` with URI `td_ent:{entity_name}`:
- `paris` → `<http://thinking-dust.org/entity/paris>`
- `south korea` → `<http://thinking-dust.org/entity/south_korea>` (spaces → underscores)
- `2.1M` → `Literal("2100000", datatype=xsd:decimal)` (numeric entities)

**Relations** become `NamedNode` with URI `td_rel:{relation_name}`:
- `capital_of` → `<http://thinking-dust.org/relation/capital_of>`
- `in` → `<http://thinking-dust.org/relation/in>`

### 2.3 Metadata Storage (Per-Fact)

Each fact gets a metadata node using the n-ary relation pattern:

```
# Fact metadata node
td:fact_1  a  td:Fact ;
    td:meta_source      "user" ;
    td:meta_proof       "paris capital_of france → france in eu" ;
    td:meta_confidence  0.95 ;
    td:meta_created     "2026-07-05T04:00:00"^^xsd:dateTime ;
    td:meta_subject     td_ent:paris ;
    td:meta_relation    td_rel:capital_of ;
    td:meta_object      td_ent:france ;
    td:meta_temporal_start  1991 ;
    td:meta_temporal_end    2026 .
```

**Query for metadata about a specific fact:**
```sparql
SELECT ?source ?proof ?confidence WHERE {
    ?fact a td:Fact ;
          td:meta_subject td_ent:paris ;
          td:meta_relation td_rel:capital_of ;
          td:meta_object td_ent:france ;
          td:meta_source ?source ;
          td:meta_proof ?proof ;
          td:meta_confidence ?confidence .
}
```

### 2.4 Named Graphs (Source Grouping)

| Graph Name | Contents | URI |
|------------|----------|-----|
| Default graph | ALL facts (for SPARQL querying) | `DefaultGraph` |
| `td_graph:user_facts` | Facts taught by user | `<http://thinking-dust.org/graph/user>` |
| `td_graph:derived_facts` | Facts derived by inference | `<http://thinking-dust.org/graph/derived>` |
| `td_graph:seed_facts` | Pre-seeded facts | `<http://thinking-dust.org/graph/seed>` |
| `td_graph:metadata` | All metadata nodes | `<http://thinking-dust.org/graph/metadata>` |

**Every fact exists in BOTH the default graph AND its source graph.** The default graph enables unified queries. Source graphs enable provenance filtering.

### 2.5 Relation Properties

```
td_rel:capital_of  td:property  "functional" ;
                   td:inverse_of  td_rel:has_capital .

td_rel:in  td:property  "transitive" .

td_rel:married_to  td:property  "symmetric" .
```

### 2.6 Composition Rules

```
td:rule_1  a  td:CompositionRule ;
    td:rule_first  td_rel:capital_of ;
    td:rule_second td_rel:in ;
    td:rule_result td_rel:in .
```

### 2.7 Temporal Data

Allen's interval relations stored as custom predicates:

```
td_ent:obama_presidency  td:temporal_start  2009 ;
                          td:temporal_end    2017 ;
                          td:allens_relation td:overlaps ;
                          td:temporal_of     td_ent:obama .
```

---

## 3. SPARQL Query Equivalents

### 3.1 Direct Fact Query (existing: `_query_knowledge_graph`)

**TD v2:** `td.ask("is Paris in the EU?")`
**Current:** BFS path search between paris and eu

**SPARQL:**
```sparql
ASK WHERE {
    td_ent:paris td_rel:in td_ent:eu .
}
```
or for multi-hop:
```sparql
ASK WHERE {
    td_ent:paris ?rel1 ?mid .
    ?mid td_rel:in td_ent:eu .
}
```

### 3.2 Inverse Query (existing: heuristic)

**TD v2:** `td.ask("What is the capital of France?")`
**Current:** Linear scan of all triples for matching object

**SPARQL (the proper way):**
```sparql
SELECT ?capital WHERE {
    ?capital td_rel:capital_of td_ent:france .
}
```

**SPARQL with inverse property path (no explicit inverse needed):**
```sparql
SELECT ?capital WHERE {
    ?capital (^td_rel:has_capital) td_ent:france .
}
```

### 3.3 Multi-Hop Transitive Chain

**TD v2:** `td.ask("Is Paris in Europe?")` → Paris→France→EU→Europe

**SPARQL with property path (the killer feature):**
```sparql
ASK WHERE {
    td_ent:paris (td_rel:capital_of/td_rel:in+) td_ent:europe .
}
```

This single pattern matches ANY length chain. No BFS needed. The `+` operator means "one or more."

### 3.4 Functional Contradiction

**TD v2:** `td.ask("Are Paris and Berlin the same?")` → NO

**SPARQL:**
```sparql
SELECT ?paris_country ?berlin_country WHERE {
    td_ent:paris td_rel:capital_of ?paris_country .
    td_ent:berlin td_rel:capital_of ?berlin_country .
    FILTER(?paris_country != ?berlin_country)
}
```

### 3.5 Proof Trace Retrieval

```sparql
SELECT ?mid ?rel1 ?rel2 WHERE {
    td_ent:paris ?rel1 ?mid .
    ?mid ?rel2 td_ent:eu .
    ?fact1 td:meta_subject td_ent:paris ; td:meta_relation ?rel1 ; td:meta_object ?mid .
    ?fact2 td:meta_subject ?mid ; td:meta_relation ?rel2 ; td:meta_object td_ent:eu .
}
```

### 3.6 Temporal Query

```sparql
SELECT ?event WHERE {
    ?event td:temporal_start ?start ;
           td:temporal_end ?end .
    FILTER(?end < 2000)
}
```

### 3.7 Source Filtering

```sparql
SELECT ?s ?p ?o WHERE {
    GRAPH td_graph:user_facts {
        ?s ?p ?o .
    }
}
```

---

## 4. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Layer 5: SPARQL Query Engine (NEW)                               │
│  pyoxigraph Store (Rust-backed, disk-persistent)                │
│  Full SPARQL 1.1: property paths, FILTER, OPTIONAL, subqueries  │
│  File: td/query/__init__.py                                     │
├─────────────────────────────────────────────────────────────────┤
│ Layer 4: Knowledge Graph + Inference Engine (existing)           │
│  BFS path search. Rule templates. Proof traces.                 │
│  NOW: also syncs to SPARQL store on add/remove.                 │
│  File: td/kg/__init__.py                                        │
├─────────────────────────────────────────────────────────────────┤
│ Layer 3: BEAGLE Word Vectors (existing)                          │
├─────────────────────────────────────────────────────────────────┤
│ Layer 2: NL Parser (existing)                                    │
├─────────────────────────────────────────────────────────────────┤
│ Layer 1: HDC + MHN + Z3 (existing)                               │
└─────────────────────────────────────────────────────────────────┘
```

**Query path (new):**
```
User: "is Paris in the EU?"
  ↓
Parser: entities = [paris, eu]
  ↓
SPARQL store: ASK WHERE { td_ent:paris (td_rel:in+) td_ent:eu }
  ↓ If no result
KG: BFS fallback (existing path)
  ↓
Answer + proof trace
```

**Teach path (new):**
```
User: "Paris is the capital of France"
  ↓
Parser: triple = (paris, capital_of, france)
  ↓
KG: add_fact() → existing SQLite + inference
  ↓
SPARQL sync: store.add(Quad(paris, capital_of, france)) + metadata
```

---

## 5. Bridge Layer API

### 5.1 Core Class: `SparqlStore`

```python
class SparqlStore:
    """Bridge between TD v2 SQLite KG and pyoxigraph SPARQL store."""

    def __init__(self, store_path: str = None):
        """Initialize pyoxigraph Store.
        Args:
            store_path: Path for disk persistence. None = in-memory.
        """
        ...

    def sync_from_kg(self, kg: KnowledgeGraph) -> int:
        """Full sync: load all triples from SQLite KG into SPARQL store.
        Returns number of quads synced.
        """
        ...

    def add_fact(self, subject: str, relation: str, obj: str,
                 source: str = "user", proof: str = "",
                 confidence: float = 1.0,
                 temporal_start: int = None, temporal_end: int = None) -> None:
        """Add a single fact to both KG and SPARQL store."""
        ...

    def remove_fact(self, subject: str, relation: str, obj: str) -> None:
        """Remove a fact from both KG and SPARQL store."""
        ...

    def query_sparql(self, sparql: str) -> list[dict]:
        """Execute raw SPARQL query. Returns list of variable bindings."""
        ...

    def ask(self, entity_a: str, entity_b: str, relation: str = None) -> SparqlResult:
        """High-level query: is entity_a related to entity_b?
        Uses SPARQL property paths for multi-hop.
        """
        ...

    def inverse_query(self, relation: str, obj: str) -> list[str]:
        """Find all subjects with given relation to object.
        SPARQL: SELECT ?s WHERE { ?s td_rel:{relation} td_ent:{obj} }
        """
        ...

    def export_turtle(self, path: str) -> None:
        """Export entire store as Turtle RDF file."""
        ...

    def import_turtle(self, path: str) -> int:
        """Import Turtle RDF file into store."""
        ...
```

### 5.2 Entity/Relation URI Mapping

```python
def entity_to_uri(entity: str) -> NamedNode:
    """Convert TD entity name to RDF URI.
    'paris' → NamedNode('http://thinking-dust.org/entity/paris')
    'south korea' → NamedNode('http://thinking-dust.org/entity/south_korea')
    """
    normalized = entity.lower().replace(" ", "_").replace("-", "_")
    return NamedNode(f"{TD_ENT}{normalized}")

def relation_to_uri(relation: str) -> NamedNode:
    """Convert TD relation name to RDF URI."""
    return NamedNode(f"{TD_REL}{relation}")

def uri_to_entity(uri: NamedNode) -> str:
    """Convert RDF URI back to TD entity name."""
    return uri.value.replace(TD_ENT, "").replace("_", " ")

def uri_to_relation(uri: NamedNode) -> str:
    """Convert RDF URI back to TD relation name."""
    return uri.value.replace(TD_REL, "")
```

### 5.3 Fact Metadata Pattern

```python
def _fact_to_quads(self, subject: str, relation: str, obj: str,
                   source: str, proof: str, confidence: float,
                   temporal_start: int = None,
                   temporal_end: int = None) -> list[Quad]:
    """Convert a TD fact to RDF quads (assertion + metadata)."""
    s_uri = entity_to_uri(subject)
    r_uri = relation_to_uri(relation)
    o_uri = entity_to_uri(obj)

    quads = []

    # 1. Main assertion — default graph
    quads.append(Quad(s_uri, r_uri, o_uri))

    # 2. Main assertion — source graph
    source_graph = NamedNode(f"{TD_GRAPH}{source}")
    quads.append(Quad(s_uri, r_uri, o_uri, source_graph))

    # 3. Metadata node
    fact_id = self._next_fact_id()
    fact_node = NamedNode(f"{TD_META}fact_{fact_id}")
    quads.append(Quad(fact_node, RDF_TYPE, TD_FACT, META_GRAPH))
    quads.append(Quad(fact_node, TD_META_SUBJECT, s_uri, META_GRAPH))
    quads.append(Quad(fact_node, TD_META_RELATION, r_uri, META_GRAPH))
    quads.append(Quad(fact_node, TD_META_OBJECT, o_uri, META_GRAPH))
    quads.append(Quad(fact_node, TD_META_SOURCE, Literal(source), META_GRAPH))
    if proof:
        quads.append(Quad(fact_node, TD_META_PROOF, Literal(proof), META_GRAPH))
    quads.append(Quad(fact_node, TD_META_CONFIDENCE,
                      Literal(confidence, datatype=XSD_FLOAT), META_GRAPH))
    if temporal_start is not None:
        quads.append(Quad(fact_node, TD_META_TEMPORAL_START,
                          Literal(temporal_start, datatype=XSD_INT), META_GRAPH))
    if temporal_end is not None:
        quads.append(Quad(fact_node, TD_META_TEMPORAL_END,
                          Literal(temporal_end, datatype=XSD_INT), META_GRAPH))

    return quads
```

---

## 6. Installation & Integration

### 6.1 Install pyoxigraph

```bash
cd ~/Documents/Thinking\ Dust/td-v2
arch -arm64 .venv-arm64/bin/pip install pyoxigraph
```

### 6.2 Integration Points

| File | Change | Description |
|------|--------|-------------|
| `td/query/__init__.py` | NEW | `SparqlStore` class — bridge layer |
| `td/kg/__init__.py` | MODIFY | `add_fact()` calls `sparql_store.sync_fact()` |
| `td/thinking.py` | MODIFY | `_query_knowledge_graph()` tries SPARQL first, falls back to BFS |
| `demos/chat_flare.py` | MODIFY | Show SPARQL query in trace mode |
| `tests/test_sparql.py` | NEW | Tests for bridge layer |
| `requirements.txt` | MODIFY | Add `pyoxigraph` |
| `pyproject.toml` | MODIFY | Add `pyoxigraph` to dependencies |

### 6.3 Backward Compatibility

- **SQLite KG remains the source of truth.** SPARQL store is a query accelerator.
- **SPARQL store is rebuilt from SQLite on startup** (sync_from_kg).
- **All existing tests pass unchanged.** SPARQL is an additional query path, not a replacement.
- **BFS fallback:** If SPARQL returns no result, existing BFS path search is used.

---

## 7. What This Enables (That Wasn't Possible Before)

| Capability | Before | After |
|-----------|--------|-------|
| Inverse query | Heuristic linear scan | `SELECT ?s WHERE { ?s td:capital_of td:france }` |
| Multi-hop | Custom BFS, max 100 hops | `(td:in+)` — unlimited, O(1) per hop |
| Paraphrase | BEAGLE only | SPARQL + BEAGLE fallback |
| Temporal range | Manual interval check | `FILTER(?start > 2000 && ?end < 2025)` |
| Export/Import | Custom pickle | Standard RDF (Turtle, JSON-LD, N-Triples) |
| Source filtering | SQL WHERE clause | `GRAPH td:derived { ?s ?p ?o }` |
| Aggregate queries | Not supported | `COUNT`, `GROUP BY`, `ORDER BY` |
| Federated queries | Not possible | Can query external SPARQL endpoints |

---

## 8. Performance Expectations

| Operation | pyoxigraph (1M triples) | Current BFS (1K triples) |
|-----------|------------------------|--------------------------|
| Direct fact lookup | <1ms | ~5ms |
| 2-hop transitive | <5ms | ~10ms |
| 6-hop transitive | <10ms | ~50ms |
| Inverse query | <2ms | ~100ms (linear scan) |
| Full-text search | <5ms | Not supported |

---

## 9. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| pyoxigraph not on M1 arm64 | Blocks installation | PyO3 builds universal wheels; test first |
| Sync drift (SQLite ≠ SPARQL) | Wrong answers | Always sync on add/remove; rebuild on startup |
| Memory at 1M triples | ~200MB | Use disk-backed `Store(path)` |
| SPARQL syntax errors | Crash | Validate queries before execution |
| Breaking existing tests | Regression | SPARQL is additive; BFS fallback preserved |

---

## 10. Implementation Order

1. **Install pyoxigraph** — verify it works on M1 arm64
2. **URI mapping** — entity/relation ↔ RDF URI conversion
3. **SparqlStore core** — init, add_fact, query_sparql
4. **Sync from KG** — bulk load all SQLite triples into SPARQL
5. **High-level queries** — ask(), inverse_query() with SPARQL
6. **Integration** — wire into thinking.py as primary query path
7. **Tests** — mirror all 437 existing tests through SPARQL
8. **Export/Import** — Turtle, JSON-LD
9. **Documentation** — update ARCHITECTURE.md, DEVELOPMENT.md

---

## References

| Reference | Year | What | Relevance |
|-----------|------|------|-----------|
| pyoxigraph | 2026 | Rust-backed SPARQL 1.1 store | Core engine |
| Oxigraph benchmarks (Trainmarks) | 2026 | 18ms @ 10M triples | Performance validation |
| W3C SPARQL 1.1 | 2013 | Query language standard | API design |
| W3C RDF 1.1 Named Graphs | 2014 | Quad-based provenance | Metadata model |
| PROV-O (W3C) | 2013 | Provenance ontology | Source/proof encoding |
| Sahaj Software | 2023 | Verb-based clause splitting | Clause segmentation |
| triplet-extract | 2025 | Stanford OpenIE port | Clause splitting reference |
| Allen (1983) | 1983 | Temporal intervals | Temporal data model |
