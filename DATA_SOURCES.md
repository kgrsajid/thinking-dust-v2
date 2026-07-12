# Data Sources Plan — TD v2 Knowledge Acquisition

_Last updated: 2026-07-12_

---

## Overview

TD v2 needs millions of facts to answer real-world questions. This document defines where to get structured knowledge, how to load it into `teach()`, and what format conversions are needed.

**Key insight:** Most structured knowledge is already in `(subject, relation, object)` format. The challenge is ID→name mapping, not sentence parsing.

---

## Source 1: Wikidata5m (PRIMARY — Ready to Use)

### What It Is

- **5 million entities**, 822 relations, **20 million triples**
- Aligned with Wikipedia (each entity has a Wikipedia page)
- Standard benchmark for KG reasoning (KEPLER paper, TACL 2021)
- Free download, CC0 license

### Downloads

| File | Size | URL | Contents |
|------|------|-----|----------|
| Transductive split | 160 MB | [download](https://www.dropbox.com/s/6sbhm0rwo4l73jq/wikidata5m_transductive.tar.gz?dl=1) | Train/valid/test triples |
| Inductive split | 160 MB | [download](https://www.dropbox.com/s/csed3cgal3m7rzo/wikidata5m_inductive.tar.gz?dl=1) | Train/valid/test triples |
| Raw triples | 168 MB | [download](https://www.dropbox.com/s/563omb11cxaqr83/wikidata5m_all_triplet.txt.gz?dl=1) | All triples (includes entities without Wikipedia) |
| Corpus (text) | 991 MB | [download](https://www.dropbox.com/s/7jp4ib8zo3i6m10/wikidata5m_text.txt.gz?dl=1) | Wikipedia descriptions per entity |
| Entity aliases | 188 MB | [download](https://www.dropbox.com/s/lnbhc8yuhit4wm5/wikidata5m_alias.tar.gz?dl=1) | Human-readable names for entities |

### Actual Format (Verified)

**Triples file** (`wikidata5m_transductive_train.txt`):
```
Q22686	P39	Q11696
Q22686	P27	Q30
Q22686	P106	Q82955
```
Format: `entity_id \t relation_id \t entity_id` (tab-separated, Wikidata Q/P IDs)

**Alias file** (`entity_aliases.del`):
```
Q22686 donnie trump	45th president of the united states	Donald John Trump	Donald Trump
```
Format: `entity_id \t alias1 \t alias2 \t alias3 ...` (tab-separated)

**Corpus file** (`wikidata5m_text.txt`):
```
Q22686	Donald John Trump (born June 14, 1946) is the 45th and current president of the United States...
```
Format: `entity_id \t description_text` (tab-separated)

**Relation aliases** (`relation_aliases.del`):
```
P39	position held	government position held
P27	country of citizenship	nationality
P106	occupation	profession
```
Format: `relation_id \t alias1 \t alias2 ...` (tab-separated)

### Conversion to TD v2 teach()

**Option A: Direct (for small datasets, <10K triples)**
```python
td.teach("Donald Trump position held President of the United States", "President of the United States")
```

**Option B: Bulk loader (for large datasets, >10K triples) — PREFERRED**
```python
from td.bulk_loader import BulkLoader

loader = BulkLoader(td, run_inference=True)
stats = loader.load_wikidata5m(
    triples_path="data/wikidata5m_transductive_train.txt",
    aliases_path="data/entity_aliases.del",
    relation_aliases_path="data/relation_aliases.del",
)
print(stats.summary())
# → Loaded 20,614,279 triples (4,594,485 entities, 822 relations) in ~30min
```

The bulk loader:
- Bypasses the parser (no sentence parsing needed)
- Maps Wikidata IDs to human-readable names via alias files
- Runs `derive_all()` once after loading (not per-fact)
- Returns stats (triples loaded, entities, relations, timing)

### Relation Name → TD v2 Property Mapping

Wikidata relation IDs need to be mapped to TD v2 relation properties:

| Wikidata ID | Name | TD v2 Property |
|-------------|------|---------------|
| P39 | position held | — (custom) |
| P27 | country of citizenship | `in` (transitive) |
| P106 | occupation | — (custom) |
| P131 | located in administrative territory | `in` (transitive) |
| P17 | country | `in` (transitive) |
| P31 | instance of | `is_a` (custom) |
| P279 | subclass of | `is_a` (transitive) |
| P361 | part of | `part_of` (transitive) |
| P527 | has part | inverse of `part_of` |
| P463 | member of | `in` (transitive) |
| P150 | contains administrative territory | inverse of `in` |
| P706 | located in/on physical feature | `in` (transitive) |
| P127 | owned by | `owned_by` |
| P355 | has subsidiary | inverse of `owned_by` |
| P169 | chief executive officer | — (custom) |
| P740 | location of formation | `created_in` |
| P159 | headquarters location | `located_in` |

### Expected Scale After Loading

| Metric | Value |
|--------|-------|
| Total triples | ~20 million |
| Unique entities | ~4.6 million |
| Unique relations | 822 |
| Storage (pyoxigraph) | ~2-5 GB |
| Query speed (estimated) | 18-50ms (pyoxigraph at 10M) |

---

## Source 2: Wikidata API (REAL-TIME — Supplementary)

### What It Is

- Live API to query Wikidata's 16 billion triples
- Free, no API key needed for basic queries
- SPARQL endpoint + REST API

### Links

| Resource | URL |
|----------|-----|
| SPARQL endpoint | `https://query.wikidata.org/sparql` |
| REST API | `https://www.wikidata.org/w/api.php` |
| Entity lookup | `https://www.wikidata.org/wiki/Special:EntityData/Q42.json` |
| API docs | `https://www.wikidata.org/wiki/Wikidata:Data_access` |

### Example SPARQL Query

```sparql
# Get all countries and their capitals
SELECT ?country ?countryLabel ?capital ?capitalLabel WHERE {
  ?country wdt:P31 wd:Q6256.        # instance of: country
  ?country wdt:P36 ?capital.         # has capital
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT 100
```

### Use Cases

- **On-demand lookup:** User asks about an entity not in local KG → fetch from Wikidata API
- **Domain expansion:** Load specific domains (all countries, all proteins, all movies)
- **Verification:** Cross-check local facts against Wikidata

---

## Source 3: Domain-Specific Datasets

### Geography (Countries, Capitals, Rivers)

**Source:** Wikidata5m filtered subset
**Filter:** `P31 = Q6256 (country)` + `P36 (capital)` + `P17 (country)`
**Expected:** ~200 countries, ~5,000 geographic entities
**Use case:** "Is Paris in Europe?" → transitive inference through country→continent

### Biology (Taxonomy, Organisms)

**Source:** Wikidata5m filtered subset
**Filter:** `P31 = Q7239 (organism)` + `P279 (subclass of)` + `P703 (found in taxon)`
**Expected:** ~100,000 species, ~500,000 triples
**Use case:** "Is a dog a mammal?" → transitive is_a chain

### Technology (Programming, Companies)

**Source:** Wikidata5m filtered subset
**Filter:** `P31 = Q7397 (software)` + `P178 (developer)` + `P277 (programming language)`
**Expected:** ~10,000 software entities
**Use case:** "Who created Python?" → 1-hop relation

### History (Events, People, Dates)

**Source:** Wikidata5m filtered subset
**Filter:** `P31 = Q5 (human)` + `P569 (date of birth)` + `P39 (position held)`
**Expected:** ~1 million people
**Use case:** "Was Obama president before Trump?" → temporal reasoning

---

## Source 4: Existing TD v2 Synthetic Corpus

### What We Have

| File | Size | Contents |
|------|------|----------|
| `data/synthetic_corpus_10k.txt` | 10K sentences | 12 domains, Kimi K2.6 generated |
| `data/word_vectors_110k.pkl` | 548 MB | BEAGLE vectors, 10K dims, 2890 vocab |
| `data/skipped_sentences.log` | 26 lines | Sentences parser couldn't extract |

### What's Missing

- Vocabulary too small (2890 words) — "used", "for", "match" not in vocab
- Need to scale to 1M+ sentences for adequate coverage
- Need domain-specific corpus aligned with Wikidata5m entities

### Scaling Plan

1. **Extract entity descriptions** from Wikidata5m corpus (991 MB)
2. **Train BEAGLE** on 1M+ sentences from Wikipedia descriptions
3. **Expected vocabulary:** 20,000-50,000 words
4. **Expected training time:** ~10-30 minutes on CPU (with Random Permutations)

---

## Source 5: Freebase (Historical — Deprecated)

### What It Was

- Google's knowledge base, 1.9 billion triples
- Superseded by Wikidata
- Still available as historical dataset

### Why NOT Use It

- Deprecated since 2016
- Wikidata is actively maintained
- Wikidata5m is a curated subset of Wikidata

---

## Loading Pipeline

```
Wikidata5m (download)
    ↓
Parse: entity_aliases.del → entity_name_map
Parse: relation_aliases.del → relation_name_map
Parse: wikidata5m_transductive_train.txt → triple_list
    ↓
For each triple (Q_subject, P_relation, Q_object):
    subject = entity_name_map[Q_subject]  # "Donald Trump"
    relation = relation_name_map[P_relation]  # "position held"
    object = entity_name_map[Q_object]  # "President of the United States"
    ↓
    td.teach(f"{subject} {relation} {object}", object)
    ↓
pyoxigraph stores as RDF triple
BEAGLE updates context vectors
    ↓
Query: "Who held the position of President?" → searches KG → "Donald Trump"
```

### Performance Estimates

| Stage | Time (20M triples) | Notes |
|-------|-------------------|-------|
| Download | ~10 min | 160 MB compressed |
| Parse aliases | ~30 sec | Tab-separated text |
| Parse triples | ~10 sec | Tab-separated text |
| Bulk load to pyoxigraph | ~30-60 min | Rust-backed, fast |
| BEAGLE training | ~30 min | 1M+ sentences, RP mode |
| **Total** | **~1-2 hours** | From download to queryable |

---

## Priority Order

| Phase | Data Source | Triples | Effort |
|-------|-----------|---------|--------|
| **Phase 1** | Wikidata5m transductive train | 20M | Download + loader script |
| **Phase 2** | Domain subsets (geography, biology) | ~500K | Filter from Phase 1 |
| **Phase 3** | Wikidata API (real-time) | 16B (live) | API integration |
| **Phase 4** | BEAGLE corpus scaling (1M+ sentences) | — | Extract from corpus file |

---

## TODO

- [ ] Download Wikidata5m transductive split (160 MB)
- [ ] Download Wikidata5m entity aliases (188 MB)
- [ ] Download Wikidata5m relation aliases (in same archive)
- [ ] Download Wikidata5m corpus (991 MB) — for BEAGLE training
- [ ] Write `scripts/load_wikidata5m.py` — parse + load into pyoxigraph
- [ ] Write `scripts/filter_domain.py` — extract domain-specific subsets
- [ ] Benchmark: query speed at 1M, 5M, 20M triples
- [ ] Scale BEAGLE training to 1M+ sentences from corpus
- [ ] Map Wikidata relation IDs to TD v2 relation properties

---

## References

| # | Source | Size | License | URL |
|---|--------|------|---------|-----|
| 1 | Wikidata5m | 5M entities, 20M triples | CC0 | `https://deepgraphlearning.github.io/project/wikidata5m` |
| 2 | Wikidata API | 16B triples | CC0 | `https://www.wikidata.org/wiki/Wikidata:Data_access` |
| 3 | Wikidata SPARQL | 16B triples | CC0 | `https://query.wikidata.org/sparql` |
| 4 | KEPLER paper | — | — | Wang et al. (2021), TACL, arXiv:1911.06136 |
| 5 | Wikidata5m-si | Semi-inductive split | CC0 | `https://github.com/uma-pi1/wikidata5m-si` |

---

_"The facts already exist. We just need to load them."_
