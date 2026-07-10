# TD v2 — Automated WSD & Parser Testing Framework

**Date:** 2026-07-09
**Status:** SPEC — for automated continuous testing

---

## 1. Problem

Current tests are static — pre-determined words, pre-written sentences, known expected outputs. This means:
- We test what we already know, not what we don't
- No coverage of unseen domains or novel polysemous words
- No measurement of generalization ability
- No automated way to detect regressions

## 2. Solution: LLM-Driven Dynamic Testing

An LLM agent (like me) dynamically:
1. Picks random polysemous words from Wikipedia
2. Fetches real Wikipedia content for each sense
3. Teaches TD v2 using that content
4. Generates novel test sentences (NOT from the glosses)
5. Evaluates correctness
6. Reports tabular results + overall score

## 3. Pipeline

```
┌─────────────────────────────────────────────────────────┐
│ STEP 1: Word Selection                                  │
│   - Pick random polysemous word from Wikipedia          │
│   - Or use WordNet to find words with ≥2 synsets        │
│   - Verify word has ≥2 distinct meanings                │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 2: Gloss Gathering                                 │
│   - Fetch Wikipedia page for each sense                 │
│   - Extract 6-8 representative sentences per sense      │
│   - Sentences must be REAL (not generated)              │
│   - Document source URL for each sense                  │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 3: Teaching                                        │
│   - Create fresh TD v2 instance (no prior knowledge)    │
│   - Teach sense glosses using td.teach() or             │
│     lesk_wsd.add_sense_example()                        │
│   - Record: senses created, triples extracted,          │
│     Lesk gloss words, BEAGLE clusters                   │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 4: Novel Test Generation                           │
│   - LLM generates 8-12 test sentences per word          │
│   - Sentences MUST NOT appear in the glosses            │
│   - Cover: different syntactic structures, edge cases,  │
│     ambiguous contexts, subordinate senses              │
│   - Assign expected sense to each sentence              │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 5: Evaluation                                      │
│   - Run each test sentence through Lesk resolution      │
│   - Record: predicted sense, expected sense, match?     │
│   - Track: accuracy, fallback rate, false positive rate │
│   - Compare against baselines: random, MFS              │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│ STEP 6: Reporting                                       │
│   - Tabular results per word, per sense                 │
│   - Overall accuracy with confidence intervals          │
│   - Error analysis: which senses are hardest?           │
│   - Regression check against previous runs              │
└─────────────────────────────────────────────────────────┘
```

## 4. Scoring Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| **Accuracy** | correct / (total - fallback) | ≥85% |
| **Fallback rate** | fallback / total | ≤20% |
| **False positive rate** | wrong_confident / total | ≤10% |
| **Per-sense recall** | correct_for_sense / total_for_sense | ≥80% per sense |
| **Novelty score** | test_sentences_not_in_glosses / total_tests | 100% |

## 5. Test Sentence Quality Criteria

LLM-generated test sentences must satisfy:

1. **Novelty:** No sentence may appear in the gloss
2. **Naturalness:** Must read like real English, not templates
3. **Syntactic diversity:** Mix declarative, passive, questions, relative clauses
4. **Lexical diversity:** Different vocabulary from glosses (tests generalization)
5. **Edge cases:** Include sentences where the sense is ambiguous or borderline
6. **Domain coverage:** Cover subordinate senses (e.g., "vampire bat" not just "bat")
7. **UNAMBIGUOUS QUESTIONS (CRITICAL):** Every test question must have ONE clear expected answer. If the question is ambiguous (e.g., "what is a match" — could be fire/sports/tool), it MUST accept ANY valid answer as correct.

### Rule: Ambiguous Questions Accept Any Answer

```
❌ BAD:  "what is a match" → expected: fire        (ambiguous — sports/tool also valid)
✅ GOOD: "what is a tennis match" → expected: sports (unambiguous — tennis = context)
✅ GOOD: "what is a match used for" → expected: fire (unambiguous — "used for" = fire context)
```

If a question IS ambiguous, mark it as `ambiguous=True` and accept any of the valid senses:
```python
('what is a match', ['fire', 'sports', 'tool'], True)  # ambiguous=True
```

**Why:** An ambiguous question with multiple valid answers is NOT a system failure — it's a test design failure. The system correctly returns the strongest association.

## 6. Baselines

| Baseline | How | Expected |
|----------|-----|----------|
| **Random** | Pick random sense | 1/N_senses accuracy |
| **Most Frequent Sense (MFS)** | Always pick sense 0 | ~50% for balanced data |
| **Lesk (our algorithm)** | Full pipeline | ≥85% target |

## 7. Regression Detection

After each code change:
1. Run the framework on the SAME set of test words
2. Compare accuracy against the previous run
3. If accuracy drops >5%, flag as regression
4. Investigate: is it a code bug or a test quality issue?

## 8. Required Output Format

Every benchmark run MUST produce two tables:

### Table 1: Teaching Triples

```
═══════════════════════════════════════════════════════════════
  PHASE 1: TEACH — sentence → triple → KG
═══════════════════════════════════════════════════════════════

    # Sense        Teach Sentence                              Extracted Triple
  --- ------------ ------------------------------------------- ---------------------------
    1 fire         matches are made of small wooden sticks      (matches, made_of, small wooden sticks)
    2 fire         wooden matches are packaged in matchboxes    (wooden matches, packaged_in, matchboxes)
    3 sports       a match is a competitive game between playe  (match, is, competitive game)
    ...
```

Columns: `#` (row number), `Sense` (expected sense label), `Teach Sentence` (input), `Extracted Triple` (s, r, o) or "(no triple extracted)".

**CRITICAL:** The Extracted Triple column MUST be shown in every benchmark run. Without it, you can't verify what was actually stored in the KG or diagnose extraction failures.

### Table 2: Questions & Answers

```
═══════════════════════════════════════════════════════════════
  PHASE 2: QUESTIONS & ANSWERS
═══════════════════════════════════════════════════════════════

    # Expected      OK?  Conf  Method              Question                               Answer
  --- ------------ ---- ------ -------------------- -------------------------------------- -------------------------
    1 fire            ✓   0.7  z3_ordered           what is a match used for?              {'type': 'generic_csp'...}
    2 sports          ✗  0.15  unknown              the team won the match                 {'type': 'unknown'...}
    ...
```

Columns: `#`, `Expected` (expected sense), `OK?` (✓/✗), `Conf` (confidence), `Method` (retrieval/sparql/z3/etc.), `Question`, `Answer` (truncated).

### Summary

```
  SCORE: 5/12 = 41.7%
  Random baseline: 33.3%
```

## 9. Example Run Output

```
╔══════════════════════════════════════════════════════════════════╗
║              TD v2 WSD BENCHMARK — 2026-07-09                   ║
╠═══════════╦═══════╦════════════╦════════════╦════════════════════╣
║ Word      ║ Senses║ Accuracy   ║ Fallback   ║ Notes              ║
╠═══════════╬═══════╬════════════╬════════════╬════════════════════╣
║ cell      ║ 3     ║ 100% (18/18)║ 0%        ║ Wikipedia glosses  ║
║ bank      ║ 2     ║ 100% (12/12)║ 0%        ║ Wikipedia glosses  ║
║ bat       ║ 2     ║ 90% (9/10) ║ 10%       ║ 1 fallback         ║
║ crane     ║ 2     ║ 100% (8/8) ║ 0%        ║                    ║
║ apple     ║ 2     ║ 100% (12/12)║ 0%        ║                    ║
║ python    ║ 2     ║ 100% (12/12)║ 0%        ║                    ║
║ mercury   ║ 2     ║ 100% (12/12)║ 0%        ║                    ║
╠═══════════╬═══════╬════════════╬════════════╬════════════════════╣
║ TOTAL     ║ 15    ║ 98.5%(83/84)║ 1.2%      ║ Random: 50%        ║
╚═══════════╩═══════╩════════════╩════════════╩════════════════════╝

Baselines: Random=50.0%, MFS=45.5%
Verdict: Lesk WSD outperforms both baselines by >48 percentage points.
```

## 9. Automation (Future)

```python
# Future: automated testing script
python tests/automated_wsd_benchmark.py \
  --words cell,bank,bat,crane,apple,python,mercury \
  --sentences-per-sense 8 \
  --test-sentences-per-word 10 \
  --gloss-source wikipedia \
  --output results/2026-07-09.json
```

This script would:
1. Fetch Wikipedia content dynamically
2. Build glosses from real text
3. Generate test sentences via LLM
4. Run the benchmark
5. Output JSON results for regression tracking

## 10. Known Limitations

1. **Gloss quality depends on Wikipedia.** Words without good Wikipedia articles get poor glosses.
2. **Test sentence quality depends on LLM.** If the LLM generates template-like sentences, the benchmark is easier.
3. **No cross-lingual testing yet.** All tests are English.
4. **No rare sense testing.** Common senses are overrepresented.
5. **Teach path doesn't use full sentences for Lesk glosses.** The `teach()` path extracts triples and loses context words. Enriching glosses manually (via `add_sense_example`) gives better results.

---

*"Test with what you don't know, not what you do."*
