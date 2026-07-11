# Preprocessing Layer — TEACH Prompt (v1 Final)

**⚠️ THIS PROMPT IS FOR TEACHING FACTS ONLY. ⚠️**
**⚠️ FOR USER QUERIES, SEE: `QUERY_PREPROCESSING_PROMPT.md` ⚠️**

**When to use this prompt:**
- User is TEACHING facts: "Paris is the capital of France"
- User is providing knowledge: "seals are marine mammals that live in cold waters"
- Input is DECLARATIVE (statements, not questions)

**When NOT to use this prompt:**
- User is ASKING questions: "What do seals eat?" → use `QUERY_PREPROCESSING_PROMPT.md`
- User is QUERYING the KG: "Is Python good for data science?" → use `QUERY_PREPROCESSING_PROMPT.md`
- Input is INTERROGATIVE (questions, requests for information)

**Why two prompts?**
- This prompt: Rule 7 says "rewrite as declarative" — correct for teach, WRONG for queries
- Query prompt: Rule 7 says "keep question mark" — correct for queries, WRONG for teach
- Using this prompt for queries causes hallucinations (LLM invents answers instead of formatting queries)
- Using query prompt for teach produces questions instead of facts

**Architecture:** User → LLM Preprocessor → TD v2 (reasoning engine)

**Version:** v1 (tested against v2 by GLM 5.2 — v1 won on all test cases)
**Model:** Gemini (selected after comparison with Kimi K2.7 Code and Kimi K2.6)

---

## Why v1 > v2

| Criteria | v1 | v2 (GLM 5.2) |
|----------|----|----|
| Underscored relations | ✅ `tool_for`, `created_in` | ❌ Plain verbs |
| Subject consistency | ✅ Same subject throughout | ⚠️ Introduces "Developers" |
| No invented entities | ✅ No new entities | ❌ "Developers" not in original |
| Parser-friendly | ✅ `Python tool_for data science` → clean triple | ⚠️ Complex SVO |
| Question handling | ✅ Keeps question form | ❌ Converts to declarative |
| Simplicity | ✅ 8 rules, concise | ⚠️ 11 rules, verbose |

---

## Model Selection (2026-07-11)

Tested three LLMs with the v1 prompt on two complex sentences.

### Test 1: "seals are marine mammals that live in cold waters along the Atlantic coast and they haul out on rocks to rest"

| # | Gemini | K2.7 Code | K2.6 |
|---|--------|-----------|------|
| 1 | `seals is_a marine mammals` | `seals is_a marine mammals` | `Seals are marine mammals.` |
| 2 | `seals live in cold waters` | `seals live in cold waters` | `Seals live in cold waters.` |
| 3 | `seals live along Atlantic coast` | `cold waters are along the Atlantic coast` ⚠️ | `Seals live along the Atlantic coast.` |
| 4 | `seals haul out on rocks` | `seals haul out on rocks` | `Seals haul out on rocks.` |
| 5 | `seals haul out to rest` | `seals rest` | `Seals rest.` |

### Test 2: "Python is a programming language that is used for data science and web development and it was created by Guido van Rossum in the Netherlands"

| # | Gemini | K2.7 Code | K2.6 |
|---|--------|-----------|------|
| 1 | `Python is_a programming language` | `Python is_a programming language` | `Python is_a programming_language.` |
| 2 | `Python tool_for data science` ✅ | `Python used_for data science` ✅ | `Python used_for data_science.` |
| 3 | `Python tool_for web development` ✅ | `Python used_for web development` ✅ | `Python used_for web_development.` |
| 4 | `Guido van Rossum created Python` | `Python created_by Guido van Rossum` | `Python created_by Guido_van_Rossum.` |
| 5 | `Python created_in the Netherlands` | `Python created_in Netherlands` | `Python created_in the_Netherlands.` |

### Model Comparison

| Criteria | Gemini | K2.7 Code | K2.6 |
|----------|--------|-----------|------|
| Underscored relations | ✅ `tool_for` | ✅ `used_for` | ✅ `used_for` |
| Entity names | ✅ Clean | ⚠️ `cold waters are along...` | ⚠️ `programming_language` (extra underscores) |
| Consistent subjects | ✅ | ✅ | ✅ |
| Speed | Fast (~28s) | Fast (~26s) | **Slow** (~180s, 2322 reasoning tokens) |
| No invented entities | ✅ | ⚠️ | ✅ |

**Selected model: Gemini** — cleanest output, fastest, no weird artifacts.

**K2.6 note:** Reasoning model (spends 2322 tokens thinking before outputting). Good for complex reasoning but overkill for simple sentence splitting. K2.7 Code is faster and equally good.

**K2.7 Code note:** Good output but creates occasional weird triples (`cold waters are along the Atlantic coast`). Not as clean as Gemini.

---

## The Prompt

```
You are a sentence simplifier for a knowledge graph engine.
Your job: break complex human input into SIMPLE atomic sentences.

## Rules

1. ONE fact per sentence. No compound sentences.
2. ALWAYS repeat the subject. Never use pronouns (he/she/it/they/that).
3. Remove filler words: "so", "like", "you know", "I mean", "basically",
   "actually", "well", "right", "I was wondering", "can you tell me".
4. Use simple SVO structure: Subject Verb Object.
5. For "is a" facts, use: "X is_a Y" (with underscore).
6. For relations, use underscored compound: "X tool_for Y", "X part_of Y".
7. Questions: rewrite as declarative if possible.
8. Output ONLY valid JSON. No explanation.

## Output Format

{"sentences": ["sentence 1", "sentence 2", ...]}

## Examples

INPUT: "So I was curious, what's the deal with matches and fire?"
OUTPUT: {"sentences": ["what is match used for"]}

INPUT: "I was wondering if you could tell me what a match is used for
starting fires?"
OUTPUT: {"sentences": ["what is match used for fire"]}

INPUT: "The cell in biology, what's it made of vs the one in prison?"
OUTPUT: {"sentences": ["what is cell in biology made of", "what is cell in prison"]}

INPUT: "Like, cell phones and stuff, they use towers right?"
OUTPUT: {"sentences": ["cell phone uses tower"]}

INPUT: "Paris is the capital of France and France is in the EU"
OUTPUT: {"sentences": ["Paris is the capital of France", "France is in the EU"]}

INPUT: "Alice and Bob went to the store"
OUTPUT: {"sentences": ["Alice went to the store", "Bob went to the store"]}

INPUT: "the thing that you strike to make fire, what's it called"
OUTPUT: {"sentences": ["what is the thing that is struck to make fire"]}

INPUT: "Python is a programming language that's used for data science
and web development"
OUTPUT: {"sentences": ["Python is_a programming language", "Python is used for data science", "Python is used for web development"]}

INPUT: "what about prison?"
OUTPUT: {"sentences": ["what about cell in prison"]}

INPUT: "how does it work?"
OUTPUT: {"sentences": ["how does it work"]}

INPUT: "the match ignites when you strike it against a rough surface"
OUTPUT: {"sentences": ["match ignites when struck against rough surface"]}

INPUT: "cranes are large wading birds with long legs that migrate
thousands of miles"
OUTPUT: {"sentences": ["cranes are large wading birds", "cranes have long legs", "cranes migrate thousands of miles"]}

INPUT: "a spring is a natural source of water from underground that is
often bottled for drinking"
OUTPUT: {"sentences": ["spring is_a natural source of water", "spring is from underground", "spring is bottled for drinking"]}

INPUT: "seals are marine mammals that live in cold waters along the
Atlantic coast and they haul out on rocks to rest"
OUTPUT: {"sentences": ["seals are marine mammals", "seals live in cold waters", "seals live along Atlantic coast", "seals haul out on rocks"]}

INPUT: "Python is a programming language that is used for data science
and web development and it was created by Guido van Rossum in the
Netherlands"
OUTPUT: {"sentences": ["Python is_a programming language", "Python tool_for data science", "Python tool_for web development", "Guido van Rossum created Python", "Python created_in the Netherlands"]}
```

---

## Parser Compatibility

The simplified sentences must follow these patterns for the parser to extract triples:

| Pattern | Example | Triple Extracted |
|---------|---------|-----------------|
| X is_a Y | "cell is_a organelle" | (cell, is_a, organelle) |
| X is the Y of Z | "Paris is the capital of France" | (Paris, capital_of, France) |
| X is in Y | "France is in the EU" | (France, in, EU) |
| X verb Y | "match ignites surface" | (match, ignites, surface) |
| X compound_rel Y | "match tool_for fire" | (match, tool_for, fire) |
| X is Y | "spring is season" | (spring, is_a, season) |

**What the parser CANNOT handle:**
- Nested relative clauses: "the thing that you strike" → simplify
- Pronouns without antecedents: "it", "they", "that" → repeat subject
- Complex coordination: "X and Y verb Z" → split into two sentences
- Passive voice without agent: "was eroded" → "flooding eroded river bank"
- Infinitive complements: "allows devices to connect" → "port connects devices"
- Multiple prepositions: "live in cold waters along the coast" → split
