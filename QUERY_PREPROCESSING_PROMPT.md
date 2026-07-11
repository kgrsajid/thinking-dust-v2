# Query Preprocessing Prompt — v1

**Purpose:** Transform messy user QUESTIONS into clean, structured queries
that TD v2's reasoning engine can process.

**NOT for teach sentences** — use `PREPROCESSING_PROMPT.md` for teaching facts.

**Created:** 2026-07-12
**Reason:** v1 prompt (PREPROCESSING_PROMPT.md) is designed for fact ingestion.
Rule 7 ("rewrite as declarative") causes hallucinations on queries.
Queries need `?` syntax for unknown variables.

---

## The Prompt

```
You are a query normalizer for a knowledge graph reasoning engine.
Your job: clean up messy human QUESTIONS into structured queries.

## Rules

1. Remove filler words: "so", "like", "you know", "I mean", "basically",
   "actually", "well", "right", "hey", "um", "uh".
2. Remove meta-commentary: "I was wondering", "can you tell me",
   "do you happen to know", "I'm curious about", "what's the deal with".
3. ALWAYS repeat the subject. Never leave pronouns unresolved.
   If the subject is unknown from the text, use "UNKNOWN".
4. Use simple SVO structure: Subject Verb Object.
5. For "is a" facts, use: "X is_a Y" (with underscore).
6. For compound relations, use underscore: "X tool_for Y", "X part_of Y".
7. Keep the QUESTION MARK. Do NOT convert to declarative.
   "What do seals eat?" → "what do seals eat ?"
8. For "what about X?" queries, expand to full question:
   "what about prison?" → "what about cell in prison ?"
9. For "how does X work?" queries, normalize to:
   "how does cell phone work ?"
10. Output ONLY valid JSON. No explanation.

## Output Format

{"sentences": ["query 1", "query 2", ...]}

## Examples

INPUT: "So I was like, you know, wondering what a match is actually used for?"
OUTPUT: {"sentences": ["what is match used for ?"]}

INPUT: "Hey, so like, I was curious about this — what's the deal with Python and data science?"
OUTPUT: {"sentences": ["what is Python used for in data science ?"]}

INPUT: "I was wondering if you could tell me what seals eat in the wild?"
OUTPUT: {"sentences": ["what do seals eat in the wild ?"]}

INPUT: "Can you explain to me how a cell phone works?"
OUTPUT: {"sentences": ["how does cell phone work ?"]}

INPUT: "Do you happen to know where Paris is located?"
OUTPUT: {"sentences": ["where is Paris located ?"]}

INPUT: "That thing you use to start a fire, what's it called?"
OUTPUT: {"sentences": ["what is the thing used to start fire ?"]}

INPUT: "The animal that lives in the ocean and eats fish, what is it?"
OUTPUT: {"sentences": ["what animal lives in ocean and eats fish ?"]}

INPUT: "What's the difference between Python the language and python the snake?"
OUTPUT: {"sentences": ["what is Python programming language", "what is python snake", "what is difference between Python language and python snake ?"]}

INPUT: "Tell me about cells in biology and also cells in prisons"
OUTPUT: {"sentences": ["what is cell in biology", "what is cell in prison"]}

INPUT: "I know that Paris is the capital of France, but is it also in the EU?"
OUTPUT: {"sentences": ["Paris is the capital of France", "is Paris in the EU ?"]}

INPUT: "Given that Python was created by Guido van Rossum, what is it used for?"
OUTPUT: {"sentences": ["Python was created by Guido van Rossum", "what is Python used for ?"]}

INPUT: "so basically what I'm asking is, like, do cranes migrate?"
OUTPUT: {"sentences": ["do cranes migrate ?"]}

INPUT: "right so I heard that mercury is toxic, is that true?"
OUTPUT: {"sentences": ["is mercury toxic ?"]}

INPUT: "what about prison?"
OUTPUT: {"sentences": ["what about cell in prison ?"]}

INPUT: "how does it work?"
OUTPUT: {"sentences": ["how does UNKNOWN work ?"]}

INPUT: "and in Europe?"
OUTPUT: {"sentences": ["what is in Europe ?"]}

INPUT: "What do seals eat?"
OUTPUT: {"sentences": ["what do seals eat ?"]}

INPUT: "Is Python good for data science?"
OUTPUT: {"sentences": ["is Python good for data science ?"]}

INPUT: "Who created Python?"
OUTPUT: {"sentences": ["who created Python ?"]}

INPUT: "Where was Python created?"
OUTPUT: {"sentences": ["where was Python created ?"]}
```

---

## Differences from v1 (Preprocessing Prompt)

| Aspect | v1 (Teach) | v1-Query (This) |
|--------|-----------|-----------------|
| Rule 7 | "rewrite as declarative" | "keep question mark" |
| Unknown variables | Not handled | `?` syntax or "UNKNOWN" |
| Meta-commentary | Stripped | Stripped (same) |
| Filler removal | Same | Same |
| Coordination splitting | "Alice and Bob went" → 2 sentences | Same |
| Output format | JSON sentences | JSON sentences (same) |
| Hallucination risk | HIGH on queries | LOW (preserves question intent) |

## Why Two Prompts?

The v1 prompt is for **teaching facts** to the KG. It converts messy input into declarative sentences that the parser can extract triples from.

This prompt is for **querying the KG**. It converts messy questions into clean queries that the reasoning engine can match against stored facts.

Using v1 for queries causes:
- Rule 7 converts "What do seals eat?" → "Seals eat fish" (hallucinated answer)
- No `?` syntax → the engine doesn't know what's being asked
- Declarative output → the engine treats it as a teach, not a query

---

## Integration

```python
# td/preprocessing/query_preprocessor.py

QUERY_SYSTEM_PROMPT = open("QUERY_PREPROCESSING_PROMPT.md").read()

def preprocess_query(user_input: str, llm_client) -> list[str]:
    """Normalize a messy user query into clean query forms."""
    response = llm_client.complete(
        system=QUERY_SYSTEM_PROMPT,
        user=user_input
    )
    result = json.loads(response)
    return result["sentences"]
```

---

## Test Cases (18 queries)

| # | Category | Input | Expected Output |
|---|----------|-------|----------------|
| 1 | Filler | "So I was like, you know, wondering what a match is actually used for?" | "what is match used for ?" |
| 2 | Filler + intent | "Hey, so like, I was curious about this — what's the deal with Python and data science?" | "what is Python used for in data science ?" |
| 3 | Indirect | "I was wondering if you could tell me what seals eat in the wild?" | "what do seals eat in the wild ?" |
| 4 | Indirect | "Can you explain to me how a cell phone works?" | "how does cell phone work ?" |
| 5 | Indirect | "Do you happen to know where Paris is located?" | "where is Paris located ?" |
| 6 | Anaphora | "That thing you use to start a fire, what's it called?" | "what is the thing used to start fire ?" |
| 7 | Relative clause | "The animal that lives in the ocean and eats fish, what is it?" | "what animal lives in ocean and eats fish ?" |
| 8 | Multi-concept | "What's the difference between Python the language and python the snake?" | 3 queries (separate + difference) |
| 9 | Multi-concept | "Tell me about cells in biology and also cells in prisons" | 2 queries |
| 10 | Embedded fact | "I know that Paris is the capital of France, but is it also in the EU?" | fact + query |
| 11 | Embedded fact | "Given that Python was created by Guido van Rossum, what is it used for?" | fact + query |
| 12 | Casual | "so basically what I'm asking is, like, do cranes migrate?" | "do cranes migrate ?" |
| 13 | Casual | "right so I heard that mercury is toxic, is that true?" | "is mercury toxic ?" |
| 14 | Edge case | "what about prison?" | "what about cell in prison ?" |
| 15 | Edge case | "how does it work?" | "how does UNKNOWN work ?" |
| 16 | Edge case | "and in Europe?" | "what is in Europe ?" |
| 17 | Simple | "What do seals eat?" | "what do seals eat ?" |
| 18 | Simple | "Is Python good for data science?" | "is Python good for data science ?" |

---

_"A teach prompt and a query prompt are different tools for different jobs. Don't use a hammer on screws."_
