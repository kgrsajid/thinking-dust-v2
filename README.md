# Thinking Dust v2 — Neuro-Symbolic Reasoning Engine

**"Computer intelligence is just human beings teaching dust how to think."**

A reasoning engine that **derives facts it was never taught**. No neural network. No GPU. No pretraining. <100K parameters. Runs on CPU in milliseconds.

## What It Does

```
teach: Paris is the capital of France | Paris
teach: France is in the EU | France is in the EU
teach: EU is part of Europe | EU is part of Europe

ask:   is Paris in the EU?
→ YES. paris capital_of france → france in eu
  Derived via logical inference (not just memory)
```

**TD was never taught "Paris is in the EU."** It derived it using transitive composition rules.

## How It Works

Four layers, each with a specific role:

| Layer | What | File | Tech |
|-------|------|------|------|
| Knowledge Graph | Store facts, derive new ones via logic | `td/kg/` | Triples + rule templates + BFS |
| Word Vectors | Paraphrase matching (semantic similarity) | `td/perception/word_vectors.py` | BEAGLE (Jones & Mewhort 2007) |
| NL Parser | Extract entities, relations, constraints | `td/perception/nl_parser.py` | CA reservoir + HDC |
| HDC/MHN/Z3 | Storage, retrieval, constraint solving | `td/perception/hdc.py`, `td/memory/mhn.py` | 10K-dim vectors + Z3 SMT |

**Query path:** Parser → KG inference → Z3 constraints → MHN retrieval → honest unknown

**Teach path:** MHN store (semantic key) + triple extraction + online BEAGLE update

See [ARCHITECTURE.md](ARCHITECTURE.md) for full documentation.

## Quick Start

```bash
git clone https://github.com/kgrsajid/thinking-dust-v2.git
cd thinking-dust-v2
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# Run the demo
.venv/bin/python3 demos/chat_flare.py --pure
```

Then type:
```
teach: Paris is the capital of France | Paris
teach: France is in the EU | France is in the EU
teach: EU is part of Europe | EU is part of Europe
is Paris in Europe?
```

## Key Features

- **Derives new facts** via transitive composition (capital_of + in → in)
- **Paraphrase matching** via BEAGLE word vectors (teach "capital of france", ask "france capital")
- **Constraint solving** via Z3 with 18 mathematical primitives
- **Honest uncertainty** — says "I don't know" instead of hallucinating
- **Proof traces** — shows the reasoning chain for every derived answer
- **Online learning** — word vectors evolve from every teach() interaction
- **Zero GPU** — pure HDC algebra + Z3, runs on any CPU
- **<100K parameters** total

## Tech Stack

`z3-solver` (Z3) · `torch` (neural components) · `torchhd` (HDC) · `numpy` (vectors)

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — Full system architecture (honest, detailed)
- [DEVELOPMENT.md](DEVELOPMENT.md) — Developer setup guide

## Research Foundation

Built on verified research from HDC/VSA literature:
- Kanerva (2009), Jones & Mewhort (2007), Ramsauer et al. (2020), Betteti et al. (2025), Lewis (2024), Fodor et al. (2025), Liu et al. (PathHD, NeurIPS 2025)

## License

MIT

---

*"Build the dust. Let it think."*
