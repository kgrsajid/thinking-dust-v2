# Thinking Dust v2

**Computer intelligence is just human beings teaching dust how to think.**

A neuro-symbolic reasoning engine with <100K parameters that runs on CPU. No GPU, no internet-scale pretraining, no attention mechanism.

## Architecture

Thinking Dust is modeled after Kahneman's System 1 / System 2:

- **TD v2 (System 1):** Fast, reactive agentic controller (~5K params). Pattern matching + memory retrieval + constraint validation.
- **TD Pro (System 2):** Slow, deliberate reasoning engine (~55K params). *Coming later.*

### TD v2 Pipeline

```
Input → HDC Encoder → CA Reservoir → MHN Retrieval
      → Hierarchical Ternary Router → [Memory | Z3 Validate | Escalate]
      → Confidence Check → Action Plan
```

| Component | Params | Role |
|-----------|--------|------|
| HDC Encoder | 0 | 10K-dim bipolar vector encoding |
| CA Reservoir (Rule 90) | 0 | Feature extraction from structured input |
| Modern Hopfield Network | 0 (runtime) | Associative memory with IDP |
| Hierarchical Ternary Router | ~5K | Domain → Task → Strategy classification |
| Z3 SMT Solver | 0 (binary) | Formal constraint validation |

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run tests
pytest

# Run demo
python demos/demo_form_automation.py
python demos/demo_monitor.py
```

## Usage

```python
from td.pipeline import TDPipeline

# Initialize
pipeline = TDPipeline()

# Train routers on synthetic data
pipeline.train_routers(epochs=50)

# Make decisions
decision = pipeline.decide("Click the submit button on the login form")

if decision.should_execute:
    execute(decision.action_plan)
elif decision.needs_confirmation:
    if ask_user(decision):
        execute(decision.action_plan)

# Learn from outcomes
pipeline.learn("Fill contact form", action_plan, "success")
```

## Key Properties

- **Deterministic:** Same input → same output, every time
- **Interpretable:** Every decision traceable to Z3 proof or energy landscape
- **Online learning:** Stores new patterns in milliseconds, zero catastrophic forgetting
- **Runs on CPU:** MacBook Air, Raspberry Pi, embedded devices
- **No LLM dependency:** Pure neuro-symbolic architecture

## Project Structure

```
td-v2/
├── td/
│   ├── perception/    # HDC, CA Reservoir, NL/DOM/API/Metrics encoders
│   ├── memory/        # Modern Hopfield Network with IDP
│   ├── routing/       # Ternary router cascade (3 levels)
│   ├── reasoning/     # Z3 bridge, constraint schemas, confidence
│   ├── learning/      # Online learning from outcomes
│   ├── utils/         # IO, visualization
│   └── pipeline.py    # Main orchestrator
├── tests/             # Full test suite
├── demos/             # Example applications
└── data/              # Training data, Z3 templates
```

## License

MIT

## Authors

- **Kazi Rabbany** — architecture, implementation
- **Kimi K2.6** — co-author of the original specification

---

*"The worm has 302 neurons and it navigates. The model has 5,000 neurons and it acts. Scale is not intelligence. Architecture is."*
