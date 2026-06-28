"""Save/load TD v2 state — router weights, vocabulary, MHN patterns."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def save_state(pipeline, path: str | Path) -> None:
    """Save full pipeline state.

    Args:
        pipeline: TDPipeline instance.
        path: Directory to save to.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Router weights
    pipeline.router.save(str(path / "router_weights.pt"))

    # Vocabulary
    pipeline.vocab.save(str(path / "concepts.json"))

    # MHN patterns
    patterns_data = []
    for p in pipeline.mhn.patterns:
        patterns_data.append({
            "key": p.key.tolist(),
            "value": p.value.tolist(),
            "composite": p.composite.tolist(),
            "metadata": p.metadata,
            "active": p.active,
        })
    with open(path / "mhn_patterns.json", "w") as f:
        json.dump(patterns_data, f)


def load_state(pipeline, path: str | Path) -> None:
    """Load full pipeline state.

    Args:
        pipeline: TDPipeline instance to load into.
        path: Directory to load from.
    """
    path = Path(path)

    router_path = path / "router_weights.pt"
    if router_path.exists():
        pipeline.router.load(str(router_path))
        pipeline.router.eval()

    vocab_path = path / "concepts.json"
    if vocab_path.exists():
        pipeline.vocab.load(str(vocab_path))

    mhn_path = path / "mhn_patterns.json"
    if mhn_path.exists():
        from ..memory.mhn import StoredPattern
        with open(mhn_path, "r") as f:
            patterns_data = json.load(f)
        pipeline.mhn.patterns = []
        for pd in patterns_data:
            p = StoredPattern(
                key=np.array(pd["key"], dtype=np.int8),
                value=np.array(pd["value"], dtype=np.int8),
                composite=np.array(pd["composite"], dtype=np.int8),
                metadata=pd["metadata"],
                active=pd["active"],
            )
            pipeline.mhn.patterns.append(p)
        pipeline.mhn._dirty = True
