"""Advice mode solver — MHN-based, NO hardcoded strategies.

All strategies stored as MHN attractors loaded from JSON seed data.
Retrieved by HDC similarity at runtime. No Python dicts of strategies.

Architecture:
    JSON seed → MHN attractors (at init)
    User query → HDC → MHN retrieve → Validate → Compose output
    User correction → learn_strategy() → new MHN attractor
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np

from .perception.hdc import similarity, bind, bundle, generate_hypervector


class AdviceSolver:
    """Solver for advice problems using ONLY MHN retrieval.

    No hardcoded strategies. All knowledge in MHN as HDC attractors.
    """

    def __init__(self, mhn, hdc_vocab, parser, seed_data_path: str | None = None):
        self.vocab = hdc_vocab
        self.parser = parser

        # Use a DEDICATED MHN for strategies so training data doesn't drown them out
        from .memory.mhn import ModernHopfieldNetwork, MHNConfig
        self.strategy_mhn = ModernHopfieldNetwork(MHNConfig(
            dim=self.vocab.dim, min_similarity=0.01, idp_enabled=False
        ))

        if seed_data_path:
            self._load_seed_data(seed_data_path)

    def _load_seed_data(self, path: str):
        """Load strategies from JSON → encode as HDC → store in MHN."""
        data_path = Path(path)
        if not data_path.exists():
            return

        with open(data_path) as f:
            strategies = json.load(f)

        for strategy in strategies:
            # Store one MHN attractor per query prototype
            for proto_text in strategy.get("query_prototypes", [strategy["title"]]):
                query_hdc = self.parser.parse(proto_text)
                strategy_hdc = self._encode_strategy(strategy)
                self.strategy_mhn.store(
                    query_hdc,
                    strategy_hdc,
                    {
                        "label": "behavioral_strategy",
                        "id": strategy["id"],
                        "title": strategy["title"],
                        "description": strategy["description"],
                        "tags": strategy.get("tags", []),
                        "effectiveness": strategy.get("effectiveness", 0.5),
                    },
                )

    def _encode_strategy(self, strategy: dict) -> np.ndarray:
        """Encode strategy content as HDC vector."""
        vectors = []

        # Title
        title = strategy["title"].lower()
        if not self.vocab.has(title):
            self.vocab.add_concept(title)
        if not self.vocab.has("strategy_title"):
            self.vocab.add_concept("strategy_title")
        vectors.append(bind(self.vocab.get("strategy_title"), self.vocab.get(title)))

        # Description (bag of words)
        desc_words = strategy.get("description", "").lower().split()
        desc_vecs = []
        for word in desc_words[:20]:
            w = re.sub(r"[^\w]", "", word)
            if w and len(w) > 2:
                if not self.vocab.has(w):
                    self.vocab.add_concept(w)
                desc_vecs.append(self.vocab.get(w))
        if desc_vecs:
            if not self.vocab.has("strategy_desc"):
                self.vocab.add_concept("strategy_desc")
            vectors.append(bind(self.vocab.get("strategy_desc"), bundle(*desc_vecs)))

        # Tags
        tag_vecs = []
        for tag in strategy.get("tags", []):
            if not self.vocab.has(tag):
                self.vocab.add_concept(tag)
            tag_vecs.append(self.vocab.get(tag))
        if tag_vecs:
            if not self.vocab.has("strategy_tags"):
                self.vocab.add_concept("strategy_tags")
            vectors.append(bind(self.vocab.get("strategy_tags"), bundle(*tag_vecs)))

        return bundle(*vectors) if vectors else generate_hypervector(self.vocab.dim)

    def solve(self, entities: dict[str, Any]) -> dict[str, Any]:
        """Generate advice by retrieving strategies from MHN."""
        trace = []

        # 1. Encode problem
        problem_hdc = self._encode_problem(entities)
        trace.append("Encoded problem as HDC vector")

        # 2. Retrieve from MHN
        results = self.strategy_mhn.retrieve(problem_hdc, top_k=5)
        trace.append(f"MHN retrieved {len(results)} candidates")

        # 3. Extract strategies from metadata
        strategies = []
        seen_ids = set()
        for _, sim, meta in results:
            if meta.get("label") != "behavioral_strategy":
                continue
            sid = meta.get("id", "")
            if sid in seen_ids:
                continue
            seen_ids.add(sid)
            strategies.append({
                "id": sid,
                "title": meta.get("title", "Unknown"),
                "description": meta.get("description", ""),
                "tags": meta.get("tags", []),
                "effectiveness": meta.get("effectiveness", 0.5),
                "retrieval_similarity": sim,
            })

        # 4. Validate relevance
        validated = self._validate_relevance(strategies, entities)
        trace.append(f"Validated {len(validated)} strategies (from {len(strategies)} retrieved)")

        # 5. Confidence
        confidence = self._calculate_confidence(validated)
        trace.append(f"Confidence: {confidence:.2f}")

        return {
            "type": "advice",
            "strategies": validated,
            "confidence": confidence,
            "caveat": "Behavioral advice retrieved from memory. Results vary by individual.",
            "reasoning_trace": trace,
            "source": "MHN" if validated else "none",
        }

    def _encode_problem(self, entities: dict) -> np.ndarray:
        """Encode problem entities into HDC vector."""
        # Use parser's entity encoding
        base = self.parser._encode_entities(entities)

        # Boost goal signals
        goal_vecs = []
        for goal in entities.get("goals", []):
            clean = goal.replace("avoid_", "").replace("?", "").strip()
            if clean and len(clean) > 2:
                if not self.vocab.has(clean):
                    self.vocab.add_concept(clean)
                goal_vecs.append(self.vocab.get(clean))

        if goal_vecs:
            if not self.vocab.has("goal_boost"):
                self.vocab.add_concept("goal_boost")
            boost = bind(self.vocab.get("goal_boost"), bundle(*goal_vecs))
            return bundle(base, boost)

        return base

    def _validate_relevance(self, strategies: list[dict], entities: dict) -> list[dict]:
        """Check strategy tags overlap with user goals. Strict filtering."""
        goals = entities.get("goals", [])
        if not goals:
            return strategies[:5]

        # Extract clean goal concepts
        goal_concepts = set()
        for g in goals:
            clean = g.replace("avoid_", "").replace("?", "").strip()
            if clean and len(clean) > 2 and clean not in ("without",):
                goal_concepts.add(clean)

        if not goal_concepts:
            return strategies[:5]

        # Only return strategies that actually overlap with goals
        relevant = []
        for s in strategies:
            tag_set = set(s.get("tags", []))
            overlap = tag_set & goal_concepts
            if overlap:
                s["relevance_score"] = len(overlap) + s.get("effectiveness", 0.5)
                relevant.append(s)

        # Sort by relevance score
        relevant.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

        # If we have relevant strategies, return them. Otherwise return top 3 anyway.
        return relevant[:5] if relevant else strategies[:3]

    def _calculate_confidence(self, strategies: list[dict]) -> float:
        if not strategies:
            return 0.3
        avg_eff = sum(s.get("effectiveness", 0.5) for s in strategies) / len(strategies)
        avg_sim = sum(s.get("retrieval_similarity", 0.3) for s in strategies) / len(strategies)
        return min(avg_eff * 0.7 + avg_sim * 0.3, 0.95)

    # ── ONLINE LEARNING ──

    def learn_strategy(self, query_text: str, strategy: dict[str, Any]):
        """Learn a new strategy — no code changes needed."""
        query_hdc = self.parser.parse(query_text)
        strategy_hdc = self._encode_strategy(strategy)
        self.strategy_mhn.store(query_hdc, strategy_hdc, {
            "label": "behavioral_strategy",
            "id": strategy.get("id", f"learned_{random.randint(1000, 9999)}"),
            "title": strategy["title"],
            "description": strategy.get("description", ""),
            "tags": strategy.get("tags", []),
            "effectiveness": strategy.get("effectiveness", 0.5),
            "source": "user_taught",
        })
