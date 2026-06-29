"""TD v2 Main Pipeline — Agentic Controller.

Wires all modules into a single decision pipeline:

    Input → Perception (CA + HDC) → MHN retrieval
          → Router cascade → Strategy execution
          → [MHN | Z3 | Escalate] → Confidence check
          → Output (action plan or escalation)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .perception.hdc import (
    ConceptVocabulary, build_default_vocabulary,
    generate_hypervector, bind, bundle, similarity,
)
from .perception.ca_reservoir import CAReservoir
from .perception.nl_parser import NLParser
from .perception.dom_encoder import DOMEncoder
from .perception.api_encoder import APIEncoder
from .perception.metrics_encoder import MetricsEncoder
from .memory.mhn import ModernHopfieldNetwork, MHNConfig
from .memory.attractor_store import AttractorStore
from .routing.hierarchical_router import HierarchicalRouter, RoutingResult
from .routing.router_train import train_router
try:
    from .reasoning.z3_bridge import Z3Bridge, Z3Result
except ImportError:
    Z3Bridge = None  # type: ignore
    class Z3Result:  # minimal stub
        pass
from .reasoning.confidence import ConfidenceScore, compute_confidence
from .reasoning.constraint_schemas import (
    WEB_FORM_CONSTRAINTS, API_SEQUENTIAL_CONSTRAINTS,
    FILE_PARSE_CONSTRAINTS, MONITOR_THRESHOLD_CONSTRAINTS,
)
from .learning.online import OnlineLearner


CONSTRAINT_MAP = {
    "Web/Form": WEB_FORM_CONSTRAINTS,
    "API/Sequential": API_SEQUENTIAL_CONSTRAINTS,
    "File/Parse": FILE_PARSE_CONSTRAINTS,
    "Monitor/Threshold": MONITOR_THRESHOLD_CONSTRAINTS,
}


@dataclass
class TDDecision:
    """Full output of the TD v2 pipeline.

    Contains the action plan, confidence, routing info, and a
    human-readable decision trace for debugging and interpretability.
    """
    action_plan: list[dict]
    confidence: ConfidenceScore
    strategy: str
    routing: RoutingResult
    mhn_hits: list[tuple[float, dict]]
    z3_validation: Z3Result | None
    trace: list[str]
    latency_ms: float

    @property
    def should_execute(self) -> bool:
        """True if confidence is high enough to execute autonomously."""
        return self.confidence.decision == "execute"

    @property
    def needs_confirmation(self) -> bool:
        """True if user should confirm before execution."""
        return self.confidence.decision == "confirm"

    @property
    def should_escalate(self) -> bool:
        """True if this task should be escalated to TD Pro."""
        return self.confidence.decision == "escalate"

    def summary(self) -> str:
        """Human-readable one-line summary."""
        return (f"TD Decision: {self.routing.domain}/{self.routing.task_type} "
                f"→ {self.strategy} (conf={self.confidence.combined:.2f}, "
                f"decision={self.confidence.decision})")

    def full_trace(self) -> str:
        """Full multi-line decision trace."""
        lines = [f"=== Thinking Dust v2 Decision ===", ""]
        lines.append(f"Domain: {self.routing.domain} (conf={self.routing.domain_confidence:.3f})")
        lines.append(f"Task Type: {self.routing.task_type} (conf={self.routing.task_type_confidence:.3f})")
        lines.append(f"Strategy: {self.strategy} (conf={self.routing.strategy_confidence:.3f})")
        lines.append(f"Combined Router Confidence: {self.routing.combined_confidence:.3f}")
        lines.append(f"MHN Similarity: {self.confidence.mhn_similarity:.3f}")
        lines.append(f"Z3 Validation: {self.z3_validation.status if self.z3_validation else 'skipped'}")
        lines.append(f"Combined Confidence: {self.confidence.combined:.3f}")
        lines.append(f"Decision: {self.confidence.decision.upper()}")
        lines.append(f"Latency: {self.latency_ms:.1f}ms")
        lines.append("")
        lines.append("Action Plan:")
        for i, step in enumerate(self.action_plan):
            lines.append(f"  {i+1}. {step}")
        lines.append("")
        lines.append("Trace:")
        for t in self.trace:
            lines.append(f"  → {t}")
        return "\n".join(lines)


class TDPipeline:
    """Full TD v2 Agentic Controller pipeline.

    Total parameters: ~5K (router weights, ternarized)
    Total memory: ~7.3MB (concept vocab + MHN patterns + Z3)
    Inference latency: <5ms end-to-end (excluding Z3 on hard queries)

    Usage:
        pipeline = TDPipeline()
        decision = pipeline.decide("Click the submit button on the login form")
        if decision.should_execute:
            # Hand action_plan to OpenClaw for execution
            execute(decision.action_plan)
        elif decision.needs_confirmation:
            if ask_user(decision):
                execute(decision.action_plan)
    """

    def __init__(
        self,
        vocabulary_path: str | Path | None = None,
        router_weights_path: str | Path | None = None,
        z3_template_dir: str | Path | None = None,
        dim: int = 10_000,
    ):
        """Initialize all TD v2 components.

        Args:
            vocabulary_path: Path to concept vocabulary JSON.
                             If None, builds default vocabulary.
            router_weights_path: Path to trained router weights.
                                 If None, uses untrained routers (call train() first).
            z3_template_dir: Directory for Z3 templates.
        """
        # Perception layer
        if vocabulary_path and Path(vocabulary_path).exists():
            self.vocab = ConceptVocabulary(path=vocabulary_path, dim=dim)
        else:
            self.vocab = build_default_vocabulary(dim=dim)

        from .perception.ca_reservoir import CAConfig
        self.ca = CAReservoir(CAConfig(input_dim=dim))
        self.nl_parser = NLParser(self.vocab)
        self.dom_encoder = DOMEncoder(self.vocab, self.ca)
        self.api_encoder = APIEncoder(self.vocab)
        self.metrics_encoder = MetricsEncoder(self.vocab)

        # Memory layer
        self.mhn = ModernHopfieldNetwork(MHNConfig(dim=dim))
        self.attractors = AttractorStore(self.mhn)

        # Routing layer
        self.router = HierarchicalRouter(input_dim=dim)
        if router_weights_path and Path(router_weights_path).exists():
            self.router.load(str(router_weights_path))
            self.router.eval()

        # Reasoning layer
        self.z3_bridge = Z3Bridge(z3_template_dir)

        # Learning layer
        self.learner = OnlineLearner(self.mhn, self.vocab)

    def perceive(self, raw_input: Any, input_type: str = "natural_language") -> np.ndarray:
        """Encode raw input into unified HDC vector.

        Args:
            raw_input: The input data.
            input_type: One of "natural_language", "dom", "api", "metrics", "file".

        Returns:
            HDC vector of shape (dim,).
        """
        if input_type == "natural_language":
            return self.nl_parser.parse(raw_input)
        elif input_type == "dom":
            return self.dom_encoder.encode(raw_input)
        elif input_type == "api":
            return self.api_encoder.encode(raw_input)
        elif input_type == "metrics":
            return self.metrics_encoder.encode(raw_input)
        elif input_type == "file":
            # File content treated as structured data
            if isinstance(raw_input, dict):
                return self.api_encoder.encode(raw_input)
            else:
                return self.nl_parser.parse(str(raw_input))
        else:
            raise ValueError(f"Unknown input_type: {input_type}")

    def decide(
        self,
        raw_input: Any,
        input_type: str = "natural_language",
        context: dict | None = None,
        dom_html: str | None = None,
    ) -> TDDecision:
        """Full decision pipeline.

        Steps:
        1. Perceive input → HDC vector
        2. Retrieve from MHN
        3. Route through cascade
        4. Execute strategy (memory/validate/escalate)
        5. Confidence scoring
        6. Return decision with trace

        Args:
            raw_input: The input (text, DOM HTML, dict, etc.).
            input_type: Type of input.
            context: Optional context dict (constraints, metadata).
            dom_html: Optional DOM HTML to bundle with NL input.

        Returns:
            TDDecision with action plan and full trace.
        """
        t0 = time.perf_counter()
        trace = []
        context = context or {}

        # 1. Perceive
        hdc_vector = self.perceive(raw_input, input_type)
        trace.append(f"Encoded {input_type} input → HDC vector")

        # Optionally bundle with DOM
        if dom_html:
            dom_vec = self.dom_encoder.encode(dom_html)
            hdc_vector = bundle(hdc_vector, dom_vec)
            trace.append("Bundled with DOM context")

        # 2. MHN retrieval
        mhn_results = self.mhn.retrieve(hdc_vector, top_k=3)
        if mhn_results:
            best_sim = mhn_results[0][1]
            trace.append(f"MHN retrieved pattern (sim={best_sim:.3f})")
        else:
            best_sim = 0.0
            trace.append("MHN: no matching patterns")

        # 3. Routing
        mhn_vec = mhn_results[0][0] if mhn_results else None
        routing = self.router.route(hdc_vector, mhn_vec)
        trace.append(f"Router: {routing.domain}/{routing.task_type}/"
                    f"{routing.strategy} (conf={routing.combined_confidence:.3f})")

        # 4. Strategy execution
        action_plan = []
        z3_result = None
        strategy = routing.strategy

        # If memory-dependent strategy but no MHN hits, escalate
        if strategy in ("MEMORY_ONLY", "MEMORY_THEN_VALIDATE") and not mhn_results:
            strategy = "ESCALATE"
            trace.append("Strategy: ESCALATE (no MHN patterns for memory strategy)")

        if strategy == "MEMORY_ONLY":
            if mhn_results:
                # Reconstruct action plan from metadata
                _, _, meta = mhn_results[0]
                action_plan = meta.get("action_plan", [
                    {"action": "retrieved", "metadata": meta}
                ])
            trace.append("Strategy: MEMORY_ONLY — using retrieved pattern")

        elif strategy == "MEMORY_THEN_VALIDATE":
            if mhn_results:
                _, _, meta = mhn_results[0]
                action_plan = meta.get("action_plan", [
                    {"action": "retrieved", "metadata": meta}
                ])
            # Z3 validation — deep-copy constraints to avoid mutating global CONSTRAINT_MAP
            constraint_key = f"{routing.domain}/{routing.task_type}"
            constraints = dict(CONSTRAINT_MAP.get(constraint_key, {}))
            extra_constraints = context.get("constraints") or {}
            constraints.update(extra_constraints)

            if constraints:
                z3_result = self.z3_bridge.validate_action(action_plan, constraints)
                trace.append(f"Z3 validation: {z3_result.status}")
            else:
                trace.append("Z3: no constraints to validate (skipped)")

        elif strategy == "ESCALATE":
            trace.append("Strategy: ESCALATE — too novel for v2")
            action_plan = [{"action": "escalate", "reason": "low_confidence"}]

        else:
            # Unknown strategy — escalate rather than returning empty plan (bug fix #9)
            trace.append(f"Strategy: UNKNOWN ({strategy}) — escalating")
            action_plan = [{"action": "escalate", "reason": f"unknown_strategy:{strategy}"}]

        # 5. Confidence scoring
        confidence = compute_confidence(routing, mhn_results, z3_result)

        # Override confidence for ESCALATE: if the system decided to escalate,
        # force confidence below 0.7 so decision is "escalate" (not "execute").
        if strategy == "ESCALATE" and confidence.combined >= 0.7:
            from dataclasses import replace
            confidence = replace(confidence, combined=0.5)

        trace.append(f"Confidence: {confidence.combined:.3f} → {confidence.decision}")

        # 6. Build decision
        latency = (time.perf_counter() - t0) * 1000

        mhn_hits = [(sim, meta) for _, sim, meta in mhn_results]

        return TDDecision(
            action_plan=action_plan,
            confidence=confidence,
            strategy=strategy,
            routing=routing,
            mhn_hits=mhn_hits,
            z3_validation=z3_result,
            trace=trace,
            latency_ms=latency,
        )

    def learn(self, situation_text: str, action_plan: list[dict],
              outcome: str, metadata: dict | None = None):
        """Learn from execution outcome.

        Encodes the situation and action plan as HDC vectors and
        stores them in MHN as a new attractor.

        Args:
            situation_text: Natural language description of the situation.
            action_plan: List of action dicts that were executed.
            outcome: "success", "failure", or "partial".
            metadata: Additional context.
        """
        situation_hdc = self.nl_parser.parse(situation_text)
        action_hdc = self.learner.encode_action_plan(action_plan)
        meta = metadata or {}
        meta["action_plan"] = action_plan
        self.learner.learn_from_outcome(situation_hdc, action_hdc, outcome, meta)

    def learn_correction(self, situation_text: str,
                         wrong_actions: list[dict],
                         correct_actions: list[dict],
                         metadata: dict | None = None):
        """Learn from a user correction.

        Args:
            situation_text: The situation description.
            wrong_actions: The incorrect action plan.
            correct_actions: The corrected action plan.
            metadata: Additional context.
        """
        situation_hdc = self.nl_parser.parse(situation_text)
        wrong_hdc = self.learner.encode_action_plan(wrong_actions)
        correct_hdc = self.learner.encode_action_plan(correct_actions)
        meta = metadata or {}
        meta["action_plan"] = correct_actions
        self.learner.learn_from_correction(
            situation_hdc, wrong_hdc, correct_hdc, meta
        )

    def train_routers(self, epochs: int = 50, lr: float = 1e-3, verbose: bool = True):
        """Train all routers on synthetic data.

        Convenience method that trains all 3 router levels.
        """
        return train_router(self.vocab, epochs=epochs, lr=lr, verbose=verbose)

    def save_state(self, path: str | Path):
        """Persist router weights and vocabulary."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save router weights
        self.router.save(str(path / "router_weights.pt"))

        # Save vocabulary
        self.vocab.save(str(path / "concepts.json"))

    def load_state(self, path: str | Path):
        """Load router weights and vocabulary."""
        path = Path(path)

        router_path = path / "router_weights.pt"
        if router_path.exists():
            self.router.load(str(router_path))
            self.router.eval()

        vocab_path = path / "concepts.json"
        if vocab_path.exists():
            self.vocab.load(str(vocab_path))
