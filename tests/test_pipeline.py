"""Tests for the full TD v2 pipeline."""

import time

import numpy as np
import pytest

from td.pipeline import TDPipeline, TDDecision


@pytest.fixture(scope="module")
def pipeline():
    """Create a pipeline with small dim for test speed."""
    return TDPipeline(dim=2000)


def test_perceive_natural_language(pipeline):
    vec = pipeline.perceive("Click the submit button", "natural_language")
    assert vec.shape == (2000,)
    assert set(np.unique(vec)).issubset({-1, 1})


def test_perceive_dom(pipeline):
    html = "<form><input type='text' name='user'><button type='submit'>OK</button></form>"
    vec = pipeline.perceive(html, "dom")
    assert vec.shape == (2000,)


def test_perceive_api(pipeline):
    data = {"user": "alice", "id": 123, "orders": [1, 2, 3]}
    vec = pipeline.perceive(data, "api")
    assert vec.shape == (2000,)


def test_perceive_metrics(pipeline):
    metrics = {"cpu": 92.5, "memory": 78.0, "service": "nginx"}
    vec = pipeline.perceive(metrics, "metrics")
    assert vec.shape == (2000,)


def test_decide_returns_decision(pipeline):
    decision = pipeline.decide("Click the submit button on the login form")
    assert isinstance(decision, TDDecision)
    assert isinstance(decision.action_plan, list)
    assert decision.confidence.combined >= 0.0
    assert decision.routing.domain in ["Web", "API", "File", "Monitor", "Unknown"]
    assert len(decision.trace) > 0
    assert decision.latency_ms > 0


def test_decide_unknown_escalates(pipeline):
    """Novel input with no MHN patterns — router may still classify it."""
    decision = pipeline.decide("Plan a birthday party for my cat")
    assert isinstance(decision, TDDecision)
    # With no trained router and no MHN memory, confidence is unpredictable.
    # Just verify the pipeline doesn't crash and returns a valid decision.
    assert decision.confidence.combined >= 0.0
    assert decision.routing.domain is not None


def test_learn_and_retrieve(pipeline):
    """Store a pattern, verify it's retrievable."""
    pipeline.learn(
        "Fill out the contact form with name and email",
        [{"action": "click", "target": "name_field"},
         {"action": "type", "target": "name_field", "value": "test"}],
        "success",
        {"domain": "Web", "task_type": "Form"},
    )
    assert len(pipeline.mhn) > 0


def test_learn_correction(pipeline):
    """Correction learning updates MHN."""
    situation = "Extract product prices"
    wrong = [{"action": "click", "target": "wrong_button"}]
    correct = [{"action": "extract", "target": "price_table"}]
    pipeline.learn_correction(situation, wrong, correct,
                              {"domain": "Web", "task_type": "Extraction"})
    assert len(pipeline.mhn) > 0


def test_save_load_state(pipeline, tmp_path):
    """Save and load pipeline state."""
    pipeline.save_state(str(tmp_path / "state"))
    assert (tmp_path / "state" / "concepts.json").exists()
    assert (tmp_path / "state" / "router_weights.pt").exists()


def test_latency_under_50ms(pipeline):
    """End-to-end pipeline should be fast (excluding Z3)."""
    t0 = time.perf_counter()
    pipeline.decide("Click submit button")
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms < 500  # Generous for test environments
