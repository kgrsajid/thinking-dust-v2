"""Tests for Z3 Bridge."""

import pytest
from z3 import sat

from td.perception.hdc import build_default_vocabulary
from td.reasoning.z3_bridge import Z3Bridge, Z3Result
from td.reasoning.constraint_schemas import WEB_FORM_CONSTRAINTS


class TestZ3Bridge:
    def test_validate_form_valid(self):
        bridge = Z3Bridge()
        action_plan = [
            {"action": "type", "target": "name_field", "value": "Alice"},
            {"action": "type", "target": "email_field", "value": "alice@test.com"},
            {"action": "click", "target": "submit_button"},
        ]
        constraints = {
            "submit_visible": True,
            "required_fields_filled": True,
            "captcha_present": False,
        }
        result = bridge.validate_action(action_plan, constraints)
        assert result.status == "sat"
        assert result.is_valid

    def test_validate_form_missing_field(self):
        bridge = Z3Bridge()
        action_plan = [
            {"action": "click", "target": "submit_button"},
        ]
        constraints = {
            "submit_visible": True,
            "required_fields_filled": False,
            "captcha_present": False,
        }
        result = bridge.validate_action(action_plan, constraints)
        assert result.status == "sat"

    def test_decompose(self):
        bridge = Z3Bridge()
        vocab = build_default_vocabulary(dim=1000)
        from td.perception.nl_parser import NLParser
        parser = NLParser(vocab)
        vec = parser.parse("Click the submit button on the login form")
        concepts = bridge.decompose(vec, vocab, threshold=0.05)
        assert len(concepts) > 0
        assert "action" in concepts or "click" in concepts

    def test_select_template(self):
        bridge = Z3Bridge()
        concepts = {"form": 0.8, "submit": 0.7, "click": 0.6, "login": 0.5}
        template = bridge.select_template(concepts)
        assert template == "form_validation"

    def test_satisfiability_score(self):
        sat_result = Z3Result(status="sat")
        assert sat_result.satisfiability_score == 1.0

        unsat_result = Z3Result(status="unsat")
        assert unsat_result.satisfiability_score == 0.0

        unknown_result = Z3Result(status="unknown")
        assert unknown_result.satisfiability_score == 0.5
