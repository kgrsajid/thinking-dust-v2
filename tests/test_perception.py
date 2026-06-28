"""Tests for perception encoders."""

import numpy as np
import pytest

from td.perception.hdc import build_default_vocabulary
from td.perception.nl_parser import NLParser
from td.perception.dom_encoder import DOMEncoder
from td.perception.api_encoder import APIEncoder
from td.perception.metrics_encoder import MetricsEncoder


class TestNLParser:
    @pytest.fixture
    def parser(self):
        vocab = build_default_vocabulary(dim=2000)
        return NLParser(vocab)

    def test_extract_concepts(self, parser):
        concepts = parser.extract_concepts("Click the submit button on the login form")
        assert "action" in concepts
        assert concepts["action"] == "click"

    def test_parse_shape(self, parser):
        vec = parser.parse("Click the submit button")
        assert vec.shape == (2000,)

    def test_parse_unknown_text(self, parser):
        vec = parser.parse("xyzzy foobar")
        assert vec.shape == (2000,)

    def test_parse_api_command(self, parser):
        concepts = parser.extract_concepts("Fetch user profile from API")
        assert "action" in concepts
        assert concepts["action"] == "fetch"

    def test_parse_monitor_command(self, parser):
        concepts = parser.extract_concepts("Restart the nginx service")
        assert "action" in concepts
        assert concepts["action"] == "restart"


class TestDOMEncoder:
    def test_encode_form(self):
        vocab = build_default_vocabulary(dim=2000)
        encoder = DOMEncoder(vocab)
        html = "<form><input type='text' name='user'><input type='password'><button type='submit'>Login</button></form>"
        vec = encoder.encode(html)
        assert vec.shape == (2000,)

    def test_encode_empty(self):
        vocab = build_default_vocabulary(dim=2000)
        encoder = DOMEncoder(vocab)
        vec = encoder.encode("<html></html>")
        assert vec.shape == (2000,)


class TestAPIEncoder:
    def test_encode_dict(self):
        vocab = build_default_vocabulary(dim=2000)
        encoder = APIEncoder(vocab)
        data = {"user": "alice", "id": 123, "active": True}
        vec = encoder.encode(data)
        assert vec.shape == (2000,)

    def test_encode_json_string(self):
        vocab = build_default_vocabulary(dim=2000)
        encoder = APIEncoder(vocab)
        vec = encoder.encode('{"name": "test", "value": 42}')
        assert vec.shape == (2000,)

    def test_encode_nested(self):
        vocab = build_default_vocabulary(dim=2000)
        encoder = APIEncoder(vocab)
        data = {"user": {"name": "Alice", "email": "alice@test.com"}, "orders": [1, 2]}
        vec = encoder.encode(data)
        assert vec.shape == (2000,)


class TestMetricsEncoder:
    def test_encode(self):
        vocab = build_default_vocabulary(dim=2000)
        encoder = MetricsEncoder(vocab)
        vec = encoder.encode({"cpu": 92.5, "memory": 45.0, "service": "nginx"})
        assert vec.shape == (2000,)

    def test_discretization(self):
        vocab = build_default_vocabulary(dim=2000)
        encoder = MetricsEncoder(vocab)
        low = encoder.encode({"cpu": 10.0})
        high = encoder.encode({"cpu": 95.0})
        # Should be different
        from td.perception.hdc import similarity
        assert similarity(low, high) < 0.5
