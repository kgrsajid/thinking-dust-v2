"""Tests for the ternary router."""

import numpy as np
import pytest
import torch

from td.routing.ternary_linear import TernaryLinear
from td.routing.router_a import RouterA, DOMAINS
from td.routing.router_b import RouterB, TASK_TYPES
from td.routing.router_c import RouterC, STRATEGIES
from td.routing.hierarchical_router import HierarchicalRouter, RoutingResult
from td.perception.hdc import generate_hypervector, build_default_vocabulary


class TestTernaryLinear:
    def test_ternarize(self):
        layer = TernaryLinear(100, 10)
        w_t = layer.get_ternary_weights()
        assert w_t.shape == (10, 100)
        assert set(w_t.unique().tolist()).issubset({-1, 0, 1})

    def test_forward(self):
        layer = TernaryLinear(10, 5)
        x = torch.randn(2, 10)
        out = layer(x)
        assert out.shape == (2, 5)

    def test_sparsity(self):
        layer = TernaryLinear(100, 50)
        w_t = layer.get_ternary_weights()
        zeros = (w_t == 0).sum().item()
        sparsity = zeros / w_t.numel()
        assert sparsity > 0.5  # Most weights should be 0

    def test_num_params(self):
        layer = TernaryLinear(100, 20)
        stats = layer.num_ternary_params()
        assert stats["total"] == 2000
        assert stats["sparsity"] > 0.5


class TestRouterA:
    def test_output_shape(self):
        router = RouterA(input_dim=1000, hidden_dim=32)
        x = torch.randn(3, 1000)
        out = router(x)
        assert out.shape == (3, 5)

    def test_classify(self):
        vocab = build_default_vocabulary(dim=1000)
        router = RouterA(input_dim=1000)
        # Parse a web-like input
        from td.perception.nl_parser import NLParser
        parser = NLParser(vocab)
        vec = parser.parse("Click the submit button on the login form")
        domain, conf, probs = router.classify(vec)
        assert domain in DOMAINS
        assert 0 <= conf <= 1
        assert probs.shape == (5,)


class TestRouterB:
    def test_output_shape(self):
        router = RouterB("Web", input_dim=1000, hidden_dim=32)
        x = torch.randn(2, 1000)
        out = router(x)
        assert out.shape == (2, 4)

    def test_classify(self):
        vocab = build_default_vocabulary(dim=1000)
        router = RouterB("Web", input_dim=1000)
        from td.perception.nl_parser import NLParser
        parser = NLParser(vocab)
        vec = parser.parse("Fill out the contact form")
        task, conf, probs = router.classify(vec)
        assert task in TASK_TYPES["Web"]


class TestRouterC:
    def test_output_shape(self):
        router = RouterC(input_dim=1000, hidden_dim=16)
        x = torch.randn(1, 1000)
        out = router(x)
        assert out.shape == (1, 3)

    def test_classify(self):
        vocab = build_default_vocabulary(dim=1000)
        router = RouterC(input_dim=1000)
        from td.perception.nl_parser import NLParser
        parser = NLParser(vocab)
        vec = parser.parse("Click the submit button")
        strategy, conf, probs = router.classify(vec)
        assert strategy in STRATEGIES


class TestHierarchicalRouter:
    def test_route(self):
        vocab = build_default_vocabulary(dim=1000)
        router = HierarchicalRouter(input_dim=1000)
        from td.perception.nl_parser import NLParser
        parser = NLParser(vocab)
        vec = parser.parse("Click the submit button on the login form")
        result = router.route(vec)
        assert isinstance(result, RoutingResult)
        assert result.domain in DOMAINS
        assert result.task_type in TASK_TYPES.get(result.domain, [])
        assert result.strategy in STRATEGIES
        assert 0 <= result.combined_confidence <= 1

    def test_save_load(self, tmp_path):
        router = HierarchicalRouter(input_dim=500)
        path = tmp_path / "router.pt"
        router.save(str(path))
        assert path.exists()
        router2 = HierarchicalRouter(input_dim=500)
        router2.load(str(path))
