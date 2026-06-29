#!/usr/bin/env python3
"""TD v2 Decomposer Demo — Scheduling Meetings with Constraints.

First working demo of the decomposer as primary reasoning path.
Tests: can TD decompose a scheduling problem and produce a plan?

Usage:
    .venv/bin/python3 demos/demo_decomposer.py
"""

import sys
import os

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from td.perception.hdc import build_default_vocabulary, similarity
from td.memory.mhn import ModernHopfieldNetwork, MHNConfig
from td.reasoning.z3_bridge import Z3Bridge
from td.routing.hierarchical_router import HierarchicalRouter
from td.routing.router_train import train_router
from td.decomposer import Decomposer


def main():
    print("=" * 70)
    print("TD v2 Decomposer Demo — Scheduling with Constraints")
    print("=" * 70)
    print()

    # Build components
    print("Building vocabulary...")
    vocab = build_default_vocabulary(dim=10_000)

    print("Training router (fallback)...")
    result = train_router(vocab, epochs=100, lr=1e-2, verbose=False)
    router = HierarchicalRouter(input_dim=10_000)
    router.router_a.load_state_dict(result["router_a"].state_dict())
    for d in result["routers_b"]:
        router.router_b_dict[d].load_state_dict(result["routers_b"][d].state_dict())
    router.router_c.load_state_dict(result["router_c"].state_dict())
    router.eval()

    print("Building decomposer...")
    decomposer = Decomposer(
        vocab=vocab,
        mhn=ModernHopfieldNetwork(MHNConfig(dim=10_000, min_similarity=0.10)),
        z3_bridge=Z3Bridge(),
        router=router,
        dim=10_000,
        similarity_threshold=0.05,
        mhn_threshold=0.20,
    )

    # Teach it all 200 hand-written examples
    print("\nLoading 200 training examples into MHN...")
    from td.training_data import EXAMPLES
    for problem_text, solution, domain, task_type in EXAMPLES:
        decomposer.learn(
            problem_text,
            {"solution": solution, "domain": domain, "task_type": task_type},
            "success",
            {"domain": domain, "task_type": task_type},
        )
    print(f"  Stored {len(decomposer.mhn.patterns)} patterns in MHN")

    # Test problems (the main demo)
    problems = [
        # The target demo from the plan
        "Schedule 5 meetings this week without conflicts, prioritize client calls, avoid Friday afternoons",

        # Variations
        "Book a meeting with Alice, Bob, and Carol next Tuesday morning",
        "I have 3 team members on vacation next week and 20 tasks to complete before the demo on Friday",
        "Find a time when all 5 interviewers are available for the candidate panel",
    ]

    print(f"\n{'=' * 70}")
    print(f"Running {len(problems)} scheduling problems through decomposer")
    print(f"{'=' * 70}\n")

    for problem in problems:
        print(f"PROBLEM: {problem}")
        print(f"{'─' * 70}")

        result = decomposer.decompose(problem)

        print(result.summary())
        print()

    # Show stats
    print(f"{'=' * 70}")
    print("DECOMPOSER STATISTICS")
    print(f"{'=' * 70}")
    stats = decomposer.get_stats()
    for k, v in stats.items():
        print(f"  {k:30s}: {v}")

    # Also test non-scheduling problems to see if decomposer is generic
    print(f"\n{'=' * 70}")
    print("NON-SCHEDULING TESTS (genericity check)")
    print(f"{'=' * 70}\n")

    generic_problems = [
        "Parse the CSV file and validate that all required columns are present",
        "Find the cheapest way to allocate budget across 5 departments",
        "Prove that if A implies B and B implies C then A implies C",
        "Convert the JSON data to CSV format with specific column mapping",
        "Debug why the function returns wrong result for negative inputs",
    ]

    for problem in generic_problems:
        print(f"PROBLEM: {problem}")
        print(f"{'─' * 70}")

        result = decomposer.decompose(problem)
        print(result.summary())
        print()

    # Final stats
    print(f"{'=' * 70}")
    print("FINAL STATISTICS")
    print(f"{'=' * 70}")
    stats = decomposer.get_stats()
    for k, v in stats.items():
        print(f"  {k:30s}: {v}")


if __name__ == "__main__":
    main()
