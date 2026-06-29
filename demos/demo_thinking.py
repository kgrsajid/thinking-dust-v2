#!/usr/bin/env python3
"""TD v2 — The Thinking Loop Demo.

Shows all four mechanisms working:
    1. IDP: State evolves through iterative retrieval
    2. HDC: Sub-problems extracted via algebraic binding
    3. Z3: Real constraint solving with integer variables
    4. Auto-storage: Every solve grows the memory

Usage:
    .venv/bin/python3 demos/demo_thinking.py "Schedule meetings with Alice and Bob"
    .venv/bin/python3 demos/demo_thinking.py --stress
    .venv/bin/python3 demos/demo_thinking.py --pure "How to schedule without procrastination"
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from td.perception.hdc import build_default_vocabulary
from td.memory.mhn import ModernHopfieldNetwork, MHNConfig
from td.thinking import ThinkingDust


class C:
    RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
    R = "\033[31m"; G = "\033[32m"; Y = "\033[33m"
    B = "\033[34m"; M = "\033[35m"; CY = "\033[36m"; GR = "\033[90m"


def build_td(pure_mode=False):
    """Build TD with minimal seed (50 patterns) or pure mode (0 patterns)."""
    print(f"{C.DIM}Initializing Thinking Dust...{C.RESET}", file=sys.stderr)
    print(f"{C.DIM}Mode: {'PURE (0 seed)' if pure_mode else 'SEED (50 innate reflexes)'}{C.RESET}", file=sys.stderr)
    vocab = build_default_vocabulary(dim=10_000)
    mhn = ModernHopfieldNetwork(MHNConfig(dim=10_000, min_similarity=0.01, idp_enabled=False))

    td = ThinkingDust(vocab=vocab, mhn=mhn, max_idp_iterations=5, pure_mode=pure_mode)
    stats = td.stats()
    print(f"{C.DIM}Memory: {stats['seed_patterns']} seed + {stats['learned_patterns']} learned = {stats['memory_size']} total{C.RESET}", file=sys.stderr)
    print(f"{C.DIM}Seed ratio: {stats['seed_ratio_pct']:.1f}%{C.RESET}\n", file=sys.stderr)
    return td


def solve_with_trace(td, problem_text):
    """Solve with full visual trace of the thinking loop."""
    print(f"{C.BOLD}{C.CY}╔══════════════════════════════════════════════════════════════╗{C.RESET}")
    print(f"{C.BOLD}{C.CY}║  ✦  Thinking Dust — The Thinking Loop                       ║{C.RESET}")
    print(f"{C.BOLD}{C.CY}╚══════════════════════════════════════════════════════════════╝{C.RESET}")
    print()
    print(f"{C.BOLD}Problem:{C.RESET} {problem_text}")
    print(f"{C.GR}{'─' * 64}{C.RESET}")
    print()

    result = td.think(problem_text)

    # Show IDP iterations
    print(f"{C.M}🌀 IDP Iterative Refinement ({result.iterations} iterations):{C.RESET}")
    for t in result.thoughts:
        marker = f"{C.G}✓" if t.converged else f"{C.Y}→"
        print(f"   {marker} Iter {t.iteration}: sim={t.retrieved_similarity:.3f} "
              f"{t.description[:50]}{C.RESET}")
    print()

    # Show HDC decomposition
    print(f"{C.B}🔗 HDC Algebraic Decomposition:{C.RESET}")
    for sp in result.sub_problems:
        sim = sp.get("retrieved_sim", 0)
        bar = "█" * int(max(0, sim) * 30)
        print(f"   [{sp['concept']:25s}] sim={sim:.3f} {C.GR}{bar}{C.RESET}")
    print()

    # Show solution
    if result.solution:
        print(f"{C.G}✅ Solution:{C.RESET}")
        formatted = result.solution.get("formatted", "")
        if formatted:
            for line in formatted.split("\n"):
                print(f"   {C.G}{line}{C.RESET}")
    else:
        print(f"{C.R}❌ No solution{C.RESET}")
    print()

    # Show memory growth with seed/learned ratio
    stats = td.stats()
    print(f"{C.GR}Memory: {stats['memory_size']} patterns "
          f"({stats['seed_patterns']} seed, {stats['learned_patterns']} learned) "
          f"seed ratio: {stats['seed_ratio_pct']:.1f}%{C.RESET}")

    # Summary
    print(f"{C.GR}{'─' * 64}{C.RESET}")
    print(f"{C.BOLD}Iterations:{C.RESET} {result.iterations}  "
          f"{C.BOLD}Confidence:{C.RESET} {result.confidence:.0%}  "
          f"{C.BOLD}Latency:{C.RESET} {result.latency_ms:.0f}ms")
    print()


def stress_test(td, n=20):
    """Run stress test showing thinking loop on novel problems."""
    novel = [
        "Schedule meetings with Alice Bob and Carol next week",
        "Allocate 5000 across marketing engineering and design",
        "Prove that if A implies B and B implies C then A implies C",
        "Convert the JSON data to CSV format",
        "Debug why the function returns wrong result for negative inputs",
        "How to schedule without procrastination",
        "Find the cheapest flight to Tokyo",
        "Validate that all email addresses in the database are correct",
        "Optimize the knapsack problem with 50 items",
        "Plan migration from monolith to microservices",
        "Schedule 10 interviews across 3 days",
        "Balance the quarterly budget after revenue drop",
        "Find all customers who bought X but not Y",
        "Plan study schedule for final exams",
        "Schedule weekly one on ones for a team of 8",
        "Debug memory leak in the Python process",
        "Find duplicate records in the customer database",
        "Allocate classroom space for 200 students",
        "Plan a dinner party for 8 people with dietary restrictions",
        "Optimize delivery routes for 3 drivers and 15 packages",
    ]

    print(f"{C.BOLD}{C.CY}Stress Test: Thinking Loop on {len(novel[:n])} Novel Problems{C.RESET}")
    print(f"{C.GR}{'=' * 64}{C.RESET}\n")

    results = {"idp_converged": 0, "z3_solved": 0, "memory_grown": 0,
               "advice": 0, "info_request": 0, "unknown": 0}
    total_iters = 0
    total_time = 0

    for i, problem in enumerate(novel[:n]):
        result = td.think(problem)
        elapsed = result.latency_ms
        total_time += elapsed
        total_iters += result.iterations

        # What happened?
        iters = result.iterations
        converged = any(t.converged for t in result.thoughts)
        has_z3 = result.solution and "z3" in str(result.solution.get("method", "")).lower()
        sol_type = result.solution.get("type", "unknown") if result.solution else "none"

        if converged:
            results["idp_converged"] += 1
        if has_z3:
            results["z3_solved"] += 1
        results["memory_grown"] += 1
        if sol_type in results:
            results[sol_type] += 1

        # Color by type
        if has_z3:
            tag = f"{C.G}Z3"
        elif sol_type == "advice":
            tag = f"{C.M}ADV"
        elif sol_type == "info_request":
            tag = f"{C.Y}EXT"
        elif converged:
            tag = f"{C.B}IDP"
        else:
            tag = f"{C.R}???"

        stats = td.stats()
        print(f"  {i+1:2d}. [{tag}{C.RESET}] {result.confidence:.0%} "
              f"iter={iters} {elapsed:5.0f}ms "
              f"mem={stats['memory_size']:4d} "
              f"seed={stats['seed_ratio_pct']:.0f}% "
              f"— {problem[:32]}")

    print(f"\n{C.GR}{'─' * 64}{C.RESET}")
    print(f"{C.BOLD}Results:{C.RESET}")
    print(f"  IDP converged:   {results['idp_converged']:2d}/{n} ({results['idp_converged']/n:.0%})")
    print(f"  Z3 solved:       {results['z3_solved']:2d}/{n} ({results['z3_solved']/n:.0%})")
    print(f"  Memory growth:   +{results['memory_grown']} patterns")
    print(f"  Avg iterations:  {total_iters/n:.1f}")
    print(f"  Avg latency:     {total_time/n:.0f}ms")
    stats = td.stats()
    print(f"  Final memory:    {stats['memory_size']} patterns")
    print(f"  Seed ratio:      {stats['seed_ratio_pct']:.1f}% (target: <5% after 1000 interactions)")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} [--pure] \"\u003cproblem\u003e\" | --stress [N]")
        print(f"\nExamples:")
        print(f"  {sys.argv[0]} \"Schedule meetings with Alice and Bob\"")
        print(f"  {sys.argv[0]} --pure \"How to schedule without procrastination\"")
        print(f"  {sys.argv[0]} --stress")
        print(f"\n--pure: Start with 0 seed patterns (research demo mode)")
        sys.exit(1)

    pure_mode = False
    args = sys.argv[1:]
    if args[0] == "--pure":
        pure_mode = True
        args = args[1:]

    if not args:
        print("Error: provide a problem text after --pure")
        sys.exit(1)

    td = build_td(pure_mode=pure_mode)

    if args[0] == "--stress":
        n = int(args[1]) if len(args) > 1 else 20
        stress_test(td, n)
    else:
        solve_with_trace(td, " ".join(args))


if __name__ == "__main__":
    main()
