#!/usr/bin/env python3
"""TD v2 CLI — Real reasoning with Z3 constraint solving.

Usage:
    .venv/bin/python3 demos/td_cli.py "Schedule 5 meetings this week"
    .venv/bin/python3 demos/td_cli.py --stress [N]
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from td.perception.hdc import build_default_vocabulary
from td.perception.nl_parser import NLParser
from td.memory.mhn import ModernHopfieldNetwork, MHNConfig
from td.reasoning.z3_bridge import Z3Bridge
from td.routing.hierarchical_router import HierarchicalRouter
from td.routing.router_train import train_router
from td.reasoning_decomposer import ReasoningDecomposer, requires_reasoning
from td.training_data import EXAMPLES


class C:
    RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
    RED = "\033[31m"; GREEN = "\033[32m"; YELLOW = "\033[33m"
    BLUE = "\033[34m"; MAGENTA = "\033[35m"; CYAN = "\033[36m"; GRAY = "\033[90m"


def build_td():
    print(f"{C.DIM}Initializing Thinking Dust...{C.RESET}", file=sys.stderr)
    vocab = build_default_vocabulary(dim=10_000)
    result = train_router(vocab, epochs=100, lr=1e-2, verbose=False)
    router = HierarchicalRouter(input_dim=10_000)
    router.router_a.load_state_dict(result["router_a"].state_dict())
    for d in result["routers_b"]:
        router.router_b_dict[d].load_state_dict(result["routers_b"][d].state_dict())
    router.router_c.load_state_dict(result["router_c"].state_dict())
    router.eval()

    decomposer = ReasoningDecomposer(
        vocab=vocab,
        mhn=ModernHopfieldNetwork(MHNConfig(dim=10_000, min_similarity=0.10)),
        z3_bridge=Z3Bridge(),
        router=router,
    )
    for problem_text, solution, domain, task_type in EXAMPLES:
        decomposer.learn(problem_text, {"solution": solution, "domain": domain}, "success",
                         {"domain": domain, "task_type": task_type})
    print(f"{C.DIM}Loaded {len(EXAMPLES)} patterns. Ready.{C.RESET}\n", file=sys.stderr)
    return decomposer


def solve_with_trace(decomposer, problem_text):
    print(f"{C.BOLD}{C.CYAN}╔══════════════════════════════════════════════════════════════╗{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}║  ✦  Thinking Dust v2 — Reasoning Engine                      ║{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}╚══════════════════════════════════════════════════════════════╝{C.RESET}")
    print()
    print(f"{C.BOLD}Problem:{C.RESET} {problem_text}")

    needs_reasoning = requires_reasoning(problem_text)
    print(f"{C.GRAY}Requires reasoning: {'YES' if needs_reasoning else 'no'}{C.RESET}")
    print(f"{C.GRAY}{'─' * 64}{C.RESET}")
    print()

    result = decomposer.solve(problem_text)
    t = result.latency_ms

    # Show trace
    print(f"{C.BLUE}📋 Reasoning Trace:{C.RESET}")
    for step in result.trace:
        print(f"   {C.GRAY}→{C.RESET} {step}")
    print()

    # Show decomposition
    if result.sub_problems:
        print(f"{C.MAGENTA}🔨 Decomposition ({len(result.sub_problems)} sub-problems):{C.RESET}")
        for i, sp in enumerate(result.sub_problems):
            status = f"{C.GREEN}✅" if sp.is_solved else f"{C.RED}❌"
            src = sp.source
            print(f"   {i+1}. {status} [{sp.prototype}] {src}{C.RESET}")
            print(f"      {C.GRAY}{sp.description[:70]}{C.RESET}")
            if sp.solution and sp.solution.get("formatted"):
                print(f"      {C.GREEN}{sp.solution['formatted']}{C.RESET}")
        print()

    # Show final solution
    if result.solution:
        print(f"{C.GREEN}✅ Solution:{C.RESET}")
        sol = result.solution

        # Advice mode
        if sol.get("type") == "advice" and sol.get("formatted"):
            print(f"   {C.BOLD}Strategy ({sol.get('category', '?')}):{C.RESET}")
            print(f"   {C.GREEN}{sol['formatted']}{C.RESET}")

        # External data
        elif sol.get("type") == "info_request" and sol.get("formatted"):
            print(f"   {C.YELLOW}{sol['formatted']}{C.RESET}")

        # Decomposition with formatted sub-solutions
        elif isinstance(sol, dict) and "steps" in sol:
            for step in sol.get("steps", []):
                step_sol = step.get("solution", {})
                if isinstance(step_sol, dict) and step_sol.get("formatted"):
                    print(f"   {C.BOLD}{step.get('sub_problem', '')[:60]}{C.RESET}")
                    print(f"   {C.GREEN}{step_sol['formatted']}{C.RESET}")
                    print()

        # Direct formatted solution
        elif isinstance(sol, dict) and sol.get("formatted"):
            print(f"   {C.GREEN}{sol['formatted']}{C.RESET}")

        else:
            for k, v in list(sol.items())[:5]:
                print(f"   {k}: {v}")
    else:
        print(f"{C.RED}❌ No solution found{C.RESET}")

    print()
    print(f"{C.GRAY}{'─' * 64}{C.RESET}")
    print(f"{C.BOLD}Latency:{C.RESET} {t:.0f}ms  "
          f"{C.BOLD}Confidence:{C.RESET} {result.confidence:.0%}  "
          f"{C.BOLD}Decomposed:{C.RESET} {'yes' if result.sub_problems else 'no'}  "
          f"{C.BOLD}Z3:{C.RESET} {'ran' if any(sp.source == 'z3' for sp in result.sub_problems) else 'no'}")
    print()


def stress_test(decomposer, n=20):
    novel = [
        "Plan a dinner party for 8 people, 2 vegetarian, 1 gluten-free, budget 200",
        "Optimize delivery routes for 3 drivers and 15 packages",
        "Debug function returns None instead of expected list on empty input",
        "Prove that if n is even then n squared is divisible by 4",
        "Allocate 5000 across marketing engineering and design with 40 30 30 split",
        "Schedule 10 interviews across 3 days with no interviewer doing more than 4",
        "Find the cheapest flight to Tokyo leaving Monday returning Friday",
        "Convert messy JSON to CSV flattening nested objects",
        "Schedule a product launch meeting with engineering and marketing teams",
        "Balance the quarterly budget after unexpected 15 percent revenue drop",
        "Find all customers who bought X but not Y in the last 30 days",
        "Prove by induction that the sum of first n odd numbers is n squared",
        "Plan migration from monolith to microservices over 6 months",
        "Debug memory leak in the long running Python process",
        "Schedule weekly one on ones for a team of 8 people",
        "Validate that all email addresses in the database are correct format",
        "Optimize the knapsack problem with 50 items and weight limit 100",
        "Plan study schedule for final exams across 5 subjects in one week",
        "Find duplicate records in the customer database and merge them",
        "Allocate classroom space for 200 students across 15 courses",
    ]

    print(f"{C.BOLD}{C.CYAN}Stress Test: {len(novel[:n])} Novel Problems — Z3 Always Runs{C.RESET}")
    print(f"{C.GRAY}{'=' * 64}{C.RESET}\n")

    results = {"decomposed": 0, "z3_solved": 0, "router": 0, "failed": 0}
    total_time = 0

    for i, problem in enumerate(novel[:n]):
        result = decomposer.solve(problem)
        elapsed = result.latency_ms
        total_time += elapsed

        has_z3 = any(sp.source == "z3" for sp in result.sub_problems)
        decomp = len(result.sub_problems) > 0

        if decomp:
            results["decomposed"] += 1
        if has_z3:
            results["z3_solved"] += 1
        if result.used_router_fallback:
            results["router"] += 1
        if not result.solution:
            results["failed"] += 1

        z3_tag = f"{C.GREEN}Z3✓" if has_z3 else f"{C.YELLOW}Z3✗"
        dec_tag = f"DEC✓" if decomp else f"DEC✗"
        print(f"  {i+1:2d}. [{z3_tag}{C.RESET}] [{dec_tag}] "
              f"{result.confidence:.0%} {elapsed:5.0f}ms — {problem[:40]}")

    total = len(novel[:n])
    print(f"\n{C.GRAY}{'─' * 64}{C.RESET}")
    print(f"{C.BOLD}Results (REAL reasoning — Z3 runs on every problem):{C.RESET}")
    print(f"  Decomposed:     {results['decomposed']:2d}/{total} ({results['decomposed']/total:.0%})")
    print(f"  Z3 solved:      {results['z3_solved']:2d}/{total} ({results['z3_solved']/total:.0%})")
    print(f"  Router fallback:{results['router']:2d}/{total} ({results['router']/total:.0%})")
    print(f"  Failed:         {results['failed']:2d}/{total} ({results['failed']/total:.0%})")
    print(f"  Avg latency:    {total_time/total:.0f}ms")
    if results['decomposed'] > 0:
        print(f"  {C.GREEN}✓ Decomposition is running ({results['decomposed']/total:.0%} of problems){C.RESET}")
    if results['z3_solved'] > 0:
        print(f"  {C.GREEN}✓ Z3 is actually solving ({results['z3_solved']/total:.0%} of problems){C.RESET}")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} \"<problem>\" | --stress [N]")
        sys.exit(1)

    decomposer = build_td()

    if sys.argv[1] == "--stress":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        stress_test(decomposer, n)
    else:
        solve_with_trace(decomposer, " ".join(sys.argv[1:]))


if __name__ == "__main__":
    main()
