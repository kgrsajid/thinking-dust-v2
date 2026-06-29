#!/usr/bin/env python3
"""TD v2 CLI — Thinking Dust with visible reasoning trace.

Usage:
    .venv/bin/python3 demos/td_cli.py "Schedule 5 meetings this week without conflicts"
    .venv/bin/python3 demos/td_cli.py "Prove that if A implies B and B implies C then A implies C"
    .venv/bin/python3 demos/td_cli.py --batch problems.txt
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from td.perception.hdc import build_default_vocabulary, similarity
from td.perception.nl_parser import NLParser
from td.memory.mhn import ModernHopfieldNetwork, MHNConfig
from td.reasoning.z3_bridge import Z3Bridge
from td.routing.hierarchical_router import HierarchicalRouter
from td.routing.router_train import train_router
from td.decomposer import Decomposer, PrototypeBank
from td.training_data import EXAMPLES


# ANSI colors
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    GRAY = "\033[90m"


def build_td():
    """Build TD with all 205 examples loaded."""
    print(f"{C.DIM}Initializing Thinking Dust...{C.RESET}", file=sys.stderr)

    vocab = build_default_vocabulary(dim=10_000)
    parser = NLParser(vocab)

    # Train router (fallback)
    result = train_router(vocab, epochs=100, lr=1e-2, verbose=False)
    router = HierarchicalRouter(input_dim=10_000)
    router.router_a.load_state_dict(result["router_a"].state_dict())
    for d in result["routers_b"]:
        router.router_b_dict[d].load_state_dict(result["routers_b"][d].state_dict())
    router.router_c.load_state_dict(result["router_c"].state_dict())
    router.eval()

    decomposer = Decomposer(
        vocab=vocab,
        mhn=ModernHopfieldNetwork(MHNConfig(dim=10_000, min_similarity=0.10)),
        z3_bridge=Z3Bridge(),
        router=router,
        dim=10_000,
        similarity_threshold=0.05,
        mhn_threshold=0.15,
    )

    # Load training data
    for problem_text, solution, domain, task_type in EXAMPLES:
        decomposer.learn(
            problem_text,
            {"solution": solution, "domain": domain, "task_type": task_type},
            "success",
            {"domain": domain, "task_type": task_type},
        )

    print(f"{C.DIM}Loaded {len(EXAMPLES)} patterns. Ready.{C.RESET}\n", file=sys.stderr)
    return decomposer


def solve_with_trace(decomposer, problem_text):
    """Solve a problem with full visual reasoning trace."""

    print(f"{C.BOLD}{C.CYAN}╔══════════════════════════════════════════════════════════════╗{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}║  ✦  Thinking Dust v2 — Reasoning Engine                      ║{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}╚══════════════════════════════════════════════════════════════╝{C.RESET}")
    print()
    print(f"{C.BOLD}Problem:{C.RESET} {problem_text}")
    print(f"{C.GRAY}{'─' * 64}{C.RESET}")
    print()

    t0 = time.perf_counter()

    # Step 1: Encode
    parser = NLParser(decomposer.vocab)
    hdc_vector = parser.parse(problem_text)
    concepts = parser.extract_concepts(problem_text)

    encode_ms = (time.perf_counter() - t0) * 1000
    print(f"{C.BLUE}🔤 Step 1: Perception{C.RESET} ({encode_ms:.1f}ms)")
    print(f"   Extracted concepts: {concepts or '(none — novel input)'}")
    print(f"   HDC vector: 10,000-dim bipolar [{int((hdc_vector > 0).sum())}+, {int((hdc_vector < 0).sum())}-]")
    print()

    # Step 2: Prototype matching
    t1 = time.perf_counter()
    proto_matches = decomposer.prototypes.match(hdc_vector, top_k=3)
    proto_ms = (time.perf_counter() - t1) * 1000

    print(f"{C.MAGENTA}🎯 Step 2: Prototype Matching{C.RESET} ({proto_ms:.1f}ms)")
    for i, (name, sim) in enumerate(proto_matches):
        marker = "►" if i == 0 else " "
        bar = "█" * int(max(0, sim) * 40)
        print(f"   {marker} {name:35s} {C.GRAY}sim={sim:.3f}{C.RESET} {bar}")
    best_proto, best_sim = proto_matches[0]
    print()

    # Step 3: Memory retrieval
    t2 = time.perf_counter()
    mhn_results = decomposer.mhn.retrieve(hdc_vector, top_k=3)
    mhn_ms = (time.perf_counter() - t2) * 1000

    print(f"{C.YELLOW}💾 Step 3: Memory Retrieval{C.RESET} ({mhn_ms:.1f}ms)")
    if mhn_results:
        for i, (vec, sim, meta) in enumerate(mhn_results[:3]):
            domain = meta.get("domain", "?")
            marker = "►" if i == 0 else " "
            bar = "█" * int(max(0, sim) * 40)
            print(f"   {marker} [{domain:12s}] sim={sim:.3f} {bar}")

        best_mhn_sim = mhn_results[0][1]
        if best_mhn_sim >= decomposer.mhn_threshold:
            print(f"   {C.GREEN}✓ Memory hit above threshold ({decomposer.mhn_threshold}){C.RESET}")

            # Show retrieved solution
            solution = mhn_results[0][2].get("solution", {})
            if solution:
                print()
                print(f"{C.GREEN}✅ Step 4: Solution Retrieved from Memory{C.RESET}")
                sol = solution.get("solution", solution)
                if isinstance(sol, dict):
                    for k, v in sol.items():
                        print(f"   {k}: {v}")
                else:
                    print(f"   {sol}")

            total_ms = (time.perf_counter() - t0) * 1000
            print()
            print(f"{C.GRAY}{'─' * 64}{C.RESET}")
            print(f"{C.BOLD}Latency:{C.RESET} {total_ms:.1f}ms  "
                  f"{C.BOLD}Source:{C.RESET} MHN (memory)  "
                  f"{C.BOLD}Confidence:{C.RESET} {best_mhn_sim:.1%}")
            return
    else:
        print(f"   {C.RED}✗ No matching patterns in memory{C.RESET}")
    print()

    # Step 4: Decompose (novel problem)
    print(f"{C.BLUE}🔨 Step 4: Decomposition{C.RESET}")
    if best_sim < decomposer.similarity_threshold:
        print(f"   {C.YELLOW}⚠ Prototype confidence too low ({best_sim:.3f} < {decomposer.similarity_threshold}){C.RESET}")
        print(f"   Falling back to router...")

        # Router fallback
        if decomposer.router is not None:
            routing = decomposer.router.route(hdc_vector)
            print(f"   {C.GRAY}Router: {routing.domain}/{routing.task_type}/{routing.strategy}{C.RESET}")

            total_ms = (time.perf_counter() - t0) * 1000
            print()
            print(f"{C.GRAY}{'─' * 64}{C.RESET}")
            print(f"{C.BOLD}Latency:{C.RESET} {total_ms:.1f}ms  "
                  f"{C.BOLD}Source:{C.RESET} Router (fallback)  "
                  f"{C.BOLD}Confidence:{C.RESET} {routing.combined_confidence:.1%}")
            return
    else:
        print(f"   Prototype: {best_proto} (sim={best_sim:.3f})")

        # Get sub-problems
        sub_descs = decomposer._generate_sub_problems(problem_text, best_proto, {})
        print(f"   Decomposed into {len(sub_descs)} sub-problems:")
        for i, desc in enumerate(sub_descs):
            sub_hdc = parser.parse(desc)
            sub_proto = decomposer.prototypes.match(sub_hdc, top_k=1)[0]
            sub_mhn = decomposer.mhn.retrieve(sub_hdc, top_k=1)

            if sub_mhn and sub_mhn[0][1] >= decomposer.mhn_threshold:
                status = f"{C.GREEN}✓ solved (MHN, sim={sub_mhn[0][1]:.2f}){C.RESET}"
            else:
                status = f"{C.YELLOW}? fallback{C.RESET}"

            print(f"   {i+1}. [{sub_proto[0]}] {desc[:55]}")
            print(f"      {status}")

    total_ms = (time.perf_counter() - t0) * 1000
    print()
    print(f"{C.GRAY}{'─' * 64}{C.RESET}")
    print(f"{C.BOLD}Latency:{C.RESET} {total_ms:.1f}ms")
    print()


def stress_test(decomposer, n_novel=20):
    """Run stress test with novel problems."""
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

    print(f"{C.BOLD}{C.CYAN}Stress Test: {len(novel)} Novel Problems{C.RESET}")
    print(f"{C.GRAY}{'=' * 64}{C.RESET}\n")

    results = {"mhn": 0, "decompose": 0, "router": 0, "unsolved": 0}
    total_time = 0

    for i, problem in enumerate(novel[:n_novel]):
        t0 = time.perf_counter()
        result = decomposer.decompose(problem)
        elapsed = (time.perf_counter() - t0) * 1000
        total_time += elapsed

        if result.used_router_fallback:
            source = "ROUTER"
            results["router"] += 1
            color = C.YELLOW
        elif result.confidence > 0.8:
            source = "MHN"
            results["mhn"] += 1
            color = C.GREEN
        elif result.sub_problems:
            source = "DECOMPOSE"
            results["decompose"] += 1
            color = C.BLUE
        else:
            source = "ROUTER"
            results["router"] += 1
            color = C.YELLOW

        proto = result.prototype_match
        sim = result.prototype_similarity
        print(f"  {color}{i+1:2d}.{C.RESET} [{source:8s}] {result.confidence:.0%} conf, "
              f"{elapsed:5.0f}ms — {problem[:45]}")

    total = sum(results.values())
    print(f"\n{C.GRAY}{'─' * 64}{C.RESET}")
    print(f"{C.BOLD}Results:{C.RESET}")
    print(f"  MHN hits:      {results['mhn']:2d}/{total} ({results['mhn']/total:.0%})")
    print(f"  Decomposed:    {results['decompose']:2d}/{total} ({results['decompose']/total:.0%})")
    print(f"  Router fallback:{results['router']:2d}/{total} ({results['router']/total:.0%})")
    print(f"  Avg latency:   {total_time/total:.0f}ms")
    fb_rate = results['router'] / total
    print(f"  Fallback rate: {fb_rate:.1%}", end="")
    if fb_rate < 0.10:
        print(f"  {C.GREEN}✓ TARGET MET (<10%){C.RESET}")
    else:
        print(f"  {C.YELLOW}(target <10%){C.RESET}")


def main():
    if len(sys.argv) < 2:
        print(f"{C.BOLD}Usage:{C.RESET}")
        print(f"  {sys.argv[0]} \"<problem text>\"")
        print(f"  {sys.argv[0]} --stress [N]")
        print(f"  {sys.argv[0]} --batch <file>")
        print()
        print(f"{C.BOLD}Examples:{C.RESET}")
        print(f"  {sys.argv[0]} \"Schedule 5 meetings this week without conflicts\"")
        print(f"  {sys.argv[0]} \"Prove that if A implies B and B implies C then A implies C\"")
        print(f"  {sys.argv[0]} \"Debug why the function returns wrong result\"")
        print(f"  {sys.argv[0]} --stress")
        sys.exit(1)

    decomposer = build_td()

    if sys.argv[1] == "--stress":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        stress_test(decomposer, n)
    elif sys.argv[1] == "--batch":
        with open(sys.argv[2]) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    solve_with_trace(decomposer, line)
                    print("\n")
    else:
        problem = " ".join(sys.argv[1:])
        solve_with_trace(decomposer, problem)


if __name__ == "__main__":
    main()
