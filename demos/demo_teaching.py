#!/usr/bin/env python3
"""TD v2 — Pure Mode Teaching Demo.

The jaw-dropping demo: TD starts with 0 knowledge, gets taught by a human,
and then gives useful advice to the next question.

Usage:
    .venv/bin/python3 demos/demo_teaching.py

Scene 1: TD doesn't know → "Teach me"
Scene 2: Human teaches → TD stores
Scene 3: Similar question → TD retrieves learned knowledge
Scene 4: Show stats — 100% learned, 0% seed
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from td.perception.hdc import build_default_vocabulary
from td.memory.mhn import ModernHopfieldNetwork, MHNConfig
from td.thinking import ThinkingDust


class C:
    RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
    R = "\033[31m"; G = "\033[32m"; Y = "\033[33m"
    B = "\033[34m"; M = "\033[35m"; CY = "\033[36m"; GR = "\033[90m"


def print_header():
    print(f"\n{C.BOLD}{C.CY}╔══════════════════════════════════════════════════════════════╗{C.RESET}")
    print(f"{C.BOLD}{C.CY}║  ✦  Thinking Dust — Teaching Demo (Pure Mode)               ║{C.RESET}")
    print(f"{C.BOLD}{C.CY}║  Watch dust learn from nothing.                             ║{C.RESET}")
    print(f"{C.BOLD}{C.CY}╚══════════════════════════════════════════════════════════════╝{C.RESET}\n")


def print_stats(td):
    s = td.stats()
    print(f"\n{C.GR}  Memory: {s['memory_size']} patterns "
          f"({s['seed_patterns']} seed, {s['learned_patterns']} learned) "
          f"| Seed ratio: {s['seed_ratio_pct']:.1f}% "
          f"| Total thoughts: {s['total_thinks']}{C.RESET}\n")


def solve_and_show(td, problem, label=""):
    """Think about a problem and show the result."""
    if label:
        print(f"{C.BOLD}{C.M}── {label} ──{C.RESET}")

    print(f"{C.BOLD}Query:{C.RESET} {problem}\n")

    result = td.think(problem)

    # Show IDP
    for t in result.thoughts:
        marker = f"{C.G}✓" if t.converged else f"{C.Y}→"
        desc = t.description[:40] if t.description else ""
        print(f"  {marker} Iter {t.iteration}: sim={t.retrieved_similarity:.3f} {desc}{C.RESET}")

    # Show solution
    if result.solution:
        formatted = result.solution.get("formatted", "")
        sol_type = result.solution.get("type", "unknown")

        if sol_type == "advice" and "No specific" in formatted:
            print(f"\n  {C.Y}🤔 TD: I don't know yet. My memory is empty.{C.RESET}")
            needs_teach = True
        elif result.confidence < 0.25:
            print(f"\n  {C.Y}🤔 TD: I'm not confident about this.{C.RESET}")
            print(f"     {formatted[:80]}")
            needs_teach = True
        else:
            print(f"\n  {C.G}💡 TD:{C.RESET}")
            for line in formatted.split("\n"):
                print(f"     {C.G}{line}{C.RESET}")
            needs_teach = False
    else:
        print(f"\n  {C.Y}🤔 TD: I don't know yet.{C.RESET}")
        needs_teach = True

    print(f"\n  {C.GR}Confidence: {result.confidence:.0%} | Latency: {result.latency_ms:.0f}ms{C.RESET}")
    print_stats(td)
    return result, needs_teach


def teach_and_show(td, problem, solution, title=None):
    """Teach TD something and show the result."""
    if title:
        print(f"{C.BOLD}{C.B}── {title} ──{C.RESET}")

    print(f"{C.BOLD}Human teaches:{C.RESET}")
    print(f"  Problem:  {problem}")
    print(f"  Solution: {solution}")

    # Extract strategy title from solution text (first few words before colon)
    strategy_title = solution.split(":")[0] if ":" in solution else solution[:20]

    result = td.teach(problem, solution, {
        "title": strategy_title,
        "effectiveness": 0.75,
    })

    print(f"\n  {C.G}✅ TD: {result['message']}{C.RESET}")
    print(f"  {C.GR}Memory: {result['memory_size']} patterns{C.RESET}\n")


def main():
    print_header()

    # ─── Initialize in PURE mode ────────────────────────────────
    print(f"{C.DIM}Initializing in PURE mode (0 seed patterns)...{C.RESET}\n")
    vocab = build_default_vocabulary(dim=10_000)
    mhn = ModernHopfieldNetwork(MHNConfig(dim=10_000, min_similarity=0.01, idp_enabled=False))
    td = ThinkingDust(vocab=vocab, mhn=mhn, max_idp_iterations=5, pure_mode=True)

    s = td.stats()
    print(f"{C.DIM}Memory: {s['memory_size']} patterns ({s['seed_patterns']} seed, {s['learned_patterns']} learned){C.RESET}")
    print(f"{C.DIM}Seed ratio: {s['seed_ratio_pct']:.1f}%{C.RESET}\n")

    # ─── Scene 1: First query — TD doesn't know ─────────────────
    print(f"{C.GR}{'═' * 64}{C.RESET}")
    result1, needs_teach = solve_and_show(
        td,
        "How to schedule without procrastination",
        label="SCENE 1: First query — empty memory"
    )

    # ─── Scene 2: Human teaches ─────────────────────────────────
    print(f"{C.GR}{'═' * 64}{C.RESET}")
    teach_and_show(
        td,
        "How to schedule without procrastination",
        "Use Pomodoro: work 25 minutes, then take a 5 minute break. Repeat 4 times.",
        title="SCENE 2: Human teaches TD"
    )

    # Also teach a couple more strategies for variety
    teach_and_show(
        td,
        "How to focus on my work and avoid distractions",
        "Eat the Frog: tackle your hardest task first thing in the morning.",
        title="SCENE 2b: Teaching more strategies"
    )

    teach_and_show(
        td,
        "How to debug code that has unexpected behavior",
        "Rubber Duck Debugging: explain your code line by line out loud to find the bug.",
        title="SCENE 2c: Teaching a debugging strategy"
    )

    # ─── Scene 3: Similar query — TD retrieves learned knowledge ─
    print(f"{C.GR}{'═' * 64}{C.RESET}")
    result3, _ = solve_and_show(
        td,
        "I keep putting off my tasks. Any tips?",
        label="SCENE 3: Similar query — does TD remember?"
    )

    # ─── Scene 4: Another learned retrieval ─────────────────────
    print(f"{C.GR}{'═' * 64}{C.RESET}")
    result4, _ = solve_and_show(
        td,
        "How to find bugs in my program",
        label="SCENE 4: Debug query — learned retrieval"
    )

    # ─── Scene 5: Novel query — TD doesn't know yet ─────────────
    print(f"{C.GR}{'═' * 64}{C.RESET}")
    result5, needs_teach5 = solve_and_show(
        td,
        "How to optimize my daily routine",
        label="SCENE 5: Novel query — TD needs more teaching"
    )

    # ─── Final Stats ────────────────────────────────────────────
    print(f"{C.GR}{'═' * 64}{C.RESET}")
    print(f"{C.BOLD}{C.CY}FINAL STATS{C.RESET}\n")
    s = td.stats()
    print(f"  Total interactions: {s['total_thinks']}")
    print(f"  Total learned:      {s['total_learned']}")
    print(f"  Memory size:        {s['memory_size']}")
    print(f"  Seed patterns:      {s['seed_patterns']}")
    print(f"  Learned patterns:   {s['learned_patterns']}")
    print(f"  Seed ratio:         {s['seed_ratio_pct']:.1f}%")
    print(f"  Avg iterations:     {s['avg_iterations']}")
    print(f"  Pure mode:          {s['pure_mode']}")

    print(f"\n{C.BOLD}{C.G}The dust started with nothing.")
    print(f"It was taught 3 strategies by a human.")
    print(f"It now gives useful advice to similar questions.")
    print(f"This is teaching dust to think.{C.RESET}\n")


if __name__ == "__main__":
    main()
