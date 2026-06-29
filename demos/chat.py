#!/usr/bin/env python3
"""TD v2 — Interactive Chat (ChatGPT-style)

Teach and think in the same session. TD learns from you in real-time.

Usage:
    .venv/bin/python3 demos/chat.py
    .venv/bin/python3 demos/chat.py --pure

Commands:
    Just type anything → TD thinks and responds
    teach: <problem> | <solution> → teach TD a new strategy
    stats → show memory stats
    quit → exit
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from td.perception.hdc import build_default_vocabulary
from td.memory.mhn import ModernHopfieldNetwork, MHNConfig
from td.thinking import ThinkingDust


def main():
    pure = "--pure" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--pure"]

    print("✦ Thinking Dust — Interactive Chat")
    print(f"   Mode: {'PURE (learn from scratch)' if pure else 'SEED (50 innate reflexes)'}")
    print()

    vocab = build_default_vocabulary(dim=10_000)
    mhn = ModernHopfieldNetwork(MHNConfig(dim=10_000, min_similarity=0.01, idp_enabled=False))
    td = ThinkingDust(vocab=vocab, mhn=mhn, max_idp_iterations=5, pure_mode=pure)

    s = td.stats()
    print(f"   Memory: {s['memory_size']} patterns ({s['seed_patterns']} seed, {s['learned_patterns']} learned)")
    print()
    print("   Type a question, or 'teach: <problem> | <solution>' to teach.")
    print("   Type 'stats' for memory, 'quit' to exit.")
    print(f"   {'─' * 60}")
    print()

    while True:
        try:
            user_input = input("you › ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("bye.")
            break

        if user_input.lower() == "stats":
            s = td.stats()
            print(f"\n  Memory:       {s['memory_size']} patterns")
            print(f"  Seed:         {s['seed_patterns']}")
            print(f"  Learned:      {s['learned_patterns']}")
            print(f"  Seed ratio:   {s['seed_ratio_pct']:.1f}%")
            print(f"  Total thinks: {s['total_thinks']}")
            print(f"  Avg iters:    {s['avg_iterations']}")
            print()
            continue

        # Teach command: "teach: how to focus | Use Pomodoro: 25min work, 5min break"
        if user_input.lower().startswith("teach:"):
            rest = user_input[6:].strip()
            if "|" not in rest:
                print("\n  Format: teach: <problem> | <solution>\n")
                continue
            problem, solution = rest.split("|", 1)
            problem = problem.strip()
            solution = solution.strip()
            if not problem or not solution:
                print("\n  Both problem and solution needed.\n")
                continue
            result = td.teach(problem, solution)
            print(f"\n  ✅ Learned! Memory: {result['memory_size']} patterns\n")
            continue

        # Think
        result = td.think(user_input)

        # Format response
        print()
        print(f"td  › ", end="")

        sol = result.solution or {}
        formatted = sol.get("formatted", "")
        sol_type = sol.get("type", "unknown")

        if result.confidence < 0.25 or (sol_type == "advice" and "No specific" in formatted):
            print(f"I don't know this one yet.")
            print(f"     Teach me with: teach: {user_input[:50]} | <your solution>")
        elif sol_type in ("schedule", "budget"):
            print()
            for line in formatted.split("\n"):
                print(f"     {line}")
            print(f"     ({result.confidence:.0%} confidence, Z3 verified)")
        elif sol_type in ("advice", "learned", "retrieved"):
            print()
            for line in formatted.split("\n"):
                print(f"     {line}")
            print(f"     ({result.confidence:.0%} confidence)")
        elif sol_type == "info_request":
            print(formatted)
            print(f"     ({result.confidence:.0%} confidence)")
        else:
            if formatted:
                print(formatted)
            else:
                print("Not sure how to handle this.")
            print(f"     ({result.confidence:.0%} confidence)")

        # Show IDP trace briefly
        if result.thoughts:
            best = max(result.thoughts, key=lambda t: t.retrieved_similarity)
            if best.retrieved_similarity > 0:
                label = best.retrieved_metadata.get("title", best.description[:30])
                print(f"     [retrieved: {label}, sim={best.retrieved_similarity:.2f}]")

        s = td.stats()
        print(f"     [memory: {s['memory_size']} | seed: {s['seed_ratio_pct']:.0f}%]")
        print()


if __name__ == "__main__":
    main()
