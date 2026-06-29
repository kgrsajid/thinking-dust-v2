#!/usr/bin/env python3
"""TD v2 — Generic Chat (universal primitives, zero hardcoding)

Teach and think with the generic reasoning engine.

Usage:
    .venv/bin/python3 demos/chat_generic.py
    .venv/bin/python3 demos/chat_generic.py --pure
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from td.perception.hdc import build_default_vocabulary
from td.memory.mhn import ModernHopfieldNetwork, MHNConfig
from td.thinking_generic import GenericThinkingDust


def main():
    pure = "--pure" in sys.argv

    print("✦ Thinking Dust — Generic Chat (universal primitives)")
    print(f"   Mode: {'PURE (learn from scratch)' if pure else 'SEED (50 innate reflexes)'}")
    print()

    vocab = build_default_vocabulary(dim=10_000)
    mhn = ModernHopfieldNetwork(MHNConfig(dim=10_000, min_similarity=0.01))
    td = GenericThinkingDust(vocab=vocab, mhn=mhn, dim=10_000, pure_mode=pure)

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
            print(f"\n  Memory:       {s['memory_size']}")
            print(f"  Seed:         {s['seed_patterns']}")
            print(f"  Learned:      {s['learned_patterns']}")
            print(f"  Seed ratio:   {s['seed_ratio_pct']:.1f}%")
            print(f"  Total thinks: {s['total_thinks']}")
            print()
            continue

        if user_input.lower().startswith("teach:"):
            rest = user_input[6:].strip()
            if "|" not in rest:
                print("\n  Format: teach: <problem> | <solution>\n")
                continue
            problem, solution = rest.split("|", 1)
            result = td.teach(problem.strip(), solution.strip())
            print(f"\n  ✅ {result['message']} Memory: {result['memory_size']}\n")
            continue

        # Think
        result = td.think(user_input)
        sol = result.solution or {}
        formatted = sol.get("formatted", "")
        sol_type = sol.get("type", "unknown")

        print()
        print(f"td  › ", end="")
        if result.confidence < 0.25 or (sol_type == "unknown"):
            print(f"I don't know this one yet.")
            print(f"     Teach me with: teach: {user_input[:50]} | <your solution>")
        elif formatted:
            print()
            for line in formatted.split("\n"):
                print(f"     {line}")
            prim = sol.get("primitives_applied", [])
            if prim:
                print(f"     ({result.confidence:.0%} confidence, primitives: {' + '.join(prim)})")
            else:
                print(f"     ({result.confidence:.0%} confidence)")
        else:
            print("Not sure how to handle this.")
            print(f"     ({result.confidence:.0%} confidence)")

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
