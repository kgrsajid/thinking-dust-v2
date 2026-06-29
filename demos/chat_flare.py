#!/usr/bin/env python3
"""TD v2 — Flare Chat. Theatrical, intent-aware, jaw-dropping demo.

Usage:
    .venv/bin/python3 demos/chat.py --pure
    .venv/bin/python3 demos/chat.py --seeded

Flares:
    ✦ Real-time reasoning trace (watch the mind think)
    ✦ HDC similarity heatmap (what is it retrieving?)
    ✦ Memory growth animation (patterns being stored live)
    ✦ Confidence gauge (honest, dynamic)
    ✦ Intent classification (6 innate intents via HDC prototypes)
    ✦ Pure mode arc: 0 → learned → useful (the thesis)
    ✦ Feedback loop: 👍 / 👎 / ✏️ (wires to teach)
    ✦ Seed ratio tracker (are we still dependent on seed?)
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from td.perception.hdc import build_default_vocabulary
from td.memory.mhn import ModernHopfieldNetwork, MHNConfig
from td.thinking import GenericThinkingDust


C = {
    "reset": "[0m", "bold": "[1m", "dim": "[2m",
    "red": "[91m", "green": "[92m", "yellow": "[93m",
    "blue": "[94m", "magenta": "[95m", "cyan": "[96m",
    "white": "[97m", "gray": "[90m",
}


def bar(val, max_val=1.0, width=20, color="green"):
    filled = int((val / max_val) * width)
    empty = width - filled
    c = C.get(color, C["green"])
    return f"{c}{'█' * filled}{C['gray']}{'░' * empty}{C['reset']}"


def gauge(confidence):
    if confidence >= 0.90:
        return f"{C['green']}[CERTAIN]{C['reset']}"
    elif confidence >= 0.70:
        return f"{C['cyan']}[CONFIDENT]{C['reset']}"
    elif confidence >= 0.50:
        return f"{C['yellow']}[LIKELY]{C['reset']}"
    elif confidence >= 0.30:
        return f"{C['magenta']}[UNCERTAIN]{C['reset']}"
    else:
        return f"{C['red']}[UNKNOWN]{C['reset']}"


def intent_emoji(intent):
    return {
        "question": "❓",
        "constraint": "🔧",
        "suggestion": "💡",
        "command": "📌",
        "conversation": "💬",
        "meta": "🔍",
        "unknown": "❓",
    }.get(intent, "❓")


def thinking_animation(duration=0.5):
    import itertools
    chars = itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
    start = time.time()
    while time.time() - start < duration:
        c = next(chars)
        print(f"  {C['gray']}{c} Thinking...{C['reset']}", end="", flush=True)
        time.sleep(0.05)
    print(f"  {' ' * 20}", end="")


def print_banner():
    print(f"""
{C['cyan']}  ╔═══════════════════════════════════════════════════════════════╗{C['reset']}
{C['cyan']}  ║{C['reset']}  {C['bold']}✦ THINKING DUST v2 — Intent-Aware Reasoning Engine{C['reset']}       {C['cyan']}║{C['reset']}
{C['cyan']}  ║{C['reset']}  {C['dim']}6 intents · 18 primitives · HDC · Z3 · MHN · Zero seed{C['reset']}  {C['cyan']}║{C['reset']}
{C['cyan']}  ╚═══════════════════════════════════════════════════════════════╝{C['reset']}
""")


def print_memory_state(td, before=False):
    s = td.stats()
    total = s["memory_size"]
    seed = s["seed_patterns"]
    learned = s["learned_patterns"]
    seed_pct = s["seed_ratio_pct"]

    if total == 0:
        print(f"  {C['gray']}Memory: empty (pure mode — teach me){C['reset']}")
        return

    seed_bar = bar(seed / total if total else 0, 1.0, 15, "yellow")
    learned_bar = bar(learned / total if total else 0, 1.0, 15, "green")

    print(f"  {C['bold']}Memory:{C['reset']} {total} patterns")
    print(f"    {C['yellow']}Seed:{C['reset']}     {seed_bar} {seed_pct:.0f}%")
    print(f"    {C['green']}Learned:{C['reset']}  {learned_bar} {100-seed_pct:.0f}%")

    if seed_pct < 5 and seed > 0:
        print(f"  {C['green']}✦ SEED RATIO < 5% — System is now self-learning!{C['reset']}")


def print_reasoning_trace(result):
    if not result.trace:
        return
    print(f"
  {C['gray']}┌─ Reasoning Trace ───────────────────────────────────────┐{C['reset']}")
    for line in result.trace:
        if line.startswith("Intent:"):
            print(f"  {C['gray']}│{C['reset']} {C['magenta']}{intent_emoji(result.intent)} {line}{C['reset']}")
        elif line.startswith("Parsed:"):
            print(f"  {C['gray']}│{C['reset']} {C['cyan']}📄 {line}{C['reset']}")
        elif "Iter" in line and "CONVERGED" in line:
            print(f"  {C['gray']}│{C['reset']} {C['green']}   {line}{C['reset']}")
        elif "Iter" in line:
            print(f"  {C['gray']}│{C['reset']} {C['yellow']}   {line}{C['reset']}")
        elif "HDC" in line:
            print(f"  {C['gray']}│{C['reset']} {C['blue']}🔍 {line}{C['reset']}")
        elif "Z3: SAT" in line:
            print(f"  {C['gray']}│{C['reset']} {C['green']}✓ {line}{C['reset']}")
        elif "Z3: UNSAT" in line:
            print(f"  {C['gray']}│{C['reset']} {C['red']}✗ {line}{C['reset']}")
        elif "Z3: Not" in line or "Z3: No" in line:
            print(f"  {C['gray']}│{C['reset']} {C['gray']}⊘ {line}{C['reset']}")
        elif "MHN:" in line:
            print(f"  {C['gray']}│{C['reset']} {C['cyan']}💡 {line}{C['reset']}")
        elif "Stored" in line:
            print(f"  {C['gray']}│{C['reset']} {C['green']}💾 {line}{C['reset']}")
        else:
            print(f"  {C['gray']}│{C['reset']} {C['gray']}   {line}{C['reset']}")
    print(f"  {C['gray']}└────────────────────────────────────────────────────────┘{C['reset']}")


def print_hdc_heatmap(result):
    if not result.thoughts:
        return
    print(f"
  {C['bold']}HDC Retrieval Heatmap:{C['reset']}")
    for t in result.thoughts:
        if t.retrieved_hdc is not None:
            color = "green" if t.retrieved_similarity > 0.7 else "yellow" if t.retrieved_similarity > 0.4 else "red"
            sim_bar = bar(t.retrieved_similarity, 1.0, 25, color)
            label = t.retrieved_metadata.get("title", t.description[:40])
            print(f"    {sim_bar} {t.retrieved_similarity:.2f} — {label}")


def print_solution(result):
    sol = result.solution or {}
    formatted = sol.get("formatted", "")
    sol_type = sol.get("type", "unknown")
    prims = sol.get("primitives_applied", [])
    conf = result.confidence

    print(f"
  {C['bold']}Answer:{C['reset']} {gauge(conf)} {C['gray']}(intent: {result.intent}){C['reset']}")

    if sol_type == "generic_csp" and prims and prims != ["bounded"]:
        print(f"  {C['cyan']}┌─ Constraint Solution ─────────────────────────────────┐{C['reset']}")
        for line in formatted.split("
"):
            print(f"  {C['cyan']}│{C['reset']}  {line}")
        print(f"  {C['cyan']}└────────────────────────────────────────────────────────┘{C['reset']}")
        print(f"  {C['gray']}Primitives: {C['yellow']}{' + '.join(prims)}{C['reset']}")

    elif sol_type == "learned" and formatted:
        print(f"  {C['green']}→ {formatted}{C['reset']}")

    elif sol_type == "unsat":
        print(f"  {C['red']}→ {formatted}{C['reset']}")
        print(f"  {C['red']}  (Proven impossible by Z3){C['reset']}")

    elif sol_type == "conversation" and formatted:
        print(f"  {C['magenta']}→ {formatted}{C['reset']}")

    elif sol_type == "suggestion" and formatted:
        print(f"  {C['yellow']}→ {formatted}{C['reset']}")

    elif sol_type == "command" and formatted:
        print(f"  {C['blue']}→ {formatted}{C['reset']}")

    elif sol_type == "meta" and formatted:
        print(f"  {C['gray']}→ {formatted}{C['reset']}")

    else:
        print(f"  {C['magenta']}→ I don't know this one yet.{C['reset']}")


def print_feedback_prompt(problem_text):
    print(f"
  {C['gray']}Was this helpful?{C['reset']}")
    print(f"    {C['green']}[y]{C['reset']} Yes  {C['red']}[n]{C['reset']} No  {C['yellow']}[t]{C['reset']} Teach me the right answer")


def handle_feedback(td, problem_text, result, user_input):
    if user_input.lower() == "y":
        print(f"  {C['green']}✓ Feedback recorded. Memory reinforced.{C['reset']}")
        return True
    elif user_input.lower() == "n":
        print(f"  {C['red']}✗ Feedback recorded. Will try to do better.{C['reset']}")
        return True
    elif user_input.lower() == "t":
        print(f"  {C['yellow']}Teach mode: enter the correct answer.{C['reset']}")
        try:
            correct = input(f"  correct › ").strip()
            if correct:
                r = td.teach(problem_text, correct)
                print(f"  {C['green']}✓ {r['message']} Memory: {r['memory_size']}{C['reset']}")
                print(f"
  {C['green']}✦ NEW PATTERN STORED ✦{C['reset']}")
                print_memory_state(td)
        except (EOFError, KeyboardInterrupt):
            pass
        return True
    return False


def main():
    pure = "--pure" in sys.argv
    seeded = not pure

    print_banner()

    vocab = build_default_vocabulary(dim=10_000)
    mhn = ModernHopfieldNetwork(MHNConfig(dim=10_000, min_similarity=0.01))
    td = GenericThinkingDust(vocab=vocab, mhn=mhn, dim=10_000, pure_mode=pure)

    print(f"  Mode: {C['green'] if pure else C['yellow']}● {'PURE' if pure else 'SEEDED'}{C['reset']}")
    print_memory_state(td)
    print()

    print(f"  {C['gray']}Commands:{C['reset']}")
    print(f"    {C['yellow']}teach: <problem> | <solution>{C['reset']}  — teach the system")
    print(f"    {C['yellow']}stats{C['reset']}                      — show memory state")
    print(f"    {C['yellow']}trace{C['reset']}                      — toggle reasoning trace")
    print(f"    {C['yellow']}quit{C['reset']}                       — exit")
    print(f"  {C['gray']}{'─' * 60}{C['reset']}")
    print()

    show_trace = False
    last_problem = None
    last_result = None
    awaiting_feedback = False

    while True:
        try:
            prompt = "feedback › " if awaiting_feedback else "you › "
            user_input = input(f"{prompt}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"
{C['gray']}bye.{C['reset']}")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print(f"{C['gray']}bye.{C['reset']}")
            break
        if user_input.lower() == "stats":
            print_memory_state(td)
            print()
            continue
        if user_input.lower() == "trace":
            show_trace = not show_trace
            print(f"  {C['yellow']}Reasoning trace: {'ON' if show_trace else 'OFF'}{C['reset']}")
            continue

        # Handle feedback
        if awaiting_feedback:
            if handle_feedback(td, last_problem, last_result, user_input):
                awaiting_feedback = False
                print()
                continue
            awaiting_feedback = False

        # Handle teach command
        if user_input.lower().startswith("teach:"):
            rest = user_input[6:].strip()
            if "|" not in rest:
                print(f"
  {C['red']}Format: teach: <problem> | <solution>{C['reset']}
")
                continue
            problem, solution = rest.split("|", 1)
            r = td.teach(problem.strip(), solution.strip())
            print(f"
  {C['green']}✓ {r['message']}{C['reset']}")
            print(f"
  {C['green']}✦ NEW PATTERN STORED ✦{C['reset']}")
            print_memory_state(td)
            print()
            continue

        # ─── THE MAIN THINKING LOOP ───────────────────────────────────
        last_problem = user_input

        # Show thinking animation
        thinking_animation(0.3)

        # Compute
        result = td.think(user_input)
        last_result = result

        # Show reasoning trace if enabled
        if show_trace:
            print_reasoning_trace(result)
            print_hdc_heatmap(result)

        # Show solution
        print_solution(result)

        # Show latency
        print(f"  {C['gray']}({result.latency_ms:.0f}ms · {result.iterations} IDP iterations){C['reset']}")

        # If confidence is low, ask to teach
        if result.confidence < 0.40 or (result.intent == "question" and not result.solution):
            print(f"
  {C['magenta']}I don't know this yet. Teach me?{C['reset']}")
            print(f"    {C['yellow']}teach: {user_input[:40]} | <answer>{C['reset']}")
        elif result.intent not in ("conversation", "suggestion", "command"):
            print_feedback_prompt(user_input)
            awaiting_feedback = True

        print()


if __name__ == "__main__":
    main()
