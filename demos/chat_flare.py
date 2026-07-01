#!/usr/bin/env python3
"""TD v2 — Flare Chat. Theatrical, intent-aware, jaw-dropping demo.

Usage:
    .venv/bin/python3 demos/chat.py --pure
    .venv/bin/python3 demos/chat.py --seeded
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from td.perception.hdc import build_default_vocabulary
from td.memory.mhn import ModernHopfieldNetwork, MHNConfig
from td.thinking import GenericThinkingDust


# ANSI colors
C = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "magenta": "\033[95m",
    "cyan": "\033[96m",
    "white": "\033[97m",
    "gray": "\033[90m",
}


def bar(val, max_val=1.0, width=20, color="green"):
    """ASCII bar chart."""
    filled = int((val / max_val) * width)
    empty = width - filled
    c = C.get(color, C["green"])
    return f"{c}{'█' * filled}{C['gray']}{'░' * empty}{C['reset']}"


def gauge(confidence):
    """Dynamic confidence gauge."""
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
    """Emoji for each intent type."""
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
    """Show a thinking animation."""
    import itertools
    chars = itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
    start = time.time()
    while time.time() - start < duration:
        c = next(chars)
        print(f"\r  {C['gray']}{c} Thinking...{C['reset']}", end="", flush=True)
        time.sleep(0.05)
    print(f"\r  {' ' * 20}\r", end="")


def print_banner():
    """Print the banner."""
    print(f"""
{C['cyan']}  ╔═══════════════════════════════════════════════════════════════╗{C['reset']}
{C['cyan']}  ║{C['reset']}  {C['bold']}✦ THINKING DUST v2 — Honest Neuro-Symbolic Engine{C['reset']}       {C['cyan']}║{C['reset']}
{C['cyan']}  ║{C['reset']}  {C['dim']}2 intents · 18 primitives · HDC · Z3 · MHN · Zero seed{C['reset']}  {C['cyan']}║{C['reset']}
{C['cyan']}  ╚═══════════════════════════════════════════════════════════════╝{C['reset']}
""")


def print_memory_state(td):
    """Show memory state with visual flair."""
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
    """Print the reasoning trace with visual hierarchy."""
    if not result.trace:
        return
    print(f"\n  {C['gray']}┌─ Reasoning Trace ───────────────────────────────────────┐{C['reset']}")
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
    """Show HDC retrieval similarity as a heatmap."""
    if not result.thoughts:
        return
    print(f"\n  {C['bold']}HDC Retrieval Heatmap:{C['reset']}")
    for t in result.thoughts:
        if t.retrieved_hdc is not None:
            color = "green" if t.retrieved_similarity > 0.7 else "yellow" if t.retrieved_similarity > 0.4 else "red"
            sim_bar = bar(t.retrieved_similarity, 1.0, 25, color)
            label = t.retrieved_metadata.get("title", t.description[:40])
            print(f"    {sim_bar} {t.retrieved_similarity:.2f} — {label}")


def print_solution(result):
    """Print the solution with dramatic flair."""
    sol = result.solution or {}
    formatted = sol.get("formatted", "")
    sol_type = sol.get("type", "unknown")
    prims = sol.get("primitives_applied", [])
    conf = result.confidence

    print(f"\n  {C['bold']}Answer:{C['reset']} {gauge(conf)} {C['gray']}(intent: {result.intent}){C['reset']}")

    if sol_type == "generic_csp" and prims and prims != ["bounded"]:
        print(f"  {C['cyan']}┌─ Constraint Solution ─────────────────────────────────┐{C['reset']}")
        for line in formatted.split("\n"):
            print(f"  {C['cyan']}│{C['reset']}  {line}")
        print(f"  {C['cyan']}└────────────────────────────────────────────────────────┘{C['reset']}")
        print(f"  {C['gray']}Primitives: {C['yellow']}{' + '.join(prims)}{C['reset']}")

    elif sol_type == "learned" and formatted:
        print(f"  {C['green']}→ {formatted}{C['reset']}")

    elif sol_type == "inferred" and formatted:
        print(f"  {C['cyan']}┌─ Reasoning Trace ──────────────────────────────────────┐{C['reset']}")
        for line in formatted.split("\n"):
            print(f"  {C['cyan']}│{C['reset']}  {line}")
        print(f"  {C['cyan']}└────────────────────────────────────────────────────────┘{C['reset']}")
        print(f"  {C['gray']}Derived via logical inference (not just memory){C['reset']}")

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


def print_feedback_prompt():
    """Show feedback options."""
    print(f"\n  {C['gray']}Was this helpful?{C['reset']}")
    print(f"    {C['green']}[y]{C['reset']} Yes  {C['red']}[n]{C['reset']} No  {C['yellow']}[t]{C['reset']} Teach me the right answer")


def handle_feedback(td, problem_text, result, user_input):
    """Handle feedback and wire to teach()."""
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
                print(f"\n  {C['green']}✦ NEW PATTERN STORED ✦{C['reset']}")
                print_memory_state(td)
        except (EOFError, KeyboardInterrupt):
            pass
        return True
    return False


def _get_state_path():
    """Get the path for the persistent state file."""
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "td_state.pkl"
    )


def _save_state(td):
    """Save TD's learned state (KG, relation properties, MHN patterns)."""
    import pickle
    state_path = _get_state_path()

    # Don't save if nothing was learned
    if not td.mhn.patterns and not td.kg.triples:
        return

    state = {
        "kg_triples": [(t.subject, t.relation, t.object, t.source, t.proof)
                       for t in td.kg.triples],
        "relation_properties": td.kg.relation_properties,
        "total_thinks": td.total_thinks,
        "total_learned": td.total_learned,
    }

    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    with open(state_path, "wb") as f:
        pickle.dump(state, f)

    total = len(td.mhn.patterns) + len(td.kg.triples)
    print(f"  {C['green']}✓ Saved {total} memories to {os.path.basename(state_path)}{C['reset']}")


def _load_state(td):
    """Load TD's learned state from file."""
    import pickle
    state_path = _get_state_path()

    if not os.path.exists(state_path):
        return False

    try:
        with open(state_path, "rb") as f:
            state = pickle.load(f)

        # Restore KG triples
        from td.kg import Triple
        for (s, r, o, source, proof) in state.get("kg_triples", []):
            td.kg.add_fact(s, r, o, source=source, proof=proof)

        # Restore relation properties
        for rel, props in state.get("relation_properties", {}).items():
            td.kg.set_relation_property(rel, *props)

        # Restore stats
        td.total_thinks = state.get("total_thinks", 0)
        td.total_learned = state.get("total_learned", 0)

        kg_count = len(td.kg.triples)
        if kg_count > 0:
            print(f"  {C['green']}✓ Loaded {kg_count} memories from previous session{C['reset']}")
        return True
    except Exception:
        return False


def main():
    pure = "--pure" in sys.argv
    seeded = not pure

    print_banner()

    vocab = build_default_vocabulary(dim=10_000)
    mhn = ModernHopfieldNetwork(MHNConfig(dim=10_000, min_similarity=0.01))

    # Load BEAGLE word vectors if available (enables paraphrase matching)
    wvm = None
    wv_paths = [
        "data/word_vectors_10k.pkl",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "word_vectors_10k.pkl"),
    ]
    for wv_path in wv_paths:
        if os.path.exists(wv_path):
            from td.perception.word_vectors import WordVectorModel
            wvm = WordVectorModel(dim=10_000)
            wvm.load(wv_path)
            print(f"  {C['green']}✦ Loaded word vectors: {wvm.stats()['vocab_size']} words, "
                  f"{wvm.stats()['trained_sentences']} sentences trained{C['reset']}")
            break

    td = GenericThinkingDust(vocab=vocab, mhn=mhn, dim=10_000, pure_mode=pure, word_vectors=wvm)

    # Load previous session's memories (persistent teaching)
    _load_state(td)

    print(f"  Mode: {C['green'] if pure else C['yellow']}● {'PURE' if pure else 'SEEDED'}{C['reset']}")
    print_memory_state(td)
    print()

    print(f"  {C['gray']}Commands:{C['reset']}")
    print(f"    {C['yellow']}teach: <problem> | <solution>{C['reset']}  — teach a fact with answer")
    print(f"    {C['yellow']}teach: <fact>{C['reset']}                   — teach a fact (triple only)")
    print(f"    {C['yellow']}relation: <name> <property>{C['reset']}     — teach relation logic")
    print(f"    {C['yellow']}stats{C['reset']}                      — show memory state")
    print(f"    {C['yellow']}save{C['reset']}                      — save memories to disk")
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
            print(f"\n{C['gray']}bye.{C['reset']}")
            _save_state(td)
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print(f"{C['gray']}bye.{C['reset']}")
            _save_state(td)
            break
        if user_input.lower() == "save":
            _save_state(td)
            continue
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
            if "|" in rest:
                # Full teach: problem | solution
                problem, solution = rest.split("|", 1)
                r = td.teach(problem.strip(), solution.strip())
                print(f"\n  {C['green']}✓ {r['message']}{C['reset']}")
            else:
                # Triple-only teach: just extract facts, no answer
                td.teach(rest, rest)
                print(f"\n  {C['green']}✓ Fact stored.{C['reset']}")

            # Check if any NEW relation was taught that has no properties
            import re
            taught_text = rest.lower()
            new_relations = set()
            for pattern_name, pattern_re in [
                ("X is the Y of Z", r'(\w+)\s+is\s+(?:the\s+)?(\w+)\s+of'),
                ("X is in Y", r'(\w+)\s+is\s+in'),
                ("X is part of Y", r'(\w+)\s+is\s+part\s+of'),
                ("X is before Y", r'(\w+)\s+is\s+before'),
                ("X is after Y", r'(\w+)\s+is\s+after'),
                ("X means Y", r'(\w+)\s+means'),
            ]:
                m = re.search(pattern_re, taught_text)
                if m:
                    # Extract relation from the match
                    if "of" in pattern_name and len(m.groups()) >= 2:
                        rel = f"{m.group(2)}_of"
                    elif "part of" in pattern_name:
                        rel = "part_of"
                    elif "in" in pattern_name:
                        rel = "in"
                    elif "before" in pattern_name:
                        rel = "before"
                    elif "after" in pattern_name:
                        rel = "after"
                    elif "means" in pattern_name:
                        rel = "means"
                    else:
                        rel = m.group(2) if len(m.groups()) >= 2 else None
                    if rel and rel not in td.kg.relation_properties:
                        new_relations.add(rel)

            # Also check for custom patterns like "X is north of Y"
            custom_m = re.search(r'(\w+)\s+is\s+(\w+)\s+of\s+(\w+)', taught_text)
            if custom_m:
                rel = f"{custom_m.group(2)}_of"
                if rel not in td.kg.relation_properties and rel not in new_relations:
                    new_relations.add(rel)

            # Ask about unknown relation properties
            for rel in new_relations:
                print(f"\n  {C['yellow']}┌─ New Relation Detected ─────────────────────────────┐{C['reset']}")
                print(f"  {C['yellow']}│{C['reset']}  I don't know how '{rel}' works yet.        {C['yellow']}│{C['reset']}")
                print(f"  {C['yellow']}│{C['reset']}                                                    {C['yellow']}│{C['reset']}")
                print(f"  {C['yellow']}│{C['reset']}  [1] Transitive — if A {rel} B and B {rel} C, {C['yellow']}│{C['reset']}")
                print(f"  {C['yellow']}│{C['reset']}      then A {rel} C                               {C['yellow']}│{C['reset']}")
                print(f"  {C['yellow']}│{C['reset']}  [2] Symmetric — if A {rel} B, then B {rel} A   {C['yellow']}│{C['reset']}")
                print(f"  {C['yellow']}│{C['reset']}  [3] Functional — each thing has only one       {C['yellow']}│{C['reset']}")
                print(f"  {C['yellow']}│{C['reset']}      (e.g., each city has one capital)          {C['yellow']}│{C['reset']}")
                print(f"  {C['yellow']}│{C['reset']}  [4] Skip — I'll teach it later                   {C['yellow']}│{C['reset']}")
                print(f"  {C['yellow']}└────────────────────────────────────────────────────┘{C['reset']}")

                while True:
                    try:
                        choice = input(f"  {C['cyan']}│ {C['reset']}choice > ").strip()
                    except (EOFError, KeyboardInterrupt):
                        choice = "4"

                    prop_map = {"1": "transitive", "2": "symmetric", "3": "functional"}
                    if choice in prop_map:
                        td.teach_relation(rel, prop_map[choice])
                        print(f"  {C['green']}✓ '{rel}' is now {prop_map[choice]}.{C['reset']}")
                        break
                    elif choice == "4":
                        print(f"  {C['gray']}Skipped. You can teach it later:{C['reset']}")
                        print(f"  {C['gray']}  relation: {rel} transitive{C['reset']}")
                        break
                    else:
                        print(f"  {C['red']}Please enter 1, 2, 3, or 4.{C['reset']}")

            print(f"\n  {C['green']}✦ NEW PATTERN STORED ✦{C['reset']}")
            print_memory_state(td)
            print()
            continue

        # ─── relation property teaching ──────────────────────────────
        if user_input.lower().startswith("relation:"):
            rest = user_input[9:].strip()
            parts = rest.split()
            if len(parts) >= 2:
                rel = parts[0]
                props = parts[1:]
                r = td.teach_relation(rel, *props)
                print(f"\n  {C['green']}✓ {r['message']}{C['reset']}")
            else:
                print(f"\n  {C['red']}Format: relation: <name> <property>{C['reset']}")
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
            print(f"\n  {C['magenta']}I don't know this yet. Teach me?{C['reset']}")
            print(f"    {C['yellow']}teach: {user_input[:40]} | <answer>{C['reset']}")
        elif result.intent not in ("conversation", "suggestion", "command"):
            print_feedback_prompt()
            awaiting_feedback = True

        print()


if __name__ == "__main__":
    main()
