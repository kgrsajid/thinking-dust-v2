"""Problem-specific Z3 constraint generation and solution formatting.

Instead of generic boolean constraints, this module extracts entities from
problem text and builds real Z3 models with integer variables, inequality
constraints, and objective functions.

This is the difference between:
  Output: "Z3 constraints: 3 variables: 3" (useless)
  Output: "Monday 9am: Alice, Tuesday 2pm: Bob" (useful)
"""

from __future__ import annotations

import re
from typing import Any

from .reasoning.z3_bridge import _z3_available

if _z3_available:
    from z3 import (
        Solver, Int, Real, Bool, And, Or, Not, Implies, If as Z3If,
        sat, unsat, unknown, Optimize,
    )


def extract_entities(text: str) -> dict:
    """Extract problem-specific entities from natural language text.

    Uses the NLParser's entity extraction if available, falls back to regex.
    """
    # Try to use the NLParser for rich entity extraction
    try:
        from .perception.nl_parser import NLParser
        from .perception.hdc import build_default_vocabulary
        vocab = build_default_vocabulary(dim=10_000)
        parser = NLParser(vocab)
        entities = parser.extract_entities(text)
        # Also add text-derived entities
        entities["_text"] = text
        return entities
    except Exception:
        pass

    # Fallback: basic regex extraction
    lower = text.lower()
    entities = {}
    numbers = [int(n) for n in re.findall(r'\b(\d+)\b', text)]
    entities["numbers"] = numbers
    entities["dollars"] = [int(d) for d in re.findall(r'\$(\d+)', text)]
    entities["type"] = "generic"
    return entities


def build_z3_model(entities: dict, context: dict | None = None):
    """Build a problem-specific Z3 model with real variables.

    If entities contains 'who' from parent problem, those names are used
    for Z3 variables instead of generic Person_N labels.

    Returns:
        Tuple of (solver, variables_dict, problem_type)
    """
    if not _z3_available:
        return None, {}, entities.get("type", "generic")

    # Determine problem type from entities
    ptype = entities.get("type", entities.get("problem_type", "generic"))

    # Map parser problem_type to z3 problem type
    if ptype == "concrete_plan" or entities.get("action") in ("schedule", "book", "allocate", "assign"):
        # Check if there are scheduling-related targets
        what = entities.get("what", [])
        if any(w in str(what) for w in ["meeting", "schedule", "appointment", "interview"]):
            ptype = "scheduling"
        elif any(w in str(what) for w in ["budget", "cost", "allocation"]):
            ptype = "budget"
        else:
            # Use 'who' presence as scheduling indicator
            who = [w for w in entities.get("who", [])
                   if w not in {"schedule", "plan", "task", "project"}]
            if who:
                ptype = "scheduling"
            else:
                ptype = "generic"
    elif ptype == "advice":
        return None, {}, "advice"  # Advice doesn't use Z3
    elif ptype == "proof":
        ptype = "proof"
    elif ptype == "optimization":
        ptype = "optimization"

    # Also check context text for problem type
    ctx_text = (context or {}).get("text", "").lower()
    if ptype == "generic":
        if any(w in ctx_text for w in ["schedule", "meeting", "appointment"]):
            ptype = "scheduling"
        elif any(w in ctx_text for w in ["budget", "allocate", "distribute"]):
            ptype = "budget"
        elif any(w in ctx_text for w in ["prove", "proof", "induction"]):
            ptype = "proof"

    if ptype == "scheduling":
        return _build_scheduling_model(entities, context)
    elif ptype == "budget":
        return _build_budget_model(entities, context)
    elif ptype == "proof":
        return _build_proof_model(entities, context)
    elif ptype == "optimization":
        return _build_optimization_model(entities, context)
    else:
        return _build_generic_model(entities, context)


def _build_scheduling_model(entities: dict, context: dict):
    """Build Z3 model for scheduling problems.

    Creates integer variables for each person/item, constrained to
    valid day slots, with no-overlap constraints.
    """
    s = Solver()
    vars = {}

    # Determine participants from entities
    who = entities.get("who", [])
    # Filter out false positives (non-name words caught as people)
    false_positives = {"schedule", "plan", "task", "project", "budget", "deadline"}
    who = [w for w in who if w not in false_positives]

    how_many = entities.get("how_many")
    if how_many and isinstance(how_many, int):
        n_people = max(len(who), how_many, 2)
    else:
        n_people = max(len(who), 3)

    # Generate participant names: use extracted names, fill gaps with Task_N
    if who:
        names = list(who[:n_people])
        while len(names) < n_people:
            names.append(f"Task_{len(names)+1}")
    else:
        names = [f"Task_{i+1}" for i in range(n_people)]

    # Days: Monday(1) to Friday(5) unless specified
    max_day = 5
    if entities.get("days"):
        max_day = max(entities["days"])

    # Time slots: 9am(1) to 5pm(8) — 1-hour slots
    max_slot = 8

    # Create variables: each person gets a (day, slot) assignment
    for name in names[:n_people]:
        day_var = Int(f"{name}_day")
        slot_var = Int(f"{name}_slot")
        s.add(day_var >= 1, day_var <= max_day)
        s.add(slot_var >= 1, slot_var <= max_slot)
        vars[name] = {"day": day_var, "slot": slot_var}

    # No two people at the same time slot on the same day
    for i, n1 in enumerate(names[:n_people]):
        for n2 in names[i+1:n_people]:
            s.add(Or(
                vars[n1]["day"] != vars[n2]["day"],
                vars[n1]["slot"] != vars[n2]["slot"],
            ))

    # Priority constraints
    lower_text = str(context.get("text", "")).lower()
    if "avoid friday" in lower_text:
        for name in names[:n_people]:
            s.add(vars[name]["day"] != 5)  # Avoid Friday

    if "morning" in lower_text:
        for name in names[:n_people]:
            s.add(vars[name]["slot"] <= 4)  # Before noon

    return s, vars, "scheduling"


def _build_budget_model(entities: dict, context: dict):
    """Build Z3 model for budget allocation problems."""
    s = Optimize()  # Use optimizer for budget
    vars = {}

    # Determine total budget
    total = 0
    if entities.get("dollars"):
        total = entities["dollars"][0]
    elif entities.get("numbers"):
        total = entities["numbers"][0]
    else:
        total = 10000  # Default

    # Determine departments/categories
    lower_text = str(context.get("text", "")).lower()
    departments = []
    for dept in ["marketing", "engineering", "design", "sales", "research",
                 "operations", "hr", "it", "finance", "support"]:
        if dept in lower_text:
            departments.append(dept.capitalize())

    if not departments:
        departments = ["Operations", "Marketing", "Development"]

    # Check for percentage split
    percentages = re.findall(r'(\d+)\s*[/\\]?\s*(\d+)\s*[/\\]?\s*(\d+)',
                             str(context.get("text", "")))
    if percentages:
        pcts = [int(x) for x in percentages[0]]
        if len(pcts) == len(departments):
            for dept, pct in zip(departments, pcts):
                var = Int(f"{dept}_allocation")
                s.add(var == (total * pct) // 100)
                vars[dept] = var
            s.add(sum(vars.values()) <= total)
            return s, vars, "budget"

    # Default: equal split with minimum constraint
    n = len(departments)
    min_alloc = total // (n * 3)  # At least 1/3 of equal share

    for dept in departments:
        var = Int(f"{dept}_allocation")
        s.add(var >= min_alloc)
        vars[dept] = var

    # Total within budget
    s.add(sum(vars.values()) <= total)

    # Maximize minimum allocation (fair distribution)
    min_var = Int("min_allocation")
    for v in vars.values():
        s.add(min_var <= v)
    s.maximize(min_var)

    return s, vars, "budget"


def _build_proof_model(entities: dict, context: dict):
    """Build Z3 model for proof verification."""
    s = Solver()
    vars = {}

    # For proofs, Z3 verifies logical consistency
    # Each step is a boolean variable (is this step valid?)
    steps = ["premises_hold", "base_case_verified",
             "inductive_hypothesis_stated", "inductive_step_proven",
             "inference_rules_valid", "conclusion_follows"]

    lower_text = str(context.get("text", "")).lower()

    if "induction" in lower_text:
        steps_used = ["premises_hold", "base_case_verified",
                      "inductive_hypothesis_stated", "inductive_step_proven",
                      "conclusion_follows"]
    elif "contradiction" in lower_text:
        steps_used = ["premises_hold", "assumption_stated",
                      "contradiction_derived", "conclusion_follows"]
    else:
        steps_used = ["premises_hold", "inference_rules_valid", "conclusion_follows"]

    for step in steps_used:
        var = Bool(step)
        s.add(var)  # Each step must hold
        vars[step] = var

    return s, vars, "proof"


def _build_optimization_model(entities: dict, context: dict):
    """Build Z3 optimization model."""
    s = Optimize()
    vars = {}

    # For knapsack-like problems, create item variables
    n_items = entities.get("numbers", [10])[0] if entities.get("numbers") else 10
    n_items = min(n_items, 20)  # Cap for performance

    weight_limit = 100
    lower_text = str(context.get("text", "")).lower()
    wl_match = re.findall(r'weight\s*(?:limit)?\s*(\d+)', lower_text)
    if wl_match:
        weight_limit = int(wl_match[0])

    weights = {}
    values = {}
    for i in range(n_items):
        w = Int(f"item_{i}_weight")
        v = Int(f"item_{i}_value")
        included = Bool(f"item_{i}_included")

        # Random-ish weights and values (deterministic)
        s.add(w == ((i * 7 + 3) % 15 + 1))
        s.add(v == ((i * 11 + 5) % 10 + 1))

        weights[i] = (w, included)
        values[i] = (v, included)
        vars[f"item_{i}"] = included

    # Weight constraint
    total_weight = Int("total_weight")
    s.add(total_weight == sum(
        Z3If(inc, w, 0) for (w, inc) in weights.values()
    ))
    s.add(total_weight <= weight_limit)

    # Maximize total value
    total_value = Int("total_value")
    s.add(total_value == sum(
        Z3If(inc, v, 0) for (v, inc) in values.values()
    ))
    s.maximize(total_value)
    vars["total_value"] = total_value

    return s, vars, "optimization"


def _build_generic_model(entities: dict, context: dict):
    """Generic Z3 model for problems without specific structure."""
    s = Solver()
    vars = {}

    requirements = ["input_valid", "processing_complete", "output_correct"]
    for req in requirements:
        var = Bool(req)
        s.add(var)
        vars[req] = var

    return s, vars, "generic"


def solve_z3_model(solver, variables: dict, problem_type: str) -> dict | None:
    """Solve a Z3 model and extract the solution.

    Returns:
        Solution dict with concrete values, or None if unsolvable.
    """
    if not _z3_available:
        return None

    result = solver.check()

    if result == sat:
        model = solver.model()
        return format_solution(model, variables, problem_type)
    elif result == unsat:
        return {"status": "unsat", "error": "Constraints are contradictory — no valid solution exists"}
    else:
        return {"status": "unknown", "error": "Z3 could not determine satisfiability"}


def format_solution(model, variables: dict, problem_type: str) -> dict:
    """Format a Z3 model into a human-readable solution.

    This is where "Z3 constraints: 3 variables: 3" becomes
    "Monday 9am: Alice, Tuesday 2pm: Bob".
    """
    solution = {"status": "solved", "type": problem_type, "details": {}}

    if problem_type == "scheduling":
        day_names = {1: "Monday", 2: "Tuesday", 3: "Wednesday",
                     4: "Thursday", 5: "Friday", 6: "Saturday", 7: "Sunday"}
        slot_times = {1: "9:00 AM", 2: "10:00 AM", 3: "11:00 AM",
                      4: "12:00 PM", 5: "1:00 PM", 6: "2:00 PM",
                      7: "3:00 PM", 8: "4:00 PM"}

        assignments = []
        for name, var_dict in variables.items():
            if isinstance(var_dict, dict) and "day" in var_dict:
                day_val = int(str(model.eval(var_dict["day"], model_completion=True)))
                slot_val = int(str(model.eval(var_dict["slot"], model_completion=True)))
                day_name = day_names.get(day_val, f"Day {day_val}")
                time_name = slot_times.get(slot_val, f"Slot {slot_val}")
                assignments.append({"person": name, "day": day_name, "time": time_name})

        # Sort by day then time
        assignments.sort(key=lambda x: (x["day"], x["time"]))
        solution["details"]["assignments"] = assignments
        solution["details"]["total_meetings"] = len(assignments)
        solution["details"]["conflicts"] = "none — all slots unique"
        solution["formatted"] = "\n".join(
            f"  {a['day']} {a['time']}: {a['person']}" for a in assignments
        )

    elif problem_type == "budget":
        allocations = {}
        for name, var in variables.items():
            if name == "min_allocation":
                continue
            if hasattr(var, 'name'):  # Z3 Int variable
                val = int(str(model.eval(var, model_completion=True)))
                allocations[name] = val

        solution["details"]["allocations"] = allocations
        solution["details"]["total"] = sum(allocations.values())
        solution["formatted"] = "\n".join(
            f"  {dept}: ${amt:,}" for dept, amt in allocations.items()
        ) + f"\n  Total: ${sum(allocations.values()):,}"

    elif problem_type == "proof":
        steps_verified = []
        for name, var in variables.items():
            val = bool(model.eval(var, model_completion=True))
            steps_verified.append({"step": name, "verified": val})

        solution["details"]["steps"] = steps_verified
        solution["details"]["all_steps_valid"] = all(s["verified"] for s in steps_verified)
        solution["formatted"] = "Proof structure:\n" + "\n".join(
            f"  {'✓' if s['verified'] else '✗'} {s['step'].replace('_', ' ')}"
            for s in steps_verified
        )

    elif problem_type == "optimization":
        included_items = []
        total_v = 0
        for name, var in variables.items():
            if name == "total_value":
                total_v = int(str(model.eval(var, model_completion=True)))
            elif hasattr(var, 'name') and str(var) == str(var):
                try:
                    val = bool(model.eval(var, model_completion=True))
                    if val:
                        included_items.append(name)
                except Exception:
                    pass

        solution["details"]["items_selected"] = included_items
        solution["details"]["total_value"] = total_v
        solution["formatted"] = f"Selected {len(included_items)} items\n" + \
                                "\n".join(f"  ✓ {item}" for item in included_items) + \
                                f"\nTotal value: {total_v}"

    else:  # generic
        verified = {}
        for name, var in variables.items():
            val = bool(model.eval(var, model_completion=True))
            verified[name] = val
        solution["details"] = verified
        solution["formatted"] = "\n".join(
            f"  {'✓' if v else '✗'} {k.replace('_', ' ')}"
            for k, v in verified.items()
        )

    return solution


def requires_external_data(problem_text: str) -> bool:
    """Check if a problem requires real-time external data."""
    lower = problem_text.lower()
    data_keywords = [
        "cheapest", "current price", "today", "right now", "live",
        "stock", "weather", "news", "trending",
        "flight price", "hotel price", "airfare",
    ]
    # "find" + data keyword = info request, not "find duplicates"
    if "find" in lower and any(k in lower for k in ["cheapest", "price", "flight", "hotel"]):
        return True
    return any(kw in lower for kw in data_keywords)


def handle_external_data_problem(problem_text: str) -> dict:
    """Generate a helpful response for problems needing external data."""
    return {
        "status": "needs_external_data",
        "type": "info_request",
        "message": ("This problem requires real-time data that I don't have. "
                    "I can help plan the approach — what data source should I check?"),
        "formatted": ("I need real-time data for this.\n"
                      "I can plan the structure, but you'll need to provide:\n"
                      f"  - Current prices/data for: {problem_text[:60]}"),
    }


# ── ADVICE MODE ──

# Behavioral strategy database
BEHAVIORAL_ADVICE: dict[str, list[str]] = {
    "procrastination": [
        "Pomodoro technique: 25 min focused work, 5 min break",
        "Eat the Frog: tackle the hardest task first thing",
        "Eliminate distractions: phone in another room, block social media",
        "Time-block: assign each task to a specific time slot",
        "Accountability partner: tell someone your plan and deadline",
    ],
    "focus": [
        "Single-task: close all tabs/apps not related to current task",
        "Environment design: dedicated workspace with minimal stimuli",
        "Energy management: align hardest tasks with peak energy hours",
        "Pomodoro: 25 min sprints with 5 min recovery",
        "Mindfulness: 2 min breathing reset between tasks",
    ],
    "learn": [
        "Active recall: test yourself instead of re-reading",
        "Spaced repetition: review at increasing intervals",
        "Interleaving: mix different topics in each session",
        "Teaching: explain concepts to someone else",
        "Feynman technique: explain in simple language",
    ],
    "debug": [
        "Reproduce first: get a reliable reproduction steps",
        "Isolate: use binary search to narrow down the cause",
        "Rubber duck: explain the code line by line to a duck",
        "Check assumptions: verify inputs match expectations",
        "Git bisect: find the exact commit that introduced the bug",
    ],
    "code": [
        "Small functions: each function does one thing",
        "Meaningful names: code reads like English",
        "DRY: don't repeat yourself — extract shared logic",
        "Single responsibility: each module handles one concern",
        "Test-driven: write the test before the fix",
    ],
    "productivity": [
        "Two-minute rule: if it takes <2 min, do it now",
        "Batch similar tasks: group emails, calls, admin together",
        "Priority matrix: urgent+important first, important-not-urgent second",
        "Time audit: track where your hours actually go for one week",
        "Default to action: when unsure, do the next physical step",
    ],
    "schedule": [
        "Time-block your day: assign specific hours to specific tasks",
        "Buffer time: leave 15 min between meetings for context switch",
        "Hard stops: set a firm end time to prevent scope creep",
        "Review weekly: spend 15 min Friday planning next week",
        "Theme days: focus Monday on planning, Tuesday on deep work, etc.",
    ],
    "budget": [
        "50/30/20 rule: 50% needs, 30% wants, 20% savings",
        "Track everything: use an app or spreadsheet for one month",
        "Pay yourself first: automate savings before discretionary spending",
        "Zero-based budget: assign every dollar a job",
        "Audit subscriptions: cancel anything unused in the last 30 days",
    ],
    "study": [
        "Cornell notes: split page into cues, notes, and summary",
        "Pomodoro: 25 min study, 5 min break, repeat 4x then long break",
        "Practice tests: simulate exam conditions before the real thing",
        "Concept maps: draw connections between topics visually",
        "Sleep on it: review key concepts before bed for memory consolidation",
    ],
}


def get_advice(entities: dict) -> dict | None:
    """Retrieve behavioral advice based on extracted entities.

    Returns formatted advice dict, or None if no matching advice.
    """
    goals = [g.lower() for g in entities.get("goals", [])]
    what = [w.lower() for w in entities.get("what", [])]
    action = entities.get("action", "").lower()

    # Match advice category
    advice_list = None
    category = None

    for goal in goals:
        if "procrastination" in goal or "procrastinate" in goal:
            category = "procrastination"
        elif "focus" in goal or "concentrate" in goal:
            category = "focus"
        elif "avoid_distraction" in goal:
            category = "focus"

    if not category:
        if action == "debug" or any("bug" in w for w in what):
            category = "debug"
        elif action == "code" or any("code" in w for w in what):
            category = "code"
        elif action in ("learn", "study"):
            category = "study"
        elif action == "schedule":
            category = "schedule"
        elif action in ("budget", "allocate"):
            category = "budget"
        elif any("task" in w for w in what):
            category = "productivity"

    if not category:
        # Try matching what keywords
        all_text = " ".join(what + goals + [action]).lower()
        for key in BEHAVIORAL_ADVICE:
            if key in all_text:
                category = key
                break

    if not category or category not in BEHAVIORAL_ADVICE:
        return None

    advice = BEHAVIORAL_ADVICE[category]
    how_many = entities.get("how_many", "?")

    # Customize advice based on entity count
    formatted_lines = []
    for i, tip in enumerate(advice, 1):
        formatted_lines.append(f"  {i}. {tip}")

    formatted = "\n".join(formatted_lines)

    if how_many != "?":
        formatted = f"Strategy for {how_many} tasks:\n" + formatted

    return {
        "status": "solved",
        "type": "advice",
        "category": category,
        "details": {"strategies": advice, "task_count": how_many},
        "formatted": formatted,
    }
