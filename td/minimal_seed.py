#!/usr/bin/env python3
"""Minimal seed data — 50 patterns max, as the strategic brief requires.

This replaces training_data.py (225 patterns) with a minimal bootstrap
of 50 innate reflexes. After 1000 interactions, seed should be <5% of total.

Rules:
- 10 sub-problem prototypes (decomposition axes)
- 10 constraint templates (Z3 scaffolding)
- 10 behavioral strategies (advice mode)
- 20 domain concepts (vocabulary)
- Total: 50 patterns max
- After 1000 interactions: seed <5% of total
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class SeedPattern:
    """A single seed pattern with HDC-encoded text and metadata."""
    text: str
    solution: str
    domain: str
    task_type: str
    metadata: dict


# ────────────────────────────────────────────
# SUB-PROBLEM PROTOTYPES (10) — decomposition axes
# ────────────────────────────────────────────
SUB_PROBLEM_PROTOTYPES = [
    SeedPattern(
        text="find all people and objects mentioned in the problem",
        solution="extract_entities",
        domain="meta",
        task_type="decomposition",
        metadata={"concept": "extract_entities", "role": "prototype"}
    ),
    SeedPattern(
        text="what limits exist what rules must be followed what cannot happen",
        solution="identify_constraints",
        domain="meta",
        task_type="decomposition",
        metadata={"concept": "identify_constraints", "role": "prototype"}
    ),
    SeedPattern(
        text="how to solve and satisfy all requirements",
        solution="find_solution",
        domain="meta",
        task_type="decomposition",
        metadata={"concept": "find_solution", "role": "prototype"}
    ),
    SeedPattern(
        text="check that the answer is correct verify no mistakes ensure all conditions met",
        solution="validate_result",
        domain="meta",
        task_type="decomposition",
        metadata={"concept": "validate_result", "role": "prototype"}
    ),
    # Additional prototype variants for robustness
    SeedPattern(
        text="who is involved and what are they doing",
        solution="extract_entities",
        domain="meta",
        task_type="decomposition",
        metadata={"concept": "extract_entities", "role": "prototype_variant"}
    ),
    SeedPattern(
        text="what are the deadlines and resource limits",
        solution="identify_constraints",
        domain="meta",
        task_type="decomposition",
        metadata={"concept": "identify_constraints", "role": "prototype_variant"}
    ),
    SeedPattern(
        text="find the optimal assignment that satisfies all constraints",
        solution="find_solution",
        domain="meta",
        task_type="decomposition",
        metadata={"concept": "find_solution", "role": "prototype_variant"}
    ),
    SeedPattern(
        text="verify that the solution meets all requirements and has no conflicts",
        solution="validate_result",
        domain="meta",
        task_type="decomposition",
        metadata={"concept": "validate_result", "role": "prototype_variant"}
    ),
    SeedPattern(
        text="break down the problem into smaller parts",
        solution="decompose",
        domain="meta",
        task_type="decomposition",
        metadata={"concept": "decompose", "role": "prototype"}
    ),
    SeedPattern(
        text="combine the sub solutions into a final answer",
        solution="compose",
        domain="meta",
        task_type="decomposition",
        metadata={"concept": "compose", "role": "prototype"}
    ),
]


# ────────────────────────────────────────────
# CONSTRAINT TEMPLATES (10) — Z3 scaffolding
# ────────────────────────────────────────────
CONSTRAINT_TEMPLATES = [
    SeedPattern(
        text="no two items can be assigned to the same time slot",
        solution="no_overlap",
        domain="constraint",
        task_type="template",
        metadata={"constraint_type": "no_overlap", "variables": "time_slot"}
    ),
    SeedPattern(
        text="item A must happen before item B",
        solution="before",
        domain="constraint",
        task_type="template",
        metadata={"constraint_type": "before", "variables": "ordered_pair"}
    ),
    SeedPattern(
        text="at least N items must be selected",
        solution="at_least",
        domain="constraint",
        task_type="template",
        metadata={"constraint_type": "at_least", "variables": "count"}
    ),
    SeedPattern(
        text="at most N items can be selected",
        solution="at_most",
        domain="constraint",
        task_type="template",
        metadata={"constraint_type": "at_most", "variables": "count"}
    ),
    SeedPattern(
        text="all items must be assigned to exactly one slot",
        solution="exactly_one",
        domain="constraint",
        task_type="template",
        metadata={"constraint_type": "exactly_one", "variables": "assignment"}
    ),
    SeedPattern(
        text="the total cost must not exceed the budget",
        solution="budget_limit",
        domain="constraint",
        task_type="template",
        metadata={"constraint_type": "budget_limit", "variables": "sum"}
    ),
    SeedPattern(
        text="allocate resources fairly across all departments",
        solution="fair_allocation",
        domain="constraint",
        task_type="template",
        metadata={"constraint_type": "fair_allocation", "variables": "distribution"}
    ),
    SeedPattern(
        text="no person can be in two places at the same time",
        solution="no_conflict",
        domain="constraint",
        task_type="template",
        metadata={"constraint_type": "no_conflict", "variables": "person_time"}
    ),
    SeedPattern(
        text="each task must be completed within its deadline",
        solution="deadline",
        domain="constraint",
        task_type="template",
        metadata={"constraint_type": "deadline", "variables": "time"}
    ),
    SeedPattern(
        text="the solution should minimize total cost or maximize total value",
        solution="optimize",
        domain="constraint",
        task_type="template",
        metadata={"constraint_type": "optimize", "variables": "objective"}
    ),
]


# ────────────────────────────────────────────
# BEHAVIORAL STRATEGIES (10) — with query prototypes
# ────────────────────────────────────────────
BEHAVIORAL_STRATEGIES = [
    SeedPattern(
        text="how to focus and avoid procrastination when working on tasks",
        solution="pomodoro",
        domain="advice",
        task_type="strategy",
        metadata={"title": "Pomodoro Technique", "effectiveness": 0.85,
                  "tags": ["focus", "time_management", "productivity"],
                  "description": "Work for 25 minutes, then take a 5-minute break. Repeat 4 times, then take a longer break."}
    ),
    SeedPattern(
        text="how to stop procrastinating and start doing the hardest task first",
        solution="eat_the_frog",
        domain="advice",
        task_type="strategy",
        metadata={"title": "Eat the Frog", "effectiveness": 0.80,
                  "tags": ["procrastination", "priority", "morning"],
                  "description": "Tackle your hardest, most important task first thing in the morning. Everything after feels easier."}
    ),
    SeedPattern(
        text="how to eliminate distractions and create a focused work environment",
        solution="eliminate_distractions",
        domain="advice",
        task_type="strategy",
        metadata={"title": "Eliminate Distractions", "effectiveness": 0.75,
                  "tags": ["focus", "environment", "productivity"],
                  "description": "Put your phone in another room, close all unnecessary tabs, and use website blockers."}
    ),
    SeedPattern(
        text="how to schedule time blocks for each task and protect them",
        solution="time_blocking",
        domain="advice",
        task_type="strategy",
        metadata={"title": "Time Blocking", "effectiveness": 0.78,
                  "tags": ["schedule", "planning", "time_management"],
                  "description": "Assign specific time slots to each task and protect them like appointments."}
    ),
    SeedPattern(
        text="how to handle quick tasks that take less than two minutes",
        solution="two_minute_rule",
        domain="advice",
        task_type="strategy",
        metadata={"title": "Two Minute Rule", "effectiveness": 0.70,
                  "tags": ["productivity", "quick_wins", "habits"],
                  "description": "If a task takes less than two minutes, do it immediately rather than scheduling it."}
    ),
    SeedPattern(
        text="how to debug code by explaining it out loud to find bugs",
        solution="rubber_duck",
        domain="advice",
        task_type="strategy",
        metadata={"title": "Rubber Duck Debugging", "effectiveness": 0.85,
                  "tags": ["debug", "learning", "problem_solving"],
                  "description": "Explain your code line by line to an inanimate object. The act of verbalizing often reveals the bug."}
    ),
    SeedPattern(
        text="how to review progress weekly and plan for the next week",
        solution="weekly_review",
        domain="advice",
        task_type="strategy",
        metadata={"title": "Weekly Review", "effectiveness": 0.80,
                  "tags": ["planning", "focus", "goals"],
                  "description": "Every Friday, review what you accomplished, what blocked you, and what matters next week."}
    ),
    SeedPattern(
        text="how to clean and normalize data before analysis or queries",
        solution="normalize_data",
        domain="advice",
        task_type="strategy",
        metadata={"title": "Normalize Before Query", "effectiveness": 0.75,
                  "tags": ["data", "validation", "cleaning"],
                  "description": "Always clean and normalize your data before any query or analysis. Remove duplicates, fix missing values."}
    ),
    SeedPattern(
        text="how to solve hard problems by taking a break and sleeping on it",
        solution="sleep_on_it",
        domain="advice",
        task_type="strategy",
        metadata={"title": "Sleep on It", "effectiveness": 0.72,
                  "tags": ["problem_solving", "creativity", "rest"],
                  "description": "If stuck on a hard problem, take a break and come back with fresh eyes."}
    ),
    SeedPattern(
        text="how to break down complex problems into fundamental truths",
        solution="first_principles",
        domain="advice",
        task_type="strategy",
        metadata={"title": "First Principles", "effectiveness": 0.82,
                  "tags": ["problem_solving", "thinking", "analysis"],
                  "description": "Break down complex problems into fundamental truths and rebuild from scratch."}
    ),
]


# ────────────────────────────────────────────
# DOMAIN CONCEPTS (20) — vocabulary
# ────────────────────────────────────────────
DOMAIN_CONCEPTS = [
    SeedPattern(text="schedule a meeting with multiple people", solution="schedule", domain="domain", task_type="concept", metadata={"concept": "schedule"}),
    SeedPattern(text="allocate budget across departments", solution="budget", domain="domain", task_type="concept", metadata={"concept": "budget"}),
    SeedPattern(text="prove a mathematical theorem by logical deduction", solution="prove", domain="domain", task_type="concept", metadata={"concept": "prove"}),
    SeedPattern(text="debug a program that has unexpected behavior", solution="debug", domain="domain", task_type="concept", metadata={"concept": "debug"}),
    SeedPattern(text="convert data from one format to another", solution="convert", domain="domain", task_type="concept", metadata={"concept": "convert"}),
    SeedPattern(text="optimize a function to minimize cost or maximize value", solution="optimize", domain="domain", task_type="concept", metadata={"concept": "optimize"}),
    SeedPattern(text="plan a project with tasks and deadlines", solution="plan", domain="domain", task_type="concept", metadata={"concept": "plan"}),
    SeedPattern(text="find duplicate records in a database", solution="find", domain="domain", task_type="concept", metadata={"concept": "find"}),
    SeedPattern(text="validate that input data meets required criteria", solution="validate", domain="domain", task_type="concept", metadata={"concept": "validate"}),
    SeedPattern(text="allocate resources across multiple teams", solution="allocate", domain="domain", task_type="concept", metadata={"concept": "allocate"}),
    SeedPattern(text="avoid procrastination and stay focused on tasks", solution="avoid", domain="domain", task_type="concept", metadata={"concept": "avoid"}),
    SeedPattern(text="eliminate distractions and create a focused work environment", solution="eliminate", domain="domain", task_type="concept", metadata={"concept": "eliminate"}),
    SeedPattern(text="organize tasks by priority and deadline", solution="organize", domain="domain", task_type="concept", metadata={"concept": "organize"}),
    SeedPattern(text="tackle the hardest problem first before easier ones", solution="tackle", domain="domain", task_type="concept", metadata={"concept": "tackle"}),
    SeedPattern(text="break a large problem into smaller manageable subproblems", solution="break", domain="domain", task_type="concept", metadata={"concept": "break"}),
    SeedPattern(text="verify that all constraints are satisfied by the solution", solution="verify", domain="domain", task_type="concept", metadata={"concept": "verify"}),
    SeedPattern(text="check for conflicts between assignments or schedules", solution="check", domain="domain", task_type="concept", metadata={"concept": "check"}),
    SeedPattern(text="distribute work evenly among team members", solution="distribute", domain="domain", task_type="concept", metadata={"concept": "distribute"}),
    SeedPattern(text="minimize the total cost of operations", solution="minimize", domain="domain", task_type="concept", metadata={"concept": "minimize"}),
    SeedPattern(text="maximize the value or efficiency of the solution", solution="maximize", domain="domain", task_type="concept", metadata={"concept": "maximize"}),
]


# ────────────────────────────────────────────
# ALL SEED PATTERNS (50 total)
# ────────────────────────────────────────────
ALL_SEED_PATTERNS = (
    SUB_PROBLEM_PROTOTYPES +      # 10
    CONSTRAINT_TEMPLATES +        # 10
    BEHAVIORAL_STRATEGIES +       # 10
    DOMAIN_CONCEPTS               # 20
)

assert len(ALL_SEED_PATTERNS) == 50, f"Expected 50 seed patterns, got {len(ALL_SEED_PATTERNS)}"

# Export for compatibility with existing code
EXAMPLES = [(p.text, p.solution, p.domain, p.task_type) for p in ALL_SEED_PATTERNS]
