"""Allen's Interval Algebra — Temporal Reasoning for TD v2.

Based on:
    Allen, J.F. (1983). "Maintaining Knowledge about Temporal Intervals."
    Communications of the ACM, 26(11), 832-843.

    Allen, J.F. (1984). "Towards a General Theory of Action and Time."
    Artificial Intelligence, 23(2), 123-154.

    TREK: Temporal Reasoning-Enhanced Knowledge Graphs (2025).
    8B model + temporal KG ≈ 671B model performance.

Allen's 13 basic relations between time intervals:
    Given intervals X = [x_s, x_e] and Y = [y_s, y_e] where x_s < x_e and y_s < y_e:

    Relation    | Definition                          | Inverse
    ------------|-------------------------------------|------------------
    before      | x_e < y_s                           | after
    after       | y_e < x_s                           | before
    meets       | x_e = y_s                           | met_by
    met_by      | y_e = x_s                           | meets
    overlaps    | x_s < y_s < x_e < y_e               | overlapped_by
    overlapped_by | y_s < x_s < y_e < x_e             | overlaps
    starts      | x_s = y_s ∧ x_e < y_e               | started_by
    started_by  | y_s = x_s ∧ y_e < x_e               | starts
    during      | y_s < x_s ∧ x_e < y_e               | contains
    contains    | x_s < y_s ∧ y_e < x_e               | during
    finishes    | x_s > y_s ∧ x_e = y_e               | finished_by
    finished_by | y_s > x_s ∧ y_e = x_e               | finishes
    equals      | x_s = y_s ∧ x_e = y_e               | equals

These 7 pairs of inverses + 1 self-inverse (equals) = 13 relations.
Every pair of intervals satisfies EXACTLY ONE of these relations.

Properties:
    - Transitive: before, after, during, contains, starts, started_by, finishes, finished_by, equals
    - Symmetric: equals
    - Asymmetric: all others
    - Reflexive: equals only

Composition Table (Allen 1983, Table 1):
    Given R1(X,Y) and R2(Y,Z), what can we say about R(X,Z)?
    Example: before ∘ before = before (if X before Y and Y before Z, then X before Z)
    Example: before ∘ during = before | meets | overlaps | starts | during
    (if X before Y and Y during Z, then X could be before, meet, overlap, start, or be during Z)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from enum import Enum


# ─── Allen's 13 Relations ────────────────────────────────────────────

class AllenRelation(Enum):
    """The 13 basic Allen interval relations."""
    BEFORE = "before"
    AFTER = "after"
    MEETS = "meets"
    MET_BY = "met_by"
    OVERLAPS = "overlaps"
    OVERLAPPED_BY = "overlapped_by"
    STARTS = "starts"
    STARTED_BY = "started_by"
    DURING = "during"
    CONTAINS = "contains"
    FINISHES = "finishes"
    FINISHED_BY = "finished_by"
    EQUALS = "equals"

    @property
    def inverse(self) -> AllenRelation:
        """Return the inverse relation."""
        return _INVERSE_MAP[self]

    @property
    def is_transitive(self) -> bool:
        return self in _TRANSITIVE_RELATIONS

    @property
    def is_symmetric(self) -> bool:
        return self == AllenRelation.EQUALS

    @property
    def is_asymmetric(self) -> bool:
        return not self.is_symmetric


# Inverse pairs (Allen 1983, Section 3)
_INVERSE_MAP = {
    AllenRelation.BEFORE: AllenRelation.AFTER,
    AllenRelation.AFTER: AllenRelation.BEFORE,
    AllenRelation.MEETS: AllenRelation.MET_BY,
    AllenRelation.MET_BY: AllenRelation.MEETS,
    AllenRelation.OVERLAPS: AllenRelation.OVERLAPPED_BY,
    AllenRelation.OVERLAPPED_BY: AllenRelation.OVERLAPS,
    AllenRelation.STARTS: AllenRelation.STARTED_BY,
    AllenRelation.STARTED_BY: AllenRelation.STARTS,
    AllenRelation.DURING: AllenRelation.CONTAINS,
    AllenRelation.CONTAINS: AllenRelation.DURING,
    AllenRelation.FINISHES: AllenRelation.FINISHED_BY,
    AllenRelation.FINISHED_BY: AllenRelation.FINISHES,
    AllenRelation.EQUALS: AllenRelation.EQUALS,
}

# Transitive relations (Allen 1983, Section 4)
_TRANSITIVE_RELATIONS = {
    AllenRelation.BEFORE,
    AllenRelation.AFTER,
    AllenRelation.DURING,
    AllenRelation.CONTAINS,
    AllenRelation.STARTS,
    AllenRelation.STARTED_BY,
    AllenRelation.FINISHES,
    AllenRelation.FINISHED_BY,
    AllenRelation.EQUALS,
}


# ─── Temporal Interval ───────────────────────────────────────────────

@dataclass
class TemporalInterval:
    """A temporal interval with start and end points.

    Allen's intervals are [start, end) where start < end.
    For discrete time (years, timestamps), both endpoints are integers.
    For continuous time, they could be floats.

    Convention:
        - None means "unbounded" (still happening, or unknown end)
        - start=None means "from the beginning of time"
        - end=None means "until further notice"
    """
    start: Optional[int] = None
    end: Optional[int] = None

    def __post_init__(self):
        if self.start is not None and self.end is not None:
            if self.start >= self.end:
                raise ValueError(f"Invalid interval: start ({self.start}) >= end ({self.end}). Allen intervals must have positive duration.")

    def is_valid(self) -> bool:
        """Check if this interval is valid (has both endpoints)."""
        return self.start is not None and self.end is not None

    def __repr__(self):
        s = self.start if self.start is not None else "-∞"
        e = self.end if self.end is not None else "∞"
        return f"[{s}, {e})"


# ─── Allen Relation Checker ─────────────────────────────────────────

def check_allen_relation(x: TemporalInterval, y: TemporalInterval) -> Optional[AllenRelation]:
    """Determine the Allen relation between two intervals.

    From Allen (1983), Table 1:
    Given X = [x_s, x_e] and Y = [y_s, y_e],

    Returns the single Allen relation that holds, or None if intervals
    are indeterminate.

    Handles open-ended intervals (None = unbounded):
        - None start: "from the beginning of time" → treated as -\u221e
        - None end: "until further notice" → treated as +\u221e
    This allows comparisons like [1976, \u221e) vs [2007, \u221e):
        since 1976 < 2007 and both end at \u221e, X FINISHES Y.

    This is a PURE FUNCTION \u2014 no side effects, deterministic.
    """
    INF = float('inf')
    NEG_INF = float('-inf')

    x_s = x.start if x.start is not None else NEG_INF
    x_e = x.end if x.end is not None else INF
    y_s = y.start if y.start is not None else NEG_INF
    y_e = y.end if y.end is not None else INF

    # EQUALS: both endpoints exactly match
    if x.start == y.start and x.end == y.end:
        return AllenRelation.EQUALS

    # Both intervals end at \u221e (open-ended): use start ordering
    # [x_s, \u221e) vs [y_s, \u221e): since x_e = y_e = \u221e
    # If x_s < y_s \u2192 X FINISHES Y (X starts earlier, both end at \u221e)
    # If x_s > y_s \u2192 X FINISHED_BY Y (X starts later, both end at \u221e)
    both_ends_open = (x.end is None and y.end is None)
    if both_ends_open:
        if x.start is not None and y.start is not None:
            if x_s < y_s:
                return AllenRelation.FINISHES
            if x_s > y_s:
                return AllenRelation.FINISHED_BY
            if x.start == y.start:  # Both None \u2192 both at -\u221e
                return AllenRelation.STARTS
        return None  # Both completely open: indeterminate

    # One or both ends bounded: standard Allen comparisons
    # BEFORE: x_e < y_s
    if x_e < y_s:
        return AllenRelation.BEFORE
    # AFTER: y_e < x_s
    if y_e < x_s:
        return AllenRelation.AFTER
    # MEETS: x_e == y_s
    if x_e == y_s:
        return AllenRelation.MEETS
    # MET_BY: y_e == x_s
    if y_e == x_s:
        return AllenRelation.MET_BY

    # DURING: y_s < x_s and x_e < y_e
    if (x.start is not None and y.start is not None and x.end is not None and y.end is not None):
        if y_s < x_s and x_e < y_e:
            return AllenRelation.DURING
    # CONTAINS: x_s < y_s and y_e < x_e
    if (x.start is not None and y.start is not None and x.end is not None and y.end is not None):
        if x_s < y_s and y_e < x_e:
            return AllenRelation.CONTAINS

    # OVERLAPS: x_s < y_s < x_e < y_e
    if (x.start is not None and y.start is not None and x.end is not None and y.end is not None):
        if x_s < y_s < x_e < y_e:
            return AllenRelation.OVERLAPS
    # OVERLAPPED_BY: y_s < x_s < y_e < x_e
    if (y.start is not None and x.start is not None and y.end is not None and x.end is not None):
        if y_s < x_s < y_e < x_e:
            return AllenRelation.OVERLAPPED_BY

    # STARTS: x_s == y_s and x_e < y_e
    if x.start is not None and y.start is not None:
        if x_s == y_s and x_e < y_e:
            return AllenRelation.STARTS
        if y_s == x_s and y_e < x_e:
            return AllenRelation.STARTED_BY

    # FINISHES / FINISHED_BY
    if (x.start is not None and y.start is not None and
            x.end is not None and y.end is not None):
        if x_s > y_s and x_e == y_e:
            return AllenRelation.FINISHES
        if y_s > x_s and y_e == x_e:
            return AllenRelation.FINISHED_BY

    return None



# ─── Allen's Composition Table ───────────────────────────────────────
#
# Allen (1983), Table 1 — "Transitions Among Relations"
# Given R1(X,Y) and R2(Y,Z), what are the possible R(X,Z)?
# Each entry is a FROZEN SET of possible Allen relations.
#
# This is the COMPLETE 13×13 composition table from the original paper.
# Some entries have multiple possible results (disjunctions).

_COMPOSITION_TABLE: dict[tuple[AllenRelation, AllenRelation], frozenset[AllenRelation]] = {}

def _init_composition_table():
    """Initialize Allen's composition table from the paper.

    Allen (1983), Table 1. Each (R1, R2) pair maps to a set of possible R.
    """
    R = AllenRelation
    table = {
        # before ∘ X
        (R.BEFORE, R.BEFORE):          frozenset({R.BEFORE}),
        (R.BEFORE, R.MEETS):           frozenset({R.BEFORE}),
        (R.BEFORE, R.OVERLAPS):        frozenset({R.BEFORE}),
        (R.BEFORE, R.STARTS):          frozenset({R.BEFORE}),
        (R.BEFORE, R.DURING):          frozenset({R.BEFORE}),
        (R.BEFORE, R.FINISHES):        frozenset({R.BEFORE}),
        (R.BEFORE, R.EQUALS):          frozenset({R.BEFORE}),
        (R.BEFORE, R.STARTED_BY):      frozenset({R.BEFORE}),
        (R.BEFORE, R.FINISHED_BY):     frozenset({R.BEFORE}),
        (R.BEFORE, R.CONTAINS):        frozenset({R.BEFORE, R.MEETS, R.OVERLAPS, R.STARTS, R.DURING, R.FINISHES, R.EQUALS, R.STARTED_BY, R.FINISHED_BY, R.CONTAINS}),
        (R.BEFORE, R.OVERLAPPED_BY):   frozenset({R.BEFORE}),
        (R.BEFORE, R.MET_BY):          frozenset({R.BEFORE}),
        (R.BEFORE, R.AFTER):           frozenset({R.BEFORE, R.MEETS, R.OVERLAPS, R.STARTS, R.DURING, R.FINISHES, R.EQUALS, R.STARTED_BY, R.FINISHED_BY, R.CONTAINS, R.OVERLAPPED_BY, R.MET_BY, R.AFTER}),

        # meets ∘ X
        (R.MEETS, R.BEFORE):           frozenset({R.BEFORE}),
        (R.MEETS, R.MEETS):            frozenset({R.BEFORE}),
        (R.MEETS, R.OVERLAPS):         frozenset({R.BEFORE}),
        (R.MEETS, R.STARTS):           frozenset({R.BEFORE, R.MEETS}),
        (R.MEETS, R.DURING):           frozenset({R.BEFORE}),
        (R.MEETS, R.FINISHES):         frozenset({R.BEFORE}),
        (R.MEETS, R.EQUALS):           frozenset({R.MEETS}),
        (R.MEETS, R.STARTED_BY):       frozenset({R.BEFORE, R.MEETS, R.OVERLAPS, R.STARTS, R.DURING, R.FINISHES}),
        (R.MEETS, R.FINISHED_BY):      frozenset({R.BEFORE, R.MEETS}),
        (R.MEETS, R.CONTAINS):         frozenset({R.BEFORE, R.MEETS, R.OVERLAPS, R.STARTS, R.DURING, R.FINISHES, R.EQUALS, R.STARTED_BY, R.FINISHED_BY, R.CONTAINS}),
        (R.MEETS, R.OVERLAPPED_BY):    frozenset({R.BEFORE, R.MEETS, R.OVERLAPS}),
        (R.MEETS, R.MET_BY):           frozenset({R.BEFORE, R.MEETS, R.OVERLAPS, R.STARTS, R.DURING}),
        (R.MEETS, R.AFTER):            frozenset({R.BEFORE, R.MEETS, R.OVERLAPS, R.STARTS, R.DURING, R.FINISHES, R.EQUALS, R.STARTED_BY, R.FINISHED_BY, R.CONTAINS, R.OVERLAPPED_BY, R.MET_BY, R.AFTER}),

        # overlaps ∘ X
        (R.OVERLAPS, R.BEFORE):        frozenset({R.BEFORE}),
        (R.OVERLAPS, R.MEETS):         frozenset({R.BEFORE}),
        (R.OVERLAPS, R.OVERLAPS):      frozenset({R.BEFORE, R.MEETS, R.OVERLAPS}),
        (R.OVERLAPS, R.STARTS):        frozenset({R.BEFORE, R.MEETS, R.OVERLAPS}),
        (R.OVERLAPS, R.DURING):        frozenset({R.BEFORE, R.MEETS, R.OVERLAPS}),
        (R.OVERLAPS, R.FINISHES):      frozenset({R.BEFORE, R.MEETS, R.OVERLAPS}),
        (R.OVERLAPS, R.EQUALS):        frozenset({R.OVERLAPS}),
        (R.OVERLAPS, R.STARTED_BY):    frozenset({R.BEFORE, R.MEETS, R.OVERLAPS, R.STARTS, R.DURING, R.FINISHES}),
        (R.OVERLAPS, R.FINISHED_BY):   frozenset({R.BEFORE, R.MEETS, R.OVERLAPS}),
        (R.OVERLAPS, R.CONTAINS):      frozenset({R.BEFORE, R.MEETS, R.OVERLAPS, R.STARTS, R.DURING, R.FINISHES, R.EQUALS, R.STARTED_BY, R.FINISHED_BY, R.CONTAINS}),
        (R.OVERLAPS, R.OVERLAPPED_BY): frozenset({R.BEFORE, R.MEETS, R.OVERLAPS, R.STARTS, R.DURING, R.FINISHES, R.EQUALS, R.STARTED_BY, R.FINISHED_BY, R.CONTAINS, R.OVERLAPPED_BY}),
        (R.OVERLAPS, R.MET_BY):        frozenset({R.BEFORE, R.MEETS, R.OVERLAPS, R.STARTS, R.DURING}),
        (R.OVERLAPS, R.AFTER):         frozenset({R.BEFORE, R.MEETS, R.OVERLAPS, R.STARTS, R.DURING, R.FINISHES, R.EQUALS, R.STARTED_BY, R.FINISHED_BY, R.CONTAINS, R.OVERLAPPED_BY, R.MET_BY, R.AFTER}),

        # starts ∘ X
        (R.STARTS, R.BEFORE):          frozenset({R.BEFORE}),
        (R.STARTS, R.MEETS):           frozenset({R.BEFORE}),
        (R.STARTS, R.OVERLAPS):        frozenset({R.BEFORE}),
        (R.STARTS, R.STARTS):          frozenset({R.STARTS}),
        (R.STARTS, R.DURING):          frozenset({R.DURING}),
        (R.STARTS, R.FINISHES):        frozenset({R.DURING}),
        (R.STARTS, R.EQUALS):          frozenset({R.STARTS}),
        (R.STARTS, R.STARTED_BY):      frozenset({R.STARTS, R.DURING, R.FINISHES, R.EQUALS, R.STARTED_BY, R.FINISHED_BY}),
        (R.STARTS, R.FINISHED_BY):     frozenset({R.DURING, R.FINISHES}),
        (R.STARTS, R.CONTAINS):        frozenset({R.DURING, R.FINISHES, R.EQUALS, R.STARTED_BY, R.FINISHED_BY, R.CONTAINS}),
        (R.STARTS, R.OVERLAPPED_BY):   frozenset({R.DURING, R.FINISHES, R.OVERLAPPED_BY}),
        (R.STARTS, R.MET_BY):          frozenset({R.DURING, R.FINISHES}),
        (R.STARTS, R.AFTER):           frozenset({R.BEFORE, R.MEETS, R.OVERLAPS, R.STARTS, R.DURING, R.FINISHES, R.EQUALS, R.STARTED_BY, R.FINISHED_BY, R.CONTAINS, R.OVERLAPPED_BY, R.MET_BY, R.AFTER}),

        # during ∘ X
        (R.DURING, R.BEFORE):          frozenset({R.BEFORE}),
        (R.DURING, R.MEETS):           frozenset({R.BEFORE}),
        (R.DURING, R.OVERLAPS):        frozenset({R.BEFORE, R.MEETS, R.OVERLAPS}),
        (R.DURING, R.STARTS):          frozenset({R.DURING}),
        (R.DURING, R.DURING):          frozenset({R.DURING}),
        (R.DURING, R.FINISHES):        frozenset({R.DURING}),
        (R.DURING, R.EQUALS):          frozenset({R.DURING}),
        (R.DURING, R.STARTED_BY):      frozenset({R.DURING, R.FINISHES}),
        (R.DURING, R.FINISHED_BY):     frozenset({R.DURING}),
        (R.DURING, R.CONTAINS):        frozenset({R.DURING, R.FINISHES, R.EQUALS, R.STARTED_BY, R.FINISHED_BY, R.CONTAINS}),
        (R.DURING, R.OVERLAPPED_BY):   frozenset({R.DURING, R.FINISHES, R.OVERLAPPED_BY}),
        (R.DURING, R.MET_BY):          frozenset({R.DURING}),
        (R.DURING, R.AFTER):           frozenset({R.BEFORE, R.MEETS, R.OVERLAPS, R.STARTS, R.DURING, R.FINISHES, R.EQUALS, R.STARTED_BY, R.FINISHED_BY, R.CONTAINS, R.OVERLAPPED_BY, R.MET_BY, R.AFTER}),

        # finishes ∘ X
        (R.FINISHES, R.BEFORE):        frozenset({R.BEFORE}),
        (R.FINISHES, R.MEETS):         frozenset({R.BEFORE}),
        (R.FINISHES, R.OVERLAPS):      frozenset({R.BEFORE}),
        (R.FINISHES, R.STARTS):        frozenset({R.DURING}),
        (R.FINISHES, R.DURING):        frozenset({R.DURING}),
        (R.FINISHES, R.FINISHES):      frozenset({R.FINISHES}),
        (R.FINISHES, R.EQUALS):        frozenset({R.FINISHES}),
        (R.FINISHES, R.STARTED_BY):    frozenset({R.DURING, R.FINISHES, R.EQUALS, R.STARTED_BY, R.FINISHED_BY}),
        (R.FINISHES, R.FINISHED_BY):   frozenset({R.FINISHES, R.EQUALS, R.FINISHED_BY}),
        (R.FINISHES, R.CONTAINS):      frozenset({R.DURING, R.FINISHES, R.EQUALS, R.STARTED_BY, R.FINISHED_BY, R.CONTAINS}),
        (R.FINISHES, R.OVERLAPPED_BY): frozenset({R.DURING, R.FINISHES, R.OVERLAPPED_BY}),
        (R.FINISHES, R.MET_BY):        frozenset({R.DURING, R.FINISHES}),
        (R.FINISHES, R.AFTER):         frozenset({R.BEFORE, R.MEETS, R.OVERLAPS, R.STARTS, R.DURING, R.FINISHES, R.EQUALS, R.STARTED_BY, R.FINISHED_BY, R.CONTAINS, R.OVERLAPPED_BY, R.MET_BY, R.AFTER}),

        # equals ∘ X (identity — R ∘ equals = R)
        (R.EQUALS, R.BEFORE):          frozenset({R.BEFORE}),
        (R.EQUALS, R.MEETS):           frozenset({R.MEETS}),
        (R.EQUALS, R.OVERLAPS):        frozenset({R.OVERLAPS}),
        (R.EQUALS, R.STARTS):          frozenset({R.STARTS}),
        (R.EQUALS, R.DURING):          frozenset({R.DURING}),
        (R.EQUALS, R.FINISHES):        frozenset({R.FINISHES}),
        (R.EQUALS, R.EQUALS):          frozenset({R.EQUALS}),
        (R.EQUALS, R.STARTED_BY):      frozenset({R.STARTED_BY}),
        (R.EQUALS, R.FINISHED_BY):     frozenset({R.FINISHED_BY}),
        (R.EQUALS, R.CONTAINS):        frozenset({R.CONTAINS}),
        (R.EQUALS, R.OVERLAPPED_BY):   frozenset({R.OVERLAPPED_BY}),
        (R.EQUALS, R.MET_BY):          frozenset({R.MET_BY}),
        (R.EQUALS, R.AFTER):           frozenset({R.AFTER}),

        # started_by ∘ X
        (R.STARTED_BY, R.BEFORE):      frozenset({R.BEFORE}),
        (R.STARTED_BY, R.MEETS):       frozenset({R.BEFORE, R.MEETS, R.OVERLAPS}),
        (R.STARTED_BY, R.OVERLAPS):    frozenset({R.BEFORE, R.MEETS, R.OVERLAPS}),
        (R.STARTED_BY, R.STARTS):      frozenset({R.STARTS, R.STARTED_BY}),
        (R.STARTED_BY, R.DURING):      frozenset({R.OVERLAPS, R.STARTS, R.DURING, R.STARTED_BY, R.FINISHES}),
        (R.STARTED_BY, R.FINISHES):    frozenset({R.OVERLAPS, R.STARTS, R.DURING, R.FINISHES, R.STARTED_BY, R.FINISHED_BY}),
        (R.STARTED_BY, R.EQUALS):      frozenset({R.STARTED_BY}),
        (R.STARTED_BY, R.STARTED_BY):  frozenset({R.STARTED_BY}),
        (R.STARTED_BY, R.FINISHED_BY): frozenset({R.OVERLAPS, R.STARTS, R.DURING, R.FINISHES, R.STARTED_BY, R.FINISHED_BY}),
        (R.STARTED_BY, R.CONTAINS):    frozenset({R.CONTAINS, R.STARTED_BY, R.FINISHED_BY, R.OVERLAPPED_BY}),
        (R.STARTED_BY, R.OVERLAPPED_BY): frozenset({R.OVERLAPS, R.OVERLAPPED_BY, R.MET_BY, R.AFTER}),
        (R.STARTED_BY, R.MET_BY):      frozenset({R.OVERLAPS, R.MET_BY, R.AFTER}),
        (R.STARTED_BY, R.AFTER):       frozenset({R.AFTER}),

        # finished_by ∘ X
        (R.FINISHED_BY, R.BEFORE):     frozenset({R.BEFORE}),
        (R.FINISHED_BY, R.MEETS):      frozenset({R.BEFORE}),
        (R.FINISHED_BY, R.OVERLAPS):   frozenset({R.BEFORE, R.MEETS, R.OVERLAPS}),
        (R.FINISHED_BY, R.STARTS):     frozenset({R.OVERLAPS, R.DURING, R.FINISHES}),
        (R.FINISHED_BY, R.DURING):     frozenset({R.OVERLAPS, R.DURING, R.FINISHES}),
        (R.FINISHED_BY, R.FINISHES):   frozenset({R.FINISHES, R.FINISHED_BY}),
        (R.FINISHED_BY, R.EQUALS):     frozenset({R.FINISHED_BY}),
        (R.FINISHED_BY, R.STARTED_BY): frozenset({R.OVERLAPS, R.DURING, R.FINISHES, R.STARTED_BY, R.FINISHED_BY, R.EQUALS, R.CONTAINS}),
        (R.FINISHED_BY, R.FINISHED_BY): frozenset({R.FINISHED_BY}),
        (R.FINISHED_BY, R.CONTAINS):   frozenset({R.CONTAINS, R.FINISHED_BY}),
        (R.FINISHED_BY, R.OVERLAPPED_BY): frozenset({R.OVERLAPS, R.OVERLAPPED_BY}),
        (R.FINISHED_BY, R.MET_BY):     frozenset({R.OVERLAPS, R.MET_BY}),
        (R.FINISHED_BY, R.AFTER):      frozenset({R.AFTER}),

        # contains ∘ X
        (R.CONTAINS, R.BEFORE):        frozenset({R.BEFORE}),
        (R.CONTAINS, R.MEETS):         frozenset({R.BEFORE, R.MEETS, R.OVERLAPS}),
        (R.CONTAINS, R.OVERLAPS):      frozenset({R.BEFORE, R.MEETS, R.OVERLAPS}),
        (R.CONTAINS, R.STARTS):        frozenset({R.STARTS, R.DURING, R.STARTED_BY}),
        (R.CONTAINS, R.DURING):        frozenset({R.DURING}),
        (R.CONTAINS, R.FINISHES):      frozenset({R.FINISHES, R.DURING, R.FINISHED_BY}),
        (R.CONTAINS, R.EQUALS):        frozenset({R.CONTAINS}),
        (R.CONTAINS, R.STARTED_BY):    frozenset({R.STARTED_BY, R.CONTAINS, R.OVERLAPPED_BY}),
        (R.CONTAINS, R.FINISHED_BY):   frozenset({R.FINISHED_BY, R.CONTAINS}),
        (R.CONTAINS, R.CONTAINS):      frozenset({R.CONTAINS}),
        (R.CONTAINS, R.OVERLAPPED_BY): frozenset({R.OVERLAPPED_BY, R.CONTAINS}),
        (R.CONTAINS, R.MET_BY):        frozenset({R.OVERLAPPED_BY, R.MET_BY, R.AFTER}),
        (R.CONTAINS, R.AFTER):         frozenset({R.AFTER}),

        # overlapped_by ∘ X
        (R.OVERLAPPED_BY, R.BEFORE):   frozenset({R.BEFORE}),
        (R.OVERLAPPED_BY, R.MEETS):    frozenset({R.BEFORE, R.MEETS}),
        (R.OVERLAPPED_BY, R.OVERLAPS): frozenset({R.BEFORE, R.MEETS, R.OVERLAPS, R.STARTS, R.DURING, R.FINISHES}),
        (R.OVERLAPPED_BY, R.STARTS):   frozenset({R.OVERLAPS, R.DURING, R.FINISHES}),
        (R.OVERLAPPED_BY, R.DURING):   frozenset({R.OVERLAPS, R.DURING, R.FINISHES}),
        (R.OVERLAPPED_BY, R.FINISHES): frozenset({R.OVERLAPS, R.FINISHES}),
        (R.OVERLAPPED_BY, R.EQUALS):   frozenset({R.OVERLAPPED_BY}),
        (R.OVERLAPPED_BY, R.STARTED_BY): frozenset({R.OVERLAPS, R.OVERLAPPED_BY, R.STARTS, R.DURING, R.FINISHES, R.STARTED_BY, R.FINISHED_BY, R.EQUALS, R.CONTAINS}),
        (R.OVERLAPPED_BY, R.FINISHED_BY): frozenset({R.OVERLAPS, R.OVERLAPPED_BY, R.FINISHES, R.FINISHED_BY}),
        (R.OVERLAPPED_BY, R.CONTAINS): frozenset({R.OVERLAPPED_BY, R.CONTAINS}),
        (R.OVERLAPPED_BY, R.OVERLAPPED_BY): frozenset({R.OVERLAPPED_BY, R.CONTAINS}),
        (R.OVERLAPPED_BY, R.MET_BY):   frozenset({R.OVERLAPPED_BY, R.MET_BY, R.AFTER}),
        (R.OVERLAPPED_BY, R.AFTER):    frozenset({R.AFTER}),

        # met_by ∘ X
        (R.MET_BY, R.BEFORE):          frozenset({R.BEFORE, R.MEETS, R.OVERLAPS, R.STARTS, R.DURING}),
        (R.MET_BY, R.MEETS):           frozenset({R.OVERLAPS, R.STARTS, R.DURING}),
        (R.MET_BY, R.OVERLAPS):        frozenset({R.OVERLAPS, R.STARTS, R.DURING}),
        (R.MET_BY, R.STARTS):          frozenset({R.OVERLAPS, R.DURING, R.FINISHES}),
        (R.MET_BY, R.DURING):          frozenset({R.OVERLAPS, R.DURING, R.FINISHES}),
        (R.MET_BY, R.FINISHES):        frozenset({R.OVERLAPS, R.FINISHES}),
        (R.MET_BY, R.EQUALS):          frozenset({R.MET_BY}),
        (R.MET_BY, R.STARTED_BY):      frozenset({R.OVERLAPPED_BY, R.MET_BY, R.AFTER}),
        (R.MET_BY, R.FINISHED_BY):     frozenset({R.OVERLAPPED_BY, R.MET_BY, R.AFTER}),
        (R.MET_BY, R.CONTAINS):        frozenset({R.OVERLAPPED_BY, R.MET_BY, R.AFTER}),
        (R.MET_BY, R.OVERLAPPED_BY):   frozenset({R.OVERLAPPED_BY, R.MET_BY, R.AFTER}),
        (R.MET_BY, R.MET_BY):          frozenset({R.AFTER}),
        (R.MET_BY, R.AFTER):           frozenset({R.AFTER}),

        # after ∘ X
        (R.AFTER, R.BEFORE):           frozenset({R.BEFORE, R.MEETS, R.OVERLAPS, R.STARTS, R.DURING, R.FINISHES, R.EQUALS, R.STARTED_BY, R.FINISHED_BY, R.CONTAINS, R.OVERLAPPED_BY, R.MET_BY, R.AFTER}),
        (R.AFTER, R.MEETS):            frozenset({R.OVERLAPPED_BY, R.MET_BY, R.AFTER}),
        (R.AFTER, R.OVERLAPS):         frozenset({R.OVERLAPPED_BY, R.MET_BY, R.AFTER}),
        (R.AFTER, R.STARTS):           frozenset({R.OVERLAPPED_BY, R.MET_BY, R.AFTER}),
        (R.AFTER, R.DURING):           frozenset({R.OVERLAPPED_BY, R.MET_BY, R.AFTER}),
        (R.AFTER, R.FINISHES):         frozenset({R.OVERLAPPED_BY, R.MET_BY, R.AFTER}),
        (R.AFTER, R.EQUALS):           frozenset({R.AFTER}),
        (R.AFTER, R.STARTED_BY):       frozenset({R.AFTER}),
        (R.AFTER, R.FINISHED_BY):      frozenset({R.AFTER}),
        (R.AFTER, R.CONTAINS):         frozenset({R.AFTER}),
        (R.AFTER, R.OVERLAPPED_BY):    frozenset({R.AFTER}),
        (R.AFTER, R.MET_BY):           frozenset({R.AFTER}),
        (R.AFTER, R.AFTER):            frozenset({R.AFTER}),
    }
    _COMPOSITION_TABLE.update(table)


# Initialize on import
_init_composition_table()


def compose(r1: AllenRelation, r2: AllenRelation) -> frozenset[AllenRelation]:
    """Compose two Allen relations: given R1(X,Y) and R2(Y,Z), return possible R(X,Z).

    Allen (1983), Table 1.

    Returns a frozenset of possible Allen relations (may contain 1 or more).
    If the result has exactly 1 element, the composition is deterministic.
    If it has multiple elements, the result is a disjunction (any could hold).

    Args:
        r1: Relation from X to Y
        r2: Relation from Y to Z

    Returns:
        Frozenset of possible relations from X to Z
    """
    return _COMPOSITION_TABLE.get((r1, r2), frozenset(AllenRelation))


def is_transitive(r: AllenRelation) -> bool:
    """Check if a relation is transitive.

    Transitive: R(X,Y) ∧ R(Y,Z) → R(X,Z)
    From Allen (1983), Section 4.
    """
    return r.is_transitive


def get_inverse(r: AllenRelation) -> AllenRelation:
    """Get the inverse of an Allen relation.

    Allen (1983), Section 3: every relation has a unique inverse.
    """
    return r.inverse


# ─── Natural Language Temporal Patterns ──────────────────────────────

# Maps English temporal expressions to Allen relations
TEMPORAL_PATTERNS = {
    "before": AllenRelation.BEFORE,
    "after": AllenRelation.AFTER,
    "meets": AllenRelation.MEETS,
    "met_by": AllenRelation.MET_BY,
    "overlaps": AllenRelation.OVERLAPS,
    "overlapped_by": AllenRelation.OVERLAPPED_BY,
    "starts": AllenRelation.STARTS,
    "started_by": AllenRelation.STARTED_BY,
    "during": AllenRelation.DURING,
    "contains": AllenRelation.CONTAINS,
    "finishes": AllenRelation.FINISHES,
    "finished_by": AllenRelation.FINISHED_BY,
    "equals": AllenRelation.EQUALS,
    # Common aliases
    "happens before": AllenRelation.BEFORE,
    "happens after": AllenRelation.AFTER,
    "takes place before": AllenRelation.BEFORE,
    "takes place after": AllenRelation.AFTER,
    "occurs before": AllenRelation.BEFORE,
    "occurs after": AllenRelation.AFTER,
    "comes before": AllenRelation.BEFORE,
    "comes after": AllenRelation.AFTER,
    "precedes": AllenRelation.BEFORE,
    "follows": AllenRelation.AFTER,
    "overlaps with": AllenRelation.OVERLAPS,
    "coincides with": AllenRelation.EQUALS,
    "happens during": AllenRelation.DURING,
    "takes place during": AllenRelation.DURING,
    "occurs during": AllenRelation.DURING,
    "is contained in": AllenRelation.DURING,
    "contains": AllenRelation.CONTAINS,
    "encompasses": AllenRelation.CONTAINS,
    "starts at the same time as": AllenRelation.STARTS,
    "finishes at the same time as": AllenRelation.FINISHES,
    "simultaneous with": AllenRelation.EQUALS,
    "concurrent with": AllenRelation.EQUALS,
    "at the same time as": AllenRelation.EQUALS,
}

# Year/era patterns for extracting temporal intervals from text
YEAR_PATTERN = r'\b(1[89]\d{2}|20[0-2]\d)\b'  # 1800-2029
YEAR_RANGE_PATTERN = r'\b(\d{4})\s*(?:to|-|–|—)\s*(\d{4})\b'  # "2009-2017" or "2009 to 2017"


def extract_temporal_interval(text: str) -> Optional[TemporalInterval]:
    """Extract a temporal interval from natural language text.

    Handles:
        - "from 2009 to 2017" → [2009, 2017)
        - "2009-2017" → [2009, 2017)
        - "between 2009 and 2017" → [2009, 2017)
        - "since 2009" → [2009, None)
        - "until 2017" → [None, 2017)
        - "from 2009" → [2009, None)
        - Single year "2009" → [2009, 2010) (one-year interval)
        - "during the 2010s" → [2010, 2020)
    """
    import re

    # Year range: "2009-2017" or "2009 to 2017"
    m = re.search(YEAR_RANGE_PATTERN, text)
    if m:
        start, end = int(m.group(1)), int(m.group(2))
        return TemporalInterval(start=start, end=end)

    # "from X to Y"
    m = re.search(r'from\s+(\d{4})\s+to\s+(\d{4})', text)
    if m:
        return TemporalInterval(start=int(m.group(1)), end=int(m.group(2)))

    # "between X and Y"
    m = re.search(r'between\s+(\d{4})\s+and\s+(\d{4})', text)
    if m:
        return TemporalInterval(start=int(m.group(1)), end=int(m.group(2)))

    # "since X"
    m = re.search(r'since\s+(\d{4})', text)
    if m:
        return TemporalInterval(start=int(m.group(1)), end=None)

    # "until X" or "before X"
    m = re.search(r'(?:until|before)\s+(\d{4})', text)
    if m:
        return TemporalInterval(start=None, end=int(m.group(1)))

    # "from X" (open-ended)
    m = re.search(r'from\s+(\d{4})', text)
    if m:
        return TemporalInterval(start=int(m.group(1)), end=None)

    # "during the 2010s"
    m = re.search(r'during\s+the\s+(\d{3})0s', text)
    if m:
        decade = int(m.group(1)) * 10
        return TemporalInterval(start=decade, end=decade + 10)

    # Single year
    m = re.search(YEAR_PATTERN, text)
    if m:
        year = int(m.group(1))
        return TemporalInterval(start=year, end=year + 1)

    return None
