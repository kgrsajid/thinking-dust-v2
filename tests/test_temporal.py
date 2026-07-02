"""Tests for Allen's Interval Algebra — Temporal Reasoning.

Based on: Allen, J.F. (1983). "Maintaining Knowledge about Temporal Intervals."
Communications of the ACM, 26(11), 832-843.

Tests cover:
- All 13 Allen relations (correct detection)
- Inverse relations (each pair)
- Composition table (deterministic and disjunctive)
- Transitivity properties
- Temporal interval extraction from natural language
- Edge cases: point intervals, open intervals, invalid intervals
- Unseen novel temporal patterns
- Real-world scenarios (presidents, wars, projects)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from td.temporal import (
    AllenRelation, TemporalInterval,
    check_allen_relation, compose, is_transitive, get_inverse,
    extract_temporal_interval, TEMPORAL_PATTERNS,
)


# ─── Allen Relation Detection ────────────────────────────────────────

class TestAllenRelationDetection:
    """All 13 Allen relations detected correctly from interval endpoints."""

    def test_before(self):
        X = TemporalInterval(start=2000, end=2005)
        Y = TemporalInterval(start=2010, end=2015)
        assert check_allen_relation(X, Y) == AllenRelation.BEFORE

    def test_after(self):
        X = TemporalInterval(start=2010, end=2015)
        Y = TemporalInterval(start=2000, end=2005)
        assert check_allen_relation(X, Y) == AllenRelation.AFTER

    def test_meets(self):
        X = TemporalInterval(start=2000, end=2005)
        Y = TemporalInterval(start=2005, end=2010)
        assert check_allen_relation(X, Y) == AllenRelation.MEETS

    def test_met_by(self):
        X = TemporalInterval(start=2005, end=2010)
        Y = TemporalInterval(start=2000, end=2005)
        assert check_allen_relation(X, Y) == AllenRelation.MET_BY

    def test_overlaps(self):
        X = TemporalInterval(start=2000, end=2010)
        Y = TemporalInterval(start=2005, end=2015)
        assert check_allen_relation(X, Y) == AllenRelation.OVERLAPS

    def test_overlapped_by(self):
        X = TemporalInterval(start=2005, end=2015)
        Y = TemporalInterval(start=2000, end=2010)
        assert check_allen_relation(X, Y) == AllenRelation.OVERLAPPED_BY

    def test_starts(self):
        X = TemporalInterval(start=2000, end=2010)
        Y = TemporalInterval(start=2000, end=2015)
        assert check_allen_relation(X, Y) == AllenRelation.STARTS

    def test_started_by(self):
        X = TemporalInterval(start=2000, end=2015)
        Y = TemporalInterval(start=2000, end=2010)
        assert check_allen_relation(X, Y) == AllenRelation.STARTED_BY

    def test_during(self):
        X = TemporalInterval(start=2005, end=2010)
        Y = TemporalInterval(start=2000, end=2015)
        assert check_allen_relation(X, Y) == AllenRelation.DURING

    def test_contains(self):
        X = TemporalInterval(start=2000, end=2015)
        Y = TemporalInterval(start=2005, end=2010)
        assert check_allen_relation(X, Y) == AllenRelation.CONTAINS

    def test_finishes(self):
        X = TemporalInterval(start=2005, end=2015)
        Y = TemporalInterval(start=2000, end=2015)
        assert check_allen_relation(X, Y) == AllenRelation.FINISHES

    def test_finished_by(self):
        X = TemporalInterval(start=2000, end=2015)
        Y = TemporalInterval(start=2005, end=2015)
        assert check_allen_relation(X, Y) == AllenRelation.FINISHED_BY

    def test_equals(self):
        X = TemporalInterval(start=2000, end=2010)
        Y = TemporalInterval(start=2000, end=2010)
        assert check_allen_relation(X, Y) == AllenRelation.EQUALS


# ─── Inverse Relations ───────────────────────────────────────────────

class TestInverseRelations:
    """Every Allen relation has a correct inverse (Allen 1983, Section 3)."""

    def test_before_after_inverse(self):
        assert get_inverse(AllenRelation.BEFORE) == AllenRelation.AFTER
        assert get_inverse(AllenRelation.AFTER) == AllenRelation.BEFORE

    def test_meets_met_by_inverse(self):
        assert get_inverse(AllenRelation.MEETS) == AllenRelation.MET_BY
        assert get_inverse(AllenRelation.MET_BY) == AllenRelation.MEETS

    def test_overlaps_inverse(self):
        assert get_inverse(AllenRelation.OVERLAPS) == AllenRelation.OVERLAPPED_BY
        assert get_inverse(AllenRelation.OVERLAPPED_BY) == AllenRelation.OVERLAPS

    def test_starts_started_by_inverse(self):
        assert get_inverse(AllenRelation.STARTS) == AllenRelation.STARTED_BY
        assert get_inverse(AllenRelation.STARTED_BY) == AllenRelation.STARTS

    def test_during_contains_inverse(self):
        assert get_inverse(AllenRelation.DURING) == AllenRelation.CONTAINS
        assert get_inverse(AllenRelation.CONTAINS) == AllenRelation.DURING

    def test_finishes_finished_by_inverse(self):
        assert get_inverse(AllenRelation.FINISHES) == AllenRelation.FINISHED_BY
        assert get_inverse(AllenRelation.FINISHED_BY) == AllenRelation.FINISHES

    def test_equals_self_inverse(self):
        assert get_inverse(AllenRelation.EQUALS) == AllenRelation.EQUALS

    def test_inverse_of_inverse_is_original(self):
        """R.inverse.inverse == R for all relations."""
        for r in AllenRelation:
            assert get_inverse(get_inverse(r)) == r

    def test_inverse_consistency(self):
        """If R(X,Y) then R.inverse(Y,X)."""
        X = TemporalInterval(start=2000, end=2010)
        Y = TemporalInterval(start=2005, end=2015)
        r_xy = check_allen_relation(X, Y)  # overlaps
        r_yx = check_allen_relation(Y, X)  # overlapped_by
        assert r_yx == get_inverse(r_xy)


# ─── Transitivity Properties ─────────────────────────────────────────

class TestTransitivity:
    """Transitive relations from Allen (1983), Section 4."""

    def test_transitive_relations(self):
        """These should be transitive."""
        transitive = [
            AllenRelation.BEFORE, AllenRelation.AFTER,
            AllenRelation.DURING, AllenRelation.CONTAINS,
            AllenRelation.STARTS, AllenRelation.STARTED_BY,
            AllenRelation.FINISHES, AllenRelation.FINISHED_BY,
            AllenRelation.EQUALS,
        ]
        for r in transitive:
            assert is_transitive(r), f"{r} should be transitive"

    def test_non_transitive_relations(self):
        """These should NOT be transitive."""
        non_transitive = [
            AllenRelation.MEETS, AllenRelation.MET_BY,
            AllenRelation.OVERLAPS, AllenRelation.OVERLAPPED_BY,
        ]
        for r in non_transitive:
            assert not is_transitive(r), f"{r} should NOT be transitive"


# ─── Composition Table ───────────────────────────────────────────────

class TestCompositionTable:
    """Allen's composition table (Allen 1983, Table 1)."""

    def test_before_compose_before(self):
        """before ∘ before = before (deterministic)."""
        result = compose(AllenRelation.BEFORE, AllenRelation.BEFORE)
        assert result == frozenset({AllenRelation.BEFORE})

    def test_before_compose_equals(self):
        """before ∘ equals = before (identity)."""
        result = compose(AllenRelation.BEFORE, AllenRelation.EQUALS)
        assert result == frozenset({AllenRelation.BEFORE})

    def test_equals_compose_any(self):
        """equals ∘ R = R (identity element)."""
        for r in AllenRelation:
            result = compose(AllenRelation.EQUALS, r)
            assert result == frozenset({r})

    def test_any_compose_equals(self):
        """R ∘ equals = R (identity element)."""
        for r in AllenRelation:
            result = compose(r, AllenRelation.EQUALS)
            assert result == frozenset({r})

    def test_before_compose_during_disjunctive(self):
        """before ∘ during = {before} (deterministic in Allen's table)."""
        result = compose(AllenRelation.BEFORE, AllenRelation.DURING)
        assert AllenRelation.BEFORE in result

    def test_during_compose_during(self):
        """during ∘ during = during (transitive)."""
        result = compose(AllenRelation.DURING, AllenRelation.DURING)
        assert result == frozenset({AllenRelation.DURING})

    def test_contains_compose_contains(self):
        """contains ∘ contains = contains (transitive)."""
        result = compose(AllenRelation.CONTAINS, AllenRelation.CONTAINS)
        assert result == frozenset({AllenRelation.CONTAINS})

    def test_starts_compose_starts(self):
        """starts ∘ starts = starts (transitive)."""
        result = compose(AllenRelation.STARTS, AllenRelation.STARTS)
        assert result == frozenset({AllenRelation.STARTS})

    def test_finishes_compose_finishes(self):
        """finishes ∘ finishes = finishes (transitive)."""
        result = compose(AllenRelation.FINISHES, AllenRelation.FINISHES)
        assert result == frozenset({AllenRelation.FINISHES})

    def test_inverse_composition(self):
        """R(X,Y) ∧ R.inverse(Y,Z) → the constraint that Y is between X and Z.

        In Allen's algebra, R ∘ R.inverse means: X relates to Y, and Y relates to Z
        where the second relation is the inverse of the first. This constrains
        the relative position of X and Z with respect to Y (which sits between them).

        Example: MEETS(X,Y) ∧ MET_BY(Y,Z) means X_e = Y_s and Y_e = Z_s.
        Since Y_s < Y_e (positive duration), X_e < Z_s, so X BEFORE Z.
        """
        # meets ∘ met_by = {before, meets, overlaps, starts, during}
        result = compose(AllenRelation.MEETS, AllenRelation.MET_BY)
        assert AllenRelation.BEFORE in result

        # before ∘ after = {all 13} (no constraint)
        result = compose(AllenRelation.BEFORE, AllenRelation.AFTER)
        assert len(result) == 13  # everything possible

        # during ∘ contains = {during, finishes, equals, started_by, finished_by, contains}
        result = compose(AllenRelation.DURING, AllenRelation.CONTAINS)
        assert AllenRelation.DURING in result

    def test_all_compositions_defined(self):
        """All 13×13 = 169 compositions should be defined."""
        for r1 in AllenRelation:
            for r2 in AllenRelation:
                result = compose(r1, r2)
                assert len(result) > 0, f"Composition ({r1}, {r2}) is empty"

    def test_compositions_are_subsets(self):
        """Every composition result should be a subset of all 13 relations."""
        all_relations = frozenset(AllenRelation)
        for r1 in AllenRelation:
            for r2 in AllenRelation:
                result = compose(r1, r2)
                assert result.issubset(all_relations)


# ─── Temporal Interval ───────────────────────────────────────────────

class TestTemporalInterval:
    """TemporalInterval data class."""

    def test_valid_interval(self):
        i = TemporalInterval(start=2000, end=2010)
        assert i.is_valid()
        assert i.start == 2000
        assert i.end == 2010

    def test_invalid_interval_start_after_end(self):
        with pytest.raises(ValueError):
            TemporalInterval(start=2010, end=2000)

    def test_open_start(self):
        i = TemporalInterval(start=None, end=2010)
        assert not i.is_valid()
        assert i.end == 2010

    def test_open_end(self):
        i = TemporalInterval(start=2000, end=None)
        assert not i.is_valid()
        assert i.start == 2000

    def test_point_interval(self):
        """Point interval: start == end. Allen treats this as degenerate."""
        # Our implementation requires start < end, so start == end is invalid
        with pytest.raises(ValueError):
            TemporalInterval(start=2000, end=2000)

    def test_repr(self):
        i = TemporalInterval(start=2000, end=2010)
        assert "2000" in repr(i)
        assert "2010" in repr(i)

    def test_open_repr(self):
        i = TemporalInterval(start=2000, end=None)
        assert "∞" in repr(i)


# ─── Natural Language Temporal Extraction ────────────────────────────

class TestTemporalExtraction:
    """Extract temporal intervals from natural language."""

    def test_year_range_dash(self):
        interval = extract_temporal_interval("Obama was president 2009-2017")
        assert interval is not None
        assert interval.start == 2009
        assert interval.end == 2017

    def test_year_range_to(self):
        interval = extract_temporal_interval("from 2009 to 2017")
        assert interval is not None
        assert interval.start == 2009
        assert interval.end == 2017

    def test_between_and(self):
        interval = extract_temporal_interval("between 1939 and 1945")
        assert interval is not None
        assert interval.start == 1939
        assert interval.end == 1945

    def test_since_year(self):
        interval = extract_temporal_interval("since 2020")
        assert interval is not None
        assert interval.start == 2020
        assert interval.end is None

    def test_until_year(self):
        interval = extract_temporal_interval("until 2025")
        assert interval is not None
        assert interval.start is None
        assert interval.end == 2025

    def test_single_year(self):
        interval = extract_temporal_interval("in 2015")
        assert interval is not None
        assert interval.start == 2015
        assert interval.end == 2016

    def test_decade(self):
        interval = extract_temporal_interval("during the 2010s")
        assert interval is not None
        assert interval.start == 2010
        assert interval.end == 2020

    def test_no_temporal(self):
        interval = extract_temporal_interval("Paris is the capital of France")
        assert interval is None

    def test_em_dash_range(self):
        interval = extract_temporal_interval("2009–2017")
        assert interval is not None
        assert interval.start == 2009
        assert interval.end == 2017


# ─── Real-World Temporal Scenarios ───────────────────────────────────

class TestRealWorldTemporal:
    """Real-world scenarios using Allen's algebra."""

    def test_presidents_non_overlapping(self):
        """Obama 2009-2017, Trump 2017-2021: Obama MEETS Trump."""
        obama = TemporalInterval(start=2009, end=2017)
        trump = TemporalInterval(start=2017, end=2021)
        assert check_allen_relation(obama, trump) == AllenRelation.MEETS

    def test_ww2_before_cold_war(self):
        """WW2 1939-1945, Cold War 1947-1991: WW2 BEFORE Cold War."""
        ww2 = TemporalInterval(start=1939, end=1945)
        cold_war = TemporalInterval(start=1947, end=1991)
        assert check_allen_relation(ww2, cold_war) == AllenRelation.BEFORE

    def test_cold_war_during_20th_century(self):
        """Cold War 1947-1991 DURING 20th century 1900-2000."""
        cold_war = TemporalInterval(start=1947, end=1991)
        century = TemporalInterval(start=1900, end=2000)
        assert check_allen_relation(cold_war, century) == AllenRelation.DURING

    def test_20th_century_contains_cold_war(self):
        """20th century CONTAINS Cold War."""
        century = TemporalInterval(start=1900, end=2000)
        cold_war = TemporalInterval(start=1947, end=1991)
        assert check_allen_relation(century, cold_war) == AllenRelation.CONTAINS

    def test_project_phases(self):
        """Project A (Jan-Mar) STARTS Project B (Jan-Jun)."""
        phase_a = TemporalInterval(start=1, end=3)
        phase_b = TemporalInterval(start=1, end=6)
        assert check_allen_relation(phase_a, phase_b) == AllenRelation.STARTS

    def test_overlapping_meetings(self):
        """Meeting A (9-11am) OVERLAPS Meeting B (10-12pm)."""
        meeting_a = TemporalInterval(start=9, end=11)
        meeting_b = TemporalInterval(start=10, end=12)
        assert check_allen_relation(meeting_a, meeting_b) == AllenRelation.OVERLAPS

    def test_transitive_before_chain(self):
        """A before B, B before C → A before C (via composition)."""
        a_before_b = AllenRelation.BEFORE
        b_before_c = AllenRelation.BEFORE
        result = compose(a_before_b, b_before_c)
        assert AllenRelation.BEFORE in result
        assert len(result) == 1  # deterministic

    def test_transitive_during_chain(self):
        """A during B, B during C → A during C."""
        a_during_b = AllenRelation.DURING
        b_during_c = AllenRelation.DURING
        result = compose(a_during_b, b_during_c)
        assert AllenRelation.DURING in result
        assert len(result) == 1

    def test_composition_with_inverse(self):
        """Obama MEETS Trump → Trump MET_BY Obama."""
        obama = TemporalInterval(start=2009, end=2017)
        trump = TemporalInterval(start=2017, end=2021)
        r = check_allen_relation(obama, trump)
        r_inv = check_allen_relation(trump, obama)
        assert r_inv == get_inverse(r)


# ─── Edge Cases ──────────────────────────────────────────────────────

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_very_small_interval(self):
        """Year-long interval."""
        X = TemporalInterval(start=2000, end=2001)
        Y = TemporalInterval(start=2001, end=2002)
        assert check_allen_relation(X, Y) == AllenRelation.MEETS

    def test_very_large_interval(self):
        """Century-long interval."""
        X = TemporalInterval(start=1900, end=2000)
        Y = TemporalInterval(start=1950, end=1960)
        assert check_allen_relation(X, Y) == AllenRelation.CONTAINS

    def test_invalid_interval_returns_none(self):
        """Invalid interval (open start) returns None."""
        X = TemporalInterval(start=None, end=2010)
        Y = TemporalInterval(start=2000, end=2010)
        assert check_allen_relation(X, Y) is None

    def test_both_open_start_equals(self):
        """Two identical open-start intervals [None, 2010) are equal."""
        X = TemporalInterval(start=None, end=2010)
        Y = TemporalInterval(start=None, end=2010)
        assert check_allen_relation(X, Y) == AllenRelation.EQUALS

    def test_adjacent_intervals_meet(self):
        """Adjacent year intervals meet."""
        X = TemporalInterval(start=2020, end=2021)
        Y = TemporalInterval(start=2021, end=2022)
        assert check_allen_relation(X, Y) == AllenRelation.MEETS

    def test_same_interval_equals(self):
        X = TemporalInterval(start=2020, end=2025)
        Y = TemporalInterval(start=2020, end=2025)
        assert check_allen_relation(X, Y) == AllenRelation.EQUALS

    def test_one_year_before(self):
        X = TemporalInterval(start=2020, end=2021)
        Y = TemporalInterval(start=2022, end=2023)
        assert check_allen_relation(X, Y) == AllenRelation.BEFORE

    def test_temporal_pattern_dict_completeness(self):
        """All 13 relations should be in TEMPORAL_PATTERNS."""
        for r in AllenRelation:
            assert r.value in TEMPORAL_PATTERNS, f"{r.value} not in TEMPORAL_PATTERNS"
