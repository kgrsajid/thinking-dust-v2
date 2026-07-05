"""Tests for clause segmentation via spaCy dependency tree."""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

pytestmark = pytest.mark.skipif(not HAS_SPACY, reason="spaCy not installed")

from td.perception.clause_segmenter import segment_text, SimpleClause


@pytest.fixture
def nlp():
    return spacy.load("en_core_web_sm")


# ─── Coordinated Objects ────────────────────────────────────────────

class TestCoordinatedObjects:
    """Split sentences with coordinated objects into multiple clauses."""

    def test_two_objects(self, nlp):
        """'Games have music and story' → 2 clauses."""
        clauses = segment_text("Games have music and a story.", nlp)
        subjects = [c.subject for c in clauses]
        objects = [c.obj for c in clauses]
        assert "games" in subjects
        assert "music" in objects
        assert "a story" in objects

    def test_three_objects(self, nlp):
        """'Games have music, a story and visuals' → 3 clauses."""
        clauses = segment_text("Games have music, a story and visuals.", nlp)
        objects = [c.obj for c in clauses]
        assert "music" in objects
        assert "a story" in objects
        assert "visuals" in objects

    def test_four_objects(self, nlp):
        """'Games have music, a story, visuals and mechanics' → 4 clauses."""
        clauses = segment_text("Games have music, a story, visuals and mechanics.", nlp)
        objects = [c.obj for c in clauses]
        assert len(objects) >= 4

    def test_coordinated_with_adjectives(self, nlp):
        """'She likes red cars and blue bikes' → 2 clauses."""
        clauses = segment_text("She likes red cars and blue bikes.", nlp)
        objects = [c.obj for c in clauses]
        assert any("car" in o for o in objects)
        assert any("bike" in o for o in objects)


# ─── Coordinated Subjects ───────────────────────────────────────────

class TestCoordinatedSubjects:
    """Split sentences with coordinated subjects into multiple clauses."""

    def test_two_subjects(self, nlp):
        """'Alice and Bob went to Paris' → 2 clauses."""
        clauses = segment_text("Alice and Bob went to Paris.", nlp)
        subjects = [c.subject for c in clauses]
        assert "alice" in subjects
        assert "bob" in subjects

    def test_three_subjects(self, nlp):
        """'Alice, Bob and Carol went to Paris' → 3 clauses."""
        clauses = segment_text("Alice, Bob and Carol went to Paris.", nlp)
        subjects = [c.subject for c in clauses]
        assert "alice" in subjects
        assert "bob" in subjects
        assert "carol" in subjects

    def test_five_subjects(self, nlp):
        """'A, B, C, D and E are in F' → 5 clauses."""
        clauses = segment_text("A, B, C, D and E are in F.", nlp)
        subjects = [c.subject for c in clauses]
        assert len(subjects) >= 5


# ─── Coordinated Verbs ──────────────────────────────────────────────

class TestCoordinatedVerbs:
    """Split sentences with coordinated verbs into multiple clauses."""

    def test_two_verbs(self, nlp):
        """'Alice loves Bob and hates Charlie' → 2 clauses."""
        clauses = segment_text("Alice loves Bob and hates Charlie.", nlp)
        assert len(clauses) == 2
        assert clauses[0].relation == "loves"
        assert clauses[1].relation == "hates"

    def test_shared_subject_two_verbs(self, nlp):
        """'Video games go above art and have a particularity' → 2 clauses."""
        clauses = segment_text("Video games go above art and have a particularity.", nlp)
        assert len(clauses) >= 2
        subjects = set(c.subject for c in clauses)
        assert len(subjects) == 1  # Same subject for both verbs


# ─── Complex Sentences ──────────────────────────────────────────────

class TestComplexSentences:
    """Split complex sentences with subordinating conjunctions."""

    def test_causal_since(self, nlp):
        """'X are Y since they produce Z' → 2 clauses."""
        clauses = segment_text(
            "Game creators are artists since they produce creative works.", nlp
        )
        assert len(clauses) >= 2

    def test_relative_clause(self, nlp):
        """'Paris which is the capital of France is beautiful' → 2 clauses."""
        clauses = segment_text(
            "Paris which is the capital of France is beautiful.", nlp
        )
        assert len(clauses) >= 2

    def test_but_conjunction(self, nlp):
        """'X is Y but Z is W' → 2 clauses."""
        clauses = segment_text("Alice is smart but Bob is fast.", nlp)
        assert len(clauses) >= 2


# ─── Game Art Text (the real test) ─────────────────────────────────

class TestGameArtText:
    """Test on the actual complex text Kazi provided."""

    def test_game_art_paragraph(self, nlp):
        """Full game art paragraph → extract multiple clauses."""
        text = ("Game creators are by definition artists since they produce "
                "creative works. To say games have no utilitarian use is a "
                "misconception of the art. Video games go above art and have "
                "a particularity: most components are modular by design or by "
                "characteristics. Games may have music, a story and visuals – "
                "each an artistic creation but which aggregate into a functional "
                "whole.")

        doc = nlp(text)
        from td.perception.clause_segmenter import segment_clauses
        clauses = segment_clauses(doc)

        # Should extract many more clauses than sentences
        sents = list(doc.sents)
        assert len(clauses) > len(sents), \
            f"Expected more clauses ({len(clauses)}) than sentences ({len(sents)})"

        # Verify key extractions
        subjects = [c.subject for c in clauses]
        objects = [c.obj for c in clauses]

        # From sentence 1: game creators are artists, game creators produce works
        assert any("game creator" in s for s in subjects)

        # From sentence 4: games have music, games have story, games have visuals
        assert "music" in objects
        assert any("story" in o for o in objects)
        assert "visuals" in objects

    def test_game_art_clauses_count(self, nlp):
        """Game art text should produce 5+ clauses from 4 sentences."""
        text = ("Game creators are artists since they produce creative works. "
                "Games have music, a story and visuals – each an artistic creation "
                "but which aggregate into a functional whole.")

        doc = nlp(text)
        from td.perception.clause_segmenter import segment_clauses
        clauses = segment_clauses(doc)

        # At least 5 clauses: creators are artists, creators produce works,
        # games have music, games have story, games have visuals
        assert len(clauses) >= 5


# ─── Edge Cases ─────────────────────────────────────────────────────

class TestClauseSegmenterEdgeCases:
    """Edge cases for clause segmentation."""

    def test_simple_sentence(self, nlp):
        """Simple SVO → 1 clause."""
        clauses = segment_text("Paris is the capital of France.", nlp)
        assert len(clauses) >= 1
        assert clauses[0].subject == "paris"

    def test_empty_text(self, nlp):
        """Empty text → 0 clauses."""
        clauses = segment_text("", nlp)
        assert len(clauses) == 0

    def test_single_word(self, nlp):
        """Single word → 0 clauses."""
        clauses = segment_text("Hello.", nlp)
        assert len(clauses) == 0

    def test_intransitive_verb(self, nlp):
        """Intransitive verb (no object) → 1 clause with empty object."""
        clauses = segment_text("Alice sleeps.", nlp)
        assert len(clauses) >= 1
        assert clauses[0].subject == "alice"

    def test_passive_voice(self, nlp):
        """Passive voice → extract subject."""
        clauses = segment_text("The ball was thrown by Alice.", nlp)
        assert len(clauses) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
