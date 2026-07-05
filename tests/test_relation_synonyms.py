"""Tests for relation synonymy detection and registry."""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from td.kg.relation_synonyms import RelationSynonymRegistry


class TestRelationSynonymRegistry:
    """Test the synonym registry."""

    def test_teach_and_lookup(self):
        reg = RelationSynonymRegistry()
        reg.teach("in", ["part of", "contains", "located in"])
        assert reg.get_canonical("part of") == "in"
        assert reg.get_canonical("in") == "in"
        assert reg.get_canonical("contains") == "in"

    def test_get_synonyms(self):
        reg = RelationSynonymRegistry()
        reg.teach("in", ["part of", "contains"])
        syns = reg.get_synonyms("in")
        assert "part of" in syns
        assert "contains" in syns
        assert "in" not in syns

    def test_are_synonyms(self):
        reg = RelationSynonymRegistry()
        reg.teach("in", ["part of", "contains"])
        assert reg.are_synonyms("in", "part of") is True
        assert reg.are_synonyms("in", "contains") is True
        assert reg.are_synonyms("in", "created_by") is False

    def test_unknown_relation(self):
        reg = RelationSynonymRegistry()
        assert reg.get_canonical("unknown") == "unknown"
        assert reg.get_synonyms("unknown") == []

    def test_multiple_groups(self):
        reg = RelationSynonymRegistry()
        reg.teach("in", ["part of", "contains"])
        reg.teach("created_by", ["made by", "built by"])
        assert reg.are_synonyms("in", "part of") is True
        assert reg.are_synonyms("created_by", "made by") is True
        assert reg.are_synonyms("in", "created_by") is False

    def test_merge_groups(self):
        """Teach overlapping synonyms — groups should merge."""
        reg = RelationSynonymRegistry()
        reg.teach("in", ["part of"])
        reg.teach("contains", ["located in"])
        # Now teach that they're all the same
        reg.teach("in", ["contains"])
        assert reg.are_synonyms("in", "located in") is True
        assert reg.are_synonyms("part of", "contains") is True

    def test_teach_preserves_existing(self):
        """Re-teaching a canonical should NOT lose existing synonyms."""
        reg = RelationSynonymRegistry()
        reg.teach("in", ["part of"])
        reg.teach("in", ["contains"])
        # "part of" should NOT be lost
        assert reg.are_synonyms("in", "part of") is True
        assert reg.are_synonyms("in", "contains") is True
        syns = reg.get_synonyms("in")
        assert "part of" in syns
        assert "contains" in syns

    def test_case_insensitive(self):
        reg = RelationSynonymRegistry()
        reg.teach("In", ["Part Of", "Contains"])
        assert reg.get_canonical("part of") == "in"
        assert reg.are_synonyms("IN", "PART OF") is True

    def test_get_all_groups(self):
        reg = RelationSynonymRegistry()
        reg.teach("in", ["part of"])
        reg.teach("created_by", ["made by"])
        groups = reg.get_all_groups()
        assert len(groups) == 2

    def test_real_world_synonyms(self):
        """Test with real-world relation synonyms."""
        reg = RelationSynonymRegistry()
        reg.teach("in", ["part of", "contains", "located in", "belongs to", "resides in"])
        reg.teach("capital_of", ["has capital", "headquarters of"])
        reg.teach("created_by", ["made by", "built by", "developed by"])
        reg.teach("depends_on", ["relies on", "requires"])
        reg.teach("married_to", ["spouse of", "wedded to"])

        # Verify all groups
        assert reg.are_synonyms("in", "resides in") is True
        assert reg.are_synonyms("capital_of", "headquarters of") is True
        assert reg.are_synonyms("depends_on", "requires") is True
        assert reg.are_synonyms("married_to", "wedded to") is True
        assert reg.are_synonyms("in", "capital_of") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
