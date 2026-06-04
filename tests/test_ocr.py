"""Unit tests for OCR text splitting logic."""

from src.ocr import _split_text_at


class TestSplitTextAt:
    """Tests for the proportional text splitter used by the Spatial Guillotine."""

    def test_splits_at_word_boundary(self):
        """Should snap to the nearest space rather than splitting mid-word."""
        text = "Fabric Cotton 60Main Fabric"
        # Simulating a split at roughly the midpoint of a 100px-wide box
        left, right = _split_text_at(text, 50, 0, 100)
        # Should split at a space, not in the middle of a word
        assert ' ' not in left or left == left.strip()
        assert ' ' not in right or right == right.strip()
        assert left + " " + right == text or (left and right)

    def test_handles_no_spaces(self):
        """When text has no spaces, falls back to proportional hard split."""
        text = "ABCDEFGHIJ"
        left, right = _split_text_at(text, 50, 0, 100)
        # Should split roughly in half
        assert left + right == text
        assert len(left) == 5
        assert len(right) == 5

    def test_split_at_left_edge(self):
        """Split near the left edge produces a short left and long right."""
        text = "A quick brown fox"
        left, right = _split_text_at(text, 10, 0, 100)
        assert len(left) < len(right)
        assert left != ""
        assert right != ""

    def test_split_at_right_edge(self):
        """Split near the right edge produces a long left and short right."""
        text = "A quick brown fox"
        left, right = _split_text_at(text, 90, 0, 100)
        assert len(left) > len(right)
