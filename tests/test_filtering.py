"""Unit tests for DNT column filtering logic."""

import numpy as np

from src.filtering import filter_translatable_blocks, _is_in_dnt_zone
from src.models import TextBlock


class TestFilterTranslatableBlocks:
    """Tests for the two-pass DNT column filter."""

    def test_no_dnt_headers_translates_everything(self):
        """When no headers match, all blocks should be translated."""
        blocks = [
            TextBlock("Hello World", 10, 10, 100, 30),
            TextBlock("Some Text", 10, 40, 100, 60),
        ]
        # Empty vertical lines mask (no grid lines)
        v_lines = np.zeros((100, 200), dtype=np.uint8)
        result = filter_translatable_blocks(blocks, ["cost"], v_lines, [])
        assert len(result) == 2

    def test_blocks_in_dnt_column_are_excluded(self):
        """Blocks beneath a DNT header in the same column should be skipped."""
        # Simulate a table with a "cost" header at row y=50
        header_block = TextBlock("Cost", 100, 50, 150, 70)
        data_block = TextBlock("$5.00", 100, 80, 150, 100)
        other_block = TextBlock("Description", 10, 80, 90, 100)

        blocks = [header_block, data_block, other_block]

        # Create vertical lines at x=95 and x=155 to bound the "cost" column
        v_lines = np.zeros((200, 300), dtype=np.uint8)
        v_lines[:, 95] = 255
        v_lines[:, 155] = 255

        # Table bounding box that contains all blocks
        largest_boxes = [{"coords": [0, 0, 300, 200], "area": 60000, "score": 0.5}]

        result = filter_translatable_blocks(blocks, ["cost"], v_lines, largest_boxes)

        # The "cost" header and "$5.00" data should be excluded,
        # but "Description" in a different column should remain
        result_texts = [b.text for b in result]
        assert "Description" in result_texts
        assert "$5.00" not in result_texts


class TestIsInDntZone:
    """Tests for the zone membership check."""

    def test_block_outside_table_is_not_in_zone(self):
        """Blocks outside the table bounds are never in a DNT zone."""
        block = TextBlock("test", 500, 500, 550, 520)
        table_bbox = [0, 0, 300, 200]
        assert _is_in_dnt_zone(block, table_bbox, 40.0, [(100, 200)]) is False

    def test_block_above_ceiling_is_not_in_zone(self):
        """Blocks above the DNT ceiling (header row area) are safe."""
        block = TextBlock("Header", 120, 20, 180, 40)
        table_bbox = [0, 0, 300, 200]
        # Ceiling at y=45 means our block at cy=30 is above
        assert _is_in_dnt_zone(block, table_bbox, 45.0, [(100, 200)]) is False

    def test_none_table_returns_false(self):
        """If no table was detected, nothing is in a DNT zone."""
        block = TextBlock("test", 50, 50, 100, 70)
        assert _is_in_dnt_zone(block, None, 40.0, [(10, 200)]) is False
