"""Do-Not-Translate (DNT) column detection and block filtering.

Tech packs contain columns of raw technical data (material codes, sizing,
costs) that must remain untranslated for factory accuracy. This module
dynamically identifies those columns at runtime by matching OCR'd header
text against the terms.json config, then builds vertical "data corridors"
that protect all content beneath the matched headers.
"""

import logging
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from src.models import TextBlock

logger = logging.getLogger(__name__)

# Pixel intensity threshold for detecting a vertical grid line at a given point
LINE_THRESHOLD = 128


def filter_translatable_blocks(
    raw_blocks: list[TextBlock],
    dnt_headers: list[str],
    v_lines: NDArray[np.uint8],
    largest_boxes: list[dict],
) -> list[TextBlock]:
    """Two-pass filter: detect DNT columns, then exclude blocks inside them.

    Pass 1: Scan all blocks for text matching dnt_headers. When found,
    locate the containing table box and trace the column boundaries using
    vertical grid lines. This builds a set of protected "column zones."

    Pass 2: For each block, check if its center falls inside any protected
    zone. If so, skip it. Everything else gets translated.
    """
    table_bbox: list[int] | None = None
    dnt_column_zones: list[tuple[int, int]] = []
    dnt_y_ceiling: float | None = None

    # --- Pass 1: Identify DNT column zones ---
    for block in raw_blocks:
        text_lower = block.text.lower()
        if not any(h in text_lower for h in dnt_headers):
            continue

        # Track the highest (topmost) DNT header position.
        # Everything above this Y coordinate is treated as header row
        # content and remains eligible for translation.
        if dnt_y_ceiling is None or block.y_min < dnt_y_ceiling:
            dnt_y_ceiling = block.y_min - 5

        # Find the table that contains this header
        if not table_bbox:
            table_bbox = _find_containing_box(block, largest_boxes)

        # Trace this header's column boundaries using vertical lines
        if table_bbox:
            zone = _detect_column_zone(block, table_bbox, v_lines)
            dnt_column_zones.append(zone)

    if dnt_column_zones:
        logger.info("Detected %d DNT column zone(s).", len(dnt_column_zones))

    # --- Pass 2: Filter blocks ---
    blocks_to_translate: list[TextBlock] = []
    for block in raw_blocks:
        if _is_in_dnt_zone(block, table_bbox, dnt_y_ceiling, dnt_column_zones):
            continue
        blocks_to_translate.append(block)

    return blocks_to_translate


def _find_containing_box(block: TextBlock, largest_boxes: list[dict]) -> list[int] | None:
    """Find the largest candidate box whose bounds contain the block's center.

    This identifies which table region the DNT header belongs to, giving us
    the outer boundary within which to search for column dividers.
    """
    for box in largest_boxes:
        bx1, by1, bx2, by2 = box["coords"]
        if bx1 <= block.cx <= bx2 and by1 <= block.cy <= by2:
            return box["coords"]
    return None


def _detect_column_zone(
    block: TextBlock,
    table_bbox: list[int],
    v_lines: NDArray[np.uint8],
) -> tuple[int, int]:
    """Trace the vertical column boundaries around a DNT header block.

    Starting from the block's center, walks left and right along the
    block's vertical midline until hitting a vertical grid line pixel.
    This gives us the exact pixel range of the column to protect.
    """
    left_bound = block.x_min
    right_bound = block.x_max
    cy = int(block.cy)

    # Walk left from center toward table edge, stop at first vertical line
    for x in range(int(block.cx), int(table_bbox[0]), -1):
        if v_lines[cy, x] > LINE_THRESHOLD:
            left_bound = x
            break

    # Walk right from center toward table edge, stop at first vertical line
    for x in range(int(block.cx), int(table_bbox[2])):
        if v_lines[cy, x] > LINE_THRESHOLD:
            right_bound = x
            break

    return (left_bound, right_bound)


def _is_in_dnt_zone(
    block: TextBlock,
    table_bbox: list[int] | None,
    dnt_y_ceiling: float | None,
    dnt_column_zones: list[tuple[int, int]],
) -> bool:
    """Check if a block should be skipped (falls in a protected DNT corridor).

    A block is protected if ALL of:
    1. It's inside the detected table region
    2. It's below the header row (below dnt_y_ceiling)
    3. Its horizontal center falls within a DNT column zone
    """
    if not table_bbox or dnt_y_ceiling is None:
        return False

    # Must be inside the table bounds
    if not (table_bbox[0] <= block.cx <= table_bbox[2]
            and table_bbox[1] <= block.cy <= table_bbox[3]):
        return False

    # Must be below the header row
    if block.cy <= dnt_y_ceiling:
        return False

    # Must fall within one of the protected column zones
    return any(z_min <= block.cx <= z_max for z_min, z_max in dnt_column_zones)
