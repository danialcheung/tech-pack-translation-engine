"""OCR extraction and merged-block slicing logic.

This module handles two core challenges:
1. Running PaddleOCR and converting raw polygon results into TextBlock objects.
2. The "Spatial Guillotine" — detecting when PaddleOCR has merged text from
   adjacent table cells into a single bounding box, and surgically splitting
   it back into separate blocks using vertical grid line intersections.
"""

import logging
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from paddleocr import PaddleOCR

from src.models import TextBlock

logger = logging.getLogger(__name__)

# Minimum block width (px) before attempting a vertical split.
# Narrow blocks can't physically span two columns.
MIN_SPLIT_WIDTH = 50

# Pixel intensity threshold for detecting a vertical grid line
LINE_INTENSITY_THRESHOLD = 128

# Inset (px) from block edges when scanning for vertical line crossings.
# Avoids detecting the column's own left/right borders as a crossing.
EDGE_INSET = 5


def create_ocr_model() -> PaddleOCR:
    """Initialize and return a PaddleOCR model.

    The model is loaded once and reused across all images in the batch
    to avoid repeated cold-start overhead (~2-3s per load).
    """
    return PaddleOCR(use_angle_cls=True, lang='en', show_log=False)


def extract_text_blocks(
    ocr_model: PaddleOCR,
    img_path: str,
    forbidden_zone: list[int],
) -> list[TextBlock]:
    """Run OCR on an image and return filtered text blocks.

    The forbidden zone excludes the main design artwork area (fashion sketches,
    technical drawings) from text extraction. This prevents the engine from
    misinterpreting sketchy linework as text or corrupting artwork regions.
    """
    results = ocr_model.ocr(img_path, cls=True)
    if not results or not results[0]:
        return []

    blocks: list[TextBlock] = []
    fz_x1, fz_y1, fz_x2, fz_y2 = forbidden_zone

    for line in results[0]:
        # PaddleOCR returns: [[4 corner points], (text_string, confidence)]
        coords, text_data = line[0], line[1]
        raw_text = text_data[0].strip()
        if not raw_text:
            continue

        # Convert polygon corners to axis-aligned bounding box
        x_min = int(min(pt[0] for pt in coords))
        y_min = int(min(pt[1] for pt in coords))
        x_max = int(max(pt[0] for pt in coords))
        y_max = int(max(pt[1] for pt in coords))

        block = TextBlock(raw_text, x_min, y_min, x_max, y_max)

        # Membership test: skip if block center falls inside the artwork zone
        if fz_x1 <= block.cx <= fz_x2 and fz_y1 <= block.cy <= fz_y2:
            continue

        blocks.append(block)

    return blocks


def slice_merged_blocks(
    blocks: list[TextBlock],
    v_lines: NDArray[np.uint8],
) -> list[TextBlock]:
    """The "Spatial Guillotine" — split blocks that straddle column borders.

    PaddleOCR aggressively clusters nearby text pixels. In tightly-formatted
    tables, it frequently merges text from two adjacent cells into one box.
    This function detects those cases by checking if the block crosses a
    vertical grid line, then splits the text proportionally at the intersection.

    A 3-character look-ahead buffer snaps to the nearest word boundary to
    avoid splitting mid-word.
    """
    sliced: list[TextBlock] = []

    for block in blocks:
        # Narrow blocks can't span two columns — skip the expensive check
        if block.width < MIN_SPLIT_WIDTH:
            sliced.append(block)
            continue

        # Scan for vertical line pixels within the block's horizontal span
        # (offset from edges to avoid detecting the column's own border)
        line_crossings = [
            x for x in range(block.x_min + EDGE_INSET, block.x_max - EDGE_INSET)
            if v_lines[int(block.cy), x] > LINE_INTENSITY_THRESHOLD
        ]

        if not line_crossings:
            sliced.append(block)
            continue

        # Average crossing position gives us the split column's X coordinate
        split_x = int(np.mean(line_crossings))
        text_left, text_right = _split_text_at(block.text, split_x, block.x_min, block.x_max)

        if text_left:
            sliced.append(TextBlock(text_left, block.x_min, block.y_min, split_x - 2, block.y_max))
        if text_right:
            sliced.append(TextBlock(text_right, split_x + 2, block.y_min, block.x_max, block.y_max))

        logger.info("Sliced merged block at X:%d -> ['%s', '%s']", split_x, text_left, text_right)

    return sliced


def _split_text_at(text: str, split_x: int, x_min: int, x_max: int) -> tuple[str, str]:
    """Split a text string proportionally based on a pixel split position.

    Uses the ratio of (split_x - x_min) / block_width to estimate where
    in the character sequence the column boundary falls, then snaps to
    the nearest whitespace within a 3-character window to avoid mid-word cuts.
    """
    ratio = (split_x - x_min) / float(x_max - x_min)
    split_index = int(len(text) * ratio)

    # Look for a word boundary near the proportional split point
    space_idx = text.rfind(' ', 0, split_index + 3)
    if space_idx == -1:
        space_idx = text.find(' ', split_index - 3)

    if space_idx != -1 and 0 < space_idx < len(text) - 1:
        return text[:space_idx].strip(), text[space_idx:].strip()

    # No word boundary found — hard split at the proportional index
    return text[:split_index].strip(), text[split_index:].strip()
