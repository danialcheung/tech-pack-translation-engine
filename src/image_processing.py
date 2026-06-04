"""Image processing: grid detection, candidate boxes, whiteout & heal.

This module handles the computer vision pipeline that doesn't involve OCR:
- Extracting the structural table grid (horizontal + vertical lines)
- Detecting candidate content regions to identify the "Forbidden Zone" (artwork)
- Erasing original English text and restoring grid lines over the cleared areas
"""

import logging

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from src.models import TextBlock

logger = logging.getLogger(__name__)

# --- Morphological line detection ---
# Kernel widths for isolating horizontal/vertical structures.
# 25px is long enough to capture table borders but short enough
# to ignore stray marks or text strokes.
H_KERNEL_WIDTH = 25
V_KERNEL_HEIGHT = 25

# Binarization threshold — pixels darker than this are considered "ink"
GRID_THRESHOLD = 150

# --- Candidate box filtering ---
# Minimum contour area to be considered a meaningful layout region
MIN_BOX_AREA = 5000
# Aspect ratio bounds to filter out extreme shapes (very tall/wide lines)
MIN_ASPECT_RATIO = 0.3
MAX_ASPECT_RATIO = 3.0

# --- Edge detection for candidate scoring ---
CANNY_LOW = 50
CANNY_HIGH = 150
DILATE_KERNEL_SIZE = 7

# Fallback forbidden zone if no candidate boxes are found
# (approximate center-right region where artwork typically lives)
DEFAULT_FORBIDDEN_ZONE = [600, 50, 1150, 450]


def extract_structural_grid(
    cv_img: NDArray[np.uint8],
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """Isolate the table's structural grid lines using directional morphology.

    This produces two masks:
    - combined_grid: all horizontal + vertical lines (used for grid restoration)
    - v_lines: vertical lines only (used by the Spatial Guillotine and DNT filtering)

    The technique works by opening the binary image with elongated kernels —
    horizontal kernels preserve only horizontal strokes, vertical kernels
    preserve only vertical strokes. This effectively strips all text while
    keeping the table skeleton intact.
    """
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, GRID_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (H_KERNEL_WIDTH, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)

    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, V_KERNEL_HEIGHT))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    return cv2.add(h_lines, v_lines), v_lines


def generate_candidate_boxes(cv_img: NDArray[np.uint8]) -> list[dict]:
    """Detect candidate content regions via multi-channel edge analysis.

    Splits the image into B/G/R channels and runs Canny edge detection on each,
    then merges results. This catches edges that might be invisible in a single
    grayscale conversion (e.g. colored artwork on a colored background).

    Each candidate is scored by edge density — the region with the highest
    density of edges is most likely the main design artwork (complex linework).

    Returns sorted list of box dicts: {coords, score, area}.
    """
    img_h, img_w = cv_img.shape[:2]

    # Multi-channel edge detection captures colored artwork better than grayscale
    b, g, r = cv2.split(cv_img)
    edges = cv2.bitwise_or(
        cv2.Canny(b, CANNY_LOW, CANNY_HIGH),
        cv2.bitwise_or(
            cv2.Canny(g, CANNY_LOW, CANNY_HIGH),
            cv2.Canny(r, CANNY_LOW, CANNY_HIGH),
        )
    )

    # Dilate to close small gaps in edge contours, forming solid regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (DILATE_KERNEL_SIZE, DILATE_KERNEL_SIZE))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidate_boxes: list[dict] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h

        # Skip tiny noise regions and full-page bounding boxes
        if area < MIN_BOX_AREA or (w >= img_w - 20 and h >= img_h - 20):
            continue
        # Skip extreme aspect ratios (likely table borders, not content)
        if not (MIN_ASPECT_RATIO <= w / float(h) <= MAX_ASPECT_RATIO):
            continue

        candidate_boxes.append({
            "coords": [x, y, x + w, y + h],
            "score": cv2.countNonZero(edges[y:y+h, x:x+w]) / float(area),
            "area": area,
        })

    # Sort by edge density — highest scoring box is likely the artwork
    candidate_boxes.sort(key=lambda item: item["score"], reverse=True)
    return candidate_boxes


def compute_forbidden_zone(candidate_boxes: list[dict], padding: int = 10) -> list[int]:
    """Determine the "Forbidden Zone" — the artwork region to protect from OCR.

    The highest-scoring candidate box (densest edges) is assumed to be the
    main technical drawing/fashion sketch. We pad it slightly to give
    a safety margin against edge-case text that hugs the artwork border.
    """
    if candidate_boxes:
        c = candidate_boxes[0]["coords"]
        return [c[0] - padding, c[1] - padding, c[2] + padding, c[3] + padding]
    return DEFAULT_FORBIDDEN_ZONE


def whiteout_and_heal(
    pil_img: Image.Image,
    cv_img: NDArray[np.uint8],
    blocks: list[TextBlock],
    grid_mask: NDArray[np.uint8],
) -> Image.Image:
    """Surgical text removal with grid line restoration.

    Two-phase process:
    1. WHITEOUT: Draw white rectangles over each text block's bounding box,
       erasing the original English text. This inevitably destroys any table
       grid lines that pass through those regions.
    2. HEAL: Blend the pre-extracted grid line mask back on top, restoring
       all horizontal/vertical borders to their original pixel positions.
       This ensures the final output maintains continuous cell borders.
    """
    clean_draw = ImageDraw.Draw(pil_img)
    for block in blocks:
        # 2px padding ensures no stray edge pixels remain
        clean_draw.rectangle(
            [block.x_min - 2, block.y_min - 2, block.x_max + 2, block.y_max + 2],
            fill="white",
        )

    # Convert back to OpenCV space for pixel-level blending
    chewed_cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Where grid_mask is white (255), use the ORIGINAL image pixels (intact lines).
    # Everywhere else, use the whited-out image (cleared text regions).
    healed_cv_img = np.where(grid_mask[:, :, None] == 255, cv_img, chewed_cv_img)

    return Image.fromarray(cv2.cvtColor(healed_cv_img, cv2.COLOR_BGR2RGB))
