"""Text rendering onto images.

After the original English text has been erased and grid lines restored,
this module draws the translated text back onto the canvas. It uses a
median-height-based font sizing strategy to maintain visual consistency
across all text blocks, then scales down individual blocks if the
translated string is wider than the original bounding box.
"""

import logging
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.models import TextBlock

logger = logging.getLogger(__name__)

# Font path relative to the project root (one level up from src/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
_FONT_PATH = os.path.join(_PROJECT_ROOT, "fonts", "simhei.ttf")

# Font sizing parameters
FONT_SCALE_FACTOR = 0.75   # Font size as fraction of box height
MIN_FONT_SIZE = 8           # Floor to keep text readable
MIN_MEDIAN_HEIGHT = 10      # Safety floor for median calculation


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load the SimHei CJK font at the given size.

    Falls back to Pillow's built-in bitmap font if the TTF file
    isn't found (e.g. during testing without the fonts/ directory).
    """
    try:
        return ImageFont.truetype(_FONT_PATH, size)
    except IOError:
        logger.warning("Font not found at '%s', using default.", _FONT_PATH)
        return ImageFont.load_default()


def render_translated_blocks(
    pil_img: Image.Image,
    translated_pairs: list[tuple[TextBlock, str]],
) -> Image.Image:
    """Render pre-translated text onto the image at each block's position.

    Uses a two-step sizing approach:
    1. Compute a "master" font size from the median height of all text boxes.
       This keeps most text visually consistent across the document.
    2. For individual blocks where the translated text is wider than the
       bounding box, scale down the font proportionally to fit.

    Text is vertically centered within each box and left-aligned with a
    2px inset to avoid clipping against the left cell border.
    """
    if not translated_pairs:
        return pil_img

    draw = ImageDraw.Draw(pil_img)

    # Base font size from the median box height across all blocks
    box_heights = [block.height for block, _ in translated_pairs]
    median_h = max(MIN_MEDIAN_HEIGHT, int(np.median(box_heights)))
    master_font_size = int(median_h * FONT_SCALE_FACTOR)

    for block, translated_text in translated_pairs:
        font_size = master_font_size
        font = _load_font(font_size)

        # Measure the rendered width of the translated string
        bbox = draw.textbbox((0, 0), translated_text, font=font)
        text_w = bbox[2] - bbox[0]

        # If translation is wider than the cell, shrink font to fit
        if text_w > block.width and text_w > 0:
            scale = block.width / float(text_w)
            font_size = max(MIN_FONT_SIZE, int(font_size * scale))
            font = _load_font(font_size)
            bbox = draw.textbbox((0, 0), translated_text, font=font)

        # Position: vertically centered in box, left-aligned with 2px inset
        text_h = bbox[3] - bbox[1]
        y_paste = (block.cy - (text_h / 2)) - bbox[1]
        x_paste = (block.x_min + 2) - bbox[0]

        draw.text((x_paste, y_paste), translated_text, fill="black", font=font)

    return pil_img
