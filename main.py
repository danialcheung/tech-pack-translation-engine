"""Tech-pack image translation engine — orchestration module.

This is the entry point for the batch translation pipeline. It coordinates
the full flow for each image:
  1. Structural grid extraction (table lines)
  2. Forbidden zone detection (artwork isolation)
  3. OCR text extraction + spatial guillotine (merged cell splitting)
  4. DNT column filtering (protecting technical data)
  5. Surgical whiteout + grid healing (erasing English text)
  6. Translation via Google Cloud API
  7. Rendered text placement (CJK characters back onto canvas)

Usage:
    python main.py                          # defaults: target=zh
    python main.py --target-lang ja         # translate to Japanese
    python main.py --input ./my_images --output ./results
    python main.py --config custom.json
"""

import argparse
import logging
import os
import sys

import cv2
from PIL import Image

from src.config import load_configs, resolve_directories, VALID_IMAGE_EXTENSIONS
from src.ocr import create_ocr_model, extract_text_blocks, slice_merged_blocks
from src.image_processing import (
    extract_structural_grid,
    generate_candidate_boxes,
    compute_forbidden_zone,
    whiteout_and_heal,
)
from src.filtering import filter_translatable_blocks
from src.translation import translate_text
from src.rendering import render_translated_blocks
from src.models import TextBlock

# Silence PaddleOCR's internal C++ logging and protobuf warnings
os.environ["GLOG_minloglevel"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Set up structured logging with timestamps and level indicators.

    Uses force=True to override any existing root logger config
    (e.g. PaddleOCR or other libraries that configure logging at import time).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def _translate_blocks(
    blocks: list[TextBlock],
    target_lang: str,
) -> list[tuple[TextBlock, str]]:
    """Translate all text blocks via the Google Cloud API.

    Decoupled from rendering so that translation and layout are independent
    concerns — makes it possible to test rendering with mock translations.
    """
    pairs: list[tuple[TextBlock, str]] = []
    for block in blocks:
        translated = translate_text(block.text, target_lang=target_lang)
        pairs.append((block, translated))
    return pairs


def process_image(
    img_path: str,
    ocr_model,
    dnt_headers: list[str],
    output_dir: str,
    target_lang: str,
) -> None:
    """Run the full translation pipeline on a single tech pack image."""
    filename = os.path.basename(img_path)

    cv_img = cv2.imread(img_path)
    if cv_img is None:
        logger.error("OpenCV could not read '%s'. Skipping.", filename)
        return

    pil_img = Image.open(img_path).convert("RGB")

    # Phase 1: Extract the structural grid (table borders) for later healing
    grid_mask, v_lines = extract_structural_grid(cv_img)

    # Phase 2: Detect candidate regions and isolate the artwork "Forbidden Zone"
    candidate_boxes = generate_candidate_boxes(cv_img)
    forbidden_zone = compute_forbidden_zone(candidate_boxes)
    largest_boxes = sorted(candidate_boxes, key=lambda b: b["area"], reverse=True)

    # Phase 3: OCR — extract text blocks, then split any that span column borders
    raw_blocks = extract_text_blocks(ocr_model, img_path, forbidden_zone)
    if not raw_blocks:
        logger.warning("No text found in '%s'. Skipping.", filename)
        return
    raw_blocks = slice_merged_blocks(raw_blocks, v_lines)

    # Phase 4: Filter out blocks that fall in protected DNT columns
    blocks_to_translate = filter_translatable_blocks(
        raw_blocks, dnt_headers, v_lines, largest_boxes
    )

    # Phase 5: Erase original text and restore grid lines over cleared areas
    pil_img = whiteout_and_heal(pil_img, cv_img, blocks_to_translate, grid_mask)

    # Phase 6 & 7: Translate and render
    logger.info("Rendering %d targeted text blocks...", len(blocks_to_translate))
    translated_pairs = _translate_blocks(blocks_to_translate, target_lang)
    pil_img = render_translated_blocks(pil_img, translated_pairs)

    # Save final output
    base_name = os.path.splitext(filename)[0]
    output_path = os.path.join(output_dir, f"translated_{base_name}.png")
    pil_img.save(output_path)
    logger.info("Complete! Saved to: %s", output_path)


def run_translation_engine(
    target_lang: str = 'zh',
    input_dir: str | None = None,
    output_dir: str | None = None,
    config_path: str = "terms.json",
) -> None:
    """Main batch processing pipeline.

    Loads config, discovers images, initializes the OCR model once,
    then iterates through each image in the input directory.
    """
    logger.info("INITIALIZING BATCH TRANSLATION ENGINE (Target: %s)", target_lang)

    dnt_headers = load_configs(config_path)

    # Use explicit dirs if provided, otherwise auto-resolve (Docker vs local)
    if input_dir is None or output_dir is None:
        resolved_input, resolved_output = resolve_directories()
        input_dir = input_dir or resolved_input
        output_dir = output_dir or resolved_output

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_dir):
        logger.error("Input directory '%s' not found.", input_dir)
        return

    image_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(VALID_IMAGE_EXTENSIONS)
    ]

    if not image_files:
        logger.error("No images found in '%s'.", input_dir)
        return

    logger.info("Found %d image(s) to process.", len(image_files))

    # Load the OCR model once — reused across all images to avoid cold-start overhead
    logger.info("Loading PaddleOCR AI Models into memory...")
    ocr_model = create_ocr_model()

    for img_index, img_path in enumerate(image_files, 1):
        filename = os.path.basename(img_path)
        logger.info("PROCESSING [%d/%d]: %s", img_index, len(image_files), filename)
        process_image(img_path, ocr_model, dnt_headers, output_dir, target_lang)

    logger.info("BATCH PROCESSING COMPLETE!")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for flexible execution."""
    parser = argparse.ArgumentParser(
        description="Translate tech pack images from English to a target language.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--target-lang",
        default="zh",
        help="Target language code (default: zh). Examples: ja, ko, fr, es.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Input directory path. Auto-detects if not specified.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory path. Auto-detects if not specified.",
    )
    parser.add_argument(
        "--config",
        default="terms.json",
        help="Path to terms.json config file (default: terms.json).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    _configure_logging()
    args = parse_args()
    run_translation_engine(
        target_lang=args.target_lang,
        input_dir=args.input,
        output_dir=args.output,
        config_path=args.config,
    )
