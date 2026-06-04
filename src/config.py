"""Configuration loading and constants.

The terms.json file decouples business terminology from source code, allowing
factory teams or product managers to update DNT (Do-Not-Translate) vocabulary
without code redeployments. See README for full config schema.
"""

import json
import logging
import os
from typing import Tuple

logger = logging.getLogger(__name__)

VALID_IMAGE_EXTENSIONS: Tuple[str, ...] = ('.png', '.jpg', '.jpeg')

# Hardcoded fallback DNT headers if config file is missing or malformed
_DEFAULT_DNT_HEADERS = ["color details", "quantity", "cost", "unit of measure"]


def load_configs(path: str = "terms.json") -> list[str]:
    """Load do-not-translate header names from the JSON config file.

    These headers identify table columns whose data should remain untranslated
    (e.g. material codes, sizing markers, cost figures). The engine will
    dynamically detect these columns at runtime and mask their vertical corridors.
    """
    if not os.path.exists(path):
        logger.warning("Config not found at '%s', using hardcoded defaults.", path)
        return _DEFAULT_DNT_HEADERS.copy()

    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error("Failed to parse config '%s': %s. Using defaults.", path, e)
        return _DEFAULT_DNT_HEADERS.copy()

    # Normalize to lowercase for case-insensitive matching against OCR output
    headers = [str(h).lower() for h in data.get("dnt_headers", [])]
    if not headers:
        logger.warning("No 'dnt_headers' found in '%s', using defaults.", path)
        return _DEFAULT_DNT_HEADERS.copy()

    return headers


def resolve_directories() -> Tuple[str, str]:
    """Resolve input/output directories with Docker and local fallbacks.

    In Docker, volumes are mounted at /input and /output.
    Locally, we look for ./input and ./output relative to CWD,
    then fall back to ../input and ../output for nested execution.
    """
    if os.path.exists("/input"):
        input_dir = "/input"
    elif os.path.exists("input"):
        input_dir = "input"
    else:
        input_dir = "../input"

    if os.path.exists("/output"):
        output_dir = "/output"
    elif os.path.exists("output"):
        output_dir = "output"
    else:
        output_dir = "../output"

    return input_dir, output_dir
