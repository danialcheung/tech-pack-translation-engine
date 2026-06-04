"""Data models used across the pipeline.

TextBlock is the core data unit — it represents a single OCR-detected text region
with its pixel bounding box. Used by every stage from extraction through rendering.
"""

from dataclasses import dataclass


@dataclass
class TextBlock:
    """A detected text block with its bounding box coordinates.

    Coordinates are in absolute pixel space relative to the source image.
    The center (cx, cy) is used for spatial membership tests — e.g. determining
    which table column a block belongs to, or whether it falls inside the
    forbidden zone (design artwork area).
    """
    text: str
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def cx(self) -> float:
        """Horizontal center — used for column zone membership checks."""
        return (self.x_min + self.x_max) / 2

    @property
    def cy(self) -> float:
        """Vertical center — used for row/zone membership checks."""
        return (self.y_min + self.y_max) / 2

    @property
    def width(self) -> int:
        """Box width in pixels (min 1 to avoid division by zero)."""
        return max(1, self.x_max - self.x_min)

    @property
    def height(self) -> int:
        """Box height in pixels (min 1 to avoid division by zero)."""
        return max(1, self.y_max - self.y_min)
