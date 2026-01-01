"""Data models for notebook parsing and representation."""

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class NotebookCell(BaseModel):
    """Represents a single notebook cell.

    Attributes:
        cell_type: Type of cell (markdown or code)
        source: Cell content as a string
        outputs: List of cell outputs (for code cells)
        execution_count: Execution number (for code cells)
    """

    cell_type: Literal["markdown", "code"]
    source: str
    outputs: list[dict] = Field(default_factory=list)
    execution_count: Optional[int] = None


class ImageOutput(BaseModel):
    """Represents an extracted image from notebook output.

    Attributes:
        cell_index: Index of the cell containing this image
        output_index: Index of the output within the cell
        format: Image format (png, jpg, svg)
        data: Raw image data as bytes
        filename: Filename for saving the image
        caption: Optional caption extracted from surrounding markdown
    """

    cell_index: int
    output_index: int
    format: str  # png, jpg, svg
    data: bytes
    filename: str
    caption: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ParsedNotebook(BaseModel):
    """Complete parsed notebook structure.

    Attributes:
        filepath: Path to the source notebook file
        cells: List of parsed notebook cells
        images: List of extracted images from outputs
        metadata: Notebook metadata dictionary
    """

    filepath: Path
    cells: list[NotebookCell]
    images: list[ImageOutput] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)
