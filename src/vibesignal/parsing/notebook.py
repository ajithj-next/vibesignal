"""Jupyter notebook parsing functionality."""

import nbformat
from pathlib import Path
from typing import Any

from vibesignal import NotebookParseError
from vibesignal.models import NotebookCell, ParsedNotebook


class NotebookParser:
    """Parser for Jupyter notebooks.

    Extracts cells and metadata from .ipynb files using nbformat.
    """

    def parse(self, filepath: Path | str) -> ParsedNotebook:
        """Parse a Jupyter notebook file.

        Args:
            filepath: Path to the .ipynb file

        Returns:
            ParsedNotebook: Parsed notebook structure

        Raises:
            NotebookParseError: If parsing fails
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise NotebookParseError(f"Notebook file not found: {filepath}")

        if not filepath.suffix == ".ipynb":
            raise NotebookParseError(f"File is not a Jupyter notebook: {filepath}")

        try:
            # Read notebook using nbformat
            with open(filepath, "r", encoding="utf-8") as f:
                nb = nbformat.read(f, as_version=4)
        except Exception as e:
            raise NotebookParseError(f"Failed to read notebook {filepath}: {e}") from e

        try:
            # Extract cells
            cells = [self._extract_cell_content(cell) for cell in nb.cells]

            # Create ParsedNotebook
            parsed = ParsedNotebook(
                filepath=filepath,
                cells=cells,
                metadata=nb.metadata if hasattr(nb, "metadata") else {},
            )

            return parsed

        except Exception as e:
            raise NotebookParseError(f"Failed to parse notebook {filepath}: {e}") from e

    def _extract_cell_content(self, cell: Any) -> NotebookCell:
        """Extract content from a single notebook cell.

        Args:
            cell: nbformat cell object

        Returns:
            NotebookCell: Parsed cell data
        """
        cell_type = cell.cell_type

        # Get source (can be string or list of strings)
        source = cell.source
        if isinstance(source, list):
            source = "".join(source)

        # Handle markdown cells
        if cell_type == "markdown":
            return NotebookCell(
                cell_type="markdown",
                source=source,
                outputs=[],
                execution_count=None,
            )

        # Handle code cells
        elif cell_type == "code":
            # Extract outputs
            outputs = []
            if hasattr(cell, "outputs"):
                outputs = [self._serialize_output(output) for output in cell.outputs]

            # Get execution count
            execution_count = getattr(cell, "execution_count", None)

            return NotebookCell(
                cell_type="code",
                source=source,
                outputs=outputs,
                execution_count=execution_count,
            )

        # For any other cell types, treat as markdown
        else:
            return NotebookCell(
                cell_type="markdown",
                source=source,
                outputs=[],
                execution_count=None,
            )

    def _serialize_output(self, output: Any) -> dict:
        """Serialize a notebook output to a dictionary.

        Args:
            output: nbformat output object

        Returns:
            dict: Serialized output data
        """
        output_dict: dict[str, Any] = {
            "output_type": output.output_type,
        }

        # Handle different output types
        if output.output_type == "stream":
            output_dict["name"] = getattr(output, "name", "stdout")
            output_dict["text"] = getattr(output, "text", "")

        elif output.output_type in ("display_data", "execute_result"):
            output_dict["data"] = getattr(output, "data", {})
            output_dict["metadata"] = getattr(output, "metadata", {})
            if hasattr(output, "execution_count"):
                output_dict["execution_count"] = output.execution_count

        elif output.output_type == "error":
            output_dict["ename"] = getattr(output, "ename", "")
            output_dict["evalue"] = getattr(output, "evalue", "")
            output_dict["traceback"] = getattr(output, "traceback", [])

        return output_dict

    def _is_reasoning_cell(self, cell: NotebookCell) -> bool:
        """Heuristic to identify potential first-principles reasoning cells.

        This is a simple heuristic - the actual reasoning extraction
        will be done by Claude AI.

        Args:
            cell: Notebook cell to check

        Returns:
            bool: True if cell likely contains reasoning
        """
        if cell.cell_type != "markdown":
            return False

        # Look for reasoning indicators
        reasoning_keywords = [
            "first principle",
            "fundamental",
            "because",
            "therefore",
            "assumption",
            "axiom",
            "proof",
            "derive",
            "reasoning",
            "insight",
        ]

        source_lower = cell.source.lower()
        return any(keyword in source_lower for keyword in reasoning_keywords)
