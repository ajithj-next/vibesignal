"""Tests for notebook parsing functionality."""

import tempfile
from pathlib import Path

import pytest

from vibesignal import NotebookParseError
from vibesignal.parsing.notebook import NotebookParser
from vibesignal.parsing.extractors import ImageExtractor


class TestNotebookParser:
    """Tests for NotebookParser class."""

    def test_parse_valid_notebook(self):
        """Test parsing a valid notebook."""
        parser = NotebookParser()
        notebook_path = Path(__file__).parent.parent / "fixtures" / "sample_notebook.ipynb"

        parsed = parser.parse(notebook_path)

        assert parsed.filepath == notebook_path
        assert len(parsed.cells) > 0
        assert parsed.metadata is not None

        # Check we have markdown and code cells
        cell_types = {cell.cell_type for cell in parsed.cells}
        assert "markdown" in cell_types
        assert "code" in cell_types

    def test_parse_nonexistent_file(self):
        """Test parsing a nonexistent file raises error."""
        parser = NotebookParser()

        with pytest.raises(NotebookParseError, match="not found"):
            parser.parse(Path("/nonexistent/notebook.ipynb"))

    def test_parse_non_notebook_file(self):
        """Test parsing a non-.ipynb file raises error."""
        parser = NotebookParser()

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            temp_path = Path(f.name)

        try:
            with pytest.raises(NotebookParseError, match="not a Jupyter notebook"):
                parser.parse(temp_path)
        finally:
            temp_path.unlink()

    def test_parse_extracts_cell_content(self):
        """Test that cell content is properly extracted."""
        parser = NotebookParser()
        notebook_path = Path(__file__).parent.parent / "fixtures" / "sample_notebook.ipynb"

        parsed = parser.parse(notebook_path)

        # Find the first markdown cell
        markdown_cells = [c for c in parsed.cells if c.cell_type == "markdown"]
        assert len(markdown_cells) > 0

        first_md = markdown_cells[0]
        assert "First Principles" in first_md.source

        # Find code cells
        code_cells = [c for c in parsed.cells if c.cell_type == "code"]
        assert len(code_cells) > 0


class TestImageExtractor:
    """Tests for ImageExtractor class."""

    def test_extract_images_from_notebook(self):
        """Test extracting images from a notebook."""
        parser = NotebookParser()
        extractor = ImageExtractor()

        notebook_path = Path(__file__).parent.parent / "fixtures" / "sample_notebook.ipynb"
        parsed = parser.parse(notebook_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            images = extractor.extract_images(parsed, tmpdir)

            # Sample notebook has one image output
            assert len(images) >= 1

            # Check image properties
            if images:
                img = images[0]
                assert img.format in ("png", "jpeg", "jpg", "svg")
                assert img.filename
                assert len(img.data) > 0

                # Check image was saved
                saved_path = Path(tmpdir) / img.filename
                assert saved_path.exists()

    def test_extract_images_creates_output_dir(self):
        """Test that extract_images creates the output directory."""
        extractor = ImageExtractor()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "new_dir" / "images"
            assert not output_dir.exists()

            # Create a minimal parsed notebook
            from vibesignal.models import ParsedNotebook

            notebook = ParsedNotebook(
                filepath=Path("test.ipynb"),
                cells=[],
                images=[],
            )

            extractor.extract_images(notebook, output_dir)
            assert output_dir.exists()
