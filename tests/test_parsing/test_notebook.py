"""Tests for notebook parsing functionality."""

import tempfile
from pathlib import Path

import pytest

from vibesignal import NotebookParseError
from vibesignal.parsing.notebook import NotebookParser
from vibesignal.parsing.extractors import ImageExtractor, EquationExtractor
from vibesignal.rendering import EquationStyle


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


class TestEquationExtractor:
    """Tests for EquationExtractor class."""

    def test_extract_equations_from_notebook(self):
        """Test extracting equations from a notebook with LaTeX."""
        parser = NotebookParser()
        extractor = EquationExtractor()

        notebook_path = Path(__file__).parent.parent / "fixtures" / "notebook_with_equations.ipynb"
        parsed = parser.parse(notebook_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            equations = extractor.extract_equations(parsed, tmpdir)

            # Notebook has 3 display equations ($$...$$)
            assert len(equations) >= 3

            # Check equation image properties
            for eq in equations:
                assert eq.format == "png"
                assert eq.filename.startswith("eq_")
                assert len(eq.data) > 0
                assert eq.caption is not None

                # Check image was saved
                saved_path = Path(tmpdir) / eq.filename
                assert saved_path.exists()

    def test_extract_equations_creates_output_dir(self):
        """Test that extract_equations creates the output directory."""
        extractor = EquationExtractor()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "new_dir" / "equations"
            assert not output_dir.exists()

            # Create a minimal parsed notebook
            from vibesignal.models import ParsedNotebook

            notebook = ParsedNotebook(
                filepath=Path("test.ipynb"),
                cells=[],
                images=[],
            )

            extractor.extract_equations(notebook, output_dir)
            assert output_dir.exists()

    def test_extract_equations_skips_short_equations(self):
        """Test that very short equations are skipped."""
        extractor = EquationExtractor(min_equation_length=5)

        from vibesignal.models import ParsedNotebook, NotebookCell

        notebook = ParsedNotebook(
            filepath=Path("test.ipynb"),
            cells=[
                NotebookCell(
                    cell_type="markdown",
                    source="Here is $x$ and also $x + y + z = 10$",
                    outputs=[],
                )
            ],
            images=[],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            equations = extractor.extract_equations(notebook, tmpdir)

            # Only the longer equation should be extracted
            assert len(equations) == 1

    def test_extract_equations_handles_display_equations(self):
        """Test extraction of display equations ($$...$$)."""
        extractor = EquationExtractor()

        from vibesignal.models import ParsedNotebook, NotebookCell

        notebook = ParsedNotebook(
            filepath=Path("test.ipynb"),
            cells=[
                NotebookCell(
                    cell_type="markdown",
                    source="The famous equation:\n\n$$E = mc^2$$\n\nIs well known.",
                    outputs=[],
                )
            ],
            images=[],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            equations = extractor.extract_equations(notebook, tmpdir)

            assert len(equations) == 1
            assert "E = mc^2" in equations[0].caption

    def test_extract_equations_ignores_code_cells(self):
        """Test that code cells are ignored."""
        extractor = EquationExtractor()

        from vibesignal.models import ParsedNotebook, NotebookCell

        notebook = ParsedNotebook(
            filepath=Path("test.ipynb"),
            cells=[
                NotebookCell(
                    cell_type="code",
                    source="# This has $\\alpha$ in a comment",
                    outputs=[],
                ),
                NotebookCell(
                    cell_type="markdown",
                    source="Real equation: $\\beta + \\gamma$",
                    outputs=[],
                )
            ],
            images=[],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            equations = extractor.extract_equations(notebook, tmpdir)

            # Only the markdown cell equation should be extracted
            assert len(equations) == 1

    def test_get_equation_count(self):
        """Test counting equations without rendering."""
        extractor = EquationExtractor()

        from vibesignal.models import ParsedNotebook, NotebookCell

        notebook = ParsedNotebook(
            filepath=Path("test.ipynb"),
            cells=[
                NotebookCell(
                    cell_type="markdown",
                    source="Equations: $\\alpha$, $\\beta$, $$\\gamma + \\delta$$",
                    outputs=[],
                )
            ],
            images=[],
        )

        count = extractor.get_equation_count(notebook)
        assert count == 3

    def test_generate_title_recognizes_common_equations(self):
        """Test that common equation patterns get meaningful titles."""
        extractor = EquationExtractor()

        # Test various equation patterns
        assert extractor._generate_title("e^{i\\pi} + 1 = 0", "display") == "Euler's Formula"
        assert extractor._generate_title("\\int_0^1 f(x) dx", "display") == "Integral"
        assert extractor._generate_title("\\sum_{n=1}^N x_n", "display") == "Summation"
        assert extractor._generate_title("\\nabla f", "inline") == "Gradient"
        assert extractor._generate_title("\\frac{df}{dx}", "inline") == "Derivative"
        assert extractor._generate_title("x + y = z", "inline") == "Equation"

    def test_equation_extractor_with_custom_style(self):
        """Test equation extractor with custom style."""
        extractor = EquationExtractor(style=EquationStyle.DARK)

        from vibesignal.models import ParsedNotebook, NotebookCell

        notebook = ParsedNotebook(
            filepath=Path("test.ipynb"),
            cells=[
                NotebookCell(
                    cell_type="markdown",
                    source="$$\\alpha + \\beta = \\gamma$$",
                    outputs=[],
                )
            ],
            images=[],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            equations = extractor.extract_equations(notebook, tmpdir)

            assert len(equations) == 1
            # Image should be created
            saved_path = Path(tmpdir) / equations[0].filename
            assert saved_path.exists()
