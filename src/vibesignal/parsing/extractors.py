"""Image and equation extraction from notebook outputs."""

import base64
import hashlib
from pathlib import Path
from typing import Optional

from PIL import Image
import io

from vibesignal import NotebookParseError
from vibesignal.models import ImageOutput, ParsedNotebook
from vibesignal.rendering import MathRenderer, EquationStyle


class ImageExtractor:
    """Extract and save images from notebook cell outputs."""

    # Supported image formats
    SUPPORTED_FORMATS = {"image/png", "image/jpeg", "image/jpg", "image/svg+xml"}

    def extract_images(
        self, notebook: ParsedNotebook, output_dir: Path | str
    ) -> list[ImageOutput]:
        """Extract all images from notebook outputs.

        Args:
            notebook: Parsed notebook to extract images from
            output_dir: Directory to save extracted images

        Returns:
            list[ImageOutput]: List of extracted images with metadata

        Raises:
            NotebookParseError: If image extraction fails
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        images: list[ImageOutput] = []

        for cell_index, cell in enumerate(notebook.cells):
            # Only code cells can have image outputs
            if cell.cell_type != "code":
                continue

            # Extract images from each output
            for output_index, output in enumerate(cell.outputs):
                image = self._extract_from_output(output, cell_index, output_index)
                if image:
                    # Save image to disk
                    saved_path = self._save_image(image, output_dir)
                    image.filename = saved_path.name
                    images.append(image)

        # Update notebook's image list
        notebook.images = images
        return images

    def _extract_from_output(
        self, output: dict, cell_index: int, output_index: int
    ) -> Optional[ImageOutput]:
        """Extract a single image from a cell output.

        Args:
            output: Output dictionary from a cell
            cell_index: Index of the cell
            output_index: Index of the output within the cell

        Returns:
            Optional[ImageOutput]: Extracted image or None if no image found
        """
        # Check if output contains image data
        if output.get("output_type") not in ("display_data", "execute_result"):
            return None

        data = output.get("data", {})

        # Try to extract image in order of preference: PNG, JPEG, SVG
        image_data = None
        format_type = None

        if "image/png" in data:
            image_data = data["image/png"]
            format_type = "png"
        elif "image/jpeg" in data:
            image_data = data["image/jpeg"]
            format_type = "jpeg"
        elif "image/jpg" in data:
            image_data = data["image/jpg"]
            format_type = "jpg"
        elif "image/svg+xml" in data:
            image_data = data["image/svg+xml"]
            format_type = "svg"

        if not image_data:
            return None

        try:
            # Decode base64 image data
            if isinstance(image_data, str):
                # Remove whitespace and newlines
                image_data = image_data.strip().replace("\n", "")
                image_bytes = base64.b64decode(image_data)
            else:
                # Already bytes
                image_bytes = image_data

            # Generate filename using hash of image data
            image_hash = hashlib.md5(image_bytes).hexdigest()[:12]
            filename = f"img_{cell_index}_{output_index}_{image_hash}.{format_type}"

            # Try to extract caption from surrounding markdown
            caption = self._try_extract_caption(cell_index, output_index)

            return ImageOutput(
                cell_index=cell_index,
                output_index=output_index,
                format=format_type,
                data=image_bytes,
                filename=filename,
                caption=caption,
            )

        except Exception as e:
            # Log but don't fail - just skip this image
            print(f"Warning: Failed to extract image from cell {cell_index}: {e}")
            return None

    def _save_image(self, image: ImageOutput, output_dir: Path) -> Path:
        """Save image to disk.

        Args:
            image: Image to save
            output_dir: Directory to save to

        Returns:
            Path: Path to saved image file

        Raises:
            NotebookParseError: If saving fails
        """
        try:
            filepath = output_dir / image.filename

            # For SVG, save directly
            if image.format == "svg":
                with open(filepath, "wb") as f:
                    f.write(image.data)
            else:
                # For raster images, use PIL to validate and save
                img = Image.open(io.BytesIO(image.data))
                img.save(filepath)

            return filepath

        except Exception as e:
            raise NotebookParseError(f"Failed to save image {image.filename}: {e}") from e

    def _try_extract_caption(self, cell_index: int, output_index: int) -> Optional[str]:
        """Try to extract a caption from surrounding markdown cells.

        This is a simple heuristic - we'll improve it later if needed.

        Args:
            cell_index: Index of the cell with the image
            output_index: Index of the output

        Returns:
            Optional[str]: Extracted caption or None
        """
        # For MVP, we'll skip caption extraction
        # This can be enhanced later by looking at surrounding markdown cells
        return None


class EquationExtractor:
    """Extract and render equations from notebook markdown cells.

    Finds LaTeX equations in markdown cells ($...$, $$...$$) and renders
    them as beautiful equation cards using MathRenderer.
    """

    def __init__(
        self,
        style: EquationStyle = EquationStyle.VIBESIGNAL,
        min_equation_length: int = 3,
    ):
        """Initialize equation extractor.

        Args:
            style: Visual style for rendered equation cards
            min_equation_length: Minimum LaTeX length to consider as equation
        """
        self.style = style
        self.min_equation_length = min_equation_length

    def extract_equations(
        self, notebook: ParsedNotebook, output_dir: Path | str
    ) -> list[ImageOutput]:
        """Extract and render all equations from notebook markdown cells.

        Args:
            notebook: Parsed notebook to extract equations from
            output_dir: Directory to save rendered equation images

        Returns:
            list[ImageOutput]: List of rendered equation images with metadata
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize renderer
        renderer = MathRenderer(output_dir=output_dir, style=self.style)

        images: list[ImageOutput] = []
        equation_count = 0

        for cell_index, cell in enumerate(notebook.cells):
            # Only process markdown cells
            if cell.cell_type != "markdown":
                continue

            # Extract equations from cell source
            equations = renderer.extract_equations(cell.source)

            for eq in equations:
                # Skip very short equations (likely not meaningful)
                if len(eq['latex']) < self.min_equation_length:
                    continue

                # Generate a descriptive title based on equation type
                title = self._generate_title(eq['latex'], eq['type'])

                try:
                    # Render equation as a card
                    image_path = renderer.render_equation_card(
                        latex=eq['latex'],
                        title=title,
                        filename=f"eq_{cell_index}_{equation_count}.png",
                    )

                    # Read the rendered image data
                    with open(image_path, 'rb') as f:
                        image_data = f.read()

                    # Create ImageOutput
                    image = ImageOutput(
                        cell_index=cell_index,
                        output_index=equation_count,  # Use equation index as output_index
                        format="png",
                        data=image_data,
                        filename=image_path.name,
                        caption=f"Equation: {eq['latex'][:50]}{'...' if len(eq['latex']) > 50 else ''}",
                    )
                    images.append(image)
                    equation_count += 1

                except Exception as e:
                    # Log but don't fail - just skip this equation
                    print(f"Warning: Failed to render equation in cell {cell_index}: {e}")
                    continue

        return images

    def _generate_title(self, latex: str, eq_type: str) -> str:
        """Generate a descriptive title for an equation.

        Args:
            latex: The LaTeX equation string
            eq_type: Type of equation ('display' or 'inline')

        Returns:
            str: A descriptive title for the equation card
        """
        # Check for common equation patterns and give them meaningful titles
        latex_lower = latex.lower().replace(" ", "")

        # Common equation patterns
        if "e^{i" in latex_lower or "e^i" in latex_lower:
            return "Euler's Formula"
        elif "=mc^2" in latex_lower or "=mc²" in latex_lower:
            return "Mass-Energy Equivalence"
        elif "\\frac{d" in latex or "\\frac{\\partial" in latex:
            return "Derivative"
        elif "\\int" in latex:
            return "Integral"
        elif "\\sum" in latex:
            return "Summation"
        elif "\\prod" in latex:
            return "Product"
        elif "\\lim" in latex:
            return "Limit"
        elif "\\nabla" in latex:
            return "Gradient"
        elif "p(a|b)" in latex_lower or "p(b|a)" in latex_lower:
            return "Bayes' Theorem"
        elif "\\sigma" in latex and "e^" in latex:
            return "Softmax Function"
        elif "\\log" in latex or "\\ln" in latex:
            return "Logarithmic Expression"
        elif "\\sqrt" in latex:
            return "Root Expression"
        elif "\\matrix" in latex or "\\begin{bmatrix}" in latex:
            return "Matrix Expression"
        elif "=" in latex:
            return "Equation"
        else:
            return "Mathematical Expression"

    def get_equation_count(self, notebook: ParsedNotebook) -> int:
        """Count the number of equations in a notebook without rendering.

        Args:
            notebook: Parsed notebook to count equations in

        Returns:
            int: Number of equations found
        """
        renderer = MathRenderer()  # Just for extraction, no rendering
        count = 0

        for cell in notebook.cells:
            if cell.cell_type == "markdown":
                equations = renderer.extract_equations(cell.source)
                count += len([eq for eq in equations
                             if len(eq['latex']) >= self.min_equation_length])

        return count
