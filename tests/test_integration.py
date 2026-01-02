"""Integration tests for the full VibeSignal pipeline."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import nbformat
import pytest

from vibesignal.workspace import WorkspaceManager


@pytest.fixture
def sample_notebook_file(tmp_path):
    """Create a sample notebook file for testing."""
    # Create a simple notebook with markdown and code cells
    nb = nbformat.v4.new_notebook()

    # Add markdown cells with first principles reasoning
    nb.cells.append(
        nbformat.v4.new_markdown_cell(
            """# First Principles: Linear Regression

## Fundamental Assumption

The core assumption is that there exists a linear relationship between input features and output.

## Derivation

Starting from the assumption that y ≈ wx + b, we can derive the optimal parameters using least squares."""
        )
    )

    # Add code cell
    nb.cells.append(nbformat.v4.new_code_cell("import numpy as np\nimport matplotlib.pyplot as plt"))

    # Add code cell with output (simulating a plot)
    code_cell = nbformat.v4.new_code_cell(
        """x = np.linspace(0, 10, 100)
y = 2 * x + 1 + np.random.randn(100) * 0.5
plt.scatter(x, y)
plt.plot(x, 2*x + 1, 'r-', label='True line')
plt.legend()
plt.show()"""
    )

    # Add a fake image output
    code_cell.outputs = [
        nbformat.v4.new_output(
            "display_data",
            data={
                "image/png": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
            },
        )
    ]

    nb.cells.append(code_cell)

    # Add more markdown
    nb.cells.append(
        nbformat.v4.new_markdown_cell(
            """## Key Insight

The least squares solution minimizes the sum of squared errors, which has a closed-form solution."""
        )
    )

    # Write notebook
    notebook_path = tmp_path / "test_notebook.ipynb"
    with open(notebook_path, "w") as f:
        nbformat.write(nb, f)

    return notebook_path


@pytest.fixture
def mock_claude_response():
    """Mock Claude API response for testing."""
    return {
        "first_principles": [
            "Linear relationship between features and target",
            "Least squares minimization",
        ],
        "insights": [
            {
                "content": "Linear regression assumes a linear relationship",
                "tweet_suggestion": "Core insight: Linear regression assumes features relate linearly to target",
                "importance": "high",
            },
            {
                "content": "Closed-form solution via least squares",
                "tweet_suggestion": "Beautiful math: Least squares has a closed-form solution!",
                "importance": "high",
            },
        ],
        "thread_structure": [
            {
                "position": 1,
                "content": "Let's explore linear regression from first principles (1/4)",
                "image_index": None,
            },
            {
                "position": 2,
                "content": "First principle: Linear relationship between features and target (2/4)",
                "image_index": 0,
            },
            {
                "position": 3,
                "content": "The least squares solution minimizes squared errors (3/4)",
                "image_index": None,
            },
            {
                "position": 4,
                "content": "This gives us a beautiful closed-form solution! (4/4)",
                "image_index": None,
            },
        ],
        "image_mappings": [
            {"image_index": 0, "suggested_position": 2, "relevance": "Shows the linear fit"}
        ],
        "hook": "Ever wondered how linear regression really works?",
    }


class TestFullPipeline:
    """Integration tests for the complete pipeline."""

    @patch("vibesignal.analysis.claude_client.anthropic.Anthropic")
    def test_end_to_end_conversion(
        self, mock_anthropic, sample_notebook_file, mock_claude_response, tmp_path
    ):
        """Test the complete end-to-end conversion pipeline."""
        from vibesignal.analysis.claude_client import ClaudeClient
        from vibesignal.generation.thread import ThreadGenerator
        from vibesignal.output.writer import OutputWriter
        from vibesignal.parsing.extractors import ImageExtractor
        from vibesignal.parsing.notebook import NotebookParser

        # Setup mock Claude client
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        mock_message = Mock()
        mock_message.content = [Mock(text=json.dumps(mock_claude_response))]
        mock_client.messages.create.return_value = mock_message

        # Create workspace for outputs
        workspace = tmp_path / "test_project"
        workspace.mkdir()
        images_dir = workspace / "images"
        images_dir.mkdir()

        # Step 1: Parse notebook
        parser = NotebookParser()
        parsed_notebook = parser.parse(sample_notebook_file)

        assert len(parsed_notebook.cells) > 0
        assert any(cell.cell_type == "markdown" for cell in parsed_notebook.cells)
        assert any(cell.cell_type == "code" for cell in parsed_notebook.cells)

        # Step 2: Extract images
        extractor = ImageExtractor()
        images = extractor.extract_images(parsed_notebook, images_dir)

        assert len(images) >= 1
        assert (images_dir / images[0].filename).exists()

        # Step 3: Analyze with Claude
        claude = ClaudeClient(api_key="test-key")
        insights = claude.analyze_notebook(parsed_notebook)

        assert "first_principles" in insights
        assert "thread_structure" in insights
        assert "hook" in insights

        # Step 4: Generate thread
        generator = ThreadGenerator()
        thread = generator.generate_thread(
            insights=insights,
            images=images,
            source_notebook=str(sample_notebook_file),
            claude_model="claude-sonnet-4-5-20250929",
        )

        assert len(thread.tweets) == 4
        assert thread.hook == "Ever wondered how linear regression really works?"
        assert thread.tweets[1].image_filename  # Second tweet should have image

        # Step 5: Write outputs
        writer = OutputWriter()

        json_path = writer.write(thread, workspace / "thread.json", format="json")
        text_path = writer.write(thread, workspace / "thread.txt", format="text")
        md_path = writer.write(thread, workspace / "thread.md", format="markdown")

        # Verify all outputs exist
        assert json_path.exists()
        assert text_path.exists()
        assert md_path.exists()

        # Verify JSON content
        with open(json_path) as f:
            thread_data = json.load(f)
        assert thread_data["metadata"]["total_tweets"] == 4
        assert len(thread_data["tweets"]) == 4

        # Verify text output has content
        text_content = text_path.read_text()
        assert "TWITTER THREAD" in text_content
        assert "linear regression" in text_content.lower()

        # Verify markdown output
        md_content = md_path.read_text()
        assert "# Twitter Thread" in md_content
        assert "## Hook" in md_content

    def test_workspace_manager_integration(self, sample_notebook_file, tmp_path):
        """Test workspace management integration."""
        workspace_manager = WorkspaceManager(base_dir=tmp_path / "workspaces")

        # Create workspace
        workspace = workspace_manager.create_workspace(
            project_name="test_project",
            source_notebook=sample_notebook_file,
            copy_source=True,
        )

        assert workspace.exists()
        assert (workspace / "images").exists()
        assert (workspace / "source").exists()
        assert (workspace / "metadata.json").exists()

        # Verify source was copied
        assert (workspace / "source" / sample_notebook_file.name).exists()

        # Verify metadata
        with open(workspace / "metadata.json") as f:
            metadata = json.load(f)
        assert metadata["project_name"] == "test_project"
        assert metadata["source_notebook"] == str(sample_notebook_file)

        # Test listing workspaces
        projects = workspace_manager.list_workspaces()
        assert "test_project" in projects

        # Test getting workspace summary
        summary = workspace_manager.get_workspace_summary("test_project")
        assert summary["project_name"] == "test_project"
        assert summary["workspace_path"] == str(workspace)

    @patch("vibesignal.analysis.claude_client.anthropic.Anthropic")
    def test_workspace_with_thread_generation(
        self, mock_anthropic, sample_notebook_file, mock_claude_response, tmp_path
    ):
        """Test workspace integration with full thread generation."""
        from vibesignal.analysis.claude_client import ClaudeClient
        from vibesignal.generation.thread import ThreadGenerator
        from vibesignal.output.writer import OutputWriter
        from vibesignal.parsing.extractors import ImageExtractor
        from vibesignal.parsing.notebook import NotebookParser

        # Setup mock
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        mock_message = Mock()
        mock_message.content = [Mock(text=json.dumps(mock_claude_response))]
        mock_client.messages.create.return_value = mock_message

        # Create workspace
        workspace_manager = WorkspaceManager(base_dir=tmp_path / "projects")
        workspace = workspace_manager.create_workspace(
            project_name="linear_regression",
            source_notebook=sample_notebook_file,
            copy_source=True,
        )

        # Run full pipeline
        parser = NotebookParser()
        parsed_notebook = parser.parse(sample_notebook_file)

        images_dir = workspace_manager.get_images_dir(workspace)
        extractor = ImageExtractor()
        images = extractor.extract_images(parsed_notebook, images_dir)

        claude = ClaudeClient(api_key="test-key")
        insights = claude.analyze_notebook(parsed_notebook)

        generator = ThreadGenerator()
        thread = generator.generate_thread(
            insights=insights,
            images=images,
            source_notebook=str(sample_notebook_file),
            claude_model="claude-sonnet-4-5-20250929",
        )

        # Write all formats
        writer = OutputWriter()
        writer.write(thread, workspace / "thread.json", format="json")
        writer.write(thread, workspace / "thread.txt", format="text")
        writer.write(thread, workspace / "thread.md", format="markdown")

        # Save metadata
        workspace_manager.save_thread_metadata(workspace, thread)

        # Verify workspace structure
        assert (workspace / "thread.json").exists()
        assert (workspace / "thread.txt").exists()
        assert (workspace / "thread.md").exists()
        assert (workspace / "images").exists()
        assert len(list((workspace / "images").glob("*"))) >= 1

        # Verify updated metadata
        with open(workspace / "metadata.json") as f:
            metadata = json.load(f)
        assert metadata["total_tweets"] == 4
        assert metadata["total_images"] >= 1
        assert metadata["hook"] == "Ever wondered how linear regression really works?"

        # Test workspace summary
        summary = workspace_manager.get_workspace_summary("linear_regression")
        assert summary["total_tweets"] == 4
        assert "json" in summary["available_outputs"]
        assert "txt" in summary["available_outputs"]
        assert "md" in summary["available_outputs"]
