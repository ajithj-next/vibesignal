"""Tests for Claude API client."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from vibesignal import ClaudeAPIError
from vibesignal.analysis.claude_client import ClaudeClient
from vibesignal.analysis.insights import InsightExtractor
from vibesignal.models import NotebookCell, ParsedNotebook


@pytest.fixture
def sample_notebook():
    """Create a sample parsed notebook for testing."""
    return ParsedNotebook(
        filepath=Path("test.ipynb"),
        cells=[
            NotebookCell(
                cell_type="markdown",
                source="# First Principles of k-NN\n\nDistance is meaning.",
            ),
            NotebookCell(
                cell_type="code",
                source="import numpy as np",
                outputs=[],
            ),
        ],
        images=[],
        metadata={},
    )


@pytest.fixture
def sample_claude_response():
    """Sample Claude API response."""
    return {
        "first_principles": [
            "Distance represents similarity in feature space",
            "Nearest neighbors share properties",
        ],
        "insights": [
            {
                "content": "k-NN makes predictions based on proximity",
                "tweet_suggestion": "Core insight: k-NN predicts by looking at nearby examples",
                "importance": "high",
            }
        ],
        "thread_structure": [
            {"position": 1, "content": "Let's explore k-NN from first principles", "image_index": None},
            {
                "position": 2,
                "content": "Distance is meaning - similar items cluster together",
                "image_index": 0,
            },
        ],
        "image_mappings": [{"image_index": 0, "suggested_position": 2, "relevance": "Shows clustering"}],
        "hook": "Ever wondered how k-NN really works?",
    }


class TestClaudeClient:
    """Tests for ClaudeClient class."""

    @patch("vibesignal.analysis.claude_client.anthropic.Anthropic")
    def test_analyze_notebook_success(self, mock_anthropic, sample_notebook, sample_claude_response):
        """Test successful notebook analysis."""
        # Setup mock
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        mock_message = Mock()
        mock_message.content = [Mock(text=json.dumps(sample_claude_response))]
        # Mock usage data for metrics tracking
        mock_message.usage = Mock()
        mock_message.usage.input_tokens = 1000
        mock_message.usage.output_tokens = 500
        mock_message.usage.cache_creation_input_tokens = 0
        mock_message.usage.cache_read_input_tokens = 0
        mock_message.id = "msg_test_123"
        mock_client.messages.create.return_value = mock_message

        # Test
        client = ClaudeClient(api_key="test-key")
        result = client.analyze_notebook(sample_notebook)

        # Verify
        assert result["first_principles"] == sample_claude_response["first_principles"]
        assert result["hook"] == sample_claude_response["hook"]
        assert len(result["thread_structure"]) == 2

        # Verify metrics were captured
        metrics = client.get_last_call_metrics()
        assert metrics is not None
        assert metrics.usage.input_tokens == 1000
        assert metrics.usage.output_tokens == 500

    @patch("vibesignal.analysis.claude_client.anthropic.Anthropic")
    def test_analyze_notebook_with_markdown_json(self, mock_anthropic, sample_notebook, sample_claude_response):
        """Test handling JSON wrapped in markdown code blocks."""
        # Setup mock
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        # Response with markdown wrapper
        markdown_response = f"Here's the analysis:\n```json\n{json.dumps(sample_claude_response)}\n```"
        mock_message = Mock()
        mock_message.content = [Mock(text=markdown_response)]
        # Mock usage data for metrics tracking
        mock_message.usage = Mock()
        mock_message.usage.input_tokens = 800
        mock_message.usage.output_tokens = 400
        mock_message.usage.cache_creation_input_tokens = 0
        mock_message.usage.cache_read_input_tokens = 0
        mock_message.id = None
        mock_client.messages.create.return_value = mock_message

        # Test
        client = ClaudeClient(api_key="test-key")
        result = client.analyze_notebook(sample_notebook)

        # Verify
        assert result["hook"] == sample_claude_response["hook"]

    @patch("vibesignal.analysis.claude_client.anthropic.Anthropic")
    def test_analyze_notebook_api_error(self, mock_anthropic, sample_notebook):
        """Test handling of API errors."""
        # Setup mock to raise a generic exception
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("API Error")

        # Test
        client = ClaudeClient(api_key="test-key")
        with pytest.raises(ClaudeAPIError, match="Failed to analyze notebook"):
            client.analyze_notebook(sample_notebook)

    def test_format_notebook_content(self, sample_notebook):
        """Test notebook content formatting."""
        client = ClaudeClient(api_key="test-key")
        content = client._format_notebook_content(sample_notebook)

        assert "MARKDOWN CELL" in content
        assert "CODE CELL" in content
        assert "First Principles" in content
        assert "import numpy" in content


class TestInsightExtractor:
    """Tests for InsightExtractor class."""

    def test_extract_insights_success(self, sample_claude_response):
        """Test successful insight extraction."""
        extractor = InsightExtractor()
        result = extractor.extract_insights(sample_claude_response)

        assert "first_principles" in result
        assert "insights" in result
        assert "thread_structure" in result
        assert "hook" in result

    def test_extract_insights_missing_fields(self):
        """Test handling of missing required fields."""
        extractor = InsightExtractor()

        with pytest.raises(ValueError, match="Missing required fields"):
            extractor.extract_insights({"first_principles": []})

    def test_validate_thread_structure_empty(self):
        """Test validation rejects empty thread structure."""
        extractor = InsightExtractor()

        with pytest.raises(ValueError, match="cannot be empty"):
            extractor._validate_thread_structure([])

    def test_validate_thread_structure_fixes_positions(self):
        """Test that validator fixes incorrect positions."""
        extractor = InsightExtractor()

        structure = [
            {"position": 5, "content": "First tweet"},
            {"position": 10, "content": "Second tweet"},
        ]

        result = extractor._validate_thread_structure(structure)

        assert result[0]["position"] == 1
        assert result[1]["position"] == 2

    def test_validate_hook_empty(self):
        """Test validation rejects empty hook."""
        extractor = InsightExtractor()

        with pytest.raises(ValueError, match="cannot be empty"):
            extractor._validate_hook("")

    def test_validate_hook_too_long(self):
        """Test hook truncation."""
        extractor = InsightExtractor()

        long_hook = "x" * 250
        result = extractor._validate_hook(long_hook)

        assert len(result) <= 200
        assert result.endswith("...")
