"""Tests for thread generation."""

from pathlib import Path

import pytest

from vibesignal import ThreadGenerationError
from vibesignal.generation.formatter import TweetFormatter
from vibesignal.generation.thread import ThreadGenerator
from vibesignal.models import ImageOutput


@pytest.fixture
def sample_insights():
    """Sample Claude insights for testing."""
    return {
        "first_principles": ["Distance is meaning", "Neighbors share properties"],
        "insights": [
            {
                "content": "k-NN predicts by proximity",
                "tweet_suggestion": "Core insight: predictions based on nearby examples",
                "importance": "high",
            }
        ],
        "thread_structure": [
            {"position": 1, "content": "Let's explore k-NN from first principles", "image_index": None},
            {
                "position": 2,
                "content": "Distance represents similarity in feature space",
                "image_index": 0,
            },
            {"position": 3, "content": "Similar items cluster together naturally", "image_index": None},
        ],
        "image_mappings": [{"image_index": 0, "suggested_position": 2, "relevance": "Shows clustering"}],
        "hook": "Ever wondered how k-NN really works?",
    }


@pytest.fixture
def sample_images():
    """Sample images for testing."""
    return [
        ImageOutput(
            cell_index=0,
            output_index=0,
            format="png",
            data=b"fake_image_data",
            filename="knn_plot.png",
        )
    ]


class TestTweetFormatter:
    """Tests for TweetFormatter class."""

    def test_format_tweet_basic(self):
        """Test basic tweet formatting."""
        formatter = TweetFormatter()
        tweet = formatter.format_tweet("Hello world", position=1, total=5)

        assert tweet.position == 1
        assert "Hello world" in tweet.text
        assert "(1/5)" in tweet.text

    def test_format_tweet_without_numbering(self):
        """Test formatting without numbering."""
        formatter = TweetFormatter()
        tweet = formatter.format_tweet(
            "Hello world", position=1, total=5, include_numbering=False
        )

        assert tweet.text == "Hello world"
        assert "(1/5)" not in tweet.text

    def test_format_tweet_with_image(self):
        """Test formatting with image."""
        formatter = TweetFormatter()
        tweet = formatter.format_tweet(
            "Check this out", position=2, total=3, image="plot.png"
        )

        assert tweet.image_filename == "plot.png"

    def test_format_tweet_too_long(self):
        """Test handling of too-long tweets."""
        formatter = TweetFormatter()
        long_text = "x" * 300

        with pytest.raises(ValueError, match="too long"):
            formatter.format_tweet(long_text, position=1, total=1, include_numbering=False)

    def test_add_thread_numbering(self):
        """Test adding thread numbering."""
        formatter = TweetFormatter()
        result = formatter.add_thread_numbering("Hello", position=1, total=10)

        assert "(1/10)" in result

    def test_add_thread_numbering_already_present(self):
        """Test numbering not duplicated if already present."""
        formatter = TweetFormatter()
        text = "Hello (1/10)"
        result = formatter.add_thread_numbering(text, position=1, total=10)

        assert result.count("(1/10)") == 1

    def test_truncate_to_fit(self):
        """Test text truncation."""
        formatter = TweetFormatter()
        long_text = "x" * 300

        result = formatter.truncate_to_fit(long_text, max_length=50)

        assert len(result) == 50
        assert result.endswith("...")

    def test_clean_text(self):
        """Test text cleaning."""
        formatter = TweetFormatter()
        messy_text = "  Hello   world  \n  with   spaces  "

        result = formatter.clean_text(messy_text)

        assert result == "Hello world with spaces"


class TestThreadGenerator:
    """Tests for ThreadGenerator class."""

    def test_generate_thread_success(self, sample_insights, sample_images):
        """Test successful thread generation."""
        generator = ThreadGenerator()
        thread = generator.generate_thread(
            insights=sample_insights,
            images=sample_images,
            source_notebook="test.ipynb",
            claude_model="claude-sonnet-4-5-20250929",
        )

        assert len(thread.tweets) == 3
        assert thread.hook == "Ever wondered how k-NN really works?"
        assert thread.metadata.source_notebook == "test.ipynb"
        assert thread.metadata.total_tweets == 3

        # Check image assignment
        tweet_with_image = thread.tweets[1]  # Position 2
        assert tweet_with_image.image_filename == "knn_plot.png"

    def test_generate_thread_with_max_tweets(self, sample_insights, sample_images):
        """Test thread generation with max tweets limit."""
        generator = ThreadGenerator()
        thread = generator.generate_thread(
            insights=sample_insights,
            images=sample_images,
            source_notebook="test.ipynb",
            claude_model="claude-sonnet-4-5-20250929",
            max_tweets=2,
        )

        assert len(thread.tweets) == 2

    def test_generate_thread_no_images(self, sample_insights):
        """Test thread generation without images."""
        generator = ThreadGenerator()
        thread = generator.generate_thread(
            insights=sample_insights,
            images=[],
            source_notebook="test.ipynb",
            claude_model="claude-sonnet-4-5-20250929",
        )

        assert len(thread.tweets) == 3
        assert thread.metadata.total_images == 0

    def test_generate_thread_invalid_insights(self):
        """Test handling of invalid insights."""
        generator = ThreadGenerator()

        with pytest.raises(ThreadGenerationError):
            generator.generate_thread(
                insights={},  # Missing required fields
                images=[],
                source_notebook="test.ipynb",
                claude_model="claude-sonnet-4-5-20250929",
            )

    def test_create_image_map(self, sample_images):
        """Test image map creation."""
        generator = ThreadGenerator()
        image_map = generator._create_image_map(sample_images)

        assert 0 in image_map
        assert image_map[0] == "knn_plot.png"

    def test_create_hook(self, sample_insights):
        """Test hook creation."""
        generator = ThreadGenerator()
        hook = generator.create_hook(sample_insights)

        assert hook == "Ever wondered how k-NN really works?"

    def test_create_hook_fallback(self):
        """Test hook creation fallback."""
        generator = ThreadGenerator()
        insights = {"first_principles": ["Test principle"]}
        hook = generator.create_hook(insights)

        assert "Test principle" in hook
