"""Thread generation from insights."""

from typing import Any, Optional

from vibesignal import ThreadGenerationError
from vibesignal.analysis.insights import InsightExtractor
from vibesignal.generation.formatter import TweetFormatter
from vibesignal.models import ImageOutput, Thread, ThreadMetadata, Tweet


class ThreadGenerator:
    """Generate Twitter thread structure from Claude insights.

    Takes insights from Claude analysis and generates a complete
    Thread with properly formatted tweets and image assignments.
    """

    def __init__(
        self,
        formatter: Optional[TweetFormatter] = None,
        insight_extractor: Optional[InsightExtractor] = None,
    ):
        """Initialize thread generator.

        Args:
            formatter: Tweet formatter (creates default if None)
            insight_extractor: Insight extractor (creates default if None)
        """
        self.formatter = formatter or TweetFormatter()
        self.extractor = insight_extractor or InsightExtractor()

    def generate_thread(
        self,
        insights: dict[str, Any],
        images: list[ImageOutput],
        source_notebook: str,
        claude_model: str,
        max_tweets: Optional[int] = None,
        include_numbering: bool = True,
    ) -> Thread:
        """Generate complete thread from insights.

        Args:
            insights: Insights dict from Claude analysis
            images: List of extracted images
            source_notebook: Path to source notebook
            claude_model: Claude model used
            max_tweets: Maximum tweets (None for no limit)
            include_numbering: Include position numbering

        Returns:
            Thread: Complete thread object

        Raises:
            ThreadGenerationError: If thread generation fails
        """
        try:
            # Extract and validate insights
            validated_insights = self.extractor.extract_insights(insights)

            # Get thread structure from Claude
            thread_structure = validated_insights["thread_structure"]

            # Limit tweets if requested
            if max_tweets and len(thread_structure) > max_tweets:
                thread_structure = thread_structure[:max_tweets]

            # Create image index map
            image_map = self._create_image_map(images)

            # Generate tweets
            tweets = []
            for tweet_data in thread_structure:
                position = tweet_data["position"]
                content = tweet_data["content"]
                image_index = tweet_data.get("image_index")

                # Get image filename if specified
                image_filename = None
                if image_index is not None and image_index in image_map:
                    image_filename = image_map[image_index]

                # Format tweet
                try:
                    tweet = self.formatter.format_tweet(
                        content=content,
                        position=position,
                        total=len(thread_structure),
                        image=image_filename,
                        include_numbering=include_numbering,
                    )
                    tweets.append(tweet)
                except ValueError as e:
                    # If tweet is too long, try to truncate
                    truncated = self.formatter.truncate_to_fit(content)
                    tweet = self.formatter.format_tweet(
                        content=truncated,
                        position=position,
                        total=len(thread_structure),
                        image=image_filename,
                        include_numbering=include_numbering,
                    )
                    tweets.append(tweet)

            # Create metadata
            metadata = ThreadMetadata(
                source_notebook=source_notebook,
                total_tweets=len(tweets),
                total_images=sum(1 for t in tweets if t.image_filename),
                claude_model=claude_model,
            )

            # Create thread
            thread = Thread(
                tweets=tweets,
                metadata=metadata,
                hook=validated_insights["hook"],
                call_to_action=validated_insights.get("call_to_action"),
            )

            return thread

        except ValueError as e:
            raise ThreadGenerationError(f"Failed to generate thread: {e}") from e
        except Exception as e:
            raise ThreadGenerationError(f"Unexpected error generating thread: {e}") from e

    def _create_image_map(self, images: list[ImageOutput]) -> dict[int, str]:
        """Create a map from image index to filename.

        Args:
            images: List of image outputs

        Returns:
            dict: Map of index -> filename
        """
        image_map = {}
        for i, image in enumerate(images):
            image_map[i] = image.filename
        return image_map

    def create_hook(self, insights: dict[str, Any]) -> str:
        """Generate an engaging opening tweet/hook.

        Args:
            insights: Insights dictionary

        Returns:
            str: Hook text
        """
        # Use hook from insights if available
        if "hook" in insights:
            return insights["hook"]

        # Fallback: create from first principle
        if insights.get("first_principles"):
            principle = insights["first_principles"][0]
            return f"Let's explore: {principle}"

        # Last resort
        return "Exploring some interesting insights"

    def assign_images_to_tweets(
        self,
        tweets: list[dict],
        images: list[ImageOutput],
        image_mappings: list[dict],
    ) -> list[dict]:
        """Assign images to tweets based on Claude's suggestions.

        Args:
            tweets: List of tweet dicts
            images: List of available images
            image_mappings: Claude's image mapping suggestions

        Returns:
            list: Updated tweet dicts with image assignments
        """
        # Create a map of suggested image positions
        image_positions = {}
        for mapping in image_mappings:
            image_idx = mapping.get("image_index")
            position = mapping.get("suggested_position")
            if image_idx is not None and position is not None:
                image_positions[position] = image_idx

        # Assign images to tweets
        for tweet in tweets:
            position = tweet["position"]
            if position in image_positions:
                image_idx = image_positions[position]
                if 0 <= image_idx < len(images):
                    tweet["image_filename"] = images[image_idx].filename

        return tweets
