"""Tweet formatting utilities."""

from typing import Optional

from vibesignal.models import Tweet


class TweetFormatter:
    """Format tweets with constraints and styling.

    Handles tweet numbering, character limits, and formatting.
    """

    MAX_TWEET_LENGTH = 275  # Safety buffer under Twitter's 280 limit

    def format_tweet(
        self,
        content: str,
        position: int,
        total: int,
        image: Optional[str] = None,
        include_numbering: bool = True,
    ) -> Tweet:
        """Format a single tweet.

        Args:
            content: Tweet content text
            position: Position in thread (1-indexed)
            total: Total tweets in thread
            image: Optional image filename
            include_numbering: Whether to add position numbering

        Returns:
            Tweet: Formatted tweet object

        Raises:
            ValueError: If content is too long even after processing
        """
        # Add numbering if requested
        if include_numbering and total > 1:
            numbered_text = self.add_thread_numbering(content, position, total)
        else:
            numbered_text = content

        # Validate length
        if len(numbered_text) > self.MAX_TWEET_LENGTH:
            # Try without numbering if it was added
            if include_numbering:
                numbered_text = content
                if len(numbered_text) > self.MAX_TWEET_LENGTH:
                    raise ValueError(
                        f"Tweet content too long ({len(numbered_text)} > {self.MAX_TWEET_LENGTH})"
                    )
            else:
                raise ValueError(
                    f"Tweet content too long ({len(numbered_text)} > {self.MAX_TWEET_LENGTH})"
                )

        return Tweet(
            position=position,
            text=numbered_text,
            image_filename=image,
        )

    def add_thread_numbering(self, text: str, position: int, total: int) -> str:
        """Add tweet position numbering in (1/N) format.

        Args:
            text: Tweet text
            position: Position in thread
            total: Total tweets

        Returns:
            str: Text with numbering added
        """
        # Check if text already has numbering
        numbering = f"({position}/{total})"
        if numbering in text or f"{position}/{total}" in text:
            return text

        # Add numbering at the end
        return f"{text} {numbering}"

    def truncate_to_fit(self, text: str, max_length: Optional[int] = None) -> str:
        """Truncate text to fit within character limit.

        Args:
            text: Text to truncate
            max_length: Maximum length (default: MAX_TWEET_LENGTH)

        Returns:
            str: Truncated text with ellipsis if needed
        """
        if max_length is None:
            max_length = self.MAX_TWEET_LENGTH

        if len(text) <= max_length:
            return text

        # Truncate with ellipsis
        return text[: max_length - 3] + "..."

    def clean_text(self, text: str) -> str:
        """Clean text for Twitter posting.

        Args:
            text: Text to clean

        Returns:
            str: Cleaned text
        """
        # Remove excessive whitespace
        text = " ".join(text.split())

        # Remove leading/trailing whitespace
        text = text.strip()

        return text
