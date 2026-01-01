"""Data models for Twitter thread generation."""

from datetime import UTC, datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class Tweet(BaseModel):
    """Single tweet in a thread.

    Attributes:
        position: 1-indexed position in thread
        text: Tweet text content
        image_filename: Optional image filename to attach
        character_count: Calculated character count
    """

    position: int
    text: str
    image_filename: Optional[str] = None
    character_count: int = Field(init=False, default=0)

    def model_post_init(self, __context) -> None:
        """Calculate character count after initialization."""
        self.character_count = len(self.text)

    @field_validator("text")
    @classmethod
    def validate_length(cls, v: str) -> str:
        """Validate tweet length (275 chars for safety buffer)."""
        if len(v) > 275:
            raise ValueError(f"Tweet exceeds 275 characters: {len(v)}")
        if not v.strip():
            raise ValueError("Tweet text cannot be empty")
        return v

    @field_validator("position")
    @classmethod
    def validate_position(cls, v: int) -> int:
        """Validate tweet position is positive."""
        if v < 1:
            raise ValueError("Tweet position must be >= 1")
        return v


class ThreadMetadata(BaseModel):
    """Metadata about the generated thread.

    Attributes:
        source_notebook: Path to source notebook
        generated_at: ISO timestamp of generation
        total_tweets: Number of tweets in thread
        total_images: Number of images in thread
        claude_model: Claude model used for generation
    """

    source_notebook: str
    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    total_tweets: int
    total_images: int
    claude_model: str


class Thread(BaseModel):
    """Complete Twitter thread.

    Attributes:
        tweets: List of tweets in order
        metadata: Thread generation metadata
        hook: Opening hook/summary line
        call_to_action: Optional CTA for end of thread
    """

    tweets: list[Tweet]
    metadata: ThreadMetadata
    hook: str
    call_to_action: Optional[str] = None

    @field_validator("tweets")
    @classmethod
    def validate_tweets(cls, v: list[Tweet]) -> list[Tweet]:
        """Validate tweets are in correct order and have valid positions."""
        if not v:
            raise ValueError("Thread must contain at least one tweet")

        # Check positions are sequential starting from 1
        for i, tweet in enumerate(v, start=1):
            if tweet.position != i:
                raise ValueError(f"Tweet positions must be sequential. Expected {i}, got {tweet.position}")

        return v

    def model_post_init(self, __context) -> None:
        """Update metadata counts after initialization."""
        self.metadata.total_tweets = len(self.tweets)
        self.metadata.total_images = sum(1 for t in self.tweets if t.image_filename)
