"""Configuration management for VibeSignal."""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class VibeSignalConfig(BaseSettings):
    """Application configuration loaded from environment variables.

    Environment variables should be prefixed with VIBESIGNAL_
    Example: VIBESIGNAL_ANTHROPIC_API_KEY=sk-ant-...

    Attributes:
        anthropic_api_key: API key for Claude (required)
        claude_model: Claude model to use
        max_tweets: Maximum number of tweets in a thread
        max_tweet_length: Maximum characters per tweet (with buffer)
        include_thread_numbering: Add position numbering to tweets
        output_format: Default output format
        image_output_dir: Directory for extracted images
    """

    # API Configuration
    anthropic_api_key: str = Field(
        description="Anthropic API key for Claude",
    )
    claude_model: str = Field(
        default="claude-sonnet-4-5-20250929",
        description="Claude model to use for analysis",
    )

    # Thread Configuration
    max_tweets: int = Field(
        default=10,
        ge=2,
        le=25,
        description="Maximum number of tweets in a thread",
    )
    max_tweet_length: int = Field(
        default=275,
        ge=100,
        le=280,
        description="Maximum characters per tweet",
    )
    include_thread_numbering: bool = Field(
        default=True,
        description="Include position numbering (1/N) in tweets",
    )

    # Output Configuration
    output_format: Literal["json", "text", "markdown"] = Field(
        default="json",
        description="Default output format",
    )
    image_output_dir: str = Field(
        default="images",
        description="Directory for extracted images",
    )

    # Twitter/X Configuration (Optional)
    twitter_api_key: str = Field(
        default="",
        description="Twitter API key",
    )
    twitter_api_secret: str = Field(
        default="",
        description="Twitter API secret",
    )
    twitter_access_token: str = Field(
        default="",
        description="Twitter access token",
    )
    twitter_access_token_secret: str = Field(
        default="",
        description="Twitter access token secret",
    )
    twitter_bearer_token: str = Field(
        default="",
        description="Twitter bearer token",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="VIBESIGNAL_",
        case_sensitive=False,
        extra="ignore",
    )


# Global config instance (lazy-loaded)
_config: VibeSignalConfig | None = None


def get_config() -> VibeSignalConfig:
    """Get or create the global configuration instance.

    Returns:
        VibeSignalConfig: The configuration object
    """
    global _config
    if _config is None:
        _config = VibeSignalConfig()
    return _config


def reset_config() -> None:
    """Reset the global configuration (useful for testing)."""
    global _config
    _config = None
