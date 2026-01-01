"""VibeSignal - Convert Jupyter notebooks to Twitter threads using AI.

First Principles reasoning. Good Vibes. Sublime Visual thinking.
"""

__version__ = "0.1.0"


class VibeSignalError(Exception):
    """Base exception for all VibeSignal errors."""

    pass


class NotebookParseError(VibeSignalError):
    """Raised when notebook parsing fails."""

    pass


class ClaudeAPIError(VibeSignalError):
    """Raised when Claude API interaction fails."""

    pass


class ThreadGenerationError(VibeSignalError):
    """Raised when thread generation fails."""

    pass


class ConfigurationError(VibeSignalError):
    """Raised when configuration is invalid or missing."""

    pass
