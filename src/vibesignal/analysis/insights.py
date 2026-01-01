"""Insight extraction and processing from Claude analysis."""

from typing import Any


class InsightExtractor:
    """Process and validate insights from Claude's analysis.

    Takes the raw Claude response and validates/processes it
    into a clean structure for thread generation.
    """

    def extract_insights(self, claude_response: dict[str, Any]) -> dict[str, Any]:
        """Extract and validate insights from Claude's response.

        Args:
            claude_response: Raw response from Claude API

        Returns:
            dict: Validated and processed insights

        Raises:
            ValueError: If response structure is invalid
        """
        # Validate required fields
        required_fields = ["first_principles", "insights", "thread_structure", "hook"]
        missing = [f for f in required_fields if f not in claude_response]
        if missing:
            raise ValueError(f"Missing required fields in Claude response: {missing}")

        # Process and validate each component
        processed = {
            "first_principles": self._validate_principles(
                claude_response.get("first_principles", [])
            ),
            "insights": self._validate_insights(claude_response.get("insights", [])),
            "thread_structure": self._validate_thread_structure(
                claude_response.get("thread_structure", [])
            ),
            "image_mappings": claude_response.get("image_mappings", []),
            "hook": self._validate_hook(claude_response.get("hook", "")),
            "call_to_action": claude_response.get("call_to_action"),
        }

        return processed

    def _validate_principles(self, principles: list[str]) -> list[str]:
        """Validate first principles list.

        Args:
            principles: List of first principles

        Returns:
            list[str]: Validated principles
        """
        if not isinstance(principles, list):
            return []

        # Filter out empty strings
        return [p.strip() for p in principles if p and p.strip()]

    def _validate_insights(self, insights: list[dict]) -> list[dict]:
        """Validate insights list.

        Args:
            insights: List of insight dictionaries

        Returns:
            list[dict]: Validated insights
        """
        if not isinstance(insights, list):
            return []

        validated = []
        for insight in insights:
            if not isinstance(insight, dict):
                continue

            # Ensure required fields
            if "content" in insight:
                validated.append({
                    "content": insight["content"],
                    "tweet_suggestion": insight.get("tweet_suggestion", insight["content"]),
                    "importance": insight.get("importance", "medium"),
                })

        return validated

    def _validate_thread_structure(self, thread_structure: list[dict]) -> list[dict]:
        """Validate thread structure.

        Args:
            thread_structure: List of tweet structure dictionaries

        Returns:
            list[dict]: Validated thread structure

        Raises:
            ValueError: If structure is invalid
        """
        if not isinstance(thread_structure, list):
            raise ValueError("thread_structure must be a list")

        if not thread_structure:
            raise ValueError("thread_structure cannot be empty")

        validated = []
        for i, tweet in enumerate(thread_structure, start=1):
            if not isinstance(tweet, dict):
                continue

            # Validate position
            position = tweet.get("position", i)
            if position != i:
                # Fix position to be sequential
                position = i

            # Validate content
            content = tweet.get("content", "")
            if not content or not content.strip():
                continue

            # Validate length (275 chars max)
            if len(content) > 275:
                # Truncate with ellipsis
                content = content[:272] + "..."

            validated.append({
                "position": position,
                "content": content.strip(),
                "image_index": tweet.get("image_index"),
            })

        if not validated:
            raise ValueError("No valid tweets in thread_structure")

        return validated

    def _validate_hook(self, hook: str) -> str:
        """Validate the hook text.

        Args:
            hook: Hook text

        Returns:
            str: Validated hook

        Raises:
            ValueError: If hook is empty or invalid
        """
        if not hook or not hook.strip():
            raise ValueError("Hook cannot be empty")

        hook = hook.strip()

        # Ensure hook isn't too long (should fit in first tweet)
        if len(hook) > 200:
            hook = hook[:197] + "..."

        return hook

    def identify_principles(self, content: str) -> list[str]:
        """Identify first principles in markdown content.

        This is a heuristic-based approach for when we don't use Claude.

        Args:
            content: Markdown content to analyze

        Returns:
            list[str]: Identified principles
        """
        principles = []
        lines = content.lower().split("\n")

        principle_indicators = [
            "first principle",
            "fundamental",
            "axiom",
            "assumption",
            "basic principle",
        ]

        for line in lines:
            for indicator in principle_indicators:
                if indicator in line:
                    # Extract the line as a principle
                    principle = line.strip("# ").strip("-").strip()
                    if principle and len(principle) > 10:
                        principles.append(principle)
                        break

        return principles[:5]  # Limit to top 5
