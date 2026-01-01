"""Claude API client for notebook analysis."""

import json
from typing import Any

import anthropic

from vibesignal import ClaudeAPIError
from vibesignal.models import ParsedNotebook


class ClaudeClient:
    """Wrapper for Claude API with notebook analysis capabilities.

    Handles communication with Claude for extracting insights and
    first principles reasoning from notebook content.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-5-20250929",
        max_tokens: int = 4000,
    ):
        """Initialize Claude client.

        Args:
            api_key: Anthropic API key
            model: Claude model to use
            max_tokens: Maximum tokens for response
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def analyze_notebook(
        self,
        notebook: ParsedNotebook,
        max_tweets: int = 10,
    ) -> dict[str, Any]:
        """Analyze notebook content and extract insights for Twitter thread.

        Args:
            notebook: Parsed notebook to analyze
            max_tweets: Maximum number of tweets for the thread

        Returns:
            dict: Structured insights with:
                - first_principles: List of identified principles
                - insights: List of key insights with tweet suggestions
                - thread_structure: Suggested breakdown into tweets
                - image_mappings: Which images map to which insights
                - hook: Engaging opening tweet idea

        Raises:
            ClaudeAPIError: If API call fails
        """
        try:
            # Build the analysis prompt
            prompt = self._build_analysis_prompt(notebook, max_tweets)

            # Call Claude API
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )

            # Extract response text
            response_text = message.content[0].text

            # Try to parse as JSON
            try:
                insights = json.loads(response_text)
            except json.JSONDecodeError:
                # If not JSON, try to extract JSON from markdown code block
                insights = self._extract_json_from_response(response_text)

            return insights

        except anthropic.APIError as e:
            raise ClaudeAPIError(f"Claude API error: {e}") from e
        except Exception as e:
            raise ClaudeAPIError(f"Failed to analyze notebook: {e}") from e

    def _build_analysis_prompt(self, notebook: ParsedNotebook, max_tweets: int) -> str:
        """Build the prompt for Claude to analyze the notebook.

        Args:
            notebook: Parsed notebook
            max_tweets: Maximum tweets for thread

        Returns:
            str: Formatted prompt
        """
        # Combine notebook cells into readable format
        notebook_content = self._format_notebook_content(notebook)

        # Count images
        num_images = len(notebook.images)

        prompt = f"""You are analyzing a Jupyter notebook that contains first principles reasoning and technical insights. Your task is to extract the key insights and structure them for a Twitter/X thread.

NOTEBOOK CONTENT:
{notebook_content}

AVAILABLE IMAGES: {num_images} images extracted from notebook outputs

TASK:
1. Identify the core first principles being explored in this notebook
2. Extract 5-10 key insights that would resonate on Twitter (technical audience)
3. Structure these insights into a thread with a maximum of {max_tweets} tweets
4. For each insight, suggest engaging, concise tweet text (max 275 characters including position numbering)
5. Suggest which images (by index 0-{num_images-1}) would best illustrate each tweet
6. Create an engaging hook for the opening tweet

REQUIREMENTS:
- Tweet text should be clear, engaging, and technically accurate
- Each tweet should stand alone but flow naturally in the thread
- Use simple language to explain complex concepts
- Include the "why" not just the "what"
- Make it visually engaging - suggest images where relevant

OUTPUT FORMAT (return as valid JSON):
{{
  "first_principles": [
    "Principle 1",
    "Principle 2"
  ],
  "insights": [
    {{
      "content": "Key insight text",
      "tweet_suggestion": "Suggested tweet text",
      "importance": "high|medium|low"
    }}
  ],
  "thread_structure": [
    {{
      "position": 1,
      "content": "Tweet text for position 1",
      "image_index": null
    }},
    {{
      "position": 2,
      "content": "Tweet text for position 2",
      "image_index": 0
    }}
  ],
  "image_mappings": [
    {{
      "image_index": 0,
      "suggested_position": 2,
      "relevance": "Shows the core concept"
    }}
  ],
  "hook": "Engaging opening line for the thread"
}}

Return ONLY the JSON structure, no additional text."""

        return prompt

    def _format_notebook_content(self, notebook: ParsedNotebook) -> str:
        """Format notebook cells into readable text.

        Args:
            notebook: Parsed notebook

        Returns:
            str: Formatted content
        """
        lines = []

        for i, cell in enumerate(notebook.cells):
            if cell.cell_type == "markdown":
                lines.append(f"\n=== MARKDOWN CELL {i} ===")
                lines.append(cell.source)
            elif cell.cell_type == "code":
                lines.append(f"\n=== CODE CELL {i} ===")
                lines.append(cell.source)

                # Note if there are outputs
                if cell.outputs:
                    has_images = any(
                        "image/png" in output.get("data", {})
                        or "image/jpeg" in output.get("data", {})
                        for output in cell.outputs
                    )
                    if has_images:
                        lines.append("[This cell has image output]")

        return "\n".join(lines)

    def _extract_json_from_response(self, response: str) -> dict[str, Any]:
        """Try to extract JSON from a response that might contain markdown.

        Args:
            response: Response text

        Returns:
            dict: Parsed JSON

        Raises:
            ClaudeAPIError: If JSON cannot be extracted
        """
        # Try to find JSON in markdown code block
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            json_str = response[start:end].strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # Try to find any JSON-like structure
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(response[start:end])
            except json.JSONDecodeError:
                pass

        raise ClaudeAPIError(f"Could not extract JSON from response: {response[:200]}")
