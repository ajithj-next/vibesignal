"""Output writing for threads in various formats."""

import json
from pathlib import Path
from typing import Literal

from vibesignal.models import Thread


class OutputWriter:
    """Write thread output in various formats.

    Supports JSON, text, and markdown output formats.
    """

    def write(
        self,
        thread: Thread,
        output_path: Path | str,
        format: Literal["json", "text", "markdown"] = "json",
    ) -> Path:
        """Write thread to file in specified format.

        Args:
            thread: Thread to write
            output_path: Path to output file
            format: Output format (json, text, or markdown)

        Returns:
            Path: Path to written file
        """
        output_path = Path(output_path)

        if format == "json":
            return self.write_json(thread, output_path)
        elif format == "text":
            return self.write_text(thread, output_path)
        elif format == "markdown":
            return self.write_markdown(thread, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def write_json(self, thread: Thread, output_path: Path) -> Path:
        """Write thread as JSON.

        Args:
            thread: Thread to write
            output_path: Output file path

        Returns:
            Path: Path to written file
        """
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert thread to dict
        thread_dict = {
            "tweets": [
                {
                    "position": t.position,
                    "text": t.text,
                    "image_filename": t.image_filename,
                    "character_count": t.character_count,
                }
                for t in thread.tweets
            ],
            "metadata": {
                "source_notebook": thread.metadata.source_notebook,
                "generated_at": thread.metadata.generated_at,
                "total_tweets": thread.metadata.total_tweets,
                "total_images": thread.metadata.total_images,
                "claude_model": thread.metadata.claude_model,
            },
            "hook": thread.hook,
            "call_to_action": thread.call_to_action,
        }

        # Write JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(thread_dict, f, indent=2, ensure_ascii=False)

        return output_path

    def write_text(self, thread: Thread, output_path: Path) -> Path:
        """Write thread as human-readable text.

        Args:
            thread: Thread to write
            output_path: Output file path

        Returns:
            Path: Path to written file
        """
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = []

        # Header
        lines.append("=" * 60)
        lines.append("TWITTER THREAD")
        lines.append("=" * 60)
        lines.append("")

        # Metadata
        lines.append(f"Source: {thread.metadata.source_notebook}")
        lines.append(f"Generated: {thread.metadata.generated_at}")
        lines.append(f"Total tweets: {thread.metadata.total_tweets}")
        lines.append(f"Total images: {thread.metadata.total_images}")
        lines.append(f"Model: {thread.metadata.claude_model}")
        lines.append("")
        lines.append("-" * 60)
        lines.append("")

        # Hook
        lines.append(f"HOOK: {thread.hook}")
        lines.append("")
        lines.append("-" * 60)
        lines.append("")

        # Tweets
        for tweet in thread.tweets:
            lines.append(f"[Tweet {tweet.position}/{thread.metadata.total_tweets}]")
            if tweet.image_filename:
                lines.append(f"[IMAGE: {tweet.image_filename}]")
            lines.append(tweet.text)
            lines.append(f"({tweet.character_count} characters)")
            lines.append("")

        # Call to action
        if thread.call_to_action:
            lines.append("-" * 60)
            lines.append(f"CTA: {thread.call_to_action}")
            lines.append("")

        lines.append("=" * 60)

        # Write file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return output_path

    def write_markdown(self, thread: Thread, output_path: Path) -> Path:
        """Write thread as markdown with image references.

        Args:
            thread: Thread to write
            output_path: Output file path

        Returns:
            Path: Path to written file
        """
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = []

        # Title and metadata
        lines.append("# Twitter Thread")
        lines.append("")
        lines.append(f"**Source:** {thread.metadata.source_notebook}  ")
        lines.append(f"**Generated:** {thread.metadata.generated_at}  ")
        lines.append(f"**Total Tweets:** {thread.metadata.total_tweets}  ")
        lines.append(f"**Total Images:** {thread.metadata.total_images}  ")
        lines.append(f"**Model:** {thread.metadata.claude_model}")
        lines.append("")

        # Hook
        lines.append("## Hook")
        lines.append("")
        lines.append(f"> {thread.hook}")
        lines.append("")

        # Tweets
        lines.append("## Thread")
        lines.append("")

        for tweet in thread.tweets:
            lines.append(f"### Tweet {tweet.position}/{thread.metadata.total_tweets}")
            lines.append("")

            if tweet.image_filename:
                lines.append(f"![Image]({tweet.image_filename})")
                lines.append("")

            lines.append(tweet.text)
            lines.append("")
            lines.append(f"*{tweet.character_count} characters*")
            lines.append("")
            lines.append("---")
            lines.append("")

        # Call to action
        if thread.call_to_action:
            lines.append("## Call to Action")
            lines.append("")
            lines.append(thread.call_to_action)
            lines.append("")

        # Write file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return output_path
