"""Terminal preview for threads using Rich."""

from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.columns import Columns
from rich.table import Table
from rich.text import Text

from vibesignal.models import Thread


class TerminalPreview:
    """Generate terminal preview of thread using Rich."""

    def __init__(self, console: Optional[Console] = None):
        """Initialize terminal preview.

        Args:
            console: Rich console to use (creates new if None)
        """
        self.console = console or Console()

    def show(self, thread: Thread, workspace: Path) -> None:
        """Show thread preview in terminal.

        Args:
            thread: Thread to preview
            workspace: Workspace containing images
        """
        # Header
        self.console.print()
        self.console.print(
            Panel.fit(
                f"[bold cyan]Thread Preview[/bold cyan]\n\n"
                f"Total Tweets: [bold]{len(thread.tweets)}[/bold]\n"
                f"Total Images: [bold]{thread.metadata.total_images}[/bold]\n"
                f"Source: [dim]{thread.metadata.source_notebook}[/dim]",
                border_style="cyan",
                title="[bold]VibeSignal Thread[/bold]",
            )
        )

        # Hook
        self.console.print()
        hook_panel = Panel(
            f"[bold white]{thread.hook}[/bold white]",
            border_style="blue",
            title="[bold blue]🎣 Hook (Thread Opener)[/bold blue]",
            title_align="left",
        )
        self.console.print(hook_panel)

        # Tweets
        self.console.print()
        self.console.print("[bold]Thread Flow:[/bold]")
        self.console.print()

        for i, tweet in enumerate(thread.tweets):
            self._show_tweet(tweet, i, workspace, thread)

        # Call to action
        if thread.call_to_action:
            self.console.print()
            cta_panel = Panel(
                f"[italic]{thread.call_to_action}[/italic]",
                border_style="green",
                title="[bold green]📢 Call to Action[/bold green]",
                title_align="left",
            )
            self.console.print(cta_panel)

        # Summary stats
        self._show_summary(thread)

    def _show_tweet(self, tweet, index: int, workspace: Path, thread: Thread) -> None:
        """Show a single tweet in the preview.

        Args:
            tweet: Tweet to show
            index: Index in thread (for indentation)
            workspace: Workspace for images
            thread: Full thread for context
        """
        # Reply indicator
        if index > 0:
            self.console.print("  [dim]↳ Reply to previous tweet[/dim]")

        # Tweet number and stats
        header = Text()
        header.append(f"Tweet {tweet.position}/{len(thread.tweets)}", style="bold cyan")
        header.append(" • ", style="dim")
        header.append(f"{tweet.character_count}/275 chars",
                     style="yellow" if tweet.character_count > 250 else "green")

        if tweet.image_filename:
            header.append(" • ", style="dim")
            header.append("📷 Image attached", style="magenta")

        # Tweet content
        content_lines = []
        content_lines.append(header)
        content_lines.append("")

        # Add tweet text with proper wrapping
        for line in tweet.text.split('\n'):
            content_lines.append(Text(line, style="white"))

        # Image info
        if tweet.image_filename:
            content_lines.append("")
            image_path = workspace / "images" / tweet.image_filename
            image_info = Text()
            image_info.append("🖼  ", style="magenta")
            image_info.append(tweet.image_filename, style="dim magenta")
            if image_path.exists():
                # Get image size
                from PIL import Image
                try:
                    with Image.open(image_path) as img:
                        image_info.append(f" ({img.width}x{img.height})", style="dim")
                except:
                    pass
            content_lines.append(image_info)

        # Create panel
        tweet_panel = Panel(
            "\n".join(str(line) for line in content_lines),
            border_style="blue" if index == 0 else "white",
            padding=(1, 2),
            expand=False,
        )

        self.console.print(tweet_panel)
        self.console.print()

    def _show_summary(self, thread: Thread) -> None:
        """Show summary statistics.

        Args:
            thread: Thread to summarize
        """
        self.console.print()
        self.console.print("[bold]📊 Thread Statistics:[/bold]")

        # Create stats table
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Total Tweets", str(len(thread.tweets)))
        table.add_row("Total Images", str(thread.metadata.total_images))

        # Character stats
        char_counts = [t.character_count for t in thread.tweets]
        table.add_row("Avg Characters", f"{sum(char_counts) / len(char_counts):.1f}")
        table.add_row("Max Characters", str(max(char_counts)))
        table.add_row("Min Characters", str(min(char_counts)))

        # Images
        tweets_with_images = sum(1 for t in thread.tweets if t.image_filename)
        table.add_row("Tweets with Images", f"{tweets_with_images}/{len(thread.tweets)}")

        self.console.print(table)

    def show_compact(self, thread: Thread) -> None:
        """Show compact preview (just tweet texts).

        Args:
            thread: Thread to preview
        """
        self.console.print()
        self.console.print(f"[bold cyan]Thread: {thread.hook}[/bold cyan]")
        self.console.print()

        for tweet in thread.tweets:
            prefix = f"[{tweet.position}/{len(thread.tweets)}]"
            image_indicator = " 📷" if tweet.image_filename else ""
            self.console.print(f"{prefix}{image_indicator} {tweet.text}")
            self.console.print()
