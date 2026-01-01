"""Command-line interface for VibeSignal."""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from vibesignal import __version__
from vibesignal.analysis.claude_client import ClaudeClient
from vibesignal.config import get_config
from vibesignal.generation.thread import ThreadGenerator
from vibesignal.output.writer import OutputWriter
from vibesignal.parsing.extractors import ImageExtractor
from vibesignal.parsing.notebook import NotebookParser

console = Console()


@click.group()
@click.version_option(version=__version__)
def main():
    """VibeSignal - Convert notebooks to Twitter threads.

    First Principles reasoning. Good Vibes. Sublime Visual thinking.
    """
    pass


@main.command()
@click.argument("notebook", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output file path (default: notebook_name_thread.json)",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["json", "text", "markdown"]),
    default=None,
    help="Output format (default: from config or json)",
)
@click.option(
    "--max-tweets",
    "-n",
    type=int,
    default=None,
    help="Maximum number of tweets (default: from config or 10)",
)
@click.option(
    "--image-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory for extracted images (default: from config or 'images')",
)
@click.option(
    "--api-key",
    envvar="VIBESIGNAL_ANTHROPIC_API_KEY",
    help="Anthropic API key (or set VIBESIGNAL_ANTHROPIC_API_KEY)",
)
def convert(
    notebook: Path,
    output: Optional[Path],
    format: Optional[str],
    max_tweets: Optional[int],
    image_dir: Optional[Path],
    api_key: Optional[str],
):
    """Convert a Jupyter notebook to a Twitter thread.

    NOTEBOOK: Path to the .ipynb file to convert
    """
    try:
        # Load config
        try:
            config = get_config()
        except Exception:
            # Config loading might fail if API key not set
            config = None

        # Override with CLI options
        if api_key:
            anthropic_api_key = api_key
        elif config:
            anthropic_api_key = config.anthropic_api_key
        else:
            console.print(
                "[red]Error:[/red] API key not found. "
                "Set VIBESIGNAL_ANTHROPIC_API_KEY environment variable or use --api-key"
            )
            sys.exit(1)

        output_format = format or (config.output_format if config else "json")
        max_tweets_limit = max_tweets or (config.max_tweets if config else 10)
        images_dir = image_dir or Path(config.image_output_dir if config else "images")
        claude_model = config.claude_model if config else "claude-sonnet-4-5-20250929"

        # Determine output path
        if not output:
            output = notebook.parent / f"{notebook.stem}_thread.{output_format}"

        # Show welcome panel
        console.print(
            Panel.fit(
                f"[bold cyan]VibeSignal[/bold cyan] v{__version__}\n"
                f"Converting: [yellow]{notebook.name}[/yellow]",
                border_style="cyan",
            )
        )

        # Process with progress indicator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Parse notebook
            task = progress.add_task("[cyan]Parsing notebook...", total=5)
            parser = NotebookParser()
            parsed_notebook = parser.parse(notebook)
            progress.advance(task)

            # Extract images
            progress.update(task, description="[cyan]Extracting images...")
            extractor = ImageExtractor()
            images = extractor.extract_images(parsed_notebook, images_dir)
            console.print(f"  Extracted {len(images)} image(s)")
            progress.advance(task)

            # Analyze with Claude
            progress.update(task, description="[cyan]Analyzing with Claude AI...")
            claude = ClaudeClient(api_key=anthropic_api_key, model=claude_model)
            insights = claude.analyze_notebook(parsed_notebook, max_tweets=max_tweets_limit)
            progress.advance(task)

            # Generate thread
            progress.update(task, description="[cyan]Generating thread...")
            generator = ThreadGenerator()
            thread = generator.generate_thread(
                insights=insights,
                images=images,
                source_notebook=str(notebook),
                claude_model=claude_model,
                max_tweets=max_tweets_limit,
            )
            progress.advance(task)

            # Write output
            progress.update(task, description="[cyan]Writing output...")
            writer = OutputWriter()
            output_path = writer.write(thread, output, format=output_format)
            progress.advance(task)

        # Success message
        console.print()
        console.print(
            Panel.fit(
                f"[green]Success![/green]\n\n"
                f"Thread: [bold]{thread.metadata.total_tweets}[/bold] tweets, "
                f"[bold]{thread.metadata.total_images}[/bold] images\n"
                f"Output: [yellow]{output_path}[/yellow]\n"
                f"Images: [yellow]{images_dir}/[/yellow]",
                border_style="green",
                title="[bold green]Thread Generated[/bold green]",
            )
        )

        # Show hook
        if thread.hook:
            console.print()
            console.print(Panel(thread.hook, title="[bold]Hook[/bold]", border_style="blue"))

    except Exception as e:
        console.print()
        console.print(
            Panel.fit(
                f"[red]Error:[/red] {str(e)}\n\n"
                f"Check your inputs and try again.",
                border_style="red",
                title="[bold red]Conversion Failed[/bold red]",
            )
        )
        sys.exit(1)


@main.command()
def config_show():
    """Show current configuration."""
    try:
        config = get_config()
        console.print(Panel.fit("[bold cyan]VibeSignal Configuration[/bold cyan]", border_style="cyan"))
        console.print()
        console.print(f"[cyan]Claude Model:[/cyan] {config.claude_model}")
        console.print(f"[cyan]Max Tweets:[/cyan] {config.max_tweets}")
        console.print(f"[cyan]Max Tweet Length:[/cyan] {config.max_tweet_length}")
        console.print(f"[cyan]Include Numbering:[/cyan] {config.include_thread_numbering}")
        console.print(f"[cyan]Output Format:[/cyan] {config.output_format}")
        console.print(f"[cyan]Image Output Dir:[/cyan] {config.image_output_dir}")
        console.print(f"[cyan]API Key Set:[/cyan] {'Yes' if config.anthropic_api_key else 'No'}")
    except Exception as e:
        console.print(f"[red]Error loading config:[/red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
