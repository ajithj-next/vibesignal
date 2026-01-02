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
from vibesignal.workspace import WorkspaceManager
from vibesignal.publishing.twitter import TwitterPublisher
from vibesignal.preview.terminal import TerminalPreview
from vibesignal.preview.html import HTMLPreview

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
    "--project-name",
    "-p",
    type=str,
    default=None,
    help="Project name (default: notebook filename)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output file path (overrides workspace path)",
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
    "--workspace-dir",
    type=click.Path(path_type=Path),
    default="projects",
    help="Base directory for project workspaces (default: projects)",
)
@click.option(
    "--api-key",
    envvar="VIBESIGNAL_ANTHROPIC_API_KEY",
    help="Anthropic API key (or set VIBESIGNAL_ANTHROPIC_API_KEY)",
)
def convert(
    notebook: Path,
    project_name: Optional[str],
    output: Optional[Path],
    format: Optional[str],
    max_tweets: Optional[int],
    workspace_dir: Path,
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
        claude_model = config.claude_model if config else "claude-sonnet-4-5-20250929"

        # Create workspace
        workspace_manager = WorkspaceManager(base_dir=workspace_dir)
        proj_name = project_name or notebook.stem
        workspace = workspace_manager.create_workspace(
            project_name=proj_name,
            source_notebook=notebook,
            copy_source=True,
        )

        # Get paths from workspace
        images_dir = workspace_manager.get_images_dir(workspace)

        # Determine output paths - use workspace unless overridden
        if output:
            output_path = output
        else:
            output_path = workspace_manager.get_output_path(workspace, output_format)

        # Show welcome panel
        console.print(
            Panel.fit(
                f"[bold cyan]VibeSignal[/bold cyan] v{__version__}\n"
                f"Converting: [yellow]{notebook.name}[/yellow]\n"
                f"Project: [yellow]{proj_name}[/yellow]\n"
                f"Workspace: [dim]{workspace}[/dim]",
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

            # Write outputs - save in multiple formats to workspace
            progress.update(task, description="[cyan]Writing outputs...")
            writer = OutputWriter()

            # Save all formats in workspace
            json_path = writer.write(thread, workspace / "thread.json", format="json")
            text_path = writer.write(thread, workspace / "thread.txt", format="text")
            md_path = writer.write(thread, workspace / "thread.md", format="markdown")

            # Also write to custom output if specified
            if output:
                writer.write(thread, output, format=output_format)

            # Save workspace metadata
            workspace_manager.save_thread_metadata(workspace, thread)

            progress.advance(task)

        # Success message
        console.print()
        console.print(
            Panel.fit(
                f"[green]Success![/green]\n\n"
                f"Thread: [bold]{thread.metadata.total_tweets}[/bold] tweets, "
                f"[bold]{thread.metadata.total_images}[/bold] images\n\n"
                f"Workspace: [yellow]{workspace}/[/yellow]\n"
                f"  ├─ thread.json\n"
                f"  ├─ thread.txt\n"
                f"  ├─ thread.md\n"
                f"  └─ images/ ({len(images)} files)",
                border_style="green",
                title=f"[bold green]{proj_name}[/bold green]",
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


@main.command()
@click.option(
    "--workspace-dir",
    type=click.Path(path_type=Path),
    default="projects",
    help="Base directory for project workspaces",
)
def list_projects(workspace_dir: Path):
    """List all thread projects."""
    workspace_manager = WorkspaceManager(base_dir=workspace_dir)
    projects = workspace_manager.list_workspaces()

    if not projects:
        console.print("[yellow]No projects found.[/yellow]")
        console.print(f"[dim]Workspace directory: {workspace_dir}[/dim]")
        return

    console.print(
        Panel.fit(
            f"[bold cyan]Thread Projects[/bold cyan]\n"
            f"[dim]{len(projects)} project(s) in {workspace_dir}[/dim]",
            border_style="cyan",
        )
    )
    console.print()

    for project in sorted(projects):
        try:
            summary = workspace_manager.get_workspace_summary(project)
            tweets = summary.get("total_tweets", "?")
            images = summary.get("total_images", "?")
            console.print(f"  [cyan]•[/cyan] [bold]{project}[/bold]")
            console.print(f"    Tweets: {tweets}, Images: {images}")
            console.print(f"    [dim]{summary['workspace_path']}[/dim]")
            console.print()
        except Exception as e:
            console.print(f"  [red]•[/red] [bold]{project}[/bold]")
            console.print(f"    [red]Error: {e}[/red]")
            console.print()


@main.command()
@click.argument("project_name")
@click.option(
    "--workspace-dir",
    type=click.Path(path_type=Path),
    default="projects",
    help="Base directory for project workspaces",
)
def show_project(project_name: str, workspace_dir: Path):
    """Show detailed information about a project."""
    try:
        workspace_manager = WorkspaceManager(base_dir=workspace_dir)
        summary = workspace_manager.get_workspace_summary(project_name)

        console.print(
            Panel.fit(
                f"[bold cyan]{project_name}[/bold cyan]\n"
                f"[dim]Created: {summary.get('created_at', 'Unknown')}[/dim]",
                border_style="cyan",
            )
        )
        console.print()

        # Thread info
        console.print("[bold]Thread Information:[/bold]")
        console.print(f"  Total Tweets: {summary.get('total_tweets', 'Not generated')}")
        console.print(f"  Total Images: {summary.get('total_images', 'Not generated')}")
        console.print(f"  Image Files: {summary.get('image_files_count', 0)}")
        console.print()

        # Files
        console.print("[bold]Available Files:[/bold]")
        for output_format in summary.get("available_outputs", []):
            console.print(f"  [green]✓[/green] thread.{output_format}")
        console.print()

        # Source
        if summary.get("source_notebook"):
            console.print("[bold]Source:[/bold]")
            console.print(f"  {summary['source_notebook']}")
            console.print()

        # Workspace path
        console.print("[bold]Workspace:[/bold]")
        console.print(f"  {summary['workspace_path']}")

    except FileNotFoundError:
        console.print(f"[red]Error:[/red] Project '{project_name}' not found")
        console.print(f"[dim]Workspace directory: {workspace_dir}[/dim]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@main.command()
@click.argument("project_name")
@click.option(
    "--workspace-dir",
    type=click.Path(path_type=Path),
    default="projects",
    help="Base directory for project workspaces",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["terminal", "html", "both"]),
    default="terminal",
    help="Preview format (default: terminal)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output path for HTML preview (default: workspace/preview.html)",
)
@click.option(
    "--open-browser",
    is_flag=True,
    help="Open HTML preview in browser",
)
def preview(
    project_name: str,
    workspace_dir: Path,
    format: str,
    output: Optional[Path],
    open_browser: bool,
):
    """Preview how a thread will look before posting.

    PROJECT_NAME: Name of the project to preview
    """
    try:
        # Get workspace
        workspace_manager = WorkspaceManager(base_dir=workspace_dir)
        try:
            workspace = workspace_manager.get_workspace(project_name)
        except FileNotFoundError:
            console.print(f"[red]Error:[/red] Project '{project_name}' not found")
            console.print(f"[dim]Workspace directory: {workspace_dir}[/dim]")
            sys.exit(1)

        # Load thread
        thread_file = workspace / "thread.json"
        if not thread_file.exists():
            console.print(f"[red]Error:[/red] Thread file not found: {thread_file}")
            console.print("[dim]Run 'vibesignal convert' first to generate the thread[/dim]")
            sys.exit(1)

        import json
        with open(thread_file) as f:
            thread_data = json.load(f)

        # Reconstruct Thread object
        from vibesignal.models import Thread, ThreadMetadata, Tweet
        tweets = [Tweet(**t) for t in thread_data["tweets"]]
        metadata = ThreadMetadata(**thread_data["metadata"])
        thread = Thread(
            tweets=tweets,
            metadata=metadata,
            hook=thread_data["hook"],
            call_to_action=thread_data.get("call_to_action"),
        )

        # Generate preview(s)
        if format in ["terminal", "both"]:
            previewer = TerminalPreview(console=console)
            previewer.show(thread, workspace)

        if format in ["html", "both"]:
            html_output = output or workspace / "preview.html"
            html_previewer = HTMLPreview()
            html_path = html_previewer.generate(thread, workspace, html_output)

            console.print()
            console.print(
                Panel.fit(
                    f"[green]HTML preview generated![/green]\n\n"
                    f"File: [yellow]{html_path}[/yellow]\n"
                    f"Open in browser to see Twitter-like mockup",
                    border_style="green",
                    title="[bold green]Preview Ready[/bold green]",
                )
            )

            # Open in browser if requested
            if open_browser:
                import webbrowser
                webbrowser.open(f"file://{html_path.absolute()}")
                console.print("[dim]Opening in browser...[/dim]")

    except Exception as e:
        console.print()
        console.print(
            Panel.fit(
                f"[red]Error:[/red] {str(e)}\n\n"
                f"Failed to generate preview.",
                border_style="red",
                title="[bold red]Preview Failed[/bold red]",
            )
        )
        import traceback
        traceback.print_exc()
        sys.exit(1)


@main.command()
@click.argument("project_name")
@click.option(
    "--workspace-dir",
    type=click.Path(path_type=Path),
    default="projects",
    help="Base directory for project workspaces",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Simulate posting without actually tweeting",
)
@click.option(
    "--delay",
    type=float,
    default=2.0,
    help="Delay in seconds between tweets (default: 2.0)",
)
def post_thread(project_name: str, workspace_dir: Path, dry_run: bool, delay: float):
    """Post a thread to Twitter/X.

    PROJECT_NAME: Name of the project to post
    """
    try:
        # Load config
        try:
            config = get_config()
        except Exception as e:
            console.print(f"[red]Error loading config:[/red] {e}")
            sys.exit(1)

        # Check Twitter credentials
        if not all([
            config.twitter_api_key,
            config.twitter_api_secret,
            config.twitter_access_token,
            config.twitter_access_token_secret,
        ]):
            console.print(
                "[red]Error:[/red] Twitter credentials not configured\n\n"
                "Set the following environment variables:\n"
                "  VIBESIGNAL_TWITTER_API_KEY\n"
                "  VIBESIGNAL_TWITTER_API_SECRET\n"
                "  VIBESIGNAL_TWITTER_ACCESS_TOKEN\n"
                "  VIBESIGNAL_TWITTER_ACCESS_TOKEN_SECRET\n"
                "  VIBESIGNAL_TWITTER_BEARER_TOKEN (optional)"
            )
            sys.exit(1)

        # Get workspace
        workspace_manager = WorkspaceManager(base_dir=workspace_dir)
        try:
            workspace = workspace_manager.get_workspace(project_name)
        except FileNotFoundError:
            console.print(f"[red]Error:[/red] Project '{project_name}' not found")
            console.print(f"[dim]Workspace directory: {workspace_dir}[/dim]")
            sys.exit(1)

        # Load thread
        thread_file = workspace / "thread.json"
        if not thread_file.exists():
            console.print(f"[red]Error:[/red] Thread file not found: {thread_file}")
            console.print("[dim]Run 'vibesignal convert' first to generate the thread[/dim]")
            sys.exit(1)

        import json
        with open(thread_file) as f:
            thread_data = json.load(f)

        # Reconstruct Thread object
        from vibesignal.models import Thread, ThreadMetadata, Tweet
        tweets = [Tweet(**t) for t in thread_data["tweets"]]
        metadata = ThreadMetadata(**thread_data["metadata"])
        thread = Thread(
            tweets=tweets,
            metadata=metadata,
            hook=thread_data["hook"],
            call_to_action=thread_data.get("call_to_action"),
        )

        # Initialize Twitter publisher
        publisher = TwitterPublisher(
            api_key=config.twitter_api_key,
            api_secret=config.twitter_api_secret,
            access_token=config.twitter_access_token,
            access_token_secret=config.twitter_access_token_secret,
            bearer_token=config.twitter_bearer_token if config.twitter_bearer_token else None,
        )

        # Verify credentials
        if not dry_run:
            try:
                user_info = publisher.verify_credentials()
                console.print(
                    f"[green]✓[/green] Authenticated as: @{user_info['username']}"
                )
            except Exception as e:
                console.print(f"[red]Error:[/red] Failed to verify Twitter credentials: {e}")
                sys.exit(1)

        # Show confirmation
        console.print()
        console.print(
            Panel.fit(
                f"[bold cyan]Ready to Post Thread[/bold cyan]\n\n"
                f"Project: [yellow]{project_name}[/yellow]\n"
                f"Tweets: [bold]{len(thread.tweets)}[/bold]\n"
                f"Images: [bold]{metadata.total_images}[/bold]\n"
                f"Mode: [{'yellow]DRY RUN' if dry_run else 'green]LIVE'}[/]\n\n"
                f"Hook: [dim]{thread.hook}[/dim]",
                border_style="cyan",
            )
        )

        if not dry_run:
            console.print()
            if not click.confirm("Post this thread to Twitter?", default=False):
                console.print("[yellow]Cancelled.[/yellow]")
                return

        # Post thread
        console.print()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"[cyan]{'Simulating' if dry_run else 'Posting'} thread...",
                total=len(thread.tweets),
            )

            published = publisher.publish_thread(
                thread=thread,
                workspace=workspace,
                dry_run=dry_run,
                delay_seconds=delay,
            )

            progress.update(task, completed=len(thread.tweets))

        # Success
        console.print()
        if dry_run:
            console.print(
                Panel.fit(
                    "[yellow]DRY RUN COMPLETE[/yellow]\n\n"
                    f"Would have posted {len(published)} tweets\n"
                    "No actual tweets were sent.",
                    border_style="yellow",
                    title="[bold yellow]Simulation[/bold yellow]",
                )
            )
        else:
            console.print(
                Panel.fit(
                    "[green]THREAD POSTED![/green]\n\n"
                    f"Posted {len(published)} tweets\n\n"
                    f"First tweet: [link={published[0]['url']}]{published[0]['url']}[/link]",
                    border_style="green",
                    title="[bold green]Success[/bold green]",
                )
            )

            # Show all tweet URLs
            console.print("\n[bold]Thread URLs:[/bold]")
            for i, tweet in enumerate(published, 1):
                console.print(f"  {i}. {tweet['url']}")

    except Exception as e:
        console.print()
        console.print(
            Panel.fit(
                f"[red]Error:[/red] {str(e)}\n\n"
                f"Failed to post thread.",
                border_style="red",
                title="[bold red]Posting Failed[/bold red]",
            )
        )
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
