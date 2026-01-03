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
from vibesignal.inspiration import QuoteDatabase, QuoteVisualizer

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
@click.option(
    "--skip-preview",
    is_flag=True,
    help="Skip the terminal preview before posting",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    help="Skip confirmation prompt (use with caution)",
)
def post_thread(
    project_name: str,
    workspace_dir: Path,
    dry_run: bool,
    delay: float,
    skip_preview: bool,
    yes: bool,
):
    """Post a thread to Twitter/X.

    PROJECT_NAME: Name of the project to post

    This command will:
    1. Show a preview of the thread
    2. Ask for confirmation before posting
    3. Post all tweets as a threaded conversation
    4. Save a publish record for later deletion if needed
    """
    try:
        # Load config
        try:
            config = get_config()
        except Exception as e:
            console.print(f"[red]Error loading config:[/red] {e}")
            sys.exit(1)

        # Check Twitter credentials
        if not dry_run and not all([
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

        # Check if already published
        publish_record_path = workspace / "publish_record.json"
        if publish_record_path.exists() and not dry_run:
            import json
            with open(publish_record_path) as f:
                record = json.load(f)
            console.print(
                Panel.fit(
                    f"[yellow]Warning:[/yellow] This thread was already published!\n\n"
                    f"Published at: {record.get('published_at', 'Unknown')}\n"
                    f"First tweet: {record.get('first_tweet_url', 'Unknown')}\n\n"
                    f"Use 'vibesignal delete-thread {project_name}' to delete it first,\n"
                    f"or use --dry-run to simulate.",
                    border_style="yellow",
                    title="[bold yellow]Already Published[/bold yellow]",
                )
            )
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

        # Show preview unless skipped
        if not skip_preview:
            console.print()
            console.print("[bold cyan]Thread Preview:[/bold cyan]")
            console.print()
            previewer = TerminalPreview(console=console)
            previewer.show_compact(thread)

        # Initialize Twitter publisher
        publisher = TwitterPublisher(
            api_key=config.twitter_api_key or "",
            api_secret=config.twitter_api_secret or "",
            access_token=config.twitter_access_token or "",
            access_token_secret=config.twitter_access_token_secret or "",
            bearer_token=config.twitter_bearer_token if config.twitter_bearer_token else None,
        )

        # Verify credentials
        user_info = None
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
                f"Mode: [{'yellow]DRY RUN' if dry_run else 'green]LIVE'}[/]\n"
                f"Account: [bold]@{user_info['username'] if user_info else 'N/A'}[/bold]\n\n"
                f"Hook: [dim]{thread.hook[:100]}{'...' if len(thread.hook) > 100 else ''}[/dim]",
                border_style="cyan",
            )
        )

        if not dry_run and not yes:
            console.print()
            console.print("[bold red]This will post to your LIVE Twitter account![/bold red]")
            console.print("[dim]You can delete the thread later with 'vibesignal delete-thread'[/dim]")
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
                    f"First tweet: [link={published[0]['url']}]{published[0]['url']}[/link]\n\n"
                    f"[dim]A publish record has been saved. You can delete this thread later with:[/dim]\n"
                    f"  vibesignal delete-thread {project_name}",
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


@main.command()
@click.argument("project_name")
@click.option(
    "--workspace-dir",
    type=click.Path(path_type=Path),
    default="projects",
    help="Base directory for project workspaces",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    help="Skip confirmation prompt",
)
@click.option(
    "--delay",
    type=float,
    default=1.0,
    help="Delay in seconds between deletions (default: 1.0)",
)
def delete_thread(project_name: str, workspace_dir: Path, yes: bool, delay: float):
    """Delete a published thread from Twitter/X.

    PROJECT_NAME: Name of the project whose thread to delete

    This command will delete all tweets in the thread that were posted
    using 'vibesignal post-thread'. It uses the saved publish record
    to identify which tweets to delete.
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
                "  VIBESIGNAL_TWITTER_ACCESS_TOKEN_SECRET"
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

        # Check for publish record
        publish_record_path = workspace / "publish_record.json"
        if not publish_record_path.exists():
            # Check if there's an archived record
            archived_path = workspace / "publish_record_deleted.json"
            if archived_path.exists():
                console.print(
                    Panel.fit(
                        "[yellow]Thread was already deleted.[/yellow]\n\n"
                        f"See {archived_path} for deletion history.",
                        border_style="yellow",
                    )
                )
            else:
                console.print(
                    Panel.fit(
                        "[yellow]No publish record found.[/yellow]\n\n"
                        "This thread may not have been published yet,\n"
                        "or the publish record was manually removed.",
                        border_style="yellow",
                    )
                )
            sys.exit(1)

        # Load publish record
        import json
        with open(publish_record_path) as f:
            record = json.load(f)

        # Initialize Twitter publisher
        publisher = TwitterPublisher(
            api_key=config.twitter_api_key,
            api_secret=config.twitter_api_secret,
            access_token=config.twitter_access_token,
            access_token_secret=config.twitter_access_token_secret,
            bearer_token=config.twitter_bearer_token if config.twitter_bearer_token else None,
        )

        # Verify credentials
        try:
            user_info = publisher.verify_credentials()
            console.print(
                f"[green]✓[/green] Authenticated as: @{user_info['username']}"
            )
        except Exception as e:
            console.print(f"[red]Error:[/red] Failed to verify Twitter credentials: {e}")
            sys.exit(1)

        # Show what will be deleted
        console.print()
        console.print(
            Panel.fit(
                f"[bold red]Delete Thread[/bold red]\n\n"
                f"Project: [yellow]{project_name}[/yellow]\n"
                f"Published at: {record.get('published_at', 'Unknown')}\n"
                f"Tweet count: [bold]{record.get('tweet_count', len(record['tweets']))}[/bold]\n"
                f"First tweet: {record.get('first_tweet_url', 'Unknown')}",
                border_style="red",
            )
        )

        console.print()
        console.print("[bold]Tweets to delete:[/bold]")
        for tweet in record["tweets"]:
            console.print(f"  {tweet['position']}. {tweet['url']}")
            console.print(f"     [dim]{tweet['text'][:60]}...[/dim]")

        if not yes:
            console.print()
            console.print("[bold red]This action cannot be undone![/bold red]")
            if not click.confirm("Delete all these tweets?", default=False):
                console.print("[yellow]Cancelled.[/yellow]")
                return

        # Delete thread
        console.print()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                "[red]Deleting tweets...",
                total=len(record["tweets"]),
            )

            results = publisher.delete_thread(workspace, delay_seconds=delay)

            progress.update(task, completed=results["total"])

        # Show results
        console.print()
        if results["failed"] > 0:
            console.print(
                Panel.fit(
                    f"[yellow]Deletion completed with errors[/yellow]\n\n"
                    f"Deleted: {results['deleted']}\n"
                    f"Already deleted: {results['already_deleted']}\n"
                    f"Failed: {results['failed']}\n\n"
                    f"Errors:\n" + "\n".join(
                        f"  - {e['id']}: {e['error']}" for e in results["errors"]
                    ),
                    border_style="yellow",
                    title="[bold yellow]Partial Deletion[/bold yellow]",
                )
            )
        else:
            console.print(
                Panel.fit(
                    f"[green]Thread deleted successfully![/green]\n\n"
                    f"Deleted: {results['deleted']} tweets\n"
                    f"Already deleted: {results['already_deleted']}",
                    border_style="green",
                    title="[bold green]Deletion Complete[/bold green]",
                )
            )

    except Exception as e:
        console.print()
        console.print(
            Panel.fit(
                f"[red]Error:[/red] {str(e)}\n\n"
                f"Failed to delete thread.",
                border_style="red",
                title="[bold red]Deletion Failed[/bold red]",
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
def publish_status(project_name: str, workspace_dir: Path):
    """Check the publish status of a project.

    PROJECT_NAME: Name of the project to check
    """
    try:
        # Get workspace
        workspace_manager = WorkspaceManager(base_dir=workspace_dir)
        try:
            workspace = workspace_manager.get_workspace(project_name)
        except FileNotFoundError:
            console.print(f"[red]Error:[/red] Project '{project_name}' not found")
            sys.exit(1)

        # Check for publish record
        publish_record_path = workspace / "publish_record.json"
        deleted_record_path = workspace / "publish_record_deleted.json"

        if publish_record_path.exists():
            import json
            with open(publish_record_path) as f:
                record = json.load(f)

            console.print(
                Panel.fit(
                    f"[green]Published[/green]\n\n"
                    f"Published at: {record.get('published_at', 'Unknown')}\n"
                    f"Username: @{record.get('username', 'Unknown')}\n"
                    f"Tweet count: {record.get('tweet_count', len(record['tweets']))}\n\n"
                    f"First tweet:\n{record.get('first_tweet_url', 'Unknown')}",
                    border_style="green",
                    title=f"[bold green]{project_name}[/bold green]",
                )
            )

            console.print("\n[bold]All tweets:[/bold]")
            for tweet in record["tweets"]:
                console.print(f"  {tweet['position']}. {tweet['url']}")

        elif deleted_record_path.exists():
            import json
            with open(deleted_record_path) as f:
                record = json.load(f)

            console.print(
                Panel.fit(
                    f"[yellow]Deleted[/yellow]\n\n"
                    f"Originally published: {record.get('published_at', 'Unknown')}\n"
                    f"Deleted at: {record.get('deleted_at', 'Unknown')}\n"
                    f"Tweet count: {record.get('tweet_count', 'Unknown')}",
                    border_style="yellow",
                    title=f"[bold yellow]{project_name}[/bold yellow]",
                )
            )

        else:
            console.print(
                Panel.fit(
                    f"[dim]Not Published[/dim]\n\n"
                    f"This thread has not been posted to Twitter yet.\n\n"
                    f"Use 'vibesignal post-thread {project_name}' to publish.",
                    border_style="dim",
                    title=f"[bold]{project_name}[/bold]",
                )
            )

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@main.command()
@click.option(
    "--style",
    "-s",
    type=click.Choice(["chalkboard", "modern", "minimal", "inaugural"]),
    default="chalkboard",
    help="Visual style for the quote card",
)
@click.option(
    "--author",
    "-a",
    type=str,
    default=None,
    help="Filter quotes by author (e.g., 'Feynman', 'Einstein')",
)
@click.option(
    "--theme",
    "-t",
    type=str,
    default=None,
    help="Filter quotes by theme (e.g., 'First Principles', 'Curiosity')",
)
@click.option(
    "--daily",
    is_flag=True,
    help="Get today's daily quote (deterministic per day)",
)
@click.option(
    "--inaugural",
    is_flag=True,
    help="Generate special inaugural post",
)
@click.option(
    "--post",
    is_flag=True,
    help="Post the inspiration to Twitter",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Simulate posting without actually tweeting",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Custom output path for the image",
)
@click.option(
    "--list-authors",
    is_flag=True,
    help="List all available authors",
)
@click.option(
    "--list-themes",
    is_flag=True,
    help="List all available themes",
)
def inspire(
    style: str,
    author: Optional[str],
    theme: Optional[str],
    daily: bool,
    inaugural: bool,
    post: bool,
    dry_run: bool,
    output: Optional[Path],
    list_authors: bool,
    list_themes: bool,
):
    """Generate and post daily inspirational quotes from scientific icons.

    Creates beautiful visual cards with quotes from Feynman, Einstein,
    Curie, Turing, and other scientific legends.

    Examples:

        vibesignal inspire                    # Random quote
        vibesignal inspire --daily            # Today's quote
        vibesignal inspire --author Feynman   # Quote from Feynman
        vibesignal inspire --inaugural --post # Post inaugural tweet
    """
    try:
        db = QuoteDatabase()

        # Handle list options
        if list_authors:
            console.print("[bold cyan]Available Authors:[/bold cyan]")
            for auth in db.authors:
                count = len(db.get_by_author(auth))
                console.print(f"  - {auth} ({count} quotes)")
            return

        if list_themes:
            console.print("[bold cyan]Available Themes:[/bold cyan]")
            for th in db.themes:
                count = len(db.get_by_theme(th))
                console.print(f"  - {th} ({count} quotes)")
            return

        # Select quote
        if inaugural:
            quote = db.get_inaugural()
            style = "inaugural"
        elif daily:
            quote = db.get_daily()
        elif author:
            quotes = db.get_by_author(author)
            if not quotes:
                console.print(f"[red]No quotes found for author: {author}[/red]")
                console.print("[dim]Use --list-authors to see available authors[/dim]")
                sys.exit(1)
            import random
            quote = random.choice(quotes)
        elif theme:
            quotes = db.get_by_theme(theme)
            if not quotes:
                console.print(f"[red]No quotes found for theme: {theme}[/red]")
                console.print("[dim]Use --list-themes to see available themes[/dim]")
                sys.exit(1)
            import random
            quote = random.choice(quotes)
        else:
            quote = db.get_random()

        # Display quote
        console.print()
        console.print(
            Panel.fit(
                f'[italic]"{quote.text}"[/italic]\n\n'
                f"[bold]{quote.author}[/bold]\n"
                f"[dim]{quote.field} | {quote.theme}[/dim]"
                + (f"\n[dim italic]({quote.context})[/dim italic]" if quote.context else ""),
                border_style="cyan",
                title="[bold cyan]Inspiration[/bold cyan]" if not inaugural else "[bold yellow]Inaugural Post[/bold yellow]",
            )
        )

        # Generate image
        console.print()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Generating visual...", total=1)

            visualizer = QuoteVisualizer(output_dir=Path("images"))
            if inaugural:
                image_path = visualizer.generate_inaugural(quote=quote, output_path=output)
            else:
                image_path = visualizer.generate(
                    quote=quote,
                    style=style,
                    output_path=output,
                )

            progress.update(task, completed=1)

        console.print(f"[green]Image saved:[/green] {image_path}")

        # Post to Twitter if requested
        if post or dry_run:
            # Compose tweet text
            if inaugural:
                tweet_text = (
                    f'"{quote.text}"\n\n'
                    f"- {quote.author}\n\n"
                    "Introducing VibeSignal: where first principles thinking meets clear communication.\n\n"
                    "Deep insights. Good vibes. Coming soon."
                )
            else:
                tweet_text = (
                    f'"{quote.text}"\n\n'
                    f"- {quote.author} ({quote.field})\n\n"
                    f"#{quote.theme.replace(' ', '')} #FirstPrinciples"
                )

            console.print()
            console.print("[bold]Tweet text:[/bold]")
            console.print(Panel(tweet_text, border_style="dim"))
            console.print(f"[dim]Characters: {len(tweet_text)}/280[/dim]")

            if len(tweet_text) > 280:
                console.print("[yellow]Warning: Tweet exceeds 280 characters, truncating...[/yellow]")
                tweet_text = tweet_text[:277] + "..."

            # Load config and post
            try:
                config = get_config()
            except Exception as e:
                console.print(f"[red]Error loading config:[/red] {e}")
                sys.exit(1)

            if not dry_run and not all([
                config.twitter_api_key,
                config.twitter_api_secret,
                config.twitter_access_token,
                config.twitter_access_token_secret,
            ]):
                console.print("[red]Twitter credentials not configured[/red]")
                sys.exit(1)

            publisher = TwitterPublisher(
                api_key=config.twitter_api_key or "",
                api_secret=config.twitter_api_secret or "",
                access_token=config.twitter_access_token or "",
                access_token_secret=config.twitter_access_token_secret or "",
                bearer_token=config.twitter_bearer_token if config.twitter_bearer_token else None,
            )

            if not dry_run:
                user_info = publisher.verify_credentials()
                console.print(f"[green]Authenticated as:[/green] @{user_info['username']}")

                if not click.confirm("\nPost this inspiration to Twitter?", default=False):
                    console.print("[yellow]Cancelled.[/yellow]")
                    return

                # Upload image and post
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task("[cyan]Posting to Twitter...", total=2)

                    # Upload image
                    media = publisher.api_v1.media_upload(filename=str(image_path))
                    progress.advance(task)

                    # Post tweet
                    response = publisher.client.create_tweet(
                        text=tweet_text,
                        media_ids=[media.media_id_string],
                    )
                    progress.advance(task)

                tweet_url = f"https://twitter.com/{user_info['username']}/status/{response.data['id']}"
                console.print()
                console.print(
                    Panel.fit(
                        f"[green]Posted successfully![/green]\n\n"
                        f"[link={tweet_url}]{tweet_url}[/link]",
                        border_style="green",
                        title="[bold green]Inspiration Posted[/bold green]",
                    )
                )
            else:
                console.print()
                console.print(
                    Panel.fit(
                        "[yellow]DRY RUN[/yellow]\n\n"
                        f"Would post tweet with image:\n{image_path}",
                        border_style="yellow",
                    )
                )

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


@main.command()
@click.argument("latex", required=False)
@click.option(
    "--title",
    "-t",
    type=str,
    default=None,
    help="Title for the equation card",
)
@click.option(
    "--description",
    "-d",
    type=str,
    default=None,
    help="Description text below the equation",
)
@click.option(
    "--style",
    "-s",
    type=click.Choice(["dark", "light", "vibesignal"]),
    default="vibesignal",
    help="Visual style for the card",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Custom output path for the image",
)
@click.option(
    "--preset",
    "-p",
    type=str,
    default=None,
    help="Use a preset equation (euler, fourier, bayes, gradient_descent, etc.)",
)
@click.option(
    "--list-presets",
    is_flag=True,
    help="List all available equation presets",
)
@click.option(
    "--post",
    is_flag=True,
    help="Post the equation card to Twitter",
)
def equation(
    latex: Optional[str],
    title: Optional[str],
    description: Optional[str],
    style: str,
    output: Optional[Path],
    preset: Optional[str],
    list_presets: bool,
    post: bool,
):
    """Generate beautiful equation cards from LaTeX.

    Creates hand-crafted visual cards with mathematical equations,
    perfect for sharing on Twitter.

    Examples:

        vibesignal equation "E = mc^2" -t "Mass-Energy Equivalence"

        vibesignal equation --preset euler

        vibesignal equation "\\sum_{n=1}^{\\infty} \\frac{1}{n^2}" -t "Basel Problem"

        vibesignal equation --list-presets
    """
    from vibesignal.rendering import MathRenderer, EquationStyle

    try:
        # Handle presets listing
        from vibesignal.rendering.math_renderer import COMMON_EQUATIONS

        if list_presets:
            console.print("[bold cyan]Available Equation Presets:[/bold cyan]\n")
            for name, eq in COMMON_EQUATIONS.items():
                display_name = name.replace('_', ' ').title()
                console.print(f"  [green]{name}[/green]")
                console.print(f"    {display_name}: ${eq}$\n")
            return

        # Get equation from preset or argument
        if preset:
            if preset not in COMMON_EQUATIONS:
                console.print(f"[red]Unknown preset: {preset}[/red]")
                console.print("[dim]Use --list-presets to see available presets[/dim]")
                sys.exit(1)
            latex_eq = COMMON_EQUATIONS[preset]
            if not title:
                title = preset.replace('_', ' ').title()
        elif latex:
            latex_eq = latex
        else:
            console.print("[red]Error: Please provide a LaTeX equation or use --preset[/red]")
            console.print("\nExamples:")
            console.print('  vibesignal equation "E = mc^2" -t "Einstein\'s Equation"')
            console.print("  vibesignal equation --preset euler")
            sys.exit(1)

        # Set default title if not provided
        if not title:
            title = "Mathematical Equation"

        # Map style string to enum
        style_map = {
            "dark": EquationStyle.DARK,
            "light": EquationStyle.LIGHT,
            "vibesignal": EquationStyle.VIBESIGNAL,
        }
        eq_style = style_map[style]

        # Set output path
        if output is None:
            output_dir = Path("images/equations")
        else:
            output_dir = output.parent

        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize renderer
        renderer = MathRenderer(output_dir=output_dir, style=eq_style)

        # Generate the card
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("[cyan]Rendering equation card...", total=None)

            if output:
                image_path = renderer.render_equation_card(
                    latex_eq,
                    title=title,
                    description=description,
                    filename=output.name,
                )
            else:
                image_path = renderer.render_equation_card(
                    latex_eq,
                    title=title,
                    description=description,
                )

        console.print()
        console.print(
            Panel.fit(
                f"[bold]{title}[/bold]\n\n"
                f"[cyan]${latex_eq}$[/cyan]\n\n"
                f"[green]Image saved to:[/green] {image_path}",
                border_style="cyan",
                title="[bold cyan]Equation Card Generated[/bold cyan]",
            )
        )

        # Post to Twitter if requested
        if post:
            config = get_config()
            publisher = TwitterPublisher(
                api_key=config.twitter_api_key,
                api_secret=config.twitter_api_secret,
                access_token=config.twitter_access_token,
                access_token_secret=config.twitter_access_token_secret,
            )

            # Verify credentials
            user_info = publisher.verify_credentials()
            console.print(f"\n[dim]Posting as @{user_info['username']}...[/dim]")

            # Create tweet text
            tweet_text = f"{title}\n\n#Math #VibeSignal"
            if description:
                tweet_text = f"{title}\n\n{description}\n\n#Math #VibeSignal"

            # Post
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task("[cyan]Posting to Twitter...", total=None)

                media = publisher.api_v1.media_upload(filename=str(image_path))
                response = publisher.client.create_tweet(
                    text=tweet_text,
                    media_ids=[media.media_id_string],
                )

            tweet_url = f"https://twitter.com/{user_info['username']}/status/{response.data['id']}"
            console.print()
            console.print(
                Panel.fit(
                    f"[green]Posted successfully![/green]\n\n"
                    f"[link={tweet_url}]{tweet_url}[/link]",
                    border_style="green",
                    title="[bold green]Equation Posted[/bold green]",
                )
            )

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
