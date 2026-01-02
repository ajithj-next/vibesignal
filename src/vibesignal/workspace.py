"""Workspace management for VibeSignal projects."""

import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

from vibesignal.models import Thread


class WorkspaceManager:
    """Manage project workspaces for thread generation.

    Each project gets a dedicated workspace with organized structure:
    projects/{project_name}/
        ├── images/          # Extracted images
        ├── thread.json      # Thread data
        ├── thread.txt       # Human-readable thread
        ├── thread.md        # Markdown version
        ├── metadata.json    # Project metadata
        └── source/          # Source notebook (optional copy)
    """

    def __init__(self, base_dir: Path | str = "projects"):
        """Initialize workspace manager.

        Args:
            base_dir: Base directory for all projects
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_workspace(
        self,
        project_name: str,
        source_notebook: Optional[Path] = None,
        copy_source: bool = True,
    ) -> Path:
        """Create a new project workspace.

        Args:
            project_name: Name for the project
            source_notebook: Path to source notebook
            copy_source: Whether to copy source notebook to workspace

        Returns:
            Path: Path to created workspace directory
        """
        # Create workspace directory
        workspace = self.base_dir / self._sanitize_name(project_name)
        workspace.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (workspace / "images").mkdir(exist_ok=True)
        (workspace / "source").mkdir(exist_ok=True)

        # Copy source notebook if requested
        if source_notebook and copy_source and source_notebook.exists():
            dest = workspace / "source" / source_notebook.name
            shutil.copy2(source_notebook, dest)

        # Create metadata
        metadata = {
            "project_name": project_name,
            "created_at": datetime.now(UTC).isoformat(),
            "source_notebook": str(source_notebook) if source_notebook else None,
            "workspace_path": str(workspace),
        }

        import json

        with open(workspace / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return workspace

    def get_workspace(self, project_name: str) -> Path:
        """Get path to existing workspace.

        Args:
            project_name: Project name

        Returns:
            Path: Workspace directory path

        Raises:
            FileNotFoundError: If workspace doesn't exist
        """
        workspace = self.base_dir / self._sanitize_name(project_name)
        if not workspace.exists():
            raise FileNotFoundError(f"Workspace not found: {workspace}")
        return workspace

    def list_workspaces(self) -> list[str]:
        """List all project workspaces.

        Returns:
            list[str]: List of project names
        """
        if not self.base_dir.exists():
            return []

        return [
            d.name
            for d in self.base_dir.iterdir()
            if d.is_dir() and (d / "metadata.json").exists()
        ]

    def get_images_dir(self, workspace: Path) -> Path:
        """Get images directory for workspace.

        Args:
            workspace: Workspace path

        Returns:
            Path: Images directory
        """
        images_dir = workspace / "images"
        images_dir.mkdir(exist_ok=True)
        return images_dir

    def get_output_path(self, workspace: Path, format: str = "json") -> Path:
        """Get output file path for workspace.

        Args:
            workspace: Workspace path
            format: Output format (json, text, markdown)

        Returns:
            Path: Output file path
        """
        extensions = {"json": ".json", "text": ".txt", "markdown": ".md"}
        ext = extensions.get(format, ".json")
        return workspace / f"thread{ext}"

    def save_thread_metadata(
        self, workspace: Path, thread: Thread, additional_data: Optional[dict] = None
    ) -> Path:
        """Save additional thread metadata to workspace.

        Args:
            workspace: Workspace path
            thread: Thread object
            additional_data: Additional metadata to save

        Returns:
            Path: Path to metadata file
        """
        import json

        # Read existing metadata
        metadata_path = workspace / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Update with thread info
        metadata.update(
            {
                "thread_generated_at": thread.metadata.generated_at,
                "total_tweets": thread.metadata.total_tweets,
                "total_images": thread.metadata.total_images,
                "claude_model": thread.metadata.claude_model,
                "hook": thread.hook,
            }
        )

        # Add any additional data
        if additional_data:
            metadata.update(additional_data)

        # Save updated metadata
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return metadata_path

    def cleanup_workspace(self, project_name: str, confirm: bool = False) -> bool:
        """Delete a workspace.

        Args:
            project_name: Project name
            confirm: Safety flag - must be True to actually delete

        Returns:
            bool: True if deleted, False otherwise
        """
        if not confirm:
            return False

        workspace = self.base_dir / self._sanitize_name(project_name)
        if workspace.exists():
            shutil.rmtree(workspace)
            return True
        return False

    def _sanitize_name(self, name: str) -> str:
        """Sanitize project name for filesystem.

        Args:
            name: Project name

        Returns:
            str: Sanitized name
        """
        # Remove invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            name = name.replace(char, "_")

        # Remove leading/trailing spaces and dots
        name = name.strip(". ")

        # Limit length
        if len(name) > 100:
            name = name[:100]

        # Ensure not empty
        if not name:
            name = "unnamed_project"

        return name

    def get_workspace_summary(self, project_name: str) -> dict:
        """Get summary information about a workspace.

        Args:
            project_name: Project name

        Returns:
            dict: Workspace summary
        """
        import json

        workspace = self.get_workspace(project_name)

        # Read metadata
        with open(workspace / "metadata.json", "r") as f:
            metadata = json.load(f)

        # Count files
        image_count = len(list((workspace / "images").glob("*"))) if (workspace / "images").exists() else 0

        outputs = []
        for ext in ["json", "txt", "md"]:
            output_file = workspace / f"thread.{ext}"
            if output_file.exists():
                outputs.append(ext)

        return {
            "project_name": project_name,
            "workspace_path": str(workspace),
            "created_at": metadata.get("created_at"),
            "source_notebook": metadata.get("source_notebook"),
            "total_tweets": metadata.get("total_tweets"),
            "total_images": metadata.get("total_images"),
            "image_files_count": image_count,
            "available_outputs": outputs,
            "metadata": metadata,
        }
