"""Visual generator for inspirational quote cards."""

import textwrap
from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from vibesignal.inspiration.quotes import Quote


Style = Literal["chalkboard", "modern", "minimal", "inaugural"]


class QuoteVisualizer:
    """Generate beautiful visual cards for quotes."""

    # Color palettes for different styles
    PALETTES = {
        "chalkboard": {
            "background": "#1a1a2e",
            "primary": "#ffd93d",
            "secondary": "#e8e8e8",
            "accent": "#6bcbff",
            "muted": "#888888",
        },
        "modern": {
            "background": "#0f0f23",
            "primary": "#00d4ff",
            "secondary": "#ffffff",
            "accent": "#ff6b6b",
            "muted": "#666666",
        },
        "minimal": {
            "background": "#ffffff",
            "primary": "#1a1a1a",
            "secondary": "#333333",
            "accent": "#0066cc",
            "muted": "#999999",
        },
        "inaugural": {
            "background": "#0d1117",
            "primary": "#58a6ff",
            "secondary": "#f0f6fc",
            "accent": "#f78166",
            "muted": "#8b949e",
            "gradient_start": "#161b22",
            "gradient_end": "#0d1117",
        },
    }

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize visualizer.

        Args:
            output_dir: Directory to save generated images.
        """
        self.output_dir = output_dir or Path("images")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        quote: Quote,
        style: Style = "chalkboard",
        output_path: Optional[Path] = None,
        show_vibesignal: bool = True,
        inaugural: bool = False,
    ) -> Path:
        """Generate a visual card for a quote.

        Args:
            quote: Quote to visualize
            style: Visual style to use
            output_path: Custom output path
            show_vibesignal: Show VibeSignal branding
            inaugural: Use special inaugural design

        Returns:
            Path: Path to generated image
        """
        if inaugural:
            style = "inaugural"

        palette = self.PALETTES[style]

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor(palette["background"])
        ax.set_facecolor(palette["background"])
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 8)
        ax.axis("off")

        if style == "chalkboard":
            self._draw_chalkboard(ax, quote, palette, show_vibesignal)
        elif style == "modern":
            self._draw_modern(ax, quote, palette, show_vibesignal)
        elif style == "minimal":
            self._draw_minimal(ax, quote, palette, show_vibesignal)
        elif style == "inaugural":
            self._draw_inaugural(ax, quote, palette, show_vibesignal)

        # Save
        if output_path is None:
            safe_author = quote.author.lower().replace(" ", "_").replace(".", "")
            output_path = self.output_dir / f"quote_{safe_author}_{style}.png"

        plt.tight_layout()
        plt.savefig(
            output_path,
            dpi=150,
            facecolor=palette["background"],
            bbox_inches="tight",
            pad_inches=0.3,
        )
        plt.close()

        return output_path

    def _wrap_text(self, text: str, width: int = 45) -> str:
        """Wrap text for display."""
        return "\n".join(textwrap.wrap(text, width=width))

    def _draw_chalkboard(
        self, ax, quote: Quote, palette: dict, show_vibesignal: bool
    ):
        """Draw chalkboard style quote card."""
        # Add chalk dust effect
        np.random.seed(42)
        dust_x = np.random.uniform(0.5, 11.5, 50)
        dust_y = np.random.uniform(0.5, 7.5, 50)
        ax.scatter(dust_x, dust_y, c=palette["secondary"], s=1, alpha=0.2)

        # Quote text
        wrapped = self._wrap_text(quote.text, width=40)
        ax.text(
            6,
            5,
            f'"{wrapped}"',
            fontsize=18,
            color=palette["secondary"],
            ha="center",
            va="center",
            fontfamily="serif",
            style="italic",
            linespacing=1.4,
        )

        # Author
        ax.text(
            6,
            2.5,
            f"- {quote.author}",
            fontsize=16,
            color=palette["primary"],
            ha="center",
            fontfamily="serif",
            fontweight="bold",
        )

        # Field and theme
        ax.text(
            6,
            2,
            f"{quote.field} | {quote.theme}",
            fontsize=11,
            color=palette["muted"],
            ha="center",
            fontfamily="sans-serif",
        )

        # VibeSignal branding
        if show_vibesignal:
            ax.text(
                6,
                0.5,
                "VibeSignal",
                fontsize=12,
                color=palette["accent"],
                ha="center",
                fontfamily="sans-serif",
                alpha=0.7,
            )

    def _draw_modern(self, ax, quote: Quote, palette: dict, show_vibesignal: bool):
        """Draw modern gradient style quote card."""
        # Add subtle gradient effect with rectangles
        for i in range(20):
            alpha = 0.03 * (1 - i / 20)
            rect = patches.Rectangle(
                (0, i * 0.4),
                12,
                0.4,
                facecolor=palette["primary"],
                alpha=alpha,
            )
            ax.add_patch(rect)

        # Quote marks
        ax.text(
            1,
            6.5,
            '"',
            fontsize=80,
            color=palette["primary"],
            ha="center",
            fontfamily="serif",
            alpha=0.3,
        )

        # Quote text
        wrapped = self._wrap_text(quote.text, width=38)
        ax.text(
            6,
            4.5,
            wrapped,
            fontsize=17,
            color=palette["secondary"],
            ha="center",
            va="center",
            fontfamily="sans-serif",
            linespacing=1.5,
        )

        # Author with accent bar
        ax.plot([4, 8], [2.3, 2.3], color=palette["accent"], linewidth=2)
        ax.text(
            6,
            1.8,
            quote.author.upper(),
            fontsize=14,
            color=palette["primary"],
            ha="center",
            fontfamily="sans-serif",
            fontweight="bold",
        )

        # Field
        ax.text(
            6,
            1.3,
            quote.field,
            fontsize=10,
            color=palette["muted"],
            ha="center",
            fontfamily="sans-serif",
        )

        if show_vibesignal:
            ax.text(
                11,
                0.3,
                "VibeSignal",
                fontsize=10,
                color=palette["muted"],
                ha="right",
                fontfamily="sans-serif",
            )

    def _draw_minimal(self, ax, quote: Quote, palette: dict, show_vibesignal: bool):
        """Draw minimal clean style quote card."""
        # Simple border
        rect = patches.FancyBboxPatch(
            (0.5, 0.5),
            11,
            7,
            boxstyle="round,pad=0.05",
            facecolor="none",
            edgecolor=palette["muted"],
            linewidth=1,
            alpha=0.3,
        )
        ax.add_patch(rect)

        # Quote text
        wrapped = self._wrap_text(quote.text, width=42)
        ax.text(
            6,
            4.5,
            f'"{wrapped}"',
            fontsize=16,
            color=palette["primary"],
            ha="center",
            va="center",
            fontfamily="Georgia",
            linespacing=1.6,
        )

        # Author
        ax.text(
            6,
            2,
            quote.author,
            fontsize=14,
            color=palette["accent"],
            ha="center",
            fontfamily="sans-serif",
            fontweight="bold",
        )

        # Thin line
        ax.plot([5, 7], [1.5, 1.5], color=palette["muted"], linewidth=0.5)

        # Theme
        ax.text(
            6,
            1.1,
            quote.theme,
            fontsize=10,
            color=palette["muted"],
            ha="center",
            fontfamily="sans-serif",
        )

        if show_vibesignal:
            ax.text(
                11,
                0.3,
                "vibesignal",
                fontsize=9,
                color=palette["muted"],
                ha="right",
                fontfamily="monospace",
                alpha=0.5,
            )

    def _draw_inaugural(
        self, ax, quote: Quote, palette: dict, show_vibesignal: bool
    ):
        """Draw special inaugural style for first post."""
        # Starfield background effect
        np.random.seed(42)
        stars_x = np.random.uniform(0, 12, 100)
        stars_y = np.random.uniform(0, 8, 100)
        stars_size = np.random.uniform(0.5, 2, 100)
        ax.scatter(stars_x, stars_y, c=palette["secondary"], s=stars_size, alpha=0.3)

        # Glowing center effect
        circle = plt.Circle(
            (6, 4), 3.5, color=palette["primary"], alpha=0.03
        )
        ax.add_patch(circle)
        circle2 = plt.Circle(
            (6, 4), 2.5, color=palette["primary"], alpha=0.05
        )
        ax.add_patch(circle2)

        # Top label
        ax.text(
            6,
            7.3,
            "INAUGURAL POST",
            fontsize=10,
            color=palette["accent"],
            ha="center",
            fontfamily="sans-serif",
            fontweight="bold",
        )

        # Quote text - larger and more prominent
        wrapped = self._wrap_text(quote.text, width=35)
        ax.text(
            6,
            4.8,
            f'"{wrapped}"',
            fontsize=20,
            color=palette["secondary"],
            ha="center",
            va="center",
            fontfamily="Georgia",
            style="italic",
            linespacing=1.5,
        )

        # Decorative line
        ax.plot([3, 9], [2.8, 2.8], color=palette["primary"], linewidth=1, alpha=0.5)

        # Author with glow effect
        ax.text(
            6,
            2.3,
            quote.author,
            fontsize=18,
            color=palette["primary"],
            ha="center",
            fontfamily="sans-serif",
            fontweight="bold",
        )

        # Field and era
        ax.text(
            6,
            1.8,
            f"{quote.field} | {quote.era}",
            fontsize=11,
            color=palette["muted"],
            ha="center",
            fontfamily="sans-serif",
        )

        # Context if available
        if quote.context:
            ax.text(
                6,
                1.3,
                f"({quote.context})",
                fontsize=9,
                color=palette["muted"],
                ha="center",
                fontfamily="sans-serif",
                style="italic",
                alpha=0.7,
            )

        # VibeSignal branding - prominent for inaugural
        if show_vibesignal:
            ax.text(
                6,
                0.4,
                "VibeSignal",
                fontsize=16,
                color=palette["primary"],
                ha="center",
                fontfamily="sans-serif",
                fontweight="bold",
            )
            ax.text(
                6,
                0.1,
                "First Principles -> Clear Insights -> Good Vibes",
                fontsize=8,
                color=palette["muted"],
                ha="center",
                fontfamily="sans-serif",
            )

    def generate_inaugural(
        self,
        quote: Optional[Quote] = None,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Generate special inaugural post image.

        Args:
            quote: Quote to use. Uses Feynman's famous quote if not provided.
            output_path: Custom output path.

        Returns:
            Path: Path to generated image.
        """
        if quote is None:
            from vibesignal.inspiration.quotes import QuoteDatabase

            db = QuoteDatabase()
            quote = db.get_inaugural()

        if output_path is None:
            output_path = self.output_dir / "vibesignal_inaugural.png"

        return self.generate(
            quote=quote,
            style="inaugural",
            output_path=output_path,
            show_vibesignal=True,
            inaugural=True,
        )
