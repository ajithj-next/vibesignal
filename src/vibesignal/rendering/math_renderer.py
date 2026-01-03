"""Mathematical equation rendering for Twitter-friendly images.

Renders LaTeX equations as high-quality PNG images optimized for
social media display. Creates beautiful, hand-crafted card-style visuals.
"""

import hashlib
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib import mathtext
from matplotlib.patches import FancyBboxPatch, Shadow
from matplotlib.transforms import Bbox
import matplotlib.patheffects as path_effects


class EquationStyle(Enum):
    """Visual styles for rendered equations."""
    DARK = "dark"           # Dark background, light text (Twitter dark mode)
    LIGHT = "light"         # Light background, dark text
    TRANSPARENT = "transparent"  # Transparent background
    VIBESIGNAL = "vibesignal"    # VibeSignal brand colors


@dataclass
class StyleConfig:
    """Configuration for equation rendering style."""
    background_color: str
    text_color: str
    accent_color: str
    transparent: bool = False
    font_size: int = 24
    padding: float = 0.3


# Predefined style configurations
STYLE_CONFIGS = {
    EquationStyle.DARK: StyleConfig(
        background_color="#0d1117",
        text_color="#f0f6fc",
        accent_color="#58a6ff",
    ),
    EquationStyle.LIGHT: StyleConfig(
        background_color="#ffffff",
        text_color="#1f2328",
        accent_color="#0969da",
    ),
    EquationStyle.TRANSPARENT: StyleConfig(
        background_color="#ffffff",
        text_color="#1f2328",
        accent_color="#0969da",
        transparent=True,
    ),
    EquationStyle.VIBESIGNAL: StyleConfig(
        background_color="#0d1117",
        text_color="#58a6ff",
        accent_color="#79c0ff",
        font_size=28,
    ),
}


class MathRenderer:
    """Render LaTeX equations as images for Twitter.

    Supports inline equations ($...$) and display equations ($$...$$).
    Generates high-quality PNG images optimized for social media.
    """

    # Regex patterns for equation detection
    DISPLAY_PATTERN = re.compile(r'\$\$(.+?)\$\$', re.DOTALL)
    INLINE_PATTERN = re.compile(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)')
    LATEX_BLOCK_PATTERN = re.compile(r'\\begin\{equation\}(.+?)\\end\{equation\}', re.DOTALL)

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        style: EquationStyle = EquationStyle.DARK,
        dpi: int = 200,
    ):
        """Initialize math renderer.

        Args:
            output_dir: Directory to save rendered images
            style: Visual style for equations
            dpi: Output resolution (higher = crisper on retina displays)
        """
        self.output_dir = output_dir or Path("images/equations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.style = style
        self.dpi = dpi
        self.style_config = STYLE_CONFIGS[style]

    def render_equation(
        self,
        latex: str,
        filename: Optional[str] = None,
        style: Optional[EquationStyle] = None,
        title: Optional[str] = None,
    ) -> Path:
        """Render a LaTeX equation to a PNG image.

        Args:
            latex: LaTeX equation string (without $ delimiters)
            filename: Output filename (auto-generated if not provided)
            style: Override default style
            title: Optional title/label above the equation

        Returns:
            Path: Path to rendered image
        """
        config = STYLE_CONFIGS[style] if style else self.style_config

        # Generate filename from equation hash if not provided
        if filename is None:
            eq_hash = hashlib.md5(latex.encode()).hexdigest()[:8]
            filename = f"eq_{eq_hash}.png"

        output_path = self.output_dir / filename

        # Create figure
        fig_height = 2.0 if title else 1.5
        fig, ax = plt.subplots(figsize=(10, fig_height))

        if not config.transparent:
            fig.patch.set_facecolor(config.background_color)
            ax.set_facecolor(config.background_color)

        # Render equation
        equation_text = f"${latex}$"

        # Position equation
        y_pos = 0.4 if title else 0.5

        ax.text(
            0.5, y_pos,
            equation_text,
            transform=ax.transAxes,
            fontsize=config.font_size,
            color=config.text_color,
            ha='center',
            va='center',
            math_fontfamily='cm',  # Computer Modern (classic LaTeX font)
        )

        # Add title if provided
        if title:
            ax.text(
                0.5, 0.85,
                title,
                transform=ax.transAxes,
                fontsize=config.font_size * 0.6,
                color=config.accent_color,
                ha='center',
                va='center',
                fontfamily='sans-serif',
                fontweight='bold',
            )

        ax.axis('off')

        # Save with proper transparency handling
        plt.tight_layout(pad=config.padding)
        plt.savefig(
            output_path,
            dpi=self.dpi,
            facecolor=fig.get_facecolor() if not config.transparent else 'none',
            transparent=config.transparent,
            bbox_inches='tight',
            pad_inches=0.2,
        )
        plt.close()

        return output_path

    def render_equation_card(
        self,
        latex: str,
        title: str,
        description: Optional[str] = None,
        filename: Optional[str] = None,
        style: Optional[EquationStyle] = None,
    ) -> Path:
        """Render an equation as a beautiful hand-crafted card.

        Creates a visually stunning card with:
        - Soft rounded corners
        - Subtle gradient background
        - Beveled edge effects
        - Hand-drawn aesthetic

        Args:
            latex: LaTeX equation string
            title: Card title
            description: Optional description below equation
            filename: Output filename
            style: Visual style

        Returns:
            Path: Path to rendered image
        """
        config = STYLE_CONFIGS[style] if style else self.style_config

        if filename is None:
            eq_hash = hashlib.md5(latex.encode()).hexdigest()[:8]
            filename = f"card_{eq_hash}.png"

        output_path = self.output_dir / filename

        # Create figure with extra space for shadow
        fig_height = 5.0 if description else 4.0
        fig, ax = plt.subplots(figsize=(12, fig_height))

        # Outer background (for shadow visibility)
        outer_bg = self._darken_color(config.background_color, 0.7)
        fig.patch.set_facecolor(outer_bg)
        ax.set_facecolor(outer_bg)

        # Card dimensions (in axes coordinates)
        card_left = 0.08
        card_bottom = 0.1
        card_width = 0.84
        card_height = 0.82

        # Draw shadow layers for depth (beveled effect)
        for i, offset in enumerate([0.025, 0.018, 0.012, 0.006]):
            shadow_alpha = 0.15 - i * 0.03
            shadow_color = self._darken_color(config.background_color, 0.3)
            shadow = FancyBboxPatch(
                (card_left + offset, card_bottom - offset),
                card_width, card_height,
                boxstyle="round,pad=0.02,rounding_size=0.04",
                facecolor=shadow_color,
                edgecolor='none',
                alpha=shadow_alpha,
                transform=ax.transAxes,
                zorder=1,
            )
            ax.add_patch(shadow)

        # Main card with gradient effect (simulated with layered patches)
        # Base card
        card = FancyBboxPatch(
            (card_left, card_bottom),
            card_width, card_height,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            facecolor=config.background_color,
            edgecolor=self._lighten_color(config.background_color, 1.3),
            linewidth=2,
            transform=ax.transAxes,
            zorder=2,
        )
        ax.add_patch(card)

        # Inner highlight for beveled effect (top-left light)
        highlight = FancyBboxPatch(
            (card_left + 0.005, card_bottom + 0.005),
            card_width - 0.01, card_height - 0.01,
            boxstyle="round,pad=0.02,rounding_size=0.035",
            facecolor='none',
            edgecolor=self._lighten_color(config.background_color, 1.5),
            linewidth=1,
            alpha=0.3,
            transform=ax.transAxes,
            zorder=3,
        )
        ax.add_patch(highlight)

        # Decorative corner accents
        corner_color = config.accent_color
        for corner_x, corner_y in [(card_left + 0.03, card_bottom + card_height - 0.06),
                                    (card_left + card_width - 0.03, card_bottom + card_height - 0.06)]:
            ax.plot([corner_x - 0.015, corner_x + 0.015], [corner_y, corner_y],
                    color=corner_color, linewidth=2, alpha=0.5, transform=ax.transAxes, zorder=4)

        # Title with subtle glow effect
        title_y = card_bottom + card_height - 0.12
        title_text = ax.text(
            0.5, title_y,
            title,
            transform=ax.transAxes,
            fontsize=config.font_size * 0.85,
            color=config.accent_color,
            ha='center',
            va='center',
            fontfamily='sans-serif',
            fontweight='bold',
            zorder=10,
        )
        title_text.set_path_effects([
            path_effects.withStroke(linewidth=3, foreground=self._darken_color(config.accent_color, 0.5), alpha=0.3),
        ])

        # Subtle divider line under title
        divider_y = title_y - 0.08
        ax.plot([0.2, 0.8], [divider_y, divider_y],
                color=config.accent_color, linewidth=1, alpha=0.3,
                transform=ax.transAxes, zorder=4)

        # Equation in center with glow
        equation_text = f"${latex}$"
        eq_y = 0.48 if description else 0.42

        # Equation glow/shadow
        for offset, alpha in [(0.003, 0.1), (0.002, 0.15), (0.001, 0.2)]:
            ax.text(
                0.5 + offset, eq_y - offset,
                equation_text,
                transform=ax.transAxes,
                fontsize=config.font_size * 1.3,
                color=config.accent_color,
                alpha=alpha,
                ha='center',
                va='center',
                math_fontfamily='cm',
                zorder=5,
            )

        # Main equation
        ax.text(
            0.5, eq_y,
            equation_text,
            transform=ax.transAxes,
            fontsize=config.font_size * 1.3,
            color=config.text_color,
            ha='center',
            va='center',
            math_fontfamily='cm',
            zorder=10,
        )

        # Description with hand-written style feel
        if description:
            ax.text(
                0.5, 0.18,
                description,
                transform=ax.transAxes,
                fontsize=config.font_size * 0.5,
                color=config.text_color,
                alpha=0.7,
                ha='center',
                va='center',
                fontfamily='sans-serif',
                style='italic',
                zorder=10,
            )

        # VibeSignal watermark (subtle)
        ax.text(
            0.95, 0.04,
            "VibeSignal",
            transform=ax.transAxes,
            fontsize=8,
            color=config.text_color,
            alpha=0.3,
            ha='right',
            va='bottom',
            fontfamily='sans-serif',
            zorder=10,
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        plt.tight_layout(pad=0)
        plt.savefig(
            output_path,
            dpi=self.dpi,
            facecolor=outer_bg,
            bbox_inches='tight',
            pad_inches=0.1,
        )
        plt.close()

        return output_path

    def _darken_color(self, hex_color: str, factor: float) -> str:
        """Darken a hex color by a factor."""
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        r = int(max(0, r * factor))
        g = int(max(0, g * factor))
        b = int(max(0, b * factor))
        return f"#{r:02x}{g:02x}{b:02x}"

    def _lighten_color(self, hex_color: str, factor: float) -> str:
        """Lighten a hex color by a factor."""
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        r = int(min(255, r * factor))
        g = int(min(255, g * factor))
        b = int(min(255, b * factor))
        return f"#{r:02x}{g:02x}{b:02x}"

    def extract_equations(self, text: str) -> list[dict]:
        """Extract equations from text content.

        Finds inline ($...$) and display ($$...$$) equations.

        Args:
            text: Text containing LaTeX equations

        Returns:
            list[dict]: List of equation data with type, latex, and position
        """
        equations = []

        # Find display equations first ($$...$$)
        for match in self.DISPLAY_PATTERN.finditer(text):
            equations.append({
                'type': 'display',
                'latex': match.group(1).strip(),
                'full_match': match.group(0),
                'start': match.start(),
                'end': match.end(),
            })

        # Find inline equations ($...$)
        # Skip positions already covered by display equations
        display_ranges = [(eq['start'], eq['end']) for eq in equations]

        for match in self.INLINE_PATTERN.finditer(text):
            # Check if this overlaps with a display equation
            overlaps = any(
                start <= match.start() < end or start < match.end() <= end
                for start, end in display_ranges
            )
            if not overlaps:
                equations.append({
                    'type': 'inline',
                    'latex': match.group(1).strip(),
                    'full_match': match.group(0),
                    'start': match.start(),
                    'end': match.end(),
                })

        # Sort by position
        equations.sort(key=lambda x: x['start'])

        return equations

    def render_equations_from_text(
        self,
        text: str,
        prefix: str = "eq",
    ) -> tuple[str, list[Path]]:
        """Extract and render all equations from text.

        Args:
            text: Text containing LaTeX equations
            prefix: Filename prefix for rendered images

        Returns:
            tuple: (modified_text with placeholders, list of image paths)
        """
        equations = self.extract_equations(text)
        if not equations:
            return text, []

        image_paths = []
        modified_text = text

        # Process in reverse order to maintain string positions
        for i, eq in enumerate(reversed(equations)):
            idx = len(equations) - 1 - i

            # Render equation
            filename = f"{prefix}_{idx}.png"
            path = self.render_equation(
                eq['latex'],
                filename=filename,
            )
            image_paths.insert(0, path)

            # Replace equation with placeholder
            placeholder = f"[Equation {idx + 1}: see image]"
            modified_text = (
                modified_text[:eq['start']] +
                placeholder +
                modified_text[eq['end']:]
            )

        return modified_text, image_paths

    def render_multi_equation_card(
        self,
        equations: list[tuple[str, str]],
        title: str,
        filename: Optional[str] = None,
    ) -> Path:
        """Render multiple equations on a single card.

        Args:
            equations: List of (label, latex) tuples
            title: Card title
            filename: Output filename

        Returns:
            Path: Path to rendered image
        """
        config = self.style_config

        if filename is None:
            content_hash = hashlib.md5(str(equations).encode()).hexdigest()[:8]
            filename = f"multi_{content_hash}.png"

        output_path = self.output_dir / filename

        n_equations = len(equations)
        fig_height = 2 + n_equations * 1.2
        fig, ax = plt.subplots(figsize=(12, fig_height))

        fig.patch.set_facecolor(config.background_color)
        ax.set_facecolor(config.background_color)

        # Title
        ax.text(
            0.5, 0.95,
            title,
            transform=ax.transAxes,
            fontsize=config.font_size * 0.8,
            color=config.accent_color,
            ha='center',
            va='top',
            fontfamily='sans-serif',
            fontweight='bold',
        )

        # Equations
        spacing = 0.8 / (n_equations + 1)
        for i, (label, latex) in enumerate(equations):
            y_pos = 0.85 - (i + 1) * spacing

            # Label
            ax.text(
                0.1, y_pos,
                f"{label}:",
                transform=ax.transAxes,
                fontsize=config.font_size * 0.5,
                color=config.text_color,
                alpha=0.7,
                ha='left',
                va='center',
                fontfamily='sans-serif',
            )

            # Equation
            ax.text(
                0.55, y_pos,
                f"${latex}$",
                transform=ax.transAxes,
                fontsize=config.font_size * 0.9,
                color=config.text_color,
                ha='center',
                va='center',
                math_fontfamily='cm',
            )

        ax.axis('off')

        plt.tight_layout()
        plt.savefig(
            output_path,
            dpi=self.dpi,
            facecolor=config.background_color,
            bbox_inches='tight',
            pad_inches=0.3,
        )
        plt.close()

        return output_path


# Common equations for quick reference
COMMON_EQUATIONS = {
    'euler': r"e^{i\pi} + 1 = 0",
    'pythagorean': r"a^2 + b^2 = c^2",
    'quadratic': r"x = \frac{-b \pm \sqrt{b^2-4ac}}{2a}",
    'einstein': r"E = mc^2",
    'schrodinger': r"i\hbar\frac{\partial}{\partial t}\Psi = \hat{H}\Psi",
    'fourier': r"f(x) = \sum_{n=-\infty}^{\infty} c_n e^{inx}",
    'gaussian': r"f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}",
    'derivative': r"\frac{df}{dx} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}",
    'integral': r"\int_a^b f(x)\,dx = F(b) - F(a)",
    'taylor': r"f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!}(x-a)^n",
    'bayes': r"P(A|B) = \frac{P(B|A)P(A)}{P(B)}",
    'entropy': r"H(X) = -\sum_{i} p(x_i) \log p(x_i)",
    'softmax': r"\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}",
    'cross_entropy': r"L = -\sum_{i} y_i \log(\hat{y}_i)",
    'gradient_descent': r"\theta_{t+1} = \theta_t - \alpha \nabla_\theta L(\theta_t)",
}


def render_common_equation(
    name: str,
    output_dir: Optional[Path] = None,
    style: EquationStyle = EquationStyle.DARK,
) -> Path:
    """Render a common equation by name.

    Args:
        name: Equation name (e.g., 'euler', 'fourier')
        output_dir: Output directory
        style: Visual style

    Returns:
        Path: Path to rendered image
    """
    if name not in COMMON_EQUATIONS:
        available = ', '.join(COMMON_EQUATIONS.keys())
        raise ValueError(f"Unknown equation: {name}. Available: {available}")

    renderer = MathRenderer(output_dir=output_dir, style=style)
    return renderer.render_equation(
        COMMON_EQUATIONS[name],
        filename=f"{name}.png",
        title=name.replace('_', ' ').title(),
    )
