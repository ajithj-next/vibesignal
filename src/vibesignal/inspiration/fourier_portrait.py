"""Fourier Transform portrait generator - creates silhouettes from FFT decomposition.

This module implements the mathematical beauty of reconstructing complex shapes
from simple sine waves - a perfect metaphor for first principles thinking.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
import matplotlib.animation as animation

# Image processing imports
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class FourierPortrait:
    """Generate portraits using Fourier series decomposition.

    Decomposes a silhouette contour into frequency components and
    reconstructs it by summing rotating circles (epicycles).
    """

    def __init__(self, n_coefficients: int = 100):
        """Initialize Fourier portrait generator.

        Args:
            n_coefficients: Number of Fourier coefficients to use
        """
        self.n_coefficients = n_coefficients
        self.coefficients = None
        self.contour = None

    def get_feynman_silhouette(self) -> np.ndarray:
        """Generate Feynman's iconic profile silhouette as a contour.

        Creates a recognizable side profile silhouette inspired by
        Feynman's characteristic features: wavy hair, strong brow,
        prominent nose, defined jaw.

        Returns:
            np.ndarray: Complex array of contour points
        """
        # Define profile as a series of control points (x, y)
        # Starting from back of head, going clockwise
        # Coordinates designed for a right-facing profile

        profile_points = [
            # Back of head/neck base
            (-0.28, 0.15),
            (-0.32, 0.30),
            (-0.35, 0.45),
            (-0.33, 0.60),

            # Back of hair - WAVY like Feynman's
            (-0.30, 0.72),
            (-0.32, 0.78),  # Wave out
            (-0.28, 0.84),  # Wave in
            (-0.31, 0.90),  # Wave out
            (-0.26, 0.96),  # Wave in
            (-0.28, 1.02),  # Wave out

            # Top of head - messy hair tufts
            (-0.22, 1.08),
            (-0.14, 1.14),  # Hair tuft up
            (-0.08, 1.12),
            (-0.02, 1.15),  # Another tuft
            (0.05, 1.13),
            (0.10, 1.16),   # Characteristic messy bit
            (0.14, 1.12),
            (0.18, 1.08),

            # Forehead - HIGH and PROMINENT (intellectual)
            (0.22, 1.00),
            (0.25, 0.92),
            (0.27, 0.84),   # Strong brow ridge
            (0.28, 0.78),

            # Brow - PRONOUNCED
            (0.29, 0.74),   # Brow bone
            (0.28, 0.70),   # Eye socket dip

            # Nose - DISTINCTIVE, slightly large
            (0.29, 0.65),   # Nose bridge start
            (0.32, 0.60),
            (0.36, 0.54),
            (0.40, 0.48),   # Nose tip - prominent!
            (0.36, 0.44),   # Under nose curve
            (0.32, 0.42),

            # Upper lip area
            (0.30, 0.40),
            (0.28, 0.38),

            # Lips - hint of a SMILE (Feynman was playful)
            (0.29, 0.36),
            (0.28, 0.34),
            (0.26, 0.33),

            # Chin - STRONG jaw
            (0.24, 0.28),
            (0.20, 0.22),
            (0.14, 0.17),   # Chin point
            (0.06, 0.14),

            # Jawline - DEFINED
            (-0.02, 0.12),
            (-0.10, 0.10),
            (-0.16, 0.08),

            # Neck - slight Adam's apple suggestion
            (-0.20, 0.04),
            (-0.22, -0.02),
            (-0.21, -0.08),
            (-0.24, -0.14),

            # Shoulder start
            (-0.28, -0.18),
            (-0.26, -0.20),

            # Close the loop back
            (-0.26, 0.0),
            (-0.28, 0.15),
        ]

        # Convert to numpy array
        points = np.array(profile_points)

        # Create smooth curve using spline interpolation
        from scipy import interpolate

        # Parametric representation
        t = np.linspace(0, 1, len(points))
        t_smooth = np.linspace(0, 1, 1000)

        # Fit cubic splines
        try:
            # Use periodic spline for closed curve
            cs_x = interpolate.CubicSpline(t, points[:, 0], bc_type='periodic')
            cs_y = interpolate.CubicSpline(t, points[:, 1], bc_type='periodic')
            x_smooth = cs_x(t_smooth)
            y_smooth = cs_y(t_smooth)
        except:
            # Fallback to regular interpolation
            x_smooth = np.interp(t_smooth, t, points[:, 0])
            y_smooth = np.interp(t_smooth, t, points[:, 1])

        # Center the contour
        x_smooth -= np.mean(x_smooth)
        y_smooth -= np.mean(y_smooth)

        # Scale to unit size
        scale = max(x_smooth.max() - x_smooth.min(),
                   y_smooth.max() - y_smooth.min())
        x_smooth /= scale
        y_smooth /= scale

        # Convert to complex representation
        contour = x_smooth + 1j * y_smooth

        return contour

    def extract_contour_from_image(
        self,
        image_path: Path,
        blur_size: int = 5,
        threshold_low: int = 50,
        threshold_high: int = 150,
        n_points: int = 1000,
    ) -> np.ndarray:
        """Extract contour from an actual photograph.

        Uses edge detection and contour finding to extract the outline
        of a person from a photograph.

        Args:
            image_path: Path to the image file
            blur_size: Gaussian blur kernel size
            threshold_low: Canny edge detection low threshold
            threshold_high: Canny edge detection high threshold
            n_points: Number of points to resample contour to

        Returns:
            np.ndarray: Complex array of contour points
        """
        if not HAS_CV2:
            raise ImportError("OpenCV (cv2) is required for image processing. Install with: pip install opencv-python")

        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

        # Use adaptive thresholding for better results on portraits
        # This creates a binary mask of the subject
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Invert if needed (we want the subject to be white)
        if np.mean(binary[:50, :50]) > 127:  # If top-left corner is bright
            binary = cv2.bitwise_not(binary)

        # Apply morphological operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            # Fallback: use Canny edge detection
            edges = cv2.Canny(blurred, threshold_low, threshold_high)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            raise ValueError("Could not find any contours in the image")

        # Get the largest contour (assumed to be the main subject)
        largest_contour = max(contours, key=cv2.contourArea)

        # Extract points
        points = largest_contour.squeeze()

        # Ensure it's 2D
        if len(points.shape) == 1:
            raise ValueError("Contour has insufficient points")

        # Resample to uniform number of points
        contour_complex = points[:, 0] + 1j * points[:, 1]

        # Resample using interpolation
        t_original = np.linspace(0, 1, len(contour_complex))
        t_new = np.linspace(0, 1, n_points)

        real_interp = np.interp(t_new, t_original, contour_complex.real)
        imag_interp = np.interp(t_new, t_original, contour_complex.imag)

        contour_resampled = real_interp + 1j * imag_interp

        # Center and normalize
        contour_resampled -= np.mean(contour_resampled)

        # Flip y-axis (image coordinates are inverted)
        contour_resampled = contour_resampled.real - 1j * contour_resampled.imag

        # Normalize scale
        scale = max(
            contour_resampled.real.max() - contour_resampled.real.min(),
            contour_resampled.imag.max() - contour_resampled.imag.min()
        )
        contour_resampled /= scale

        return contour_resampled

    def extract_edge_contour_from_image(
        self,
        image_path: Path,
        n_points: int = 2000,
        edge_method: str = "canny",
    ) -> np.ndarray:
        """Extract artistic edge contour from photograph using edge detection.

        This method creates a more detailed, line-art style contour
        by using edge detection rather than binary thresholding.

        Args:
            image_path: Path to the image file
            n_points: Number of points to resample to
            edge_method: Edge detection method ("canny" or "sobel")

        Returns:
            np.ndarray: Complex array of contour points
        """
        if not HAS_CV2:
            raise ImportError("OpenCV required")

        # Read and preprocess
        img = cv2.imread(str(image_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Blur slightly
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

        # Edge detection
        if edge_method == "canny":
            # Auto-calculate thresholds
            median = np.median(blurred)
            lower = int(max(0, 0.7 * median))
            upper = int(min(255, 1.3 * median))
            edges = cv2.Canny(blurred, lower, upper)
        else:
            # Sobel
            sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobelx**2 + sobely**2)
            edges = (edges / edges.max() * 255).astype(np.uint8)
            _, edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)

        # Dilate slightly to connect edges
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Find all contours
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        if not contours:
            raise ValueError("No edges found")

        # Combine all significant contours into one path
        # Sort by area and take the largest ones
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Take the main contour (largest)
        main_contour = contours[0].squeeze()

        if len(main_contour.shape) == 1 or len(main_contour) < 10:
            # Combine multiple contours
            all_points = []
            for c in contours[:20]:  # Take top 20 contours
                pts = c.squeeze()
                if len(pts.shape) == 2 and len(pts) > 5:
                    all_points.extend(pts.tolist())
            main_contour = np.array(all_points)

        # Convert to complex
        contour_complex = main_contour[:, 0] + 1j * main_contour[:, 1]

        # Order points by angle from center (to form a proper contour)
        center = np.mean(contour_complex)
        angles = np.angle(contour_complex - center)
        sorted_indices = np.argsort(angles)
        contour_complex = contour_complex[sorted_indices]

        # Resample
        t_original = np.linspace(0, 1, len(contour_complex))
        t_new = np.linspace(0, 1, n_points)

        real_interp = np.interp(t_new, t_original, contour_complex.real)
        imag_interp = np.interp(t_new, t_original, contour_complex.imag)

        contour_resampled = real_interp + 1j * imag_interp

        # Center, flip y, normalize
        contour_resampled -= np.mean(contour_resampled)
        contour_resampled = contour_resampled.real - 1j * contour_resampled.imag

        scale = max(
            contour_resampled.real.max() - contour_resampled.real.min(),
            contour_resampled.imag.max() - contour_resampled.imag.min()
        )
        contour_resampled /= scale

        return contour_resampled

    def compute_fft(self, contour: np.ndarray) -> np.ndarray:
        """Compute Fourier coefficients for the contour.

        Args:
            contour: Complex array representing the contour

        Returns:
            np.ndarray: Fourier coefficients
        """
        self.contour = contour
        N = len(contour)

        # Compute DFT coefficients
        coefficients = np.fft.fft(contour) / N

        # Sort by magnitude (importance)
        magnitudes = np.abs(coefficients)
        sorted_indices = np.argsort(magnitudes)[::-1]

        # Keep top n_coefficients
        keep_indices = sorted_indices[:self.n_coefficients]

        # Store with their frequencies
        self.coefficients = []
        for idx in keep_indices:
            freq = idx if idx <= N//2 else idx - N
            self.coefficients.append({
                'freq': freq,
                'coef': coefficients[idx],
                'magnitude': magnitudes[idx],
                'phase': np.angle(coefficients[idx])
            })

        # Sort by frequency for nice visualization
        self.coefficients.sort(key=lambda x: abs(x['freq']))

        return np.array([c['coef'] for c in self.coefficients])

    def reconstruct(self, t: float, n_terms: Optional[int] = None) -> complex:
        """Reconstruct point on contour at parameter t.

        Args:
            t: Parameter value (0 to 2*pi)
            n_terms: Number of terms to use (default: all)

        Returns:
            complex: Reconstructed point
        """
        if self.coefficients is None:
            raise ValueError("Must compute FFT first")

        if n_terms is None:
            n_terms = len(self.coefficients)

        result = 0j
        for i, coef_data in enumerate(self.coefficients[:n_terms]):
            freq = coef_data['freq']
            coef = coef_data['coef']
            result += coef * np.exp(1j * freq * t)

        return result

    def get_epicycles(self, t: float, n_terms: Optional[int] = None) -> list:
        """Get epicycle positions for visualization.

        Args:
            t: Parameter value
            n_terms: Number of epicycles

        Returns:
            list: List of (center, radius, point) tuples
        """
        if self.coefficients is None:
            raise ValueError("Must compute FFT first")

        if n_terms is None:
            n_terms = len(self.coefficients)

        epicycles = []
        center = 0j

        for i, coef_data in enumerate(self.coefficients[:n_terms]):
            freq = coef_data['freq']
            coef = coef_data['coef']
            radius = abs(coef)

            # Current position on this epicycle
            point = center + coef * np.exp(1j * freq * t)

            epicycles.append({
                'center': center,
                'radius': radius,
                'point': point,
                'freq': freq
            })

            center = point

        return epicycles

    def generate_static_image(
        self,
        output_path: Path,
        n_terms: int = 50,
        show_epicycles: bool = True,
        style: str = "dark"
    ) -> Path:
        """Generate static image of the Fourier portrait.

        Args:
            output_path: Where to save the image
            n_terms: Number of Fourier terms to use
            show_epicycles: Whether to show the epicycle circles
            style: Visual style ("dark", "blueprint", "neon")

        Returns:
            Path: Path to saved image
        """
        # Color schemes
        styles = {
            "dark": {
                "bg": "#0d1117",
                "line": "#58a6ff",
                "circle": "#30363d",
                "accent": "#f78166",
                "text": "#f0f6fc",
                "glow": "#58a6ff"
            },
            "blueprint": {
                "bg": "#1a237e",
                "line": "#ffffff",
                "circle": "#3949ab",
                "accent": "#ff8a65",
                "text": "#ffffff",
                "glow": "#82b1ff"
            },
            "neon": {
                "bg": "#0a0a0a",
                "line": "#00ff88",
                "circle": "#333333",
                "accent": "#ff0088",
                "text": "#ffffff",
                "glow": "#00ff88"
            }
        }

        colors = styles.get(style, styles["dark"])

        fig, ax = plt.subplots(figsize=(12, 12))
        fig.patch.set_facecolor(colors["bg"])
        ax.set_facecolor(colors["bg"])

        # Generate full contour
        t_values = np.linspace(0, 2 * np.pi, 1000)
        reconstructed = np.array([self.reconstruct(t, n_terms) for t in t_values])

        # Plot the portrait
        ax.plot(
            reconstructed.real,
            reconstructed.imag,
            color=colors["line"],
            linewidth=2.5,
            alpha=0.9
        )

        # Add glow effect
        for lw, alpha in [(8, 0.1), (6, 0.15), (4, 0.2)]:
            ax.plot(
                reconstructed.real,
                reconstructed.imag,
                color=colors["glow"],
                linewidth=lw,
                alpha=alpha
            )

        # Show epicycles at a specific time
        if show_epicycles:
            t_show = np.pi / 4  # Show epicycles at this time
            epicycles = self.get_epicycles(t_show, min(n_terms, 20))

            for epi in epicycles:
                if epi['radius'] > 0.01:  # Skip tiny circles
                    circle = Circle(
                        (epi['center'].real, epi['center'].imag),
                        epi['radius'],
                        fill=False,
                        color=colors["circle"],
                        linewidth=0.5,
                        alpha=0.5
                    )
                    ax.add_patch(circle)

            # Draw the epicycle arm
            centers = [0j] + [e['point'] for e in epicycles]
            ax.plot(
                [c.real for c in centers],
                [c.imag for c in centers],
                color=colors["accent"],
                linewidth=1,
                alpha=0.7
            )

        ax.set_aspect('equal')
        ax.axis('off')

        # Set limits with padding
        margin = 0.2
        x_range = reconstructed.real.max() - reconstructed.real.min()
        y_range = reconstructed.imag.max() - reconstructed.imag.min()
        ax.set_xlim(
            reconstructed.real.min() - margin * x_range,
            reconstructed.real.max() + margin * x_range
        )
        ax.set_ylim(
            reconstructed.imag.min() - margin * y_range,
            reconstructed.imag.max() + margin * y_range
        )

        plt.tight_layout()
        plt.savefig(
            output_path,
            dpi=150,
            facecolor=colors["bg"],
            bbox_inches='tight',
            pad_inches=0.1
        )
        plt.close()

        return output_path

    def generate_logo(
        self,
        output_path: Path,
        n_terms: int = 80,
        include_equation: bool = True,
        include_branding: bool = True
    ) -> Path:
        """Generate the VibeSignal logo with Fourier Feynman.

        Args:
            output_path: Where to save the logo
            n_terms: Number of Fourier terms
            include_equation: Show the Fourier equation
            include_branding: Show VibeSignal text

        Returns:
            Path: Path to saved logo
        """
        fig, ax = plt.subplots(figsize=(12, 10))

        # Dark gradient background
        bg_color = "#0d1117"
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)

        # Generate contour
        t_values = np.linspace(0, 2 * np.pi, 1000)
        reconstructed = np.array([self.reconstruct(t, n_terms) for t in t_values])

        # Center and scale
        center_x = (reconstructed.real.max() + reconstructed.real.min()) / 2
        center_y = (reconstructed.imag.max() + reconstructed.imag.min()) / 2
        reconstructed = reconstructed - complex(center_x, center_y)

        # Primary portrait line with glow
        line_color = "#58a6ff"
        glow_color = "#58a6ff"

        # Multiple glow layers
        for lw, alpha in [(12, 0.05), (8, 0.1), (5, 0.15), (3, 0.3)]:
            ax.plot(
                reconstructed.real,
                reconstructed.imag,
                color=glow_color,
                linewidth=lw,
                alpha=alpha,
                solid_capstyle='round'
            )

        # Main line
        ax.plot(
            reconstructed.real,
            reconstructed.imag,
            color=line_color,
            linewidth=2,
            alpha=0.95,
            solid_capstyle='round'
        )

        # Add Fourier equation if requested
        if include_equation:
            equation = r"$f(t) = \sum_{n} c_n \cdot e^{i n t}$"
            ax.text(
                0.5, 0.08,
                equation,
                transform=ax.transAxes,
                fontsize=16,
                color="#8b949e",
                ha='center',
                fontfamily='serif',
                style='italic'
            )

        # Add branding
        if include_branding:
            ax.text(
                0.5, 0.95,
                "VibeSignal",
                transform=ax.transAxes,
                fontsize=28,
                color="#58a6ff",
                ha='center',
                fontfamily='sans-serif',
                fontweight='bold'
            )
            ax.text(
                0.5, 0.90,
                "First Principles Thinking",
                transform=ax.transAxes,
                fontsize=12,
                color="#8b949e",
                ha='center',
                fontfamily='sans-serif'
            )

        ax.set_aspect('equal')
        ax.axis('off')

        # Set limits
        margin = 0.3
        x_range = reconstructed.real.max() - reconstructed.real.min()
        y_range = reconstructed.imag.max() - reconstructed.imag.min()
        max_range = max(x_range, y_range)

        ax.set_xlim(-max_range/2 - margin, max_range/2 + margin)
        ax.set_ylim(-max_range/2 - margin, max_range/2 + margin)

        plt.tight_layout()
        plt.savefig(
            output_path,
            dpi=200,
            facecolor=bg_color,
            bbox_inches='tight',
            pad_inches=0.2
        )
        plt.close()

        return output_path


def create_feynman_fft_logo(output_path: Optional[Path] = None, n_terms: int = 80) -> Path:
    """Convenience function to create the Feynman FFT logo.

    Args:
        output_path: Where to save (default: images/feynman_fft_logo.png)
        n_terms: Number of Fourier terms

    Returns:
        Path: Path to saved logo
    """
    if output_path is None:
        output_path = Path("images/feynman_fft_logo.png")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    portrait = FourierPortrait(n_coefficients=n_terms + 20)
    contour = portrait.get_feynman_silhouette()
    portrait.compute_fft(contour)

    return portrait.generate_logo(output_path, n_terms=n_terms)


class MultiPathFourierPortrait:
    """Generate portraits using multiple Fourier paths for detailed features.

    Instead of a single contour, this uses multiple paths to capture
    different facial features - like an artist drawing with multiple strokes.
    """

    def __init__(self, n_coefficients: int = 60):
        self.n_coefficients = n_coefficients
        self.paths = {}  # name -> (contour, coefficients)

    def get_feynman_multipath(self) -> dict:
        """Generate Feynman's profile as multiple distinct paths.

        Returns:
            dict: Named paths for different features
        """
        from scipy import interpolate

        def smooth_path(points: list, n_points: int = 300) -> np.ndarray:
            """Convert control points to smooth contour."""
            pts = np.array(points)
            t = np.linspace(0, 1, len(pts))
            t_smooth = np.linspace(0, 1, n_points)

            try:
                cs_x = interpolate.CubicSpline(t, pts[:, 0], bc_type='natural')
                cs_y = interpolate.CubicSpline(t, pts[:, 1], bc_type='natural')
                x_smooth = cs_x(t_smooth)
                y_smooth = cs_y(t_smooth)
            except:
                x_smooth = np.interp(t_smooth, t, pts[:, 0])
                y_smooth = np.interp(t_smooth, t, pts[:, 1])

            # Close the path
            x_smooth = np.append(x_smooth, x_smooth[0])
            y_smooth = np.append(y_smooth, y_smooth[0])

            return x_smooth + 1j * y_smooth

        paths = {}

        # 1. OUTER PROFILE - the main silhouette
        profile = [
            # Neck base
            (-0.25, 0.05),
            (-0.30, 0.25),
            (-0.32, 0.45),
            # Back of head
            (-0.30, 0.65),
            (-0.28, 0.80),
            # Hair - WILD and WAVY (signature Feynman)
            (-0.30, 0.88),
            (-0.25, 0.95),
            (-0.28, 1.02),
            (-0.20, 1.08),
            (-0.10, 1.14),
            (0.0, 1.12),
            (0.10, 1.15),
            (0.18, 1.10),
            # Forehead
            (0.24, 1.00),
            (0.27, 0.88),
            (0.28, 0.76),
            # Brow
            (0.26, 0.70),
            # Nose - prominent
            (0.30, 0.62),
            (0.38, 0.50),
            (0.32, 0.44),
            # Upper lip
            (0.28, 0.40),
            # Lips with slight smile
            (0.26, 0.36),
            # Chin
            (0.20, 0.25),
            (0.10, 0.15),
            # Jaw
            (-0.05, 0.10),
            (-0.18, 0.06),
            # Back to neck
            (-0.22, 0.02),
        ]
        paths['profile'] = smooth_path(profile, 400)

        # 2. EYE - a key feature for likeness
        eye = [
            (0.18, 0.68),
            (0.22, 0.70),
            (0.26, 0.68),
            (0.24, 0.66),
            (0.20, 0.66),
            (0.18, 0.68),
        ]
        paths['eye'] = smooth_path(eye, 100)

        # 3. EYEBROW - expressive Feynman brow
        eyebrow = [
            (0.14, 0.74),
            (0.18, 0.77),
            (0.24, 0.78),
            (0.28, 0.76),
            (0.24, 0.75),
            (0.18, 0.74),
        ]
        paths['eyebrow'] = smooth_path(eyebrow, 80)

        # 4. NOSTRIL detail
        nostril = [
            (0.30, 0.47),
            (0.33, 0.48),
            (0.34, 0.46),
            (0.32, 0.44),
            (0.30, 0.45),
        ]
        paths['nostril'] = smooth_path(nostril, 60)

        # 5. SMILE LINE - the Feynman grin
        smile = [
            (0.24, 0.36),
            (0.20, 0.34),
            (0.16, 0.33),
            (0.18, 0.35),
            (0.22, 0.36),
        ]
        paths['smile'] = smooth_path(smile, 60)

        # 6. EAR hint
        ear = [
            (-0.15, 0.55),
            (-0.18, 0.60),
            (-0.20, 0.65),
            (-0.18, 0.68),
            (-0.14, 0.62),
            (-0.15, 0.55),
        ]
        paths['ear'] = smooth_path(ear, 80)

        # 7. HAIR TEXTURE strands - adds the wild hair feeling
        hair1 = [
            (-0.22, 1.00),
            (-0.18, 1.06),
            (-0.12, 1.10),
            (-0.08, 1.08),
            (-0.14, 1.04),
            (-0.20, 1.00),
        ]
        paths['hair1'] = smooth_path(hair1, 60)

        hair2 = [
            (0.02, 1.08),
            (0.08, 1.14),
            (0.14, 1.12),
            (0.10, 1.08),
            (0.04, 1.06),
        ]
        paths['hair2'] = smooth_path(hair2, 60)

        return paths

    def compute_all_fft(self, paths: dict) -> None:
        """Compute FFT for all paths."""
        for name, contour in paths.items():
            N = len(contour)
            coefficients = np.fft.fft(contour) / N

            # Sort by magnitude
            magnitudes = np.abs(coefficients)
            sorted_indices = np.argsort(magnitudes)[::-1]
            keep_indices = sorted_indices[:self.n_coefficients]

            coef_list = []
            for idx in keep_indices:
                freq = idx if idx <= N//2 else idx - N
                coef_list.append({
                    'freq': freq,
                    'coef': coefficients[idx],
                    'magnitude': magnitudes[idx],
                })
            coef_list.sort(key=lambda x: abs(x['freq']))

            self.paths[name] = {
                'contour': contour,
                'coefficients': coef_list
            }

    def reconstruct_path(self, name: str, n_terms: int = None) -> np.ndarray:
        """Reconstruct a single path."""
        if name not in self.paths:
            raise ValueError(f"Unknown path: {name}")

        coefficients = self.paths[name]['coefficients']
        if n_terms is None:
            n_terms = len(coefficients)

        t_values = np.linspace(0, 2 * np.pi, 500)
        result = np.zeros(len(t_values), dtype=complex)

        for i, t in enumerate(t_values):
            for coef_data in coefficients[:n_terms]:
                freq = coef_data['freq']
                coef = coef_data['coef']
                result[i] += coef * np.exp(1j * freq * t)

        return result

    def generate_multipath_logo(
        self,
        output_path: Path,
        n_terms_main: int = 80,
        n_terms_detail: int = 40,
    ) -> Path:
        """Generate logo with multiple FFT paths.

        Args:
            output_path: Where to save
            n_terms_main: Terms for main profile
            n_terms_detail: Terms for detail features
        """
        fig, ax = plt.subplots(figsize=(12, 12))

        bg_color = "#0d1117"
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)

        # Color scheme
        main_color = "#58a6ff"
        detail_color = "#79c0ff"
        accent_color = "#a5d6ff"

        # Draw each path
        feature_terms = {
            'profile': n_terms_main,
            'eye': n_terms_detail,
            'eyebrow': n_terms_detail,
            'nostril': n_terms_detail // 2,
            'smile': n_terms_detail // 2,
            'ear': n_terms_detail // 2,
            'hair1': n_terms_detail // 2,
            'hair2': n_terms_detail // 2,
        }

        feature_styles = {
            'profile': {'color': main_color, 'linewidth': 2.5, 'glow': True},
            'eye': {'color': detail_color, 'linewidth': 1.5, 'glow': True},
            'eyebrow': {'color': detail_color, 'linewidth': 1.2, 'glow': False},
            'nostril': {'color': accent_color, 'linewidth': 1.0, 'glow': False},
            'smile': {'color': accent_color, 'linewidth': 1.2, 'glow': False},
            'ear': {'color': accent_color, 'linewidth': 0.8, 'glow': False},
            'hair1': {'color': detail_color, 'linewidth': 1.0, 'glow': False},
            'hair2': {'color': detail_color, 'linewidth': 1.0, 'glow': False},
        }

        all_points = []

        for name, n_terms in feature_terms.items():
            if name not in self.paths:
                continue

            path = self.reconstruct_path(name, n_terms)
            all_points.extend(path)

            style = feature_styles.get(name, {'color': main_color, 'linewidth': 1.0, 'glow': False})

            # Glow effect for main features
            if style['glow']:
                for lw, alpha in [(style['linewidth'] * 4, 0.05),
                                   (style['linewidth'] * 3, 0.1),
                                   (style['linewidth'] * 2, 0.15)]:
                    ax.plot(path.real, path.imag, color=style['color'],
                            linewidth=lw, alpha=alpha, solid_capstyle='round')

            # Main line
            ax.plot(path.real, path.imag, color=style['color'],
                    linewidth=style['linewidth'], alpha=0.9, solid_capstyle='round')

        # Branding
        ax.text(0.5, 0.95, "VibeSignal",
                transform=ax.transAxes, fontsize=28, color="#58a6ff",
                ha='center', fontfamily='sans-serif', fontweight='bold')

        ax.text(0.5, 0.90, "First Principles Thinking",
                transform=ax.transAxes, fontsize=12, color="#8b949e",
                ha='center', fontfamily='sans-serif')

        # Equation
        equation = r"$f(t) = \sum_{n} c_n \cdot e^{i n t}$"
        ax.text(0.5, 0.05, equation,
                transform=ax.transAxes, fontsize=14, color="#8b949e",
                ha='center', fontfamily='serif', style='italic')

        ax.set_aspect('equal')
        ax.axis('off')

        # Set limits from all points
        all_points = np.array(all_points)
        margin = 0.15
        x_min, x_max = all_points.real.min(), all_points.real.max()
        y_min, y_max = all_points.imag.min(), all_points.imag.max()
        x_range = x_max - x_min
        y_range = y_max - y_min

        ax.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
        ax.set_ylim(y_min - margin * y_range, y_max + margin * y_range)

        plt.tight_layout()
        plt.savefig(output_path, dpi=200, facecolor=bg_color,
                    bbox_inches='tight', pad_inches=0.2)
        plt.close()

        return output_path


def create_multipath_feynman_logo(output_path: Optional[Path] = None) -> Path:
    """Create the Feynman FFT logo with multiple paths for better likeness."""
    if output_path is None:
        output_path = Path("images/feynman_multipath_logo.png")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    portrait = MultiPathFourierPortrait(n_coefficients=100)
    paths = portrait.get_feynman_multipath()
    portrait.compute_all_fft(paths)

    return portrait.generate_multipath_logo(output_path)


def create_vibesignal_logo(output_path: Optional[Path] = None) -> Path:
    """Create the official VibeSignal logo - Feynman emerging from signal waves.

    This version emphasizes the 'signal' aspect - showing the mathematical
    beauty of Fourier decomposition with wave patterns radiating from the portrait.
    """
    if output_path is None:
        output_path = Path("images/vibesignal_logo.png")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 10))

    # Colors
    bg_color = "#0d1117"
    primary = "#58a6ff"
    secondary = "#79c0ff"
    accent = "#f78166"
    muted = "#8b949e"

    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # Generate the Feynman profile
    portrait = MultiPathFourierPortrait(n_coefficients=80)
    paths = portrait.get_feynman_multipath()
    portrait.compute_all_fft(paths)

    # Draw signal waves emanating from behind the portrait
    # These represent the Fourier components
    np.random.seed(42)
    for i in range(12):
        freq = (i + 1) * 0.5
        amplitude = 0.15 / (i + 1)
        phase = np.random.uniform(0, 2 * np.pi)
        x_offset = -0.5 - i * 0.08

        y = np.linspace(-0.4, 1.3, 200)
        x = x_offset + amplitude * np.sin(freq * y * 10 + phase)

        alpha = 0.4 / (i + 1) + 0.1
        ax.plot(x, y, color=secondary, linewidth=1.5, alpha=alpha)

    # Draw the portrait paths
    feature_terms = {
        'profile': 60,
        'eye': 30,
        'eyebrow': 25,
        'nostril': 15,
        'smile': 15,
        'ear': 15,
        'hair1': 20,
        'hair2': 20,
    }

    all_points = []
    for name, n_terms in feature_terms.items():
        if name not in portrait.paths:
            continue

        path = portrait.reconstruct_path(name, n_terms)
        all_points.extend(path)

        # Style based on feature
        if name == 'profile':
            # Multiple glow layers for main profile
            for lw, alpha in [(10, 0.05), (7, 0.1), (4, 0.2)]:
                ax.plot(path.real, path.imag, color=primary,
                        linewidth=lw, alpha=alpha, solid_capstyle='round')
            ax.plot(path.real, path.imag, color=primary,
                    linewidth=2.5, alpha=0.95, solid_capstyle='round')
        elif name in ['eye', 'eyebrow']:
            ax.plot(path.real, path.imag, color=secondary,
                    linewidth=1.5, alpha=0.85, solid_capstyle='round')
        else:
            ax.plot(path.real, path.imag, color=secondary,
                    linewidth=1.0, alpha=0.6, solid_capstyle='round')

    # Add signal equation particles/dots emanating
    np.random.seed(123)
    n_particles = 50
    for _ in range(n_particles):
        x = np.random.uniform(-0.6, -0.2)
        y = np.random.uniform(0.0, 1.0)
        size = np.random.uniform(2, 8)
        alpha = np.random.uniform(0.2, 0.5)
        ax.scatter([x], [y], c=secondary, s=size, alpha=alpha)

    # Branding - prominent
    ax.text(0.5, 0.94, "VibeSignal",
            transform=ax.transAxes, fontsize=36, color=primary,
            ha='center', fontfamily='sans-serif', fontweight='bold')

    ax.text(0.5, 0.88, "First Principles → Clear Insights → Good Vibes",
            transform=ax.transAxes, fontsize=11, color=muted,
            ha='center', fontfamily='sans-serif')

    # Fourier equation at bottom
    equation = r"$\mathrm{Signal} \rightarrow \sum c_n e^{int} \rightarrow \mathrm{Portrait}$"
    ax.text(0.5, 0.04, equation,
            transform=ax.transAxes, fontsize=13, color=muted,
            ha='center', fontfamily='serif')

    ax.set_aspect('equal')
    ax.axis('off')

    # Set limits
    all_points = np.array(all_points)
    margin = 0.25
    x_min, x_max = all_points.real.min() - 0.3, all_points.real.max()
    y_min, y_max = all_points.imag.min(), all_points.imag.max()
    x_range = x_max - x_min
    y_range = y_max - y_min

    ax.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
    ax.set_ylim(y_min - margin * y_range, y_max + margin * y_range)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, facecolor=bg_color,
                bbox_inches='tight', pad_inches=0.2)
    plt.close()

    return output_path
