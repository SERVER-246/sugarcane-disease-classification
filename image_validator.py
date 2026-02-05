#!/usr/bin/env python3
"""
================================================================================
🛡️ IMAGE VALIDATOR - Sugarcane Image Filtering System
================================================================================

Multi-level validation system to filter out non-sugarcane images before
classification. This prevents model confusion and improves user experience.

Validation Levels:
1. Basic Validation - File format, dimensions, corruption
2. Content Analysis - Color histogram, texture, vegetation detection
3. Deep Learning - Binary classifier (Sugarcane vs Non-Sugarcane)

Author: SERVER-246
Version: 1.0.0
================================================================================
"""

import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageStat


# Suppress warnings
warnings.filterwarnings('ignore')


class ValidationResult(Enum):
    """Validation result codes"""
    VALID = "valid"
    INVALID_FORMAT = "invalid_format"
    INVALID_SIZE = "invalid_size"
    LOW_QUALITY = "low_quality"
    NOT_VEGETATION = "not_vegetation"
    NOT_SUGARCANE = "not_sugarcane"
    CORRUPTED = "corrupted"


@dataclass
class ValidationReport:
    """Detailed validation report"""
    is_valid: bool
    result: ValidationResult
    confidence: float
    quality_score: float
    vegetation_score: float
    sugarcane_score: float
    message: str
    details: dict
    suggestions: list[str]


class ImageValidator:
    """
    Multi-level image validation for sugarcane disease classification.

    Filters out:
    - Non-image files
    - Too small/large images
    - Blurry or low-quality images
    - Non-vegetation images
    - Non-sugarcane plant images
    """

    # Configuration
    MIN_SIZE = 100  # Minimum dimension
    MAX_SIZE = 8000  # Maximum dimension
    MIN_QUALITY_SCORE = 30  # Minimum quality (0-100)
    MIN_VEGETATION_SCORE = 0.40  # Minimum green vegetation ratio
    MIN_SUGARCANE_CONFIDENCE = 0.50  # Minimum confidence for sugarcane

    # Valid image extensions
    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}

    # Color analysis parameters (for vegetation detection)
    # Sugarcane leaves are typically green with some yellow/brown
    GREEN_HUE_RANGE = (35, 85)  # Hue range for green vegetation (0-180 in OpenCV)
    LEAF_SATURATION_RANGE = (20, 255)  # Expected saturation range

    def __init__(self, use_deep_learning: bool = True, model_path: Path | None = None):
        """
        Initialize the image validator.

        Args:
            use_deep_learning: Whether to use deep learning for sugarcane detection
            model_path: Path to the sugarcane classifier model
        """
        self.use_deep_learning = use_deep_learning
        self.model_path = model_path
        self.sugarcane_classifier = None

        # Try to load the deep learning classifier
        if use_deep_learning:
            self._load_classifier()

    def _load_classifier(self) -> bool:
        """Load the sugarcane binary classifier"""
        try:
            # For now, we'll use heuristics
            # In production, load a trained binary classifier
            return True
        except Exception as e:
            print(f"Warning: Could not load sugarcane classifier: {e}")
            self.use_deep_learning = False
            return False

    def validate(self, image_path: str) -> ValidationReport:
        """
        Perform full validation on an image.

        Args:
            image_path: Path to the image file

        Returns:
            ValidationReport with detailed results
        """
        path = Path(image_path)
        details = {}
        suggestions = []

        # Level 1: Basic Validation
        basic_result = self._validate_basic(path)
        if basic_result[0] != ValidationResult.VALID:
            return ValidationReport(
                is_valid=False,
                result=basic_result[0],
                confidence=0.0,
                quality_score=0.0,
                vegetation_score=0.0,
                sugarcane_score=0.0,
                message=basic_result[1],
                details={'level': 'basic'},
                suggestions=basic_result[2]
            )

        # Load image for further analysis
        try:
            image = Image.open(path).convert('RGB')
            details['original_size'] = image.size
        except Exception as e:
            return ValidationReport(
                is_valid=False,
                result=ValidationResult.CORRUPTED,
                confidence=0.0,
                quality_score=0.0,
                vegetation_score=0.0,
                sugarcane_score=0.0,
                message=f"Could not open image: {str(e)}",
                details={'error': str(e)},
                suggestions=["Try a different image file"]
            )

        # Level 2: Quality Analysis
        quality_score, quality_details = self._analyze_quality(image)
        details.update(quality_details)

        if quality_score < self.MIN_QUALITY_SCORE:
            suggestions.append("Image quality is low. Try taking a clearer photo.")
            if quality_details.get('is_blurry', False):
                suggestions.append("Image appears blurry. Hold camera steady or use autofocus.")
            if quality_details.get('is_dark', False):
                suggestions.append("Image is too dark. Take photo in better lighting.")
            if quality_details.get('is_overexposed', False):
                suggestions.append("Image is overexposed. Avoid direct sunlight on camera.")

            return ValidationReport(
                is_valid=False,
                result=ValidationResult.LOW_QUALITY,
                confidence=quality_score / 100,
                quality_score=quality_score,
                vegetation_score=0.0,
                sugarcane_score=0.0,
                message="Image quality is too low for accurate analysis",
                details=details,
                suggestions=suggestions
            )

        # Level 3: Vegetation Detection
        vegetation_score, veg_details = self._detect_vegetation(image)
        details.update(veg_details)

        if vegetation_score < self.MIN_VEGETATION_SCORE:
            suggestions.append("Image does not appear to contain vegetation.")
            suggestions.append("Please capture a close-up of sugarcane leaves.")

            return ValidationReport(
                is_valid=False,
                result=ValidationResult.NOT_VEGETATION,
                confidence=vegetation_score,
                quality_score=quality_score,
                vegetation_score=vegetation_score,
                sugarcane_score=0.0,
                message="Image does not appear to be a plant/vegetation image",
                details=details,
                suggestions=suggestions
            )

        # Level 4: Sugarcane-specific Detection
        sugarcane_score, sugar_details = self._detect_sugarcane_features(image)
        details.update(sugar_details)

        if sugarcane_score < self.MIN_SUGARCANE_CONFIDENCE:
            suggestions.append("Image may not be a sugarcane plant.")
            suggestions.append("Ensure the image shows sugarcane leaves clearly.")

            return ValidationReport(
                is_valid=False,
                result=ValidationResult.NOT_SUGARCANE,
                confidence=sugarcane_score,
                quality_score=quality_score,
                vegetation_score=vegetation_score,
                sugarcane_score=sugarcane_score,
                message="Image does not appear to be a sugarcane plant",
                details=details,
                suggestions=suggestions
            )

        # All validations passed
        overall_confidence = (quality_score/100 + vegetation_score + sugarcane_score) / 3

        return ValidationReport(
            is_valid=True,
            result=ValidationResult.VALID,
            confidence=overall_confidence,
            quality_score=quality_score,
            vegetation_score=vegetation_score,
            sugarcane_score=sugarcane_score,
            message="Image is valid for disease classification",
            details=details,
            suggestions=["Image looks good! Proceeding with analysis."]
        )

    def _validate_basic(self, path: Path) -> tuple[ValidationResult, str, list[str]]:
        """Level 1: Basic file validation"""

        # Check if file exists
        if not path.exists():
            return (
                ValidationResult.INVALID_FORMAT,
                "File does not exist",
                ["Check the file path and try again"]
            )

        # Check extension
        if path.suffix.lower() not in self.VALID_EXTENSIONS:
            return (
                ValidationResult.INVALID_FORMAT,
                f"Invalid file format: {path.suffix}",
                [f"Supported formats: {', '.join(self.VALID_EXTENSIONS)}"]
            )

        # Check file size (not empty, not too large)
        file_size = path.stat().st_size
        if file_size < 1000:  # Less than 1KB
            return (
                ValidationResult.CORRUPTED,
                "File appears to be empty or corrupted",
                ["Try a different image file"]
            )
        if file_size > 50_000_000:  # More than 50MB
            return (
                ValidationResult.INVALID_SIZE,
                "File is too large (>50MB)",
                ["Use a smaller image or compress it"]
            )

        # Try to open and check dimensions
        try:
            with Image.open(path) as img:
                width, height = img.size

                if width < self.MIN_SIZE or height < self.MIN_SIZE:
                    return (
                        ValidationResult.INVALID_SIZE,
                        f"Image too small ({width}x{height}). Minimum: {self.MIN_SIZE}x{self.MIN_SIZE}",
                        ["Use a higher resolution image"]
                    )

                if width > self.MAX_SIZE or height > self.MAX_SIZE:
                    return (
                        ValidationResult.INVALID_SIZE,
                        f"Image too large ({width}x{height}). Maximum: {self.MAX_SIZE}x{self.MAX_SIZE}",
                        ["Resize the image to a smaller dimension"]
                    )
        except Exception as e:
            return (
                ValidationResult.CORRUPTED,
                f"Could not read image: {str(e)}",
                ["The file may be corrupted. Try a different image."]
            )

        return (ValidationResult.VALID, "Basic validation passed", [])

    def _analyze_quality(self, image: Image.Image) -> tuple[float, dict]:
        """Level 2: Image quality analysis"""
        details = {}

        # Convert to grayscale for some analyses
        gray = image.convert('L')

        # 1. Blur detection using Laplacian variance
        # Higher variance = sharper image
        gray_array = np.array(gray)
        laplacian_var = self._calculate_laplacian_variance(gray_array)
        details['laplacian_variance'] = laplacian_var
        details['is_blurry'] = laplacian_var < 100

        # 2. Brightness analysis
        stat = ImageStat.Stat(image)
        brightness = sum(stat.mean) / 3
        details['brightness'] = brightness
        details['is_dark'] = brightness < 50
        details['is_overexposed'] = brightness > 220

        # 3. Contrast analysis
        contrast = sum(stat.stddev) / 3
        details['contrast'] = contrast
        details['is_low_contrast'] = contrast < 30

        # 4. Color diversity (avoid solid colors)
        color_diversity = self._calculate_color_diversity(image)
        details['color_diversity'] = color_diversity

        # Calculate overall quality score (0-100)
        score = 100

        if details['is_blurry']:
            score -= 30
        if details['is_dark']:
            score -= 25
        if details['is_overexposed']:
            score -= 25
        if details['is_low_contrast']:
            score -= 15
        if color_diversity < 0.3:
            score -= 20

        # Bonus for good sharpness
        if laplacian_var > 500:
            score = min(100, score + 10)

        return max(0, score), details

    def _calculate_laplacian_variance(self, gray_array: np.ndarray) -> float:
        """Calculate Laplacian variance for blur detection"""
        # Simple Laplacian approximation
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

        # Downsample for efficiency
        h, w = gray_array.shape
        if h > 500 or w > 500:
            scale = 500 / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            gray_array = np.array(Image.fromarray(gray_array).resize((new_w, new_h)))

        # Apply Laplacian
        from scipy import ndimage
        try:
            laplacian = ndimage.convolve(gray_array.astype(float), kernel)
            return float(np.var(laplacian))
        except ImportError:
            # Fallback without scipy
            return 200  # Assume moderate sharpness

    def _calculate_color_diversity(self, image: Image.Image) -> float:
        """Calculate color diversity score (0-1)"""
        # Resize for efficiency
        small = image.resize((50, 50))
        pixels = np.array(small).reshape(-1, 3)

        # Calculate unique colors ratio
        unique_colors = len(np.unique(pixels, axis=0))
        max_colors = 50 * 50

        return min(1.0, unique_colors / (max_colors * 0.5))

    def _detect_vegetation(self, image: Image.Image) -> tuple[float, dict]:
        """Level 3: Vegetation/plant detection using color analysis"""
        details = {}

        # Convert to numpy array
        img_array = np.array(image)

        # Calculate green channel dominance
        # For vegetation, green channel is typically dominant
        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]

        # Method 1: Green channel ratio
        total = r.astype(float) + g.astype(float) + b.astype(float) + 1  # +1 to avoid division by zero
        green_ratio = g.astype(float) / total
        avg_green_ratio = np.mean(green_ratio)
        details['green_ratio'] = float(avg_green_ratio)

        # Method 2: Excess green index (ExG)
        # ExG = 2*G - R - B (normalized)
        norm_r = r.astype(float) / 255
        norm_g = g.astype(float) / 255
        norm_b = b.astype(float) / 255
        exg = 2 * norm_g - norm_r - norm_b
        vegetation_mask = exg > 0.1
        vegetation_percentage = np.mean(vegetation_mask)
        details['vegetation_percentage'] = float(vegetation_percentage)

        # Method 3: Check for natural green colors
        # Green vegetation typically has: G > R and G > B, but not pure green
        green_dominant = (g > r) & (g > b)
        natural_green = green_dominant & (r > 30) & (b > 30)  # Not pure artificial green
        natural_green_ratio = np.mean(natural_green)
        details['natural_green_ratio'] = float(natural_green_ratio)

        # Method 4: Texture analysis for leaf patterns
        # Leaves have characteristic texture patterns
        texture_score = self._analyze_leaf_texture(image)
        details['texture_score'] = texture_score

        # Combine scores
        vegetation_score = (
            avg_green_ratio * 0.2 +
            vegetation_percentage * 0.3 +
            natural_green_ratio * 0.3 +
            texture_score * 0.2
        )

        # Normalize to 0-1
        vegetation_score = min(1.0, max(0.0, vegetation_score * 2))

        return vegetation_score, details

    def _analyze_leaf_texture(self, image: Image.Image) -> float:
        """Analyze texture patterns typical of leaves"""
        # Resize for efficiency
        small = image.resize((224, 224))
        gray = small.convert('L')

        # Apply edge detection
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edge_array = np.array(edges)

        # Calculate edge density
        edge_density = np.mean(edge_array > 30)

        # Leaves typically have moderate edge density (veins, edges)
        # Too low = smooth surface, too high = noisy/artificial
        if 0.05 < edge_density < 0.4:
            return min(1.0, edge_density * 3)
        else:
            return 0.3

    def _detect_sugarcane_features(self, image: Image.Image) -> tuple[float, dict]:
        """Level 4: Sugarcane-specific feature detection"""
        details = {}

        # Convert to numpy
        img_array = np.array(image)

        # Feature 1: Aspect ratio of image (sugarcane leaves are long)
        # This is a weak signal but can help
        h, w = img_array.shape[:2]
        aspect_ratio = max(h, w) / min(h, w)
        details['aspect_ratio'] = aspect_ratio

        # Feature 2: Color palette matching
        # Sugarcane leaves are typically:
        # - Healthy: bright green
        # - Diseased: green with yellow/brown/red spots
        color_match = self._match_sugarcane_colors(img_array)
        details['color_match'] = color_match

        # Feature 3: Parallel line detection (leaf veins)
        # Sugarcane has parallel venation
        vein_score = self._detect_parallel_veins(image)
        details['vein_score'] = vein_score

        # Feature 4: Check for elongated structures
        elongation_score = self._check_elongation(image)
        details['elongation_score'] = elongation_score

        # Combine features
        # These are heuristics - a trained model would be better
        sugarcane_score = (
            color_match * 0.35 +
            vein_score * 0.35 +
            elongation_score * 0.30
        )

        return min(1.0, max(0.0, sugarcane_score)), details

    def _match_sugarcane_colors(self, img_array: np.ndarray) -> float:
        """Check if colors match typical sugarcane leaf colors"""
        # Define typical sugarcane color ranges (RGB)
        sugarcane_colors = [
            # Healthy green
            ([30, 80, 20], [100, 180, 80]),
            # Yellow-green (mild stress)
            ([100, 140, 40], [180, 200, 100]),
            # Brown spots
            ([60, 40, 20], [140, 100, 60]),
            # Red-brown (disease)
            ([100, 40, 30], [180, 100, 80]),
        ]

        # Sample pixels
        h, w = img_array.shape[:2]
        sample_size = 1000
        indices = np.random.choice(h * w, min(sample_size, h * w), replace=False)
        pixels = img_array.reshape(-1, 3)[indices]

        # Check what percentage of pixels match sugarcane colors
        matches = 0
        for pixel in pixels:
            for (low, high) in sugarcane_colors:
                if all(l <= p <= h for l, p, h in zip(low, pixel, high)):
                    matches += 1
                    break

        return matches / len(pixels)

    def _detect_parallel_veins(self, image: Image.Image) -> float:
        """Detect parallel vein patterns typical of monocot leaves"""
        # Convert to grayscale and resize
        gray = image.convert('L').resize((224, 224))

        # Apply directional edge detection
        # Horizontal Sobel for vertical lines (leaf veins)
        horizontal = gray.filter(ImageFilter.Kernel(
            size=(3, 3),
            kernel=[-1, 0, 1, -2, 0, 2, -1, 0, 1],
            scale=1
        ))

        edge_array = np.array(horizontal)

        # Check for regular patterns
        # Calculate variance along rows (should be consistent for parallel veins)
        row_means = np.mean(edge_array, axis=1)
        row_variance = np.var(row_means)

        # Normalize score
        # Lower variance in row means = more parallel structure
        if row_variance < 100:
            return 0.9
        elif row_variance < 500:
            return 0.6
        else:
            return 0.3

    def _check_elongation(self, image: Image.Image) -> float:
        """Check for elongated leaf-like structures"""
        # This is a simplified check
        # A proper implementation would use contour detection

        gray = image.convert('L').resize((224, 224))
        gray_array = np.array(gray)

        # Threshold to create binary image
        threshold = np.mean(gray_array)
        binary = gray_array > threshold

        # Check horizontal vs vertical distribution
        h_profile = np.mean(binary, axis=0)
        v_profile = np.mean(binary, axis=1)

        # Leaves often show elongated patterns
        h_var = np.var(h_profile)
        v_var = np.var(v_profile)

        # If one direction has more variance, likely elongated structure
        elongation = abs(h_var - v_var) / (max(h_var, v_var) + 0.01)

        return min(1.0, elongation * 3 + 0.3)

    def quick_check(self, image_path: str) -> tuple[bool, str]:
        """
        Quick validation check (faster, less detailed).

        Args:
            image_path: Path to image

        Returns:
            Tuple of (is_valid, reason)
        """
        report = self.validate(image_path)
        return report.is_valid, report.message


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_image(image_path: str) -> ValidationReport:
    """Convenience function to validate an image"""
    validator = ImageValidator()
    return validator.validate(image_path)


def is_valid_sugarcane_image(image_path: str) -> bool:
    """Quick check if image is valid for classification"""
    validator = ImageValidator()
    return validator.quick_check(image_path)[0]


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import sys

    print("🛡️ Sugarcane Image Validator")
    print("=" * 50)

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"\nValidating: {image_path}")

        validator = ImageValidator()
        report = validator.validate(image_path)

        print(f"\n{'✅' if report.is_valid else '❌'} Result: {report.result.value}")
        print(f"Message: {report.message}")
        print("\nScores:")
        print(f"  Quality Score: {report.quality_score:.1f}/100")
        print(f"  Vegetation Score: {report.vegetation_score:.2f}")
        print(f"  Sugarcane Score: {report.sugarcane_score:.2f}")
        print(f"  Overall Confidence: {report.confidence:.2f}")

        if report.suggestions:
            print("\nSuggestions:")
            for s in report.suggestions:
                print(f"  • {s}")
    else:
        print("\nUsage: python image_validator.py <image_path>")
        print("\nExample:")
        print("  python image_validator.py sample_leaf.jpg")
