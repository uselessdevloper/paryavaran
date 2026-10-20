"""
opencv_analyzer.py - Substantive OpenCV 5 Computer Vision Engine for Aerial Imagery
====================================================================================
Performs multi-spectral color-space transformation, contrast-limited adaptive
histogram equalization (CLAHE), morphological contour analysis, and edge-density
estimation on aerial/satellite tiles to extract ground-truth urban cover.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class VisionAnalysisResult:
    """Structured output from the OpenCV 5 aerial inspection."""
    canopy_fraction: float          # Percentage of detected tree/greenspace cover (0.0 to 1.0)
    bare_ground_fraction: float     # Percentage of unpaved, plantable bare soil (0.0 to 1.0)
    impervious_fraction: float      # Percentage of concrete/asphalt/built footprint (0.0 to 1.0)
    water_fraction: float           # Percentage of surface water bodies (0.0 to 1.0)
    edge_density: float             # Structural structural complexity from Canny edge detector
    largest_bare_parcel_sqm: float  # Area in square meters of the single largest continuous plantable parcel
    num_plantable_parcels: int      # Number of distinct plantable parcels (> 100 sqm)
    dominant_class: str             # "Dense Canopy", "Bare Ground", "Built Urban", or "Water Body"
    annotated_image: np.ndarray     # Rendered OpenCV visualization with masks & contours
    green_mask: np.ndarray          # Binary segmentation mask for vegetation
    bare_mask: np.ndarray           # Binary segmentation mask for bare land
    water_mask: np.ndarray          # Binary segmentation mask for water


class OpenCVAerialAnalyzer:
    """OpenCV 5 Aerial & Satellite Computer Vision Engine."""

    def __init__(
        self,
        tile_size_meters: float = 250.0,
        clahe_clip_limit: float = 2.5,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8),
    ):
        """
        Args:
            tile_size_meters: Physical dimension of the square aerial tile in meters (default 250m).
            clahe_clip_limit: Threshold for contrast limiting in atmospheric smog removal.
            clahe_tile_grid_size: Grid size for CLAHE localized histogram equalization.
        """
        self.tile_size_meters = tile_size_meters
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_grid_size,
        )

    def enhance_contrast_and_dehaze(self, bgr_image: np.ndarray) -> np.ndarray:
        """Apply CLAHE in CIE-Lab color space to suppress Delhi atmospheric smog and haze."""
        lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        enhanced_l = self.clahe.apply(l_channel)
        enhanced_lab = cv2.merge([enhanced_l, a_channel, b_channel])
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    def segment_vegetation(self, bgr_image: np.ndarray) -> np.ndarray:
        """
        Extract active vegetation canopy using dual-space (HSV + Excess Green Index) analysis.
        Uses morphological opening to eliminate isolated pixel noise and morphological closing
        to consolidate tree clusters.
        """
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

        # Primary green spectrum in HSV
        lower_green = np.array([30, 35, 35], dtype=np.uint8)
        upper_green = np.array([88, 255, 255], dtype=np.uint8)
        hsv_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Excess Green Index (ExG = 2G - R - B) to capture shaded canopy
        b, g, r = cv2.split(bgr_image.astype(np.int16))
        exg = 2 * g - r - b
        exg_mask = np.where(exg > 15, 255, 0).astype(np.uint8)

        # Blend both representations
        combined = cv2.bitwise_and(hsv_mask, exg_mask)

        # Morphological filtering
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
        return cleaned

    def segment_bare_ground(self, bgr_image: np.ndarray, green_mask: np.ndarray) -> np.ndarray:
        """
        Segment exposed soil, unpaved vacant lots, and fallow ground suitable for Miyawaki forests.
        Excludes areas already identified as vegetation.
        """
        blurred = cv2.GaussianBlur(bgr_image, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        # Brown/tan/yellow soil color band in HSV (excludes low-saturation asphalt)
        lower_soil = np.array([8, 42, 50], dtype=np.uint8)
        upper_soil = np.array([30, 220, 240], dtype=np.uint8)
        soil_mask = cv2.inRange(hsv, lower_soil, upper_soil)

        # Ensure no overlap with vegetation
        soil_mask = cv2.bitwise_and(soil_mask, cv2.bitwise_not(green_mask))

        # Morphological consolidation into continuous parcels
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        cleaned = cv2.morphologyEx(soil_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        return cleaned

    def segment_water_bodies(self, bgr_image: np.ndarray) -> np.ndarray:
        """Detect surface water bodies (e.g. Yamuna river, wetlands, retention lakes)."""
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        lower_water = np.array([92, 40, 30], dtype=np.uint8)
        upper_water = np.array([135, 255, 230], dtype=np.uint8)
        water_mask = cv2.inRange(hsv, lower_water, upper_water)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        return cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    def compute_edge_density(self, bgr_image: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute Canny structural edge density to quantify built-up urban complexity
        (rooftop boundaries, roads, concrete outlines).
        """
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
        edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
        edge_density = float(np.count_nonzero(edges)) / float(edges.size)
        return edge_density, edges

    def analyze_aerial_tile(
        self,
        bgr_image: np.ndarray,
        tile_id: str = "DELHI_MICRO_ZONE",
    ) -> VisionAnalysisResult:
        """
        Run the complete OpenCV 5 substantive computer vision pipeline on an aerial tile.
        """
        h, w = bgr_image.shape[:2]
        total_pixels = float(h * w)
        pixel_area_sqm = (self.tile_size_meters * self.tile_size_meters) / total_pixels

        # 1. Atmospheric enhancement & haze reduction
        enhanced = self.enhance_contrast_and_dehaze(bgr_image)

        # 2. Multi-spectral visual segmentation
        green_mask = self.segment_vegetation(enhanced)
        water_mask = self.segment_water_bodies(enhanced)
        bare_mask = self.segment_bare_ground(enhanced, green_mask)

        # 3. Structural edge density
        edge_density, edges = self.compute_edge_density(enhanced)

        # 4. Built-up / impervious surface deduction
        # Impervious is non-green, non-water, non-bare with substantial edge or grayscale properties
        non_permeable = cv2.bitwise_not(cv2.bitwise_or(green_mask, cv2.bitwise_or(bare_mask, water_mask)))
        impervious_mask = non_permeable

        # 5. Calculate coverage fractions
        canopy_fraction = float(np.count_nonzero(green_mask)) / total_pixels
        bare_ground_fraction = float(np.count_nonzero(bare_mask)) / total_pixels
        water_fraction = float(np.count_nonzero(water_mask)) / total_pixels
        impervious_fraction = float(np.count_nonzero(impervious_mask)) / total_pixels

        # 6. Extract plantable bare parcels using OpenCV contour analysis
        contours, _ = cv2.findContours(bare_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        plantable_parcels = []
        largest_bare_parcel_sqm = 0.0

        for cnt in contours:
            area_px = cv2.contourArea(cnt)
            area_sqm = area_px * pixel_area_sqm
            if area_sqm >= 100.0:  # Minimum viable planting parcel (100 sqm)
                plantable_parcels.append((cnt, area_sqm))
                if area_sqm > largest_bare_parcel_sqm:
                    largest_bare_parcel_sqm = area_sqm

        # 7. Determine dominant visual class
        fractions = {
            "Dense Canopy": canopy_fraction,
            "Bare Ground": bare_ground_fraction,
            "Built Urban": impervious_fraction,
            "Water Body": water_fraction,
        }
        dominant_class = max(fractions, key=fractions.get)

        # 8. Render annotated visualization
        annotated = enhanced.copy()

        # Highlight detected canopy with translucent green
        green_overlay = np.zeros_like(annotated)
        green_overlay[green_mask > 0] = [34, 139, 34]  # Forest green
        annotated = cv2.addWeighted(annotated, 1.0, green_overlay, 0.45, 0)

        # Highlight detected bare plantable land with translucent amber
        bare_overlay = np.zeros_like(annotated)
        bare_overlay[bare_mask > 0] = [0, 165, 255]  # Orange/amber
        annotated = cv2.addWeighted(annotated, 1.0, bare_overlay, 0.40, 0)

        # Draw contours around plantable parcels (> 100 sqm)
        for cnt, area_sqm in plantable_parcels:
            cv2.drawContours(annotated, [cnt], -1, (0, 255, 255), 2)  # Yellow border
            # Label parcel centroid
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                label = f"{int(area_sqm)}m²"
                cv2.putText(
                    annotated, label, (cx - 20, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA
                )

        # Header metrics overlay
        header_text = (
            f"OpenCV 5 Vision | Canopy: {canopy_fraction*100:.1f}% | "
            f"Bare Ground: {bare_ground_fraction*100:.1f}% | "
            f"Built: {impervious_fraction*100:.1f}%"
        )
        cv2.rectangle(annotated, (0, 0), (w, 32), (20, 20, 20), -1)
        cv2.putText(
            annotated, header_text, (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 200), 1, cv2.LINE_AA
        )

        return VisionAnalysisResult(
            canopy_fraction=round(canopy_fraction, 4),
            bare_ground_fraction=round(bare_ground_fraction, 4),
            impervious_fraction=round(impervious_fraction, 4),
            water_fraction=round(water_fraction, 4),
            edge_density=round(edge_density, 4),
            largest_bare_parcel_sqm=round(largest_bare_parcel_sqm, 1),
            num_plantable_parcels=len(plantable_parcels),
            dominant_class=dominant_class,
            annotated_image=annotated,
            green_mask=green_mask,
            bare_mask=bare_mask,
            water_mask=water_mask,
        )
