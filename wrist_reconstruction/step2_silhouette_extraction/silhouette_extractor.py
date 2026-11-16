"""
=============================================================================
Step 2: 2D Silhouette & Contour Extraction Engine
=============================================================================
Extracts high-precision outer silhouettes from clinical AP and Lateral images,
eliminates radio-opaque artifacts/labels, bridges joint spaces into solid
anatomical contours, computes continuous metric cross-section profiles (ribs),
and maps patient-specific contours into physical millimeter space.
=============================================================================
"""

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class MetricSlice:
    """
    Cross-sectional anatomical slice at a given physical height Z (in mm).
    """
    z_mm: float                 # Height from forearm base (0.0 to Z_max)
    width_mm: float             # Physical width from AP silhouette (X-axis)
    depth_mm: float             # Physical depth from Lateral silhouette (Y-axis)
    ap_center_offset_mm: float  # Centroid X offset relative to base axis
    lat_center_offset_mm: float # Centroid Y offset relative to base axis
    perimeter_est_mm: float     # Ramanujan elliptical perimeter estimate
    area_est_mm2: float         # Elliptical cross-section area in mm^2


@dataclass
class AnatomicalLandmarks:
    """
    Key longitudinal landmark levels along the hand/wrist axis (in mm from base).
    """
    forearm_base_z_mm: float
    wrist_crease_z_mm: float
    palm_max_z_mm: float
    fingertip_apex_z_mm: float
    total_hand_length_mm: float


class SilhouetteExtractor:
    """
    Extracts, cleans, and converts 2D AP and Lateral images into calibrated
    metric silhouettes and anatomical cross-sectional slice profiles.
    """

    def __init__(self, slice_step_mm: float = 2.0):
        """
        :param slice_step_mm: Vertical interval (in mm) between cross-sectional slices.
        """
        self.slice_step_mm = slice_step_mm

    def segment_hand(self, image_path: Path, view_name: str) -> np.ndarray:
        """
        Performs artifact-resistant segmentation of the hand/wrist envelope.
        Uses adaptive vertical illumination gradient thresholding to capture
        delicate distal finger phalanges (including ring and little finger tips)
        while cleanly isolating the forearm and eliminating external markers ('R').
        """
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot load image at {image_path}")

        h, w = img.shape
        blurred = cv2.bilateralFilter(img, 7, 50, 50)

        # Check for light background (e.g. inverted scan or photographic document)
        corner_samples = np.concatenate([
            blurred[0:20, 0:20].flatten(),
            blurred[0:20, -20:].flatten(),
            blurred[-20:, 0:20].flatten(),
            blurred[-20:, -20:].flatten()
        ])
        if float(np.median(corner_samples)) > 140:
            blurred = 255 - blurred

        # Dynamic vertical exposure gradient estimation from top and bottom corners
        corner_top = np.concatenate([blurred[0:25, 0:35].flatten(), blurred[0:25, -35:].flatten()])
        corner_bot = np.concatenate([blurred[-25:, 0:35].flatten(), blurred[-25:, -35:].flatten()])
        bg_top = float(np.median(corner_top))
        bg_bot = float(np.median(corner_bot))

        offset = 12.0 if view_name == "AP" else 15.0
        t_top = bg_top + offset
        t_bot = bg_bot + offset + (5.0 if view_name == "AP" else 0.0)

        thresh_map = np.tile(np.linspace(t_top, t_bot, h)[:, np.newaxis], (1, w))
        thresh = np.where(blurred >= thresh_map, 255, 0).astype(np.uint8)

        # Morphological closing to seal internal joint cavities
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)

        # Connected components: select the component connecting to the forearm base
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
        hand_idx = None
        max_area = 0
        for idx in range(1, num_labels):
            area = stats[idx, cv2.CC_STAT_AREA]
            y_bot = stats[idx, cv2.CC_STAT_TOP] + stats[idx, cv2.CC_STAT_HEIGHT]
            if y_bot >= h - 10:
                if area > max_area:
                    max_area = area
                    hand_idx = idx

        if hand_idx is None:
            hand_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

        mask = np.where(labels == hand_idx, 255, 0).astype(np.uint8)

        # Fill interior cavities to ensure a solid volume
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        solid_mask = np.zeros_like(mask)
        if contours:
            largest_c = max(contours, key=cv2.contourArea)
            cv2.drawContours(solid_mask, [largest_c], -1, 255, -1)
        else:
            solid_mask = mask

        return solid_mask

    def extract_smooth_contour(self, mask: np.ndarray, epsilon: float = 1.2) -> np.ndarray:
        """
        Extracts the external boundary polygon and applies Douglas-Peucker
        simplification to eliminate single-pixel stairstepping.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            raise ValueError("No contour found in mask.")
        largest_c = max(contours, key=cv2.contourArea)
        smoothed = cv2.approxPolyDP(largest_c, epsilon, closed=True)
        return smoothed

    @staticmethod
    def get_row_bounds(mask: np.ndarray, y: int) -> Optional[Tuple[int, int, float]]:
        """
        Returns (min_x, max_x, center_x) of foreground pixels at row y.
        """
        if y < 0 or y >= mask.shape[0]:
            return None
        row = mask[y, :]
        cols = np.where(row > 0)[0]
        if len(cols) == 0:
            return None
        return int(cols[0]), int(cols[-1]), float(cols[0] + cols[-1]) / 2.0

    def compute_metric_slices(
        self,
        ap_mask: np.ndarray,
        lat_mask: np.ndarray,
        ap_scale: float,
        lat_scale: float
    ) -> Tuple[List[MetricSlice], AnatomicalLandmarks]:
        """
        Computes metric cross-sectional slices along the hand from the forearm base
        to the fingertips, normalizing the physical height Z across both views.
        """
        # Find vertical extents in pixels
        ap_y_fg = np.where(np.any(ap_mask > 0, axis=1))[0]
        lat_y_fg = np.where(np.any(lat_mask > 0, axis=1))[0]

        ap_y_top, ap_y_bot = int(ap_y_fg[0]), int(ap_y_fg[-1])
        lat_y_top, lat_y_bot = int(lat_y_fg[0]), int(lat_y_fg[-1])

        # Physical heights in mm
        ap_total_mm = (ap_y_bot - ap_y_top) * ap_scale
        lat_total_mm = (lat_y_bot - lat_y_top) * lat_scale

        # Canonical height: average of the two views
        total_length_mm = (ap_total_mm + lat_total_mm) / 2.0

        slices: List[MetricSlice] = []
        z_values = np.arange(0.0, total_length_mm, self.slice_step_mm)

        # Baseline centroids (at forearm base)
        base_ap_bounds = self.get_row_bounds(ap_mask, ap_y_bot - 2)
        base_lat_bounds = self.get_row_bounds(lat_mask, lat_y_bot - 2)
        base_ap_cx = base_ap_bounds[2] if base_ap_bounds else (ap_mask.shape[1] / 2.0)
        base_lat_cx = base_lat_bounds[2] if base_lat_bounds else (lat_mask.shape[1] / 2.0)

        # Locate wrist crease (narrowest neck in lower region: 10% - 35% from base)
        min_width_mm = float("inf")
        wrist_crease_z = 0.0

        # Locate palm maximum width (40% - 65% from base)
        max_palm_mm = 0.0
        palm_max_z = 0.0

        for z in z_values:
            # Map physical height z (from bottom=0 to top=total_length_mm) to image y rows
            norm_pos = z / total_length_mm  # 0.0 (base) to 1.0 (apex)
            ap_y = int(round(ap_y_bot - norm_pos * (ap_y_bot - ap_y_top)))
            lat_y = int(round(lat_y_bot - norm_pos * (lat_y_bot - lat_y_top)))

            ap_bounds = self.get_row_bounds(ap_mask, ap_y)
            lat_bounds = self.get_row_bounds(lat_mask, lat_y)

            if ap_bounds is not None and lat_bounds is not None:
                w_px = max(1.0, ap_bounds[1] - ap_bounds[0])
                d_px = max(1.0, lat_bounds[1] - lat_bounds[0])

                w_mm = w_px * ap_scale
                d_mm = d_px * lat_scale

                offset_x_mm = (ap_bounds[2] - base_ap_cx) * ap_scale
                offset_y_mm = (lat_bounds[2] - base_lat_cx) * lat_scale

                # Ramanujan perimeter
                a = w_mm / 2.0
                b = d_mm / 2.0
                h = ((a - b) ** 2) / ((a + b) ** 2) if (a + b) > 0 else 0
                perimeter_mm = math.pi * (a + b) * (1.0 + (3.0 * h) / (10.0 + math.sqrt(max(0.001, 4.0 - 3.0 * h))))
                area_mm2 = math.pi * a * b

                slices.append(MetricSlice(
                    z_mm=round(float(z), 2),
                    width_mm=round(float(w_mm), 2),
                    depth_mm=round(float(d_mm), 2),
                    ap_center_offset_mm=round(float(offset_x_mm), 2),
                    lat_center_offset_mm=round(float(offset_y_mm), 2),
                    perimeter_est_mm=round(float(perimeter_mm), 2),
                    area_est_mm2=round(float(area_mm2), 2)
                ))

                # Track landmarks
                rel_height = z / total_length_mm
                if 0.10 <= rel_height <= 0.35:
                    if w_mm < min_width_mm:
                        min_width_mm = w_mm
                        wrist_crease_z = float(z)

                if 0.35 <= rel_height <= 0.65:
                    if w_mm > max_palm_mm:
                        max_palm_mm = w_mm
                        palm_max_z = float(z)

        landmarks = AnatomicalLandmarks(
            forearm_base_z_mm=0.0,
            wrist_crease_z_mm=round(wrist_crease_z, 2),
            palm_max_z_mm=round(palm_max_z, 2),
            fingertip_apex_z_mm=round(total_length_mm, 2),
            total_hand_length_mm=round(total_length_mm, 2)
        )

        return slices, landmarks

    def draw_contour_overlay(
        self,
        img_path: Path,
        mask: np.ndarray,
        contour: np.ndarray,
        view_name: str,
        scale_mm_per_px: float,
        output_path: Path
    ) -> Path:
        """
        Renders a high-contrast clinical overlay with smooth neon boundary contours
        and key anatomical markers.
        """
        img = cv2.imread(str(img_path))
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Neon color palette
        COLOR_CONTOUR = (50, 255, 100)      # Neon green
        COLOR_PANEL = (25, 25, 25)
        COLOR_TEXT = (255, 255, 255)
        COLOR_ACCENT = (0, 220, 255)

        # Draw smooth boundary
        cv2.polylines(img, [contour], True, COLOR_CONTOUR, 2, cv2.LINE_AA)

        # Draw HUD Panel
        h_img, w_img = img.shape[:2]
        hud_w, hud_h = min(280, w_img - 20), 80
        cv2.rectangle(img, (10, 10), (10 + hud_w, 10 + hud_h), COLOR_PANEL, -1)
        cv2.rectangle(img, (10, 10), (10 + hud_w, 10 + hud_h), COLOR_CONTOUR, 1)

        cv2.putText(img, f"STEP 2: {view_name} SILHOUETTE", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)
        cv2.putText(img, f"Scale: {scale_mm_per_px:.4f} mm/px", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_ACCENT, 1, cv2.LINE_AA)
        cv2.putText(img, "Artifacts Removed | Continuous", (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 255, 180), 1, cv2.LINE_AA)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), img)
        return output_path

    def plot_metric_profiles(
        self,
        slices: List[MetricSlice],
        landmarks: AnatomicalLandmarks,
        output_path: Path
    ) -> Path:
        """
        Generates a metric profile chart of Width and Depth along the hand's
        vertical height (in millimeters).
        """
        z_vals = [s.z_mm for s in slices]
        w_vals = [s.width_mm for s in slices]
        d_vals = [s.depth_mm for s in slices]

        plt.figure(figsize=(9, 5), facecolor="#1e1e1e")
        ax = plt.gca()
        ax.set_facecolor("#282828")

        plt.plot(z_vals, w_vals, color="#34d399", linewidth=2.5, label="AP Width (X-axis, mm)")
        plt.plot(z_vals, d_vals, color="#38bdf8", linewidth=2.5, label="Lateral Depth (Y-axis, mm)")

        # Landmark indicator lines
        plt.axvline(landmarks.wrist_crease_z_mm, color="#f59e0b", linestyle="--", alpha=0.8,
                    label=f"Wrist Crease ({landmarks.wrist_crease_z_mm:.1f} mm)")
        plt.axvline(landmarks.palm_max_z_mm, color="#ec4899", linestyle="--", alpha=0.8,
                    label=f"Palm Arch Max ({landmarks.palm_max_z_mm:.1f} mm)")

        plt.title("Anatomical Cross-Section Metric Profile (Forearm Base to Fingertips)",
                  color="white", fontsize=12, fontweight="bold", pad=12)
        plt.xlabel("Height Z from Forearm Base (mm)", color="white", fontsize=10)
        plt.ylabel("Physical Dimension (mm)", color="white", fontsize=10)

        plt.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#555555")

        plt.grid(True, color="#404040", linestyle=":", alpha=0.6)
        plt.legend(facecolor="#1e1e1e", edgecolor="#555555", labelcolor="white", fontsize=9)
        plt.tight_layout()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_path), dpi=200, facecolor="#1e1e1e")
        plt.close()
        return output_path

    def process(
        self,
        ap_image_path: Path,
        lat_image_path: Path,
        calibration_manifest_path: Path,
        output_dir: Path
    ) -> Dict[str, Any]:
        """
        Executes Step 2:
        1. Loads calibration scales from Step 1
        2. Segments solid silhouettes for AP and Lateral views
        3. Extracts vector polygons
        4. Slices cross-sections in physical millimeters
        5. Generates high-contrast overlays & metric plot
        6. Exports JSON manifest for Step 3
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load Step 1 calibration
        with open(calibration_manifest_path, "r", encoding="utf-8") as f:
            calib_data = json.load(f)

        ap_scale = calib_data["calibrations"]["AP"]["scale_mm_per_pixel"]
        lat_scale = calib_data["calibrations"]["Lateral"]["scale_mm_per_pixel"]

        # 2. Segment silhouettes
        ap_mask = self.segment_hand(ap_image_path, "AP")
        lat_mask = self.segment_hand(lat_image_path, "Lateral")

        # Save masks
        ap_mask_path = output_dir / "step2_ap_mask.png"
        lat_mask_path = output_dir / "step2_lat_mask.png"
        cv2.imwrite(str(ap_mask_path), ap_mask)
        cv2.imwrite(str(lat_mask_path), lat_mask)

        # 3. Extract smooth contour polygons
        ap_contour = self.extract_smooth_contour(ap_mask)
        lat_contour = self.extract_smooth_contour(lat_mask)

        # 4. Compute metric slices and landmarks
        slices, landmarks = self.compute_metric_slices(
            ap_mask=ap_mask,
            lat_mask=lat_mask,
            ap_scale=ap_scale,
            lat_scale=lat_scale
        )

        # 5. Overlays & plots
        ap_overlay_path = output_dir / "step2_ap_contour_overlay.png"
        lat_overlay_path = output_dir / "step2_lat_contour_overlay.png"
        profile_plot_path = output_dir / "step2_metric_profiles.png"

        self.draw_contour_overlay(ap_image_path, ap_mask, ap_contour, "AP", ap_scale, ap_overlay_path)
        self.draw_contour_overlay(lat_image_path, lat_mask, lat_contour, "Lateral", lat_scale, lat_overlay_path)
        self.plot_metric_profiles(slices, landmarks, profile_plot_path)

        # 6. Export metric slices manifest
        slices_manifest = {
            "step": "Step 2: 2D Silhouette & Metric Contour Extraction",
            "landmarks_mm": asdict(landmarks),
            "slice_step_mm": self.slice_step_mm,
            "total_slices": len(slices),
            "slices": [asdict(s) for s in slices],
            "artifacts": {
                "ap_mask": str(ap_mask_path),
                "lat_mask": str(lat_mask_path),
                "ap_overlay": str(ap_overlay_path),
                "lat_overlay": str(lat_overlay_path),
                "profile_plot": str(profile_plot_path)
            }
        }

        manifest_path = output_dir / "step2_metric_slices.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(slices_manifest, f, indent=2)

        return {
            "status": "SUCCESS",
            "manifest_path": str(manifest_path),
            "landmarks": landmarks,
            "num_slices": len(slices),
            "artifacts": slices_manifest["artifacts"]
        }

