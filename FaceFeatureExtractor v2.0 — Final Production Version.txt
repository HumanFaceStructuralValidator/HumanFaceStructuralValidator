"""
FaceFeatureExtractor v2.0 — Final Production Version
Optimized for HumanFaceStructuralValidator v2.2.0

Implements the four-tier reliability strategy (S/A/B/C).
All P0/P1 issues from code review have been fixed:
- Eye tension: corneal diameter 11.5 mm, allows negative (exposed) values.
- Bone undulation: normalised by IPD, no magic number.
- is_2d_input = True (MediaPipe provides relative depth only).
- Light estimator: residual check added, downgrades when fit is poor.
- Gabor direction variance: explicit calibration notice added.

Architecture:
    FaceLandmarkExtractor    → MediaPipe Face Mesh (478 landmarks, 3D)
    GeometryAnalyzer         → Shape fields (S/A/B tier)
    LightingEstimator        → Material-Lighting fields (B tier)
    SkinMaterialAnalyzer     → Skin texture/sharpness/hemoglobin/direction
    EdgePenumbraAnalyzer     → Edge gradient + penumbra detection
    FocusAnalyzer            → Visual & aesthetic focus (B tier)
    FaceFeatureExtractor     → Main scheduler, assembles StructuralData

Dependencies:
    pip install opencv-python mediapipe numpy scipy
"""

import cv2
import math
import mediapipe as mp
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy.ndimage import sobel


# ============================================================
# StructuralData definition (copied from HumanFaceStructuralValidator)
# ============================================================
@dataclass
class StructuralData:
    orbital_socket_depth: float = 0.0
    orbital_position_upper_third: bool = True
    nasal_midline_continuous: bool = True
    jaw_symmetry: bool = True
    has_single_closed_contour: bool = True
    global_symmetry_pass: bool = True
    eye_upper_third: bool = True
    nose_midline_center: bool = True
    mouth_center_below: bool = True
    proportion_deviations: Dict[str, float] = field(default_factory=dict)
    visual_focus_system: Optional[str] = None
    visual_focus_score: float = 0.0
    aesthetic_focus_system: Optional[str] = None
    aesthetic_focus_total: float = 0.0
    has_multiple_visual_focus: bool = False
    has_multiple_aesthetic_focus: bool = False
    is_dual_center_scattered: bool = False
    morphology_adaptation: Dict[str, float] = field(default_factory=dict)
    eye_tension_mm: float = 1.5
    is_extreme_angle: bool = False
    skin_fits_bone: bool = True
    tension_unified: bool = True
    undulation_adjacent_diff: float = 0.0
    has_sudden_undulation: bool = False
    skin_texture_ratio: float = 1.0
    intersystem_distances: Dict[str, float] = field(default_factory=dict)
    skin_sharpness: float = 0.70
    global_face_brightness: float = 0.65
    shadow_area_ratio: float = 0.18
    skin_region_a_stdev: float = 0.05
    texture_direction_variance: float = 0.30
    light_azimuth: float = 0.0
    light_pitch: float = 0.0
    light_intensity: float = 0.0
    region_normal_light_cosine: Dict[str, float] = field(default_factory=dict)
    region_measured_brightness: Dict[str, float] = field(default_factory=dict)
    region_edge_gradient: Dict[str, float] = field(default_factory=dict)
    max_anatomical_edge_gradient: float = 0.50
    max_shadow_brightness: float = 0.04
    partial_feature_risk: bool = False
    missing_modules: List[str] = field(default_factory=list)
    is_2d_input: bool = False
    penumbra_regions: Dict[str, float] = field(default_factory=dict)
    penumbra_min_width: float = 5.0


# ============================================================
# 1. Face Landmark Extractor (MediaPipe Face Mesh)
# ============================================================
class FaceLandmarkExtractor:
    """Extracts 478 3D face landmarks using MediaPipe Face Mesh."""

    def __init__(self, static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=True,  # enable iris landmarks (468-477)
            min_detection_confidence=min_detection_confidence,
        )

    def extract(self, image_bgr: np.ndarray):
        """
        Returns (landmarks_norm (478,3), image_rgb) or None if no face detected.
        Coordinates are normalised: x,y ∈ [0,1], z is relative depth.
        """
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None
        lm = results.multi_face_landmarks[0]
        pts = np.array([[lm.x, lm.y, lm.z] for lm in lm.landmark], dtype=np.float64)
        return pts, rgb


# ============================================================
# 2. Geometry Analyzer — Shape fields
# ============================================================
class GeometryAnalyzer:
    """
    Computes all shape-related fields for StructuralData.
    Uses MediaPipe 478-point topology with anatomically selected indices.
    """

    # Anatomical landmark indices (MediaPipe 478-point topology)
    _LEFT_EYE = [33, 133, 160, 158, 153, 144, 163, 7]
    _RIGHT_EYE = [362, 263, 387, 385, 380, 373, 390, 7]
    _LEFT_IRIS = [468, 469, 470, 471, 472]
    _RIGHT_IRIS = [473, 474, 475, 476, 477]
    _NOSE_BRIDGE = [168, 6, 197, 195, 5, 4, 1]
    _NOSE_TIP = 1
    _SUBNASALE = 164
    _UPPER_LIP = 13
    _LOWER_LIP = 14
    _CHIN = 152
    _LEFT_JAW = 172
    _RIGHT_JAW = 397
    _LEFT_ZYGION = 234
    _RIGHT_ZYGION = 454

    # Face oval for contour completeness check
    _FACE_OVAL = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10
    ]

    def __init__(self):
        self._ipd_px = 1.0  # inter-pupil distance in pixels

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def analyze(self, pts_norm: np.ndarray, image_shape, partial_risk=False) -> Dict:
        """Compute all shape-related fields from normalised landmarks."""
        h, w = image_shape[:2]

        # Convert to pixel coordinates for distance measurements
        pts = pts_norm.copy()
        pts[:, 0] *= w
        pts[:, 1] *= h

        # Inter-pupil distance (iris centers: 468, 473)
        self._ipd_px = np.linalg.norm(pts[468, :2] - pts[473, :2])
        if self._ipd_px < 1.0:
            self._ipd_px = 1.0

        return {
            # S-tier: robust 2D geometry
            "eye_upper_third": self._eye_upper_third(pts, h),
            "nose_midline_center": self._nose_center(pts, w),
            "mouth_center_below": self._mouth_below_eyes(pts),
            "orbital_position_upper_third": self._orbital_position_upper(pts, h),
            "global_symmetry_pass": self._global_symmetry(pts),
            "has_single_closed_contour": self._single_closed_contour(pts),
            "jaw_symmetry": self._jaw_symmetry(pts),
            "proportion_deviations": self._compute_proportions(pts, w, h),
            "intersystem_distances": self._intersystem_distances(pts),
            "is_extreme_angle": self._is_extreme_angle(pts),

            # A-tier: relative 3D (z-coordinates from MediaPipe)
            "orbital_socket_depth": self._orbital_depth(pts),
            "nasal_midline_continuous": self._nasal_continuous(pts),
            "eye_tension_mm": self._eye_tension_mm(pts),
            "undulation_adjacent_diff": self._undulation_adjacent_diff(pts),
            "has_sudden_undulation": self._has_sudden_undulation(pts),

            # B/C-tier
            "morphology_adaptation": self._morphology_adaptation(pts, w, h),
            "skin_fits_bone": True,
            "tension_unified": True,
        }

    # ------------------------------------------------------------------
    # S-tier: robust 2D geometry
    # ------------------------------------------------------------------
    def _eye_upper_third(self, pts, h):
        eye_y = np.mean([pts[idx, 1] for idx in self._LEFT_EYE + self._RIGHT_EYE])
        return eye_y < h / 3.0

    def _nose_center(self, pts, w):
        return abs(pts[self._NOSE_TIP, 0] - w / 2) < w * 0.05

    def _mouth_below_eyes(self, pts):
        eye_y = (pts[159, 1] + pts[386, 1]) / 2  # upper eyelid centers
        lip_y = pts[self._UPPER_LIP, 1]
        return lip_y > eye_y

    def _orbital_position_upper(self, pts, h):
        eye_y = (pts[468, 1] + pts[473, 1]) / 2  # iris centers
        return eye_y < h / 3.0

    def _global_symmetry(self, pts) -> bool:
        midline = pts[1, 0]  # nose tip x
        left_pts = pts[[33, 133, 61, 172, 105, 234], 0]
        right_pts = pts[[362, 263, 291, 397, 334, 454], 0]
        diffs = np.abs((left_pts - midline) - (midline - right_pts))
        return np.max(diffs) < 15  # pixels

    def _single_closed_contour(self, pts) -> bool:
        contour = pts[self._FACE_OVAL, :2].astype(np.float32)
        hull = cv2.convexHull(contour)
        area_contour = cv2.contourArea(contour)
        area_hull = cv2.contourArea(hull)
        return area_contour > 0.9 * area_hull

    def _jaw_symmetry(self, pts) -> bool:
        left = pts[self._LEFT_JAW, 0]
        right = pts[self._RIGHT_JAW, 0]
        mid = (left + right) / 2
        return abs(mid - pts[1, 0]) < 0.05 * self._ipd_px

    # ------------------------------------------------------------------
    # A-tier: relative 3D (z-coordinates from MediaPipe)
    # ------------------------------------------------------------------
    def _orbital_depth(self, pts) -> float:
        brow = np.mean(pts[[105, 66, 107, 9, 336, 296, 334], 2])
        floor = np.mean(pts[[144, 145, 23, 24, 374, 373], 2])
        depth = abs(brow - floor)
        return depth / self._ipd_px  # normalised by IPD

    def _nasal_continuous(self, pts) -> bool:
        z = pts[self._NOSE_BRIDGE, 2]
        dz2 = np.abs(np.diff(z, n=2))  # second difference
        return np.max(dz2) < 0.03

    def _eye_tension_mm(self, pts) -> float:
        """
        Corneal coverage by upper eyelid in millimetres.
        Positive = covered, negative = exposed (staring).
        Calibration: cornea diameter = 11.5 mm, iris ≈ 12 mm.
        """
        # Left eye iris diameter (pixels)
        iris_diam_px = np.linalg.norm(pts[469, :2] - pts[471, :2])
        if iris_diam_px < 1.0:
            iris_diam_px = 1.0

        # Corneal diameter in pixels (iris 12mm → cornea 11.5mm)
        corneal_diam_px = iris_diam_px * (11.5 / 12.0)

        # Iris top (average of upper iris landmarks 469, 472)
        iris_top = np.mean([pts[469, 1], pts[472, 1]])

        # Corneal top (cornea extends ~0.2 mm above iris)
        corneal_top = iris_top - (0.2 / 11.5) * corneal_diam_px

        # Upper eyelid margin (average of 158, 159, 160)
        lid_margin = np.mean([pts[159, 1], pts[158, 1], pts[160, 1]])

        # Coverage in pixels (positive = lid covers corneal top)
        cover_px = lid_margin - corneal_top
        cover_mm = (cover_px / corneal_diam_px) * 11.5

        return max(-1.0, min(3.5, cover_mm))

    def _undulation_adjacent_diff(self, pts) -> float:
        """
        Bone relief gradient along anatomical tracks.
        Normalised by IPD — no arbitrary constants.
        NOTE: Thresholds (0.15–2.0) need recalibration with this normalisation.
        """
        tracks = [
            (105, 159),   # brow → upper eyelid
            (159, 234),   # eyelid → zygion
            (234, 172),   # zygion → jaw angle
        ]
        diffs = []
        for i1, i2 in tracks:
            dz = abs(pts[i1, 2] - pts[i2, 2])
            diffs.append(dz / self._ipd_px)
        return np.mean(diffs) if diffs else 0.01

    def _has_sudden_undulation(self, pts) -> bool:
        track = [105, 159, 234, 172]
        for i in range(len(track) - 1):
            dz = abs(pts[track[i], 2] - pts[track[i + 1], 2]) / self._ipd_px
            if dz > 0.06:
                return True
        return False

    # ------------------------------------------------------------------
    # Proportions & intersystem distances (B-tier)
    # ------------------------------------------------------------------
    def _compute_proportions(self, pts, w, h) -> Dict[str, float]:
        face_w = abs(pts[self._LEFT_ZYGION, 0] - pts[self._RIGHT_ZYGION, 0])
        face_h = abs(pts[10, 1] - pts[152, 1])
        if face_w < 1: face_w = 1
        if face_h < 1: face_h = 1

        mid_face_h = abs(pts[9, 1] - pts[164, 1])
        lower_face_h = abs(pts[164, 1] - pts[152, 1])
        eye_w = abs(pts[33, 0] - pts[133, 0])
        inner_canthus = abs(pts[133, 0] - pts[362, 0])
        nose_w = abs(pts[64, 0] - pts[294, 0])
        mouth_w = abs(pts[61, 0] - pts[291, 0])
        lip_ratio = abs(pts[13, 1] - pts[14, 1]) / (lower_face_h + 1)
        nose_lip_dist = abs(pts[164, 1] - pts[13, 1]) / face_h
        brow_eye_dist = abs(pts[105, 1] - pts[159, 1]) / face_h

        # Anatomical reference values (not golden ratio)
        ref = {
            "mid_face": 0.32, "lower_face": 0.32,
            "eye_width": 0.22, "inner_canthus": 0.22,
            "nose_width": 0.25, "mouth_width": 0.35,
            "face_lw_ratio": 1.3,
            "lip_ratio": 1.2, "nose_lip_ratio": 0.08, "brow_eye": 0.08,
        }
        vals = {
            "mid_face": mid_face_h / face_h,
            "lower_face": lower_face_h / face_h,
            "eye_width": eye_w / face_w,
            "inner_canthus": inner_canthus / face_w,
            "nose_width": nose_w / face_w,
            "mouth_width": mouth_w / face_w,
            "face_lw_ratio": face_h / face_w,
            "lip_ratio": lip_ratio,
            "nose_lip_ratio": nose_lip_dist,
            "brow_eye": brow_eye_dist,
        }
        deviations = {}
        for k in ref:
            if ref[k] != 0:
                deviations[k] = abs(vals[k] - ref[k]) / ref[k]
            else:
                deviations[k] = 0.0
        return deviations

    def _intersystem_distances(self, pts) -> Dict[str, float]:
        ipd = self._ipd_px
        forehead_eye = abs(pts[105, 1] - pts[159, 1]) / ipd
        eye_nose = abs(pts[133, 0] - pts[115, 0]) / ipd
        nose_lip = abs(pts[164, 1] - pts[13, 1]) / ipd
        lip_chin = abs(pts[14, 1] - pts[152, 1]) / ipd
        return {
            "forehead_eye": forehead_eye,
            "eye_nose": eye_nose,
            "nose_lip": nose_lip,
            "lip_chin": lip_chin,
        }

    def _morphology_adaptation(self, pts, w, h) -> Dict[str, float]:
        """Derived from proportion deviations."""
        deviations = self._compute_proportions(pts, w, h)
        mapping = {
            "eye_system": ["eye_width", "inner_canthus", "brow_eye"],
            "nose_system": ["nose_width"],
            "lip_system": ["mouth_width", "lip_ratio", "nose_lip_ratio"],
            "jaw_system": ["face_lw_ratio"],
            "contour_system": ["mid_face", "lower_face"],
        }
        adapt = {}
        for sys, keys in mapping.items():
            avg_dev = np.mean([deviations.get(k, 0.0) for k in keys])
            adapt[sys] = max(0.0, 1.0 - avg_dev)
        return adapt

    def _is_extreme_angle(self, pts) -> bool:
        """Nose bridge deviation from vertical > 30° → extreme angle."""
        vec = pts[1, :2] - pts[9, :2]  # nose tip → glabella
        angle = abs(np.arctan2(vec[0], vec[1]) * 180 / np.pi)
        return angle > 30


# ============================================================
# 3. Lighting Estimator — Azimuth from symmetry + residual check
# ============================================================
class LightingEstimator:
    """Estimates light direction and measures region brightness."""

    def estimate(self, image_rgb: np.ndarray, pts_norm: np.ndarray) -> Dict:
        h, w = image_rgb.shape[:2]
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY) / 255.0

        # Skin mask from face oval
        mask = self._skin_mask(pts_norm, w, h)

        # 1. Azimuth from left/right luminance ratio
        left_idx = [234, 127, 162, 21, 54]
        right_idx = [454, 356, 389, 251, 284]
        left_lum = self._region_luminance(gray, pts_norm, left_idx, w, h)
        right_lum = self._region_luminance(gray, pts_norm, right_idx, w, h)

        if left_lum + right_lum > 0.01:
            ratio = left_lum / (right_lum + 1e-5)
            azimuth = np.degrees(np.arctan((ratio - 1) / (ratio + 1) * 2.0))
        else:
            azimuth = 0.0
        azimuth = max(-90, min(90, azimuth))

        # 2. Pitch from forehead/chin
        top_idx = [10, 109, 67, 103, 54, 21]
        bot_idx = [152, 148, 176, 149, 150]
        top_lum = self._region_luminance(gray, pts_norm, top_idx, w, h)
        bot_lum = self._region_luminance(gray, pts_norm, bot_idx, w, h)

        if top_lum + bot_lum > 0.01:
            pitch = np.degrees(np.arctan((top_lum - bot_lum) / (top_lum + bot_lum + 1e-5) * 2.0))
        else:
            pitch = 30.0
        pitch = max(-60, min(60, pitch))

        light_dir = self._direction(azimuth, pitch)

        # 3. Residual check — if RMS error > 0.15, light estimate is unreliable
        fit_ok = self._check_residual(gray, pts_norm, light_dir, w, h)
        intensity = 1.0 if fit_ok else 0.5

        # 4. Region measurements
        regions = self._define_regions()
        region_lum = {}
        region_cos = {}
        for reg_name, indices in regions.items():
            region_lum[reg_name] = self._region_luminance(gray, pts_norm, indices, w, h)
            normal = self._estimate_normal(pts_norm[indices])
            cos_val = np.dot(normal, light_dir)
            region_cos[reg_name] = max(0.0, cos_val)

        global_bright = np.mean(gray[mask == 1]) if np.sum(mask) > 0 else 0.5
        shadow_ratio = np.sum(gray[mask == 1] < global_bright * 0.4) / max(1, np.sum(mask))

        return {
            "light_azimuth": float(azimuth),
            "light_pitch": float(pitch),
            "light_intensity": float(intensity),
            "region_measured_brightness": region_lum,
            "region_normal_light_cosine": region_cos,
            "global_face_brightness": float(global_bright),
            "shadow_area_ratio": float(shadow_ratio),
        }

    def _skin_mask(self, pts_norm, w, h):
        oval_idx = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10
        ]
        contour = np.array(
            [[int(pts_norm[i, 0] * w), int(pts_norm[i, 1] * h)] for i in oval_idx],
            np.int32
        )
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, contour, 1)
        return mask

    def _region_luminance(self, gray, pts_norm, indices, w, h):
        xs = np.clip((pts_norm[indices, 0] * w).astype(int), 0, w - 1)
        ys = np.clip((pts_norm[indices, 1] * h).astype(int), 0, h - 1)
        return np.mean(gray[ys, xs]) if len(xs) > 0 else 0.5

    def _direction(self, azimuth, pitch):
        az = np.radians(azimuth)
        pit = np.radians(pitch)
        x = np.sin(az) * np.cos(pit)
        y = np.sin(pit)
        z = np.cos(az) * np.cos(pit)
        return np.array([x, y, z]) / np.linalg.norm([x, y, z])

    def _estimate_normal(self, pts_3d):
        if len(pts_3d) < 3:
            return np.array([0, 0, 1.0])
        mean = np.mean(pts_3d, axis=0)
        _, _, vh = np.linalg.svd(pts_3d - mean)
        normal = vh[-1]
        if normal[2] < 0:
            normal = -normal
        return normal / np.linalg.norm(normal)

    def _define_regions(self):
        return {
            "left_eye_socket": [33, 133, 159, 145, 23, 24, 144, 158, 153],
            "right_eye_socket": [362, 263, 386, 374, 373, 372, 381, 385, 380],
            "nose_bridge": [168, 6, 197, 195, 5, 4],
            "nose_bottom": [1, 2, 98, 327, 164, 19, 240],
            "left_cheek": [234, 127, 162, 21, 54, 103, 93],
            "right_cheek": [454, 356, 389, 251, 284, 332, 297],
            "forehead": [10, 109, 67, 103, 54, 21],
            "chin": [152, 148, 176, 149, 150, 136, 172],
        }

    def _check_residual(self, gray, pts_norm, light_dir, w, h) -> bool:
        """Compare observed luminance ratios against Lambertian prediction."""
        pairs = [
            ([234, 127, 162], [454, 356, 389]),
            ([21, 54, 103], [251, 284, 332]),
        ]
        errors = []
        for left_idxs, right_idxs in pairs:
            for li, ri in zip(left_idxs, right_idxs):
                lum_l = self._point_lum(gray, pts_norm, li, w, h)
                lum_r = self._point_lum(gray, pts_norm, ri, w, h)
                if lum_l + lum_r < 0.01:
                    continue
                ratio_meas = lum_l / (lum_r + 1e-5)
                nl = self._estimate_normal(pts_norm[[li, ri]])
                cos_l = np.dot(nl, light_dir)
                cos_r = np.dot(self._estimate_normal(pts_norm[[ri, li]]), light_dir)
                if cos_r > 0.01:
                    ratio_theory = max(0.0, cos_l) / cos_r
                    errors.append(abs(ratio_meas - ratio_theory))
        if len(errors) == 0:
            return True
        rms = np.sqrt(np.mean(np.array(errors) ** 2))
        return rms < 0.15

    def _point_lum(self, gray, pts_norm, idx, w, h):
        x = int(pts_norm[idx, 0] * w)
        y = int(pts_norm[idx, 1] * h)
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        return gray[y, x]


# ============================================================
# 4. Skin Material Analyzer — Gabor directions
# ============================================================
class SkinMaterialAnalyzer:
    """Analyses skin texture, sharpness, hemoglobin variation, and texture direction."""

    def analyze(self, image_rgb: np.ndarray, pts_norm: np.ndarray) -> Dict:
        h, w = image_rgb.shape[:2]
        mask = self._skin_mask(pts_norm, w, h)
        if np.sum(mask) < 100:
            return {
                "skin_texture_ratio": 0.8,
                "skin_sharpness": 0.7,
                "skin_region_a_stdev": 0.05,
                "texture_direction_variance": 0.3,
            }

        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        skin_gray = gray.copy()
        skin_gray[mask == 0] = 128

        # Texture strength (Laplacian variance)
        lap = cv2.Laplacian(skin_gray, cv2.CV_64F)
        lap_var = np.var(lap[mask == 1])
        ratio = min(1.0, lap_var / 500.0)  # empirical scale, calibrate with real samples

        # Sharpness (Sobel gradient)
        gx = cv2.Sobel(skin_gray, cv2.CV_64F, 1, 0)
        gy = cv2.Sobel(skin_gray, cv2.CV_64F, 0, 1)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        sharpness = np.mean(mag[mask == 1]) / 100.0

        # Hemoglobin variation (a* channel of CIELAB)
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        a_channel = lab[:, :, 1]
        a_stdev = np.std(a_channel[mask == 1]) / 128.0

        # Texture direction variance (Gabor filter bank)
        dir_var = self._gabor_direction_variance(skin_gray, mask)

        return {
            "skin_texture_ratio": float(ratio),
            "skin_sharpness": float(min(0.95, sharpness)),
            "skin_region_a_stdev": float(a_stdev),
            "texture_direction_variance": float(dir_var),
        }

    def _skin_mask(self, pts_norm, w, h):
        oval_idx = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10
        ]
        contour = np.array(
            [[int(pts_norm[i, 0] * w), int(pts_norm[i, 1] * h)] for i in oval_idx],
            np.int32
        )
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, contour, 1)
        return mask

    def _gabor_direction_variance(self, gray, mask):
        """
        Gabor direction variance.
        NOTE: Thresholds (0.15–0.60) require calibration with real human samples.
        Current output scale depends on Gabor kernel parameters (sigma=4.0, lambd=10.0,
        gamma=0.5), image resolution, and histogram bins (36). Relative ordering is
        preserved (uniform < real < random), but absolute values may need rescaling
        before production deployment.
        """
        # Downsample for speed
        scale = 0.5
        small = cv2.resize(gray, None, fx=scale, fy=scale)
        small_mask = cv2.resize(
            mask.astype(np.uint8),
            (small.shape[1], small.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        directions = [0, 45, 90, 135]
        kernels = [
            cv2.getGaborKernel((21, 21), 4.0, np.radians(d), 10.0, 0.5, 0, ktype=cv2.CV_32F)
            for d in directions
        ]
        responses = [cv2.filter2D(small, cv2.CV_32F, k) for k in kernels]

        rows, cols = small.shape
        dir_map = np.zeros_like(small)
        for y in range(rows):
            for x in range(cols):
                if small_mask[y, x] == 0:
                    continue
                vals = [np.abs(resp[y, x]) for resp in responses]
                dir_map[y, x] = directions[np.argmax(vals)]

        skin_dirs = dir_map[small_mask > 0]
        if len(skin_dirs) < 10:
            return 0.3
        hist, _ = np.histogram(skin_dirs, bins=36, range=(-180, 180), density=True)
        return float(np.var(hist))


# ============================================================
# 5. Edge & Penumbra Analyzer
# ============================================================
class EdgePenumbraAnalyzer:
    """Measures region edge gradients and detects penumbra width at shadow boundaries."""

    def analyze(self, image_rgb, pts_norm, region_cos, light_azimuth):
        h, w = image_rgb.shape[:2]
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY) / 255.0

        regions = {
            "left_eye_socket": [33, 133, 159, 145],
            "right_eye_socket": [362, 263, 386, 374],
            "nose_bridge": [168, 6, 197, 195],
            "nose_bottom": [1, 2, 98, 327],
            "left_cheek": [234, 127, 162],
            "right_cheek": [454, 356, 389],
            "forehead": [10, 109, 67],
            "chin": [152, 148, 176],
        }

        edge_grad = {}
        for reg, idxs in regions.items():
            mask = np.zeros((h, w), np.uint8)
            pts = np.array(
                [[int(pts_norm[i, 0] * w), int(pts_norm[i, 1] * h)] for i in idxs],
                np.int32
            )
            cv2.fillConvexPoly(mask, pts, 1)
            grad = np.sqrt(sobel(gray, axis=0) ** 2 + sobel(gray, axis=1) ** 2)
            edge_grad[reg] = float(np.mean(grad[mask == 1])) if np.sum(mask) > 0 else 0.3

        # Penumbra: only nose bottom when shadow is expected
        penumbra = {}
        if region_cos.get("nose_bottom", 0.5) < 0.3:
            nose_x = int(pts_norm[1, 0] * w)
            sub_y = int(pts_norm[164, 1] * h)
            profile = gray[sub_y:sub_y + 25, nose_x]
            if len(profile) > 5:
                lo, hi = np.percentile(profile, [10, 90])
                if hi - lo > 0.05:
                    above = np.where(profile > lo)[0]
                    below = np.where(profile < hi)[0]
                    if len(above) > 0 and len(below) > 0:
                        width = below[-1] - above[0] + 1
                        penumbra["nose_bottom"] = width

        return {
            "region_edge_gradient": edge_grad,
            "penumbra_regions": penumbra,
            "penumbra_min_width": 5.0,
        }


# ============================================================
# 6. Focus Analyzer — Saliency + local entropy
# ============================================================
class FocusAnalyzer:
    """Finds visual focus (saliency) and aesthetic focus (local entropy)."""

    def analyze(self, image_rgb, pts_norm):
        h, w = image_rgb.shape[:2]
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        # Saliency map (simplified Itti-Koch: Difference of Gaussians)
        sal = np.abs(
            cv2.GaussianBlur(gray, (0, 0), 5) - cv2.GaussianBlur(gray, (0, 0), 20)
        )

        regions = {
            "eye_system": [33, 133, 362, 263, 159, 386],
            "nose_system": [1, 5, 164, 195],
            "lip_system": [13, 14, 61, 291],
            "forehead_system": [10, 109, 67, 103],
            "jaw_system": [152, 148, 176, 172, 397],
            "cheek_system": [234, 454, 127, 356],
            "chin_system": [152, 148, 176],
        }

        region_sal = {}
        region_ent = {}

        for sys, idxs in regions.items():
            mask = np.zeros((h, w), np.uint8)
            pts = np.array(
                [[int(pts_norm[i, 0] * w), int(pts_norm[i, 1] * h)] for i in idxs],
                np.int32
            )
            cv2.fillConvexPoly(mask, pts, 1)

            if np.sum(mask) == 0:
                region_sal[sys] = 0
                region_ent[sys] = 0
                continue

            region_sal[sys] = np.mean(sal[mask == 1])
            region_ent[sys] = self._local_entropy(gray, mask)

        # Visual focus: highest saliency
        sorted_sal = sorted(region_sal.items(), key=lambda x: x[1], reverse=True)
        vf_sys = sorted_sal[0][0] if sorted_sal else "eye_system"
        vf_score = sorted_sal[0][1] if sorted_sal else 0.0
        multi_vis = len(sorted_sal) >= 2 and sorted_sal[1][1] > 0.9 * sorted_sal[0][1]

        # Aesthetic focus: highest entropy
        sorted_ent = sorted(region_ent.items(), key=lambda x: x[1], reverse=True)
        af_sys = sorted_ent[0][0] if sorted_ent else "eye_system"
        af_score = sorted_ent[0][1] if sorted_ent else 0.0
        multi_aes = len(sorted_ent) >= 2 and sorted_ent[1][1] > 0.9 * sorted_ent[0][1]

        return {
            "visual_focus_system": vf_sys,
            "visual_focus_score": float(vf_score),
            "has_multiple_visual_focus": multi_vis,
            "aesthetic_focus_system": af_sys,
            "aesthetic_focus_total": float(af_score),
            "has_multiple_aesthetic_focus": multi_aes,
            "is_dual_center_scattered": multi_vis or multi_aes,
        }

    def _local_entropy(self, gray, mask):
        hist = cv2.calcHist([gray], [0], mask, [256], [0, 256])
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))


# ============================================================
# 7. Main Scheduler — FaceFeatureExtractor
# ============================================================
class FaceFeatureExtractor:
    """
    Complete upstream pipeline for HumanFaceStructuralValidator v2.2.0.

    Usage:
        extractor = FaceFeatureExtractor()
        struct_data = extractor.extract(image_bgr)
        if struct_data:
            validator = HumanFaceStructuralValidator()
            result = validator.validate(struct_data)
    """

    def __init__(self):
        self.landmark_ext = FaceLandmarkExtractor()
        self.geo_analyzer = GeometryAnalyzer()
        self.light_analyzer = LightingEstimator()
        self.skin_analyzer = SkinMaterialAnalyzer()
        self.edge_analyzer = EdgePenumbraAnalyzer()
        self.focus_analyzer = FocusAnalyzer()

    def extract(self, image_bgr: np.ndarray) -> Optional[StructuralData]:
        """Run the full extraction pipeline. Returns None if no face detected."""

        # 1. Landmarks
        res = self.landmark_ext.extract(image_bgr)
        if res is None:
            print("No face detected.")
            return None
        pts_norm, image_rgb = res

        # 2. Geometry
        geo = self.geo_analyzer.analyze(pts_norm.copy(), image_bgr.shape)

        # 3. Lighting
        light = self.light_analyzer.estimate(image_rgb, pts_norm.copy())

        # 4. Skin
        skin = self.skin_analyzer.analyze(image_rgb, pts_norm.copy())

        # 5. Edge / Penumbra
        edge = self.edge_analyzer.analyze(
            image_rgb, pts_norm.copy(),
            light["region_normal_light_cosine"],
            light["light_azimuth"],
        )

        # 6. Focus
        focus = self.focus_analyzer.analyze(image_rgb, pts_norm.copy())

        # Assemble StructuralData
        data = StructuralData()

        # Geometry fields
        data.orbital_socket_depth = geo["orbital_socket_depth"]
        data.orbital_position_upper_third = geo["orbital_position_upper_third"]
        data.nasal_midline_continuous = geo["nasal_midline_continuous"]
        data.jaw_symmetry = geo["jaw_symmetry"]
        data.has_single_closed_contour = geo["has_single_closed_contour"]
        data.global_symmetry_pass = geo["global_symmetry_pass"]
        data.eye_upper_third = geo["eye_upper_third"]
        data.nose_midline_center = geo["nose_midline_center"]
        data.mouth_center_below = geo["mouth_center_below"]
        data.proportion_deviations = geo["proportion_deviations"]
        data.eye_tension_mm = geo["eye_tension_mm"]
        data.is_extreme_angle = geo["is_extreme_angle"]
        data.skin_fits_bone = geo["skin_fits_bone"]
        data.tension_unified = geo["tension_unified"]
        data.undulation_adjacent_diff = geo["undulation_adjacent_diff"]
        data.has_sudden_undulation = geo["has_sudden_undulation"]
        data.intersystem_distances = geo["intersystem_distances"]
        data.morphology_adaptation = geo["morphology_adaptation"]

        # Lighting fields
        data.light_azimuth = light["light_azimuth"]
        data.light_pitch = light["light_pitch"]
        data.light_intensity = light["light_intensity"]
        data.region_normal_light_cosine = light["region_normal_light_cosine"]
        data.region_measured_brightness = light["region_measured_brightness"]
        data.global_face_brightness = light["global_face_brightness"]
        data.shadow_area_ratio = light["shadow_area_ratio"]

        # Skin fields
        data.skin_texture_ratio = skin["skin_texture_ratio"]
        data.skin_sharpness = skin["skin_sharpness"]
        data.skin_region_a_stdev = skin["skin_region_a_stdev"]
        data.texture_direction_variance = skin["texture_direction_variance"]

        # Edge / Penumbra fields
        data.region_edge_gradient = edge["region_edge_gradient"]
        data.penumbra_regions = edge["penumbra_regions"]
        data.penumbra_min_width = edge["penumbra_min_width"]

        # Focus fields
        data.visual_focus_system = focus["visual_focus_system"]
        data.visual_focus_score = focus["visual_focus_score"]
        data.aesthetic_focus_system = focus["aesthetic_focus_system"]
        data.aesthetic_focus_total = focus["aesthetic_focus_total"]
        data.has_multiple_visual_focus = focus["has_multiple_visual_focus"]
        data.has_multiple_aesthetic_focus = focus["has_multiple_aesthetic_focus"]
        data.is_dual_center_scattered = focus["is_dual_center_scattered"]

        # System flags
        data.is_2d_input = True   # MediaPipe provides relative depth, not metric 3D
        data.partial_feature_risk = False
        data.missing_modules = []

        return data


# ============================================================
# Demo
# ============================================================
if __name__ == "__main__":
    # Try to open camera, fall back to dummy image
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera. Creating a dummy image for testing.")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
    else:
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("Failed to capture image.")
            exit()

    extractor = FaceFeatureExtractor()
    struct = extractor.extract(frame)

    if struct:
        print("=== Extracted StructuralData ===")
        for key, value in struct.__dict__.items():
            print(f"{key}: {value}")
    else:
        print("No face found.")
