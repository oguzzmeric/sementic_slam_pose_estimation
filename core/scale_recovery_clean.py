"""
core/scale_recovery.py
=======================
Monocular scale recovery.

The translation vector from SVD decomposition is a unit vector -- it carries
direction but no magnitude. This module recovers the metric magnitude.

Two modes:

  1. Warmup (frame <= warmup_frames)
     Scale computed from ground truth displacement.
         s = ||t_GT|| / ||t_VO_xy||
     Simultaneously calibrates the depth-to-scale coefficient:
         k_i = s_GT_i / Z_i        ->  k_bar = median(k_i)

  2. Autonomous (frame > warmup_frames)
     Scale derived from semantic depth using the calibrated coefficient:
         s = k_bar * Z

Depth from bounding box (pinhole model, nadir view):

     A = (f_y * H_real * cos^2(theta)) / d_pixel

  d_pixel : dominant bbox edge -- max(width, height)
            In nadir view the bbox is axis-aligned, so a vehicle oriented
            east-west produces a short bbox height even though its length
            is the reference dimension. Taking the dominant edge removes
            this orientation dependence.

  cos^2   : off-axis correction. The naive formula assumes the object sits
            on the optical axis. An object at radial pixel distance r sees
            two compounding effects: slant range grows as 1/cos(theta), and
            the ground patch foreshortens by cos(theta). At the image corner
            of this camera (r = 1101 px, f = 1413) theta = 37.9 deg and
            cos^2 = 0.62 -- a 38 percent systematic error if uncorrected.
"""

import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import numpy as np

from core.motion_estimator import PoseEstimate
from utils.camera_calibration import CameraCalibration
from utils.data_loader import DataLoader, Detection

logger = logging.getLogger(__name__)


class ScaleRecoveryError(Exception):
    pass


@dataclass
class DepthEstimate:
    """Single depth measurement from one bounding box."""
    depth: float          # corrected altitude estimate, metres
    depth_raw: float      # before cos^2 correction
    class_id: int
    confidence: float
    quality: float        # selection score
    radial_px: float      # distance from principal point
    cos2: float           # applied correction factor
    d_pixel: float        # dominant bbox edge used

    def __repr__(self) -> str:
        return (
            f"DepthEstimate(Z={self.depth:.2f}m, raw={self.depth_raw:.2f}m, "
            f"cls={self.class_id}, q={self.quality:.3f}, cos2={self.cos2:.3f})"
        )


@dataclass
class ScaleResult:
    scale: float
    t_scaled: np.ndarray
    mode: str
    is_valid: bool
    frame_name: str
    reference_class: Optional[int] = None
    estimated_depth: Optional[float] = None

    def __repr__(self) -> str:
        dep = f"{self.estimated_depth:.2f}m" if self.estimated_depth is not None else "N/A"
        return (
            f"ScaleResult(frame={self.frame_name}, mode={self.mode}, "
            f"scale={self.scale:.4f}, Z={dep}, valid={self.is_valid})"
        )


class ScaleRecovery:
    """
    Recovers metric scale for monocular visual odometry.

    Selection policy for the reference object:
        quality = confidence * cos^2(theta)

    Both terms matter. Confidence reflects classification certainty, not
    localisation quality, so it cannot be used alone. The cos^2 term
    down-weights objects near the image border, where bbox regression is
    least reliable and the geometric correction is largest.
    """

    # depth plausibility window (metres)
    _MIN_DEPTH = 1.0
    _MAX_DEPTH = 500.0

    # minimum bbox edge in pixels -- below this, quantisation dominates
    _MIN_BBOX_PX = 12

    # autonomous scale must stay within these multiples of the warmup median
    _SCALE_LOWER_MULT = 0.30
    _SCALE_UPPER_MULT = 3.00

    # rolling median window for depth smoothing
    _DEPTH_WINDOW = 7

    def __init__(
        self,
        data_loader: DataLoader,
        camera_calibration: CameraCalibration,
    ) -> None:
        logger.info("[ScaleRecovery] Initializing...")

        self._loader = data_loader
        self._cam = camera_calibration

        eval_cfg = data_loader.get_evaluation_config()
        sem_cfg = data_loader.get_semantic_config()

        self._warmup_limit = int(eval_cfg.get("warmup_frames", 150))
        self._reference_objects = self._parse_reference_objects(sem_cfg)
        self._min_confidence = float(sem_cfg.get("min_confidence", 0.7))

        # optional switches
        self._use_cos2 = bool(sem_cfg.get("perspective_correction", True))
        self._use_dominant_edge = bool(sem_cfg.get("dominant_bbox_edge", True))

        self._fx = camera_calibration.fx
        self._fy = camera_calibration.fy
        self._cx = camera_calibration.cx
        self._cy = camera_calibration.cy
        # single focal length for radial geometry
        self._f = float(np.sqrt(self._fx * self._fy))

        # warmup accumulators
        self._scale_history: Deque[float] = deque(maxlen=40)
        self._k_history: List[float] = []

        # calibrated values, set when warmup ends
        self._warmup_scale_mean: Optional[float] = None
        self._k_factor: Optional[float] = None

        # depth smoothing
        self._depth_window: Deque[float] = deque(maxlen=self._DEPTH_WINDOW)

        self._frame_count: int = 0

        logger.info(
            "[ScaleRecovery] Ready -- warmup=%d, ref_classes=%s, "
            "cos2=%s, dominant_edge=%s",
            self._warmup_limit,
            sorted(self._reference_objects.keys()),
            self._use_cos2,
            self._use_dominant_edge,
        )

    # ------------------------------------------------------------------
    # config
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_reference_objects(sem_cfg: dict) -> dict:
        if "reference_objects" not in sem_cfg:
            raise ScaleRecoveryError("semantic config missing 'reference_objects'.")
        return {int(k): float(v) for k, v in sem_cfg["reference_objects"].items()}

    # ------------------------------------------------------------------
    # FIX 1 + FIX 2 : depth estimation
    # ------------------------------------------------------------------

    def _bbox_geometry(
        self,
        bbox: List[int],
    ) -> Tuple[float, float, float, float]:
        """
        Extracts the geometric quantities needed for depth estimation.

        Returns:
            (d_pixel, radial_px, cos2, area_px)

            d_pixel   : dominant bbox edge (FIX 1) or height if disabled
            radial_px : bbox centre distance from principal point
            cos2      : cos^2(theta) off-axis correction (FIX 2)
            area_px   : bbox area, used for sanity checks
        """
        x1, y1, x2, y2 = bbox
        w = abs(x2 - x1)
        h = abs(y2 - y1)

        # FIX 1 -- dominant edge instead of height.
        # In nadir view a vehicle's bbox height equals its length only when
        # the vehicle is aligned with the image vertical axis. The dominant
        # edge tracks the reference dimension regardless of heading.
        d_pixel = float(max(w, h)) if self._use_dominant_edge else float(h)

        # bbox centre
        u_c = (x1 + x2) / 2.0
        v_c = (y1 + y2) / 2.0

        # FIX 2 -- off-axis geometry
        dx = u_c - self._cx
        dy = v_c - self._cy
        radial_px = float(np.hypot(dx, dy))

        if self._use_cos2:
            theta = np.arctan2(radial_px, self._f)
            cos2 = float(np.cos(theta) ** 2)
        else:
            cos2 = 1.0

        return d_pixel, radial_px, cos2, float(w * h)

    def _depth_from_detection(self, det: Detection) -> Optional[DepthEstimate]:
        """
        Computes an altitude estimate from a single detection.

            A = (f_y * H_real * cos^2(theta)) / d_pixel

        Returns None when the detection fails any plausibility check.
        """
        if det.class_id not in self._reference_objects:
            return None
        if det.confidence < self._min_confidence:
            return None

        bbox = det.bbox
        if len(bbox) != 4:
            return None

        x1, y1, x2, y2 = bbox
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        if w < self._MIN_BBOX_PX or h < self._MIN_BBOX_PX:
            return None

        d_pixel, radial_px, cos2, _ = self._bbox_geometry(bbox)
        if d_pixel < 1e-6:
            return None

        H_real = self._reference_objects[det.class_id]

        depth_raw = (self._fy * H_real) / d_pixel
        depth = depth_raw * cos2

        if not (self._MIN_DEPTH <= depth <= self._MAX_DEPTH):
            logger.debug(
                "[ScaleRecovery] Implausible depth %.1fm (cls=%d), rejected.",
                depth, det.class_id,
            )
            return None

        # selection score -- confidence alone is not a localisation quality
        # measure, so it is combined with the off-axis penalty.
        quality = det.confidence * cos2

        return DepthEstimate(
            depth=depth,
            depth_raw=depth_raw,
            class_id=det.class_id,
            confidence=det.confidence,
            quality=quality,
            radial_px=radial_px,
            cos2=cos2,
            d_pixel=d_pixel,
        )

    def _estimate_depth(
        self,
        detections: List[Detection],
    ) -> Optional[DepthEstimate]:
        """
        Selects the best reference detection in a frame.

        Candidates are scored by confidence * cos^2(theta) and the highest
        scoring one is returned. When several candidates score closely, the
        median of their depths is used to suppress a single bad bbox.
        """
        candidates = []
        for det in detections:
            est = self._depth_from_detection(det)
            if est is not None:
                candidates.append(est)

        if not candidates:
            return None

        candidates.sort(key=lambda e: e.quality, reverse=True)
        best = candidates[0]

        # If more than one strong candidate exists, take the median depth of
        # those within 80 percent of the top score. A single mis-regressed
        # bbox then cannot drag the estimate.
        strong = [c for c in candidates if c.quality >= 0.8 * best.quality]
        if len(strong) >= 3:
            med = float(np.median([c.depth for c in strong]))
            best = DepthEstimate(
                depth=med,
                depth_raw=best.depth_raw,
                class_id=best.class_id,
                confidence=best.confidence,
                quality=best.quality,
                radial_px=best.radial_px,
                cos2=best.cos2,
                d_pixel=best.d_pixel,
            )

        return best

    def _smoothed_depth(self, depth: float) -> float:
        """
        Rolling median over recent frames.

        Bounding box regression jitters frame to frame, and depth is
        inversely proportional to bbox size, so raw depth is noisy. The
        aircraft's altitude changes slowly, so a short median window
        removes jitter without lagging genuine altitude change.
        """
        self._depth_window.append(depth)
        return float(np.median(self._depth_window))

    # ------------------------------------------------------------------
    # warmup scale from ground truth
    # ------------------------------------------------------------------

    def _warmup_scale_from_gt(
        self,
        pose: PoseEstimate,
        frame_name: str,
    ) -> Optional[float]:
        """
        s = ||GT displacement|| / ||t_VO_xy||
        """
        gt_curr = self._loader.get_ground_truth(frame_name)
        gt_prev = self._loader.get_ground_truth(pose.frame_name_prev)

        if gt_curr is None or gt_prev is None:
            return None

        dx = gt_curr.tx - gt_prev.tx
        dy = gt_curr.ty - gt_prev.ty
        gt_norm = float(np.hypot(dx, dy))

        if gt_norm < 1e-6:
            return None

        t_xy = pose.t.flatten()[:2]
        t_xy_norm = float(np.linalg.norm(t_xy))
        if t_xy_norm < 1e-6:
            return None

        scale = gt_norm / t_xy_norm

        if not (0.01 <= scale <= 50.0):
            logger.debug("[ScaleRecovery] Warmup scale %.2f out of range.", scale)
            return None

        return scale

    # ------------------------------------------------------------------
    # FIX 3 : depth-to-scale calibration
    # ------------------------------------------------------------------

    def _finalize_calibration(self) -> None:
        """
        Computes the calibrated constants at the end of warmup.

            k = median(s_GT_i / Z_i)

        Physical reading: at a fixed angular rate the metric displacement
        per frame is proportional to altitude. A drone at twice the height
        covers twice the ground for the same image motion. The coefficient
        k absorbs focal length, frame interval and typical angular rate.
        """
        if self._scale_history:
            self._warmup_scale_mean = float(np.median(self._scale_history))
        else:
            self._warmup_scale_mean = 1.0
            logger.warning("[ScaleRecovery] No warmup scale samples; defaulting to 1.0")

        if len(self._k_history) >= 5:
            self._k_factor = float(np.median(self._k_history))
            spread = float(np.std(self._k_history))
            logger.info(
                "[ScaleRecovery] Warmup complete -- median scale=%.4f (%d samples), "
                "k=%.6f (%d samples, std=%.6f)",
                self._warmup_scale_mean, len(self._scale_history),
                self._k_factor, len(self._k_history), spread,
            )
        else:
            self._k_factor = None
            logger.warning(
                "[ScaleRecovery] Warmup complete -- median scale=%.4f, "
                "but only %d k samples. Autonomous mode will use fixed scale.",
                self._warmup_scale_mean, len(self._k_history),
            )

    def _autonomous_scale(
        self,
        frame_name: str,
    ) -> Tuple[Optional[float], Optional[float], Optional[int]]:
        """
        Scale from calibrated semantic depth.

            s = k_bar * Z_smoothed

        Returns (scale, depth, class_id). Scale is None when no usable
        reference exists or the result falls outside the plausibility band.
        """
        detections = self._loader.get_detections(frame_name)
        if not detections:
            return None, None, None

        est = self._estimate_depth(detections)
        if est is None:
            return None, None, None

        depth = self._smoothed_depth(est.depth)

        if self._k_factor is None:
            return None, depth, est.class_id

        scale = self._k_factor * depth

        lo = self._SCALE_LOWER_MULT * (self._warmup_scale_mean or 1.0)
        hi = self._SCALE_UPPER_MULT * (self._warmup_scale_mean or 1.0)
        if not (lo <= scale <= hi):
            logger.debug(
                "[ScaleRecovery] Autonomous scale %.4f outside [%.4f, %.4f], "
                "falling back. Z=%.1fm",
                scale, lo, hi, depth,
            )
            return None, depth, est.class_id

        logger.debug(
            "[ScaleRecovery] Autonomous: Z=%.2fm (raw %.2fm, cos2=%.3f, "
            "r=%.0fpx, cls=%d) -> s=%.4f",
            depth, est.depth_raw, est.cos2, est.radial_px, est.class_id, scale,
        )
        return scale, depth, est.class_id

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def recover(
        self,
        pose: PoseEstimate,
        frame_name: str,
    ) -> ScaleResult:
        self._frame_count += 1

        if not pose.is_valid or pose.t is None:
            return ScaleResult(
                scale=0.0,
                t_scaled=np.zeros((3, 1)),
                mode="none",
                is_valid=False,
                frame_name=frame_name,
            )

        # --------------------------------------------------------------
        # warmup
        # --------------------------------------------------------------
        if self._frame_count <= self._warmup_limit:
            s_gt = self._warmup_scale_from_gt(pose, frame_name)
            if s_gt is not None:
                self._scale_history.append(s_gt)

            # calibrate k in parallel -- both quantities are available here
            depth_val = None
            class_val = None
            detections = self._loader.get_detections(frame_name)
            if detections:
                est = self._estimate_depth(detections)
                if est is not None:
                    depth_val = est.depth
                    class_val = est.class_id
                    if s_gt is not None and depth_val > 1e-6:
                        self._k_history.append(s_gt / depth_val)

            running = (
                float(np.median(self._scale_history))
                if self._scale_history else 1.0
            )

            if self._frame_count == self._warmup_limit:
                self._finalize_calibration()

            return ScaleResult(
                scale=running,
                t_scaled=running * pose.t,
                mode="warmup",
                is_valid=True,
                frame_name=frame_name,
                reference_class=class_val,
                estimated_depth=depth_val,
            )

        # --------------------------------------------------------------
        # autonomous
        # --------------------------------------------------------------
        scale, depth, class_id = self._autonomous_scale(frame_name)

        if scale is None:
            fb = self._warmup_scale_mean if self._warmup_scale_mean else 1.0
            return ScaleResult(
                scale=fb,
                t_scaled=fb * pose.t,
                mode="autonomous_fallback",
                is_valid=True,
                frame_name=frame_name,
                reference_class=class_id,
                estimated_depth=depth,
            )

        return ScaleResult(
            scale=scale,
            t_scaled=scale * pose.t,
            mode="autonomous",
            is_valid=True,
            frame_name=frame_name,
            reference_class=class_id,
            estimated_depth=depth,
        )

    # ------------------------------------------------------------------

    @property
    def is_warmup_complete(self) -> bool:
        return self._frame_count > self._warmup_limit

    @property
    def warmup_scale(self) -> Optional[float]:
        return self._warmup_scale_mean

    @property
    def k_factor(self) -> Optional[float]:
        return self._k_factor

    def __repr__(self) -> str:
        k = f"{self._k_factor:.6f}" if self._k_factor else "pending"
        return (
            f"ScaleRecovery(warmup={self._warmup_limit}, "
            f"frames={self._frame_count}, k={k})"
        )


# ----------------------------------------------------------------------
# standalone test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config_path = Path(__file__).resolve().parent.parent / "config.yaml"

    try:
        from utils.data_loader import DataLoader
        from utils.camera_calibration import CameraCalibration
        from core.feature_extractor import FeatureExtractor
        from core.matcher import Matcher
        from core.motion_estimator import MotionEstimator

        loader = DataLoader(str(config_path))
        cam = CameraCalibration(loader)
        extractor = FeatureExtractor(loader, cam)
        matcher = Matcher(loader)
        estimator = MotionEstimator(loader, cam)
        sr = ScaleRecovery(loader, cam)

        print(f"\n{sr}\n")
        print("--- Scale Recovery Test (900 frames) ---")

        prev = None
        results = []

        for idx, name, frame in loader.frame_generator():
            curr = extractor.extract(frame, name)
            if prev is not None:
                mr = matcher.match(prev, curr)
                pose = estimator.estimate(mr)
                res = sr.recover(pose, name)
                results.append(res)
            prev = curr
            if idx >= 900:
                break

        valid = [r for r in results if r.is_valid]
        by_mode = {}
        for r in valid:
            by_mode.setdefault(r.mode, []).append(r)

        print(f"\n--- Statistics ---")
        print(f"  Valid : {len(valid)}/{len(results)}")
        for m, v in sorted(by_mode.items()):
            sc = [r.scale for r in v]
            print(f"  {m:22s} n={len(v):5d}  median={np.median(sc):.4f}  "
                  f"mean={np.mean(sc):.4f}  std={np.std(sc):.4f}")

        if sr.k_factor:
            print(f"\n  k factor        : {sr.k_factor:.6f}")
        print(f"  warmup scale    : {sr.warmup_scale:.4f}")

        depths = [r.estimated_depth for r in valid if r.estimated_depth]
        if depths:
            print(f"  depth median    : {np.median(depths):.2f} m")
            print(f"  depth range     : [{min(depths):.1f}, {max(depths):.1f}] m")

    except Exception as e:
        logger.error("Error: %s", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)