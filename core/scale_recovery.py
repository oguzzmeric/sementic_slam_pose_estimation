"""
core/scale_recovery.py
=======================
Monocular scale recovery.

The translation vector from SVD decomposition is a unit vector -- it carries
direction but no magnitude. This module recovers the metric magnitude.

Deployment model
----------------
In a real mission the module engages when GNSS is lost, not at take-off.
At that moment the aircraft still has:
    - last GNSS fix           -> starting position
    - barometer / rangefinder -> altitude, unaffected by the outage
    - IMU, magnetometer

So the warmup phase does not have to sit at the start of the flight.
`warmup_start` lets it begin anywhere, simulating the outage instant.

During the warmup window the reference signal (ground truth here, GNSS in
deployment) is used to learn two things:
    s_ref  : metric displacement per frame
    k      : depth-to-scale coefficient, k = s_ref / Z

After the window closes the reference is never read again.

Altitude sources
----------------
    semantic : Z estimated from YOLO bounding boxes
    gt       : Z read from the reference channel (barometer equivalent)
    hybrid   : gt when available, semantic otherwise

Depth from bounding box (pinhole model, nadir view):

     A = (f_y * H_real * cos^2(theta)) / d_pixel

  d_pixel : dominant bbox edge -- max(width, height). In nadir view the
            bbox is axis-aligned, so a vehicle heading east-west yields a
            short bbox height even though length is the reference metric.
  cos^2   : off-axis correction. The naive formula assumes the object lies
            on the optical axis. Slant range grows as 1/cos(theta) and the
            ground patch foreshortens by cos(theta); the two compound.
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
    """Single altitude measurement from one bounding box."""
    depth: float          # corrected altitude estimate, metres
    depth_raw: float      # before cos^2 correction
    class_id: int
    confidence: float
    quality: float        # selection score
    radial_px: float      # distance from principal point
    cos2: float           # applied correction factor
    d_pixel: float        # dominant bbox edge used

    def __repr__(self) -> str:
        return (f"DepthEstimate(Z={self.depth:.2f}m, raw={self.depth_raw:.2f}m, "
                f"cls={self.class_id}, q={self.quality:.3f})")


@dataclass
class ScaleResult:
    scale: float
    t_scaled: np.ndarray
    mode: str
    is_valid: bool
    frame_name: str
    reference_class: Optional[int] = None
    estimated_depth: Optional[float] = None
    depth_source: Optional[str] = None

    def __repr__(self) -> str:
        dep = f"{self.estimated_depth:.2f}m" if self.estimated_depth is not None else "N/A"
        return (f"ScaleResult(frame={self.frame_name}, mode={self.mode}, "
                f"scale={self.scale:.4f}, Z={dep}, src={self.depth_source}, "
                f"valid={self.is_valid})")


class ScaleRecovery:
    """
    Recovers metric scale for monocular visual odometry.

    Timeline:
        frame < warmup_start                -> pre-warmup, unit scale
        warmup_start <= frame < warmup_end  -> learning from reference
        frame >= warmup_end                 -> autonomous
    """

    _MIN_DEPTH = 1.0
    _MAX_DEPTH = 500.0
    _MIN_BBOX_PX = 12
    _SCALE_LOWER_MULT = 0.30
    _SCALE_UPPER_MULT = 3.00
    _DEPTH_WINDOW = 7
    _MIN_K_SAMPLES = 5

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

        self._warmup_start = int(eval_cfg.get("warmup_start", 0))
        self._warmup_len = int(eval_cfg.get("warmup_frames", 150))
        self._warmup_end = self._warmup_start + self._warmup_len

        self._altitude_source = str(eval_cfg.get("altitude_source", "semantic")).lower()
        if self._altitude_source not in ("semantic", "gt", "hybrid"):
            raise ScaleRecoveryError(
                f"altitude_source must be semantic/gt/hybrid, "
                f"got '{self._altitude_source}'"
            )

        # This dataset uses a NED-style convention: Z grows downward and is
        # measured relative to the launch point. A barometer reports absolute
        # altitude directly, so this conversion is dataset-specific.
        self._gt_z_down = bool(eval_cfg.get("gt_z_is_down", False))
        self._gt_z_offset = float(eval_cfg.get("gt_z_offset", 0.0))

        self._reference_objects = self._parse_reference_objects(sem_cfg)
        self._min_confidence = float(sem_cfg.get("min_confidence", 0.7))
        self._use_cos2 = bool(sem_cfg.get("perspective_correction", True))
        self._use_dominant_edge = bool(sem_cfg.get("dominant_bbox_edge", True))

        self._fx = camera_calibration.fx
        self._fy = camera_calibration.fy
        self._cx = camera_calibration.cx
        self._cy = camera_calibration.cy
        self._f = float(np.sqrt(self._fx * self._fy))

        # learned during warmup
        self._scale_history: Deque[float] = deque(maxlen=60)
        self._k_history: List[float] = []
        self._warmup_scale_mean: Optional[float] = None
        self._k_factor: Optional[float] = None
        self._calibrated: bool = False
        self._flow_scale = bool(eval_cfg.get("optical_flow_scale", False))

        # diagnostics: bbox depth vs reference altitude
        self._depth_pairs: List[Tuple[float, float]] = []

        self._depth_window: Deque[float] = deque(maxlen=self._DEPTH_WINDOW)
        self._frame_count: int = 0

        logger.info(
            "[ScaleRecovery] Ready -- warmup [%d, %d), altitude_source=%s, "
            "cos2=%s, dominant_edge=%s, z_down=%s, z_offset=%.2f",
            self._warmup_start, self._warmup_end, self._altitude_source,
            self._use_cos2, self._use_dominant_edge,
            self._gt_z_down, self._gt_z_offset,
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
    # bbox geometry
    # ------------------------------------------------------------------

    def _bbox_geometry(self, bbox: List[int]) -> Tuple[float, float, float]:
        """
        Returns (d_pixel, radial_px, cos2).

        d_pixel   : dominant bbox edge, orientation independent
        radial_px : bbox centre distance from principal point
        cos2      : cos^2(theta) off-axis correction factor
        """
        x1, y1, x2, y2 = bbox
        w = abs(x2 - x1)
        h = abs(y2 - y1)

        # Dominant edge instead of height. In nadir view a vehicle's bbox
        # height equals its length only when the vehicle is aligned with the
        # image vertical axis; the dominant edge tracks the reference
        # dimension regardless of heading.
        d_pixel = float(max(w, h)) if self._use_dominant_edge else float(h)

        u_c = (x1 + x2) / 2.0
        v_c = (y1 + y2) / 2.0
        radial_px = float(np.hypot(u_c - self._cx, v_c - self._cy))

        if self._use_cos2:
            theta = np.arctan2(radial_px, self._f)
            cos2 = float(np.cos(theta) ** 2)
        else:
            cos2 = 1.0

        return d_pixel, radial_px, cos2

    def _depth_from_detection(self, det: Detection) -> Optional[DepthEstimate]:
        """
        Altitude estimate from a single detection.

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
        w, h = abs(x2 - x1), abs(y2 - y1)
        if w < self._MIN_BBOX_PX or h < self._MIN_BBOX_PX:
            return None

        d_pixel, radial_px, cos2 = self._bbox_geometry(bbox)
        if d_pixel < 1e-6:
            return None

        H_real = self._reference_objects[det.class_id]
        depth_raw = (self._fy * H_real) / d_pixel
        depth = depth_raw * cos2

        if not (self._MIN_DEPTH <= depth <= self._MAX_DEPTH):
            return None

        # Confidence measures classification certainty, not localisation
        # quality, so it is combined with the off-axis penalty.
        quality = det.confidence * cos2

        return DepthEstimate(
            depth=depth, depth_raw=depth_raw, class_id=det.class_id,
            confidence=det.confidence, quality=quality,
            radial_px=radial_px, cos2=cos2, d_pixel=d_pixel,
        )

    def _semantic_depth(self, frame_name: str) -> Optional[DepthEstimate]:
        """
        Altitude estimated from YOLO bounding boxes.

        Picks the best reference detection in the frame. Candidates are
        scored by

            quality = confidence * cos^2(theta)

        Confidence alone is the wrong criterion -- it measures classification
        certainty, not localisation quality. A detector can be 95 percent
        sure an object is a car while placing the box ten pixels off. The
        cos^2 term penalises objects near the image border, where bbox
        regression is least reliable and the geometric correction largest.

        When three or more strong candidates exist their median depth is
        used, so a single mis-regressed box cannot drag the estimate.
        """
        detections = self._loader.get_detections(frame_name)
        if not detections:
            return None

        cands = [e for e in (self._depth_from_detection(d) for d in detections)
                 if e is not None]
        if not cands:
            return None

        cands.sort(key=lambda e: e.quality, reverse=True)
        best = cands[0]

        # Median of the strong candidates. Two candidates would make the
        # median equal the mean, so the guard starts at three.
        strong = [c for c in cands if c.quality >= 0.8 * best.quality]
        if len(strong) >= 3:
            best = DepthEstimate(
                depth=float(np.median([c.depth for c in strong])),
                depth_raw=best.depth_raw, class_id=best.class_id,
                confidence=best.confidence, quality=best.quality,
                radial_px=best.radial_px, cos2=best.cos2, d_pixel=best.d_pixel,
            )
        return best

    # ------------------------------------------------------------------
    # altitude
    # ------------------------------------------------------------------

    def _gt_altitude(self, frame_name: str) -> Optional[float]:
        """
        Altitude above ground from the reference channel.

        This dataset uses a NED-style convention: Z grows downward and is
        measured relative to the launch point. Absolute altitude is

            z_down_positive : Z_agl = offset - z_gt
            z_up_positive   : Z_agl = offset + z_gt

        In deployment a barometer reports absolute altitude directly and
        this conversion is not needed.
        """
        gt = self._loader.get_ground_truth(frame_name)
        if gt is None:
            return None

        z_rel = float(gt.tz)
        z = (self._gt_z_offset - z_rel) if self._gt_z_down \
            else (self._gt_z_offset + z_rel)

        if not (self._MIN_DEPTH <= z <= self._MAX_DEPTH):
            return None
        return z

    def _altitude(self, frame_name: str) -> Tuple[Optional[float], Optional[int], str]:
        """
        Resolves altitude according to the configured source.

        Returns (altitude, class_id, source_label).
        """
        if self._altitude_source == "gt":
            z = self._gt_altitude(frame_name)
            return (z, None, "gt") if z is not None else (None, None, "none")

        if self._altitude_source == "semantic":
            est = self._semantic_depth(frame_name)
            return (est.depth, est.class_id, "semantic") if est else (None, None, "none")

        # hybrid
        z = self._gt_altitude(frame_name)
        if z is not None:
            return z, None, "gt"
        est = self._semantic_depth(frame_name)
        return (est.depth, est.class_id, "semantic") if est else (None, None, "none")

    def _smoothed(self, depth: float) -> float:
        """
        Rolling median. Bbox regression jitters frame to frame and depth is
        inversely proportional to bbox size, so raw depth is noisy. Altitude
        itself changes slowly, so a short window removes jitter without lag.
        """
        self._depth_window.append(depth)
        return float(np.median(self._depth_window))

    # ------------------------------------------------------------------
    # warmup
    # ------------------------------------------------------------------

    def _reference_scale(self, pose: PoseEstimate, frame_name: str) -> Optional[float]:
        """
        Metric displacement between two frames, from the reference channel.

            s = ||delta_ref|| / ||t||

        t is a unit vector, so the full 3D norm is used -- projecting onto
        XY would discard the vertical component the reference provides, and
        would also inflate the scale because ||t_xy|| < 1 whenever t_z != 0.
        """
        gt_curr = self._loader.get_ground_truth(frame_name)
        gt_prev = self._loader.get_ground_truth(pose.frame_name_prev)
        if gt_curr is None or gt_prev is None:
            return None

        delta = gt_curr.as_vector() - gt_prev.as_vector()
        d_norm = float(np.linalg.norm(delta))
        if d_norm < 1e-9:
            return None

        t_norm = float(np.linalg.norm(pose.t))
        if t_norm < 1e-9:
            return None

        scale = d_norm / t_norm
        if not (1e-4 <= scale <= 50.0):
            logger.debug("[ScaleRecovery] Reference scale %.5f out of range.", scale)
            return None
        return scale

    def _finalize(self) -> None:
        """
        Closes the warmup window and fixes the calibrated constants.

            k = median(s_ref_i / Z_i)

        Physical reading: at a fixed angular rate the metric displacement
        per frame is proportional to altitude. A drone at twice the height
        covers twice the ground for the same image motion. The coefficient
        k absorbs focal length, frame interval and typical angular rate.

        Sanity check: k * Z should reproduce the warmup scale.
        """
        self._calibrated = True

        if self._scale_history:
            self._warmup_scale_mean = float(np.median(self._scale_history))
        else:
            self._warmup_scale_mean = 1.0
            logger.warning("[ScaleRecovery] No warmup scale samples; defaulting to 1.0")

        if len(self._k_history) >= self._MIN_K_SAMPLES:
            self._k_factor = float(np.median(self._k_history))
            logger.info(
                "[ScaleRecovery] Warmup closed -- scale=%.5f (%d), "
                "k=%.6f (%d, std=%.6f)",
                self._warmup_scale_mean, len(self._scale_history),
                self._k_factor, len(self._k_history), float(np.std(self._k_history)),
            )
        else:
            self._k_factor = None
            logger.warning(
                "[ScaleRecovery] Warmup closed -- scale=%.5f, only %d k samples. "
                "Autonomous mode falls back to fixed scale.",
                self._warmup_scale_mean, len(self._k_history),
            )

        if len(self._depth_pairs) >= 5:
            a = np.array(self._depth_pairs)
            ratio = a[:, 0] / np.maximum(a[:, 1], 1e-9)
            logger.info(
                "[ScaleRecovery] Depth check -- bbox/reference ratio: "
                "median=%.3f mean=%.3f std=%.3f (n=%d)",
                float(np.median(ratio)), float(np.mean(ratio)),
                float(np.std(ratio)), len(a),
            )
 
    # ------------------------------------------------------------------
    # autonomous
    # ------------------------------------------------------------------

    def _autonomous(
        self, frame_name: str,
    ) -> Tuple[Optional[float], Optional[float], Optional[int], str]:
        """
        Scale from calibrated altitude:  s = k * Z

        Rejected when the result falls outside a plausibility band around
        the warmup median, so a single bad measurement cannot corrupt the
        trajectory.
        """
        z, cls, src = self._altitude(frame_name)
        if z is None:
            return None, None, None, src
        
        z = self._smoothed(z)

        if self._k_factor is None:
            return None, z, cls, src

        scale = self._k_factor * z

        base = self._warmup_scale_mean or 1.0
        lo, hi = self._SCALE_LOWER_MULT * base, self._SCALE_UPPER_MULT * base
        if not (lo <= scale <= hi):
            logger.debug(
                "[ScaleRecovery] Scale %.5f outside [%.5f, %.5f], "
                "falling back. Z=%.2f",
                scale, lo, hi, z,
            )
            return None, z, cls, src

        return scale, z, cls, src

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def recover(self, pose: PoseEstimate, frame_name: str, match_result=None) -> ScaleResult:
        self._frame_count += 1
        idx = self._frame_count

        if not pose.is_valid or pose.t is None:
            return ScaleResult(
                scale=0.0, t_scaled=np.zeros((3, 1)), mode="none",
                is_valid=False, frame_name=frame_name,
            )

        # --------------------------------------------------------------
        # before the warmup window opens
        # --------------------------------------------------------------
        if idx < self._warmup_start:
            return ScaleResult(
                scale=1.0, t_scaled=pose.t.copy(), mode="pre_warmup",
                is_valid=True, frame_name=frame_name,
            )

        # --------------------------------------------------------------
        # inside the warmup window -- learn
        # --------------------------------------------------------------
        if idx < self._warmup_end:
            s_ref = self._reference_scale(pose, frame_name)
            if s_ref is not None:
                self._scale_history.append(s_ref)

            z, cls, src = self._altitude(frame_name)
            if s_ref is not None and z is not None and z > 1e-9:
                self._k_history.append(s_ref / z)

            # Diagnostic: how well does bbox depth track reference altitude.
            # This is what caught the NED sign error -- the ratio came out
            # at 134 instead of 1.0.
            if self._altitude_source != "gt":
                est = self._semantic_depth(frame_name)
                z_ref = self._gt_altitude(frame_name)
                if est is not None and z_ref is not None:
                    self._depth_pairs.append((est.depth, z_ref))

            running = float(np.median(self._scale_history)) if self._scale_history else 1.0

            if idx == self._warmup_end - 1:
                self._finalize()

            return ScaleResult(
                scale=running, t_scaled=running * pose.t, mode="warmup",
                is_valid=True, frame_name=frame_name,
                reference_class=cls, estimated_depth=z, depth_source=src,
            )

        # --------------------------------------------------------------
        # autonomous -- reference channel is no longer read
        # --------------------------------------------------------------
        if not self._calibrated:
            self._finalize()

        if self._flow_scale and match_result is not None:
            z, cls, src = self._altitude(frame_name)
            if z is not None:
                z = self._smoothed(z)
                flow = float(np.median(np.linalg.norm(
                    match_result.pts_curr - match_result.pts_prev, axis=1)))
                scale = flow * z / self._f
                base = self._warmup_scale_mean or 1.0
                if self._SCALE_LOWER_MULT * base <= scale <= self._SCALE_UPPER_MULT * base:
                    return ScaleResult(
                        scale=scale, t_scaled=scale * pose.t, mode="autonomous",
                        is_valid=True, frame_name=frame_name,
                        reference_class=cls, estimated_depth=z, depth_source=src)

        scale, z, cls, src = self._autonomous(frame_name)

        if scale is None:
            fb = self._warmup_scale_mean or 1.0
            return ScaleResult(
                scale=fb, t_scaled=fb * pose.t, mode="autonomous_fallback",
                is_valid=True, frame_name=frame_name,
                reference_class=cls, estimated_depth=z, depth_source=src,
            )

        return ScaleResult(
            scale=scale, t_scaled=scale * pose.t, mode="autonomous",
            is_valid=True, frame_name=frame_name,
            reference_class=cls, estimated_depth=z, depth_source=src,
        )

    # ------------------------------------------------------------------

    @property
    def is_warmup_complete(self) -> bool:
        return self._frame_count >= self._warmup_end

    @property
    def warmup_scale(self) -> Optional[float]:
        return self._warmup_scale_mean

    @property
    def k_factor(self) -> Optional[float]:
        return self._k_factor

    @property
    def depth_check(self) -> Optional[dict]:
        """bbox depth vs reference altitude, collected during warmup."""
        if len(self._depth_pairs) < 5:
            return None
        a = np.array(self._depth_pairs)
        ratio = a[:, 0] / np.maximum(a[:, 1], 1e-9)
        return {
            "n": len(a),
            "median": float(np.median(ratio)),
            "mean": float(np.mean(ratio)),
            "std": float(np.std(ratio)),
            "bbox_median": float(np.median(a[:, 0])),
            "ref_median": float(np.median(a[:, 1])),
        }

    def __repr__(self) -> str:
        k = f"{self._k_factor:.6f}" if self._k_factor else "pending"
        return (f"ScaleRecovery(warmup=[{self._warmup_start},{self._warmup_end}), "
                f"src={self._altitude_source}, frames={self._frame_count}, k={k})")


# ----------------------------------------------------------------------
# standalone test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

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

        # quick sanity: both altitude sources on the same frames
        print("--- altitude sources ---")
        for nm in loader.frame_list[:5]:
            est = sr._semantic_depth(nm.name)
            gz = sr._gt_altitude(nm.name)
            if est is None:
                print(f"  {nm.name:24s} semantic=None      gt={gz}")
            else:
                print(f"  {nm.name:24s} semantic={est.depth:6.2f} "
                      f"(cls={est.class_id}, q={est.quality:.3f})   gt={gz}")

        LIMIT = 900
        print(f"\n--- Scale Recovery Test ({LIMIT} frames) ---")

        prev = None
        results = []
        for idx, name, frame in loader.frame_generator():
            curr = extractor.extract(frame, name)
            if prev is not None:
                pose = estimator.estimate(matcher.match(prev, curr))
                results.append(sr.recover(pose, name))
            prev = curr
            if idx >= LIMIT:
                break

        valid = [r for r in results if r.is_valid]
        by_mode = {}
        for r in valid:
            by_mode.setdefault(r.mode, []).append(r)

        print(f"\n--- Statistics ---")
        print(f"  Valid : {len(valid)}/{len(results)}")
        for m in sorted(by_mode):
            v = [r.scale for r in by_mode[m]]
            print(f"  {m:22s} n={len(v):5d}  median={np.median(v):.5f}  "
                  f"mean={np.mean(v):.5f}  std={np.std(v):.5f}")

        print()
        print(f"  warmup scale : {sr.warmup_scale}")
        print(f"  k factor     : {sr.k_factor}")

        dz = [r.estimated_depth for r in valid if r.estimated_depth]
        if dz:
            print(f"  altitude     : median {np.median(dz):.2f} m  "
                  f"range [{min(dz):.1f}, {max(dz):.1f}]")

        dc = sr.depth_check
        if dc:
            print()
            print("  --- bbox depth vs reference altitude ---")
            print(f"  samples      : {dc['n']}")
            print(f"  ratio        : median {dc['median']:.3f}  "
                  f"mean {dc['mean']:.3f}  std {dc['std']:.3f}")
            print(f"  bbox median  : {dc['bbox_median']:.2f} m")
            print(f"  ref  median  : {dc['ref_median']:.2f} m")
            print("  (ratio 1.0 = bbox depth matches reference)")

    except Exception as e:
        logger.error("Error: %s", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)