"""
core/scale_recovery.py
=======================
Scale recovery module.
Solves the fundamental monocular constraint: t vector from SVD is unit vector,
no absolute metric scale exists.

Two modes:
    1. Warmup (frame < warmup_limit):
       Real metric scale computed from ground-truth CSV.
       s = ||t_GT|| / ||t_est_xy||

    2. Autonomous (frame >= warmup_limit):
       Uses warmup median scale as base.
       Depth from YOLO bbox is used for reference only.
       Z = (f_y * H_real) / h_pixel
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
class ScaleResult:
    scale: float
    t_scaled: np.ndarray
    mode: str
    is_valid: bool
    frame_name: str
    reference_class: Optional[int] = None
    estimated_depth: Optional[float] = None

    def __repr__(self) -> str:
        depth_str = f"{self.estimated_depth:.2f}m" if self.estimated_depth is not None else "N/A"
        return (
            f"ScaleResult("
            f"frame={self.frame_name}, "
            f"mode={self.mode}, "
            f"scale={self.scale:.4f}, "
            f"depth={depth_str}, "
            f"valid={self.is_valid})"
        )


class ScaleRecovery:
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

        self._warmup_limit = self._parse_warmup_limit(eval_cfg)
        self._reference_objects = self._parse_reference_objects(sem_cfg)
        self._min_confidence = float(sem_cfg.get("min_confidence", 0.7))
        self._fy = camera_calibration.fy

        self._scale_history: Deque[float] = deque(maxlen=20)
        self._warmup_scale_mean: Optional[float] = None
        self._frame_count: int = 0

        logger.info(
            "[ScaleRecovery] Ready — warmup_limit=%d, reference_classes=%s",
            self._warmup_limit,
            list(self._reference_objects.keys()),
        )

    @staticmethod
    def _parse_warmup_limit(eval_cfg: dict) -> int:
        if "warmup_frames" not in eval_cfg:
            raise ScaleRecoveryError("evaluation config missing 'warmup_frames'.")
        return int(eval_cfg["warmup_frames"])

    @staticmethod
    def _parse_reference_objects(sem_cfg: dict) -> dict:
        if "reference_objects" not in sem_cfg:
            raise ScaleRecoveryError("semantic config missing 'reference_objects'.")
        return {int(k): float(v) for k, v in sem_cfg["reference_objects"].items()}

    def _compute_warmup_scale(
        self,
        pose: PoseEstimate,
        frame_name: str,
    ) -> Optional[float]:
        gt_curr = self._loader.get_ground_truth(frame_name)
        gt_prev = self._loader.get_ground_truth(pose.frame_name_prev)

        if gt_curr is None or gt_prev is None:
            logger.warning("[ScaleRecovery] GT not found: %s or %s", frame_name, pose.frame_name_prev)
            return None

        gt_displacement = gt_curr.as_vector() - gt_prev.as_vector()
        gt_norm = np.linalg.norm(gt_displacement)

        if gt_norm < 1e-9:
            logger.debug("[ScaleRecovery] GT displacement near zero: %s", frame_name)
            return None

        t_xy = pose.t.flatten()[:2]
        t_xy_norm = np.linalg.norm(t_xy)

        if t_xy_norm < 1e-9:
            logger.warning("[ScaleRecovery] VO t_xy near zero: %s", frame_name)
            return None

        scale = gt_norm / t_xy_norm

        if not (0.01 <= scale <= 50.0):
            logger.debug("[ScaleRecovery] Unreasonable warmup scale (%.2f), skipping.", scale)
            return None

        logger.debug("[ScaleRecovery] Warmup scale: GT=%.4f, t_xy=%.4f, s=%.4f", gt_norm, t_xy_norm, scale)
        return scale

    def _estimate_depth_from_bbox(
        self,
        detections: List[Detection],
    ) -> Tuple[Optional[float], Optional[int]]:
        best_depth = None
        best_class = None
        best_confidence = -1.0

        for det in detections:
            if det.class_id not in self._reference_objects:
                continue
            if det.confidence < self._min_confidence:
                continue

            x1, y1, x2, y2 = det.bbox
            h_pixel = abs(y2 - y1)
            w_pixel = abs(x2 - x1)

            if h_pixel < 10 or w_pixel < 10:
                continue

            H_real = self._reference_objects[det.class_id]
            depth = (self._fy * H_real) / h_pixel

            if not (1.0 <= depth <= 500.0):
                continue

            if det.confidence > best_confidence:
                best_confidence = det.confidence
                best_depth = depth
                best_class = det.class_id

        return best_depth, best_class

    def _compute_autonomous_scale(
        self,
        pose: PoseEstimate,
        frame_name: str,
    ) -> Tuple[Optional[float], Optional[float], Optional[int]]:
        detections = self._loader.get_detections(frame_name)

        if not detections:
            logger.debug("[ScaleRecovery] No detections: %s", frame_name)
            return None, None, None

        depth, class_id = self._estimate_depth_from_bbox(detections)

        if depth is None:
            logger.debug("[ScaleRecovery] No valid reference object: %s", frame_name)
            return None, None, None

        # Use warmup median scale as base — depth is reference only
        if self._warmup_scale_mean is not None:
            scale = self._warmup_scale_mean
        else:
            scale = 1.0

        logger.debug(
            "[ScaleRecovery] Autonomous scale: depth=%.2fm, class=%d, scale=%.4f",
            depth, class_id, scale,
        )
        return scale, depth, class_id

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

        # Warmup mode
        if self._frame_count <= self._warmup_limit:
            scale = self._compute_warmup_scale(pose, frame_name)

            if scale is not None:
                self._scale_history.append(scale)

            if len(self._scale_history) > 0:
                mean_scale = float(np.median(self._scale_history))
            else:
                mean_scale = 1.0

            if self._frame_count == self._warmup_limit:
                self._warmup_scale_mean = float(np.median(self._scale_history))
                logger.info(
                    "[ScaleRecovery] Warmup complete. Median scale=%.4f (%d samples)",
                    self._warmup_scale_mean,
                    len(self._scale_history),
                )

            t_scaled = mean_scale * pose.t
            return ScaleResult(
                scale=mean_scale,
                t_scaled=t_scaled,
                mode="warmup",
                is_valid=True,
                frame_name=frame_name,
            )

        # Autonomous mode
        scale, depth, class_id = self._compute_autonomous_scale(pose, frame_name)

        if scale is None:
            fallback_scale = self._warmup_scale_mean if self._warmup_scale_mean else 1.0
            t_scaled = fallback_scale * pose.t
            return ScaleResult(
                scale=fallback_scale,
                t_scaled=t_scaled,
                mode="autonomous_fallback",
                is_valid=True,
                frame_name=frame_name,
                reference_class=class_id,
                estimated_depth=depth,
            )

        t_scaled = scale * pose.t
        return ScaleResult(
            scale=scale,
            t_scaled=t_scaled,
            mode="autonomous",
            is_valid=True,
            frame_name=frame_name,
            reference_class=class_id,
            estimated_depth=depth,
        )

    @property
    def is_warmup_complete(self) -> bool:
        return self._frame_count > self._warmup_limit

    @property
    def warmup_scale(self) -> Optional[float]:
        return self._warmup_scale_mean

    def __repr__(self) -> str:
        return (
            f"ScaleRecovery("
            f"warmup_limit={self._warmup_limit}, "
            f"frame_count={self._frame_count}, "
            f"warmup_complete={self.is_warmup_complete})"
        )


if __name__ == "__main__":
    import sys
    from pathlib import Path

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config_path = Path(__file__).resolve().parent.parent / "config.yaml"

    try:
        from utils.data_loader import DataLoader, DataLoaderError
        from utils.camera_calibration import CameraCalibration, CameraCalibrationError
        from core.feature_extractor import FeatureExtractor
        from core.matcher import Matcher
        from core.motion_estimator import MotionEstimator

        loader = DataLoader(str(config_path))
        cam = CameraCalibration(loader)
        extractor = FeatureExtractor(loader, cam)
        matcher = Matcher(loader)
        estimator = MotionEstimator(loader, cam)
        scale_recovery = ScaleRecovery(loader, cam)

        print(f"\n{scale_recovery}")
        print("\n--- Scale Recovery Test (first 300 frames) ---")

        prev_features = None
        results = []

        for idx, name, frame in loader.frame_generator():
            curr_features = extractor.extract(frame, name)

            if prev_features is not None:
                match_result = matcher.match(prev_features, curr_features)
                pose = estimator.estimate(match_result)
                scale_result = scale_recovery.recover(pose, name)
                results.append(scale_result)

                if idx % 50 == 0:
                    print(f"\n[{idx}] {scale_result}")
                    if scale_result.is_valid and pose.is_valid:
                        print(f"     t_unit  : {pose.t.T}")
                        print(f"     t_scaled: {scale_result.t_scaled.T}")

            prev_features = curr_features

            if idx >= 300:
                break

        valid = [r for r in results if r.is_valid]
        warmup = [r for r in valid if r.mode == "warmup"]
        autonomous = [r for r in valid if r.mode == "autonomous"]
        fallback = [r for r in valid if r.mode == "autonomous_fallback"]

        print(f"\n--- Statistics ---")
        print(f"  Valid              : {len(valid)}/{len(results)}")
        print(f"  Warmup             : {len(warmup)}")
        print(f"  Autonomous         : {len(autonomous)}")
        print(f"  Autonomous fallback: {len(fallback)}")
        if warmup:
            scales = [r.scale for r in warmup]
            print(f"  Warmup scale mean  : {np.mean(scales):.4f}")
            print(f"  Warmup scale std   : {np.std(scales):.4f}")
        if scale_recovery.warmup_scale:
            print(f"  Warmup median scale: {scale_recovery.warmup_scale:.4f}")
        if autonomous:
            depths = [r.estimated_depth for r in autonomous if r.estimated_depth]
            print(f"  Mean autonomous Z  : {np.mean(depths):.2f}m")

    except Exception as e:
        logger.error("Error: %s", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)