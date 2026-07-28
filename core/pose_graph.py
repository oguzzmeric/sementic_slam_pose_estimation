"""
core/pose_graph.py
===================
Pose graph and trajectory accumulation module.

Math:
    T_world = T_world * T_local
    T_local = [R | t_scaled]
              [0 |     1   ]
    position = T_world[:3, 3]

Coordinate frame flags (config.yaml evaluation section):
    force_2d : zero out Z accumulation
    swap_xy  : swap X and Y axes (camera -> world frame)
    flip_y   : negate Y axis after swap

ATE:
    ATE = sqrt(1/N * sum(||t_GT_i - t_est_i||^2))
"""

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from core.motion_estimator import PoseEstimate
from core.scale_recovery import ScaleResult
from utils.data_loader import DataLoader

logger = logging.getLogger(__name__)


class PoseGraphError(Exception):
    pass


@dataclass
class TrajectoryPoint:
    frame_name: str
    frame_idx: int
    position: np.ndarray
    R_world: np.ndarray
    scale: float
    mode: str
    is_valid: bool

    def __repr__(self) -> str:
        pos = self.position
        return (
            f"TrajectoryPoint("
            f"frame={self.frame_name}, "
            f"pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}], "
            f"scale={self.scale:.4f}, "
            f"mode={self.mode})"
        )


class PoseGraph:
    def __init__(self, data_loader: DataLoader) -> None:
        logger.info("[PoseGraph] Initializing...")

        self._loader = data_loader
        self._eval_cfg = data_loader.get_evaluation_config()

        # Coordinate frame flags
        self._force_2d = bool(self._eval_cfg.get("force_2d", False))
        self._swap_xy  = bool(self._eval_cfg.get("swap_xy",  False))
        self._flip_y   = bool(self._eval_cfg.get("flip_y",   False))

        # Global pose — 4x4 homogeneous transformation
        self._T_world = np.eye(4, dtype=np.float64)

        self._trajectory: List[TrajectoryPoint] = []
        self._frame_idx: int = 0

        logger.info(
            "[PoseGraph] Ready — force_2d=%s, swap_xy=%s, flip_y=%s",
            self._force_2d, self._swap_xy, self._flip_y,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_local_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        return T

    def _apply_coordinate_transform(self, position: np.ndarray) -> np.ndarray:
        """
        Applies coordinate frame corrections to position vector.

        Order:
            1. swap_xy  : position = [Y, X, Z]
            2. flip_y   : position[1] = -position[1]

        Args:
            position: Raw [x, y, z] from T_world.

        Returns:
            Corrected position vector.
        """
        pos = position.copy()

        if self._swap_xy:
            pos[0], pos[1] = pos[1].copy(), pos[0].copy()

        if self._flip_y:
            pos[1] = -pos[1]

        return pos

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        pose: PoseEstimate,
        scale_result: ScaleResult,
        frame_name: str,
    ) -> TrajectoryPoint:
        self._frame_idx += 1

        # Invalid pose — position unchanged
        if not pose.is_valid or not scale_result.is_valid:
            position = self._apply_coordinate_transform(self._T_world[:3, 3])
            R_world = self._T_world[:3, :3].copy()
            point = TrajectoryPoint(
                frame_name=frame_name,
                frame_idx=self._frame_idx,
                position=position,
                R_world=R_world,
                scale=0.0,
                mode="invalid",
                is_valid=False,
            )
            self._trajectory.append(point)
            return point

        # Build translation vector
        t_vec = scale_result.t_scaled.flatten().copy()

        # Zero out Z if force_2d
        if self._force_2d:
            t_vec[2] = 0.0

        # Build and apply local transform
        T_local = self._build_local_transform(pose.R, t_vec)
        self._T_world = self._T_world @ T_local

        # Zero out accumulated Z drift if force_2d
        if self._force_2d:
            self._T_world[2, 3] = 0.0

        # Extract and transform position
        raw_position = self._T_world[:3, 3].copy()
        position = self._apply_coordinate_transform(raw_position)

        R_world = self._T_world[:3, :3].copy()

        point = TrajectoryPoint(
            frame_name=frame_name,
            frame_idx=self._frame_idx,
            position=position,
            R_world=R_world,
            scale=scale_result.scale,
            mode=scale_result.mode,
            is_valid=True,
        )
        self._trajectory.append(point)

        logger.debug(
            "[PoseGraph] %s raw=[%.3f,%.3f,%.3f] transformed=[%.3f,%.3f,%.3f]",
            frame_name,
            raw_position[0], raw_position[1], raw_position[2],
            position[0], position[1], position[2],
        )
        return point

    # ------------------------------------------------------------------
    # Trajectory access
    # ------------------------------------------------------------------

    @property
    def trajectory(self) -> List[TrajectoryPoint]:
        return self._trajectory

    @property
    def valid_trajectory(self) -> List[TrajectoryPoint]:
        return [p for p in self._trajectory if p.is_valid]

    @property
    def current_position(self) -> np.ndarray:
        raw = self._T_world[:3, 3].copy()
        return self._apply_coordinate_transform(raw)

    @property
    def current_rotation(self) -> np.ndarray:
        return self._T_world[:3, :3].copy()

    def get_positions_array(self) -> np.ndarray:
        valid = self.valid_trajectory
        if not valid:
            return np.empty((0, 3))
        return np.array([p.position for p in valid])

    def get_gt_array(self, data_loader: DataLoader) -> np.ndarray:
        gt_positions = []
        for point in self.valid_trajectory:
            gt = data_loader.get_ground_truth(point.frame_name)
            if gt is not None:
                gt_positions.append([gt.tx, gt.ty, 0.0])
            else:
                gt_positions.append([0.0, 0.0, 0.0])
        return np.array(gt_positions)

    def compute_ate(self, data_loader: DataLoader) -> Optional[float]:
        """
        Absolute Trajectory Error (ATE).
        ATE = sqrt(1/N * sum(||t_GT_i - t_est_i||^2))
        Both trajectories normalized to origin.
        Only X, Y used (GT has no Z).
        """
        est_positions = self.get_positions_array()
        gt_positions = self.get_gt_array(data_loader)

        if len(est_positions) < 2 or len(gt_positions) < 2:
            logger.warning("[PoseGraph] Not enough points for ATE.")
            return None

        N = min(len(est_positions), len(gt_positions))
        est = est_positions[:N, :2]
        gt = gt_positions[:N, :2]

        est = est - est[0]
        gt = gt - gt[0]

        errors = np.linalg.norm(est - gt, axis=1)
        ate = float(np.sqrt(np.mean(errors ** 2)))

        logger.info("[PoseGraph] ATE = %.4f m (%d points)", ate, N)
        return ate

    def save_trajectory(self, output_path: str) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_name", "x", "y", "z", "scale", "mode", "is_valid"])
            for point in self._trajectory:
                writer.writerow([
                    point.frame_name,
                    f"{point.position[0]:.6f}",
                    f"{point.position[1]:.6f}",
                    f"{point.position[2]:.6f}",
                    f"{point.scale:.6f}",
                    point.mode,
                    point.is_valid,
                ])

        logger.info("[PoseGraph] Saved: %s (%d points)", path, len(self._trajectory))

    def __repr__(self) -> str:
        return (
            f"PoseGraph("
            f"total={len(self._trajectory)}, "
            f"valid={len(self.valid_trajectory)}, "
            f"force_2d={self._force_2d}, "
            f"swap_xy={self._swap_xy}, "
            f"flip_y={self._flip_y})"
        )


# ------------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------------

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
        from core.scale_recovery import ScaleRecovery

        loader = DataLoader(str(config_path))
        cam = CameraCalibration(loader)
        extractor = FeatureExtractor(loader, cam)
        matcher = Matcher(loader)
        estimator = MotionEstimator(loader, cam)
        scale_recovery = ScaleRecovery(loader, cam)
        pose_graph = PoseGraph(loader)

        print(f"\n{pose_graph}")
        print("\n--- Pose Graph Test (first 300 frames) ---")

        prev_features = None

        for idx, name, frame in loader.frame_generator():
            curr_features = extractor.extract(frame, name)

            if prev_features is not None:
                match_result = matcher.match(prev_features, curr_features)
                pose = estimator.estimate(match_result)
                scale_result = scale_recovery.recover(pose, name)
                traj_point = pose_graph.update(pose, scale_result, name)

                if idx % 50 == 0:
                    print(f"\n[{idx}] {traj_point}")

            prev_features = curr_features

            if idx >= 300:
                break

        positions = pose_graph.get_positions_array()

        print(f"\n--- Trajectory Statistics ---")
        print(f"  Total frames  : {len(pose_graph.trajectory)}")
        print(f"  Valid frames  : {len(pose_graph.valid_trajectory)}")
        if len(positions) > 0:
            print(f"  X range       : [{positions[:,0].min():.3f}, {positions[:,0].max():.3f}] m")
            print(f"  Y range       : [{positions[:,1].min():.3f}, {positions[:,1].max():.3f}] m")
            print(f"  Z range       : [{positions[:,2].min():.3f}, {positions[:,2].max():.3f}] m")
            print(f"  Final pos     : {pose_graph.current_position}")

        ate = pose_graph.compute_ate(loader)
        if ate is not None:
            print(f"\n  ATE           : {ate:.4f} m")

        pose_graph.save_trajectory("data/trajectory_output.csv")
        print(f"\n  Saved: data/trajectory_output.csv")

    except Exception as e:
        logger.error("Error: %s", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)
