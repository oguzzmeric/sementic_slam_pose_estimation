"""
core/pose_graph.py
===================
Pose graph and trajectory accumulation module.

Local VO chain:
    T_world = T_world * T_local
    T_local = [R | t_scaled]
              [0 |     1   ]
    raw_position = T_world[:3, 3]

The chain above lives in an arbitrary "local" frame -- its origin and axis
orientation come from the first camera pose, not from the GT/world frame.
Reporting raw_position directly (or guessing a fixed axis swap) only works
by luck. Instead, while GT is available (scale_result.mode == "warmup"),
every (raw_position, GT) pair is collected; the moment GT stops being
available a Umeyama similarity transform (scale + rotation + translation,
i.e. Sim(3)) is fit once from those pairs and reused for every subsequent
frame:

    position = s * R_align @ raw_position + t_align

This replaces the old swap_xy/flip_y flags (a hand-guessed 90-degree axis
swap, fit to one dataset) with a proper least-squares fit -- and gives a
natural hook for relocalization: whenever GT reappears later, the same
mechanism can refit and reset accumulated drift.

Coordinate frame flags (config.yaml evaluation section):
    force_2d : zero out Z accumulation (per-step, independent of the fit)

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

        self._force_2d = bool(self._eval_cfg.get("force_2d", False))
        self._min_sim3_samples = int(self._eval_cfg.get("min_sim3_samples", 10))

        self._T_world = np.eye(4, dtype=np.float64)
        self._trajectory: List[TrajectoryPoint] = []
        self._frame_idx: int = 0

        # Sim(3) alignment: local-frame -> GT-frame. Fit once from
        # (raw_position, GT) pairs collected while GT is available.
        self._local_gt_pairs: List[tuple] = []
        self._sim3_fit: Optional[tuple] = None  # (s, R, t) once fit

        logger.info(
            "[PoseGraph] Ready -- force_2d=%s, min_sim3_samples=%d",
            self._force_2d, self._min_sim3_samples,
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

    @staticmethod
    def _umeyama(P: np.ndarray, Q: np.ndarray) -> Optional[tuple]:
        """
        Least-squares similarity transform P -> Q: finds (s, R, t) minimizing
        sum ||s*R@p_i + t - q_i||^2. Standard Umeyama/Kabsch solution via SVD.

        Returns None if P is degenerate (near-zero spread -- SVD would be
        numerically meaningless).
        """
        Pt = P.T
        Qt = Q.T
        n = Pt.shape[1]
        mu_p = Pt.mean(axis=1, keepdims=True)
        mu_q = Qt.mean(axis=1, keepdims=True)
        Pc = Pt - mu_p
        Qc = Qt - mu_q

        var_p = (Pc ** 2).sum() / n
        if var_p < 1e-9:
            return None

        W = Qc @ Pc.T / n
        U, D, Vt = np.linalg.svd(W)
        S = np.eye(3)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            S[2, 2] = -1
        R = U @ S @ Vt
        s = float(np.trace(np.diag(D) @ S) / var_p)
        t = (mu_q - s * R @ mu_p).flatten()
        return s, R, t

    def _fit_sim3(self) -> Optional[tuple]:
        if len(self._local_gt_pairs) < self._min_sim3_samples:
            return None
        P = np.array([p[0] for p in self._local_gt_pairs])
        Q = np.array([p[1] for p in self._local_gt_pairs])
        return self._umeyama(P, Q)

    def _report_position(self, raw_position: np.ndarray) -> np.ndarray:
        """Maps a local-frame position into the GT/world frame using the
        fitted Sim(3) transform. Before a fit exists (not enough GT samples
        collected yet) the raw local position is returned as-is."""
        if self._sim3_fit is None:
            return raw_position.copy()
        s, R, t = self._sim3_fit
        return s * (R @ raw_position) + t

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

        if not pose.is_valid or not scale_result.is_valid:
            position = self._report_position(self._T_world[:3, 3])
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


        t_vec = scale_result.t_scaled.flatten().copy()

        if self._force_2d:
            # Zeroing Z shortens the XY norm. Rescale so the planar step
            # keeps the metric length that scale recovery assigned to it.
            n_before = float(np.linalg.norm(t_vec))
            t_vec[2] = 0.0
            xy = float(np.linalg.norm(t_vec))
            if xy > 1e-9 and n_before > 1e-9:
                t_vec *= (n_before / xy)

        T_local = self._build_local_transform(pose.R, t_vec)
        self._T_world = self._T_world @ T_local

        if self._force_2d:
            self._T_world[2, 3] = 0.0

        raw_position = self._T_world[:3, 3].copy()

        gt = self._loader.get_ground_truth(frame_name)
        if scale_result.mode == "warmup" and gt is not None:
            # GT is available -- collect a correspondence sample and report
            # GT directly (no need to estimate what we already know).
            self._local_gt_pairs.append((raw_position.copy(), gt.as_vector()))
            position = gt.as_vector().copy()
        else:
            if self._sim3_fit is None:
                self._sim3_fit = self._fit_sim3()
            position = self._report_position(raw_position)

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
            "[PoseGraph] %s pos=[%.3f, %.3f, %.3f]",
            frame_name, position[0], position[1], position[2],
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
        return self._report_position(raw)

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
                gt_positions.append([gt.tx, gt.ty, gt.tz])   # tz artik gercek
            else:
                gt_positions.append([0.0, 0.0, 0.0])
        return np.array(gt_positions)

    def compute_ate(self, data_loader: DataLoader, use_3d: Optional[bool] = None) -> Optional[float]:
        """
        ATE = sqrt(1/N * sum(||t_GT_i - t_est_i||^2))

        use_3d None ise config'deki ate_3d degeri kullanilir.
        Iki veri setini karsilastirirken ayni metrigi kullan --
        yeni setin 3B ATE'si ile eski setin 2B ATE'si yan yana
        konulmaz.
        """
        if use_3d is None:
            use_3d = bool(self._eval_cfg.get("ate_3d", False))

        est_positions = self.get_positions_array()
        gt_positions = self.get_gt_array(data_loader)

        if len(est_positions) < 2 or len(gt_positions) < 2:
            logger.warning("[PoseGraph] Not enough points for ATE.")
            return None

        N = min(len(est_positions), len(gt_positions))
        dims = 3 if use_3d else 2
        est = est_positions[:N, :dims]
        gt = gt_positions[:N, :dims]

        est = est - est[0]
        gt = gt - gt[0]

        errors = np.linalg.norm(est - gt, axis=1)
        ate = float(np.sqrt(np.mean(errors ** 2)))

        logger.info("[PoseGraph] ATE(%dD) = %.4f m (%d points)", dims, ate, N)
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
        fit = "pending" if self._sim3_fit is None else f"s={self._sim3_fit[0]:.4f}"
        return (
            f"PoseGraph("
            f"total={len(self._trajectory)}, "
            f"valid={len(self.valid_trajectory)}, "
            f"force_2d={self._force_2d}, "
            f"sim3={fit})"
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