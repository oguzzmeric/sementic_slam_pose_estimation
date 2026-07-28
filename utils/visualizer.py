"""
utils/visualizer.py
====================
Trajectory visualization and analysis module.
Plots estimated trajectory vs ground truth.
Computes ATE per frame and saves comparison plots.
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class VisualizerError(Exception):
    pass


class Visualizer:
    def __init__(self, output_dir: str = "data/viz") -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("[Visualizer] Output dir: %s", self._output_dir)

    def plot_trajectory_2d(
        self,
        est_positions: np.ndarray,
        gt_positions: np.ndarray,
        title: str = "Trajectory Comparison",
        filename: str = "trajectory_2d.png",
    ) -> None:
        """
        Plots estimated vs GT trajectory in 2D (X-Y plane).

        Args:
            est_positions : (N x 3) estimated positions array
            gt_positions  : (N x 3) GT positions array
            title         : Plot title
            filename      : Output filename
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise VisualizerError("matplotlib not installed. Run: pip install matplotlib")

        if len(est_positions) == 0 or len(gt_positions) == 0:
            logger.warning("[Visualizer] Empty positions array, skipping plot.")
            return

        N = min(len(est_positions), len(gt_positions))
        est = est_positions[:N]
        gt = gt_positions[:N]

        # Normalize to origin
        est_norm = est - est[0]
        gt_norm = gt - gt[0]

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # --- Left: Raw trajectory ---
        ax = axes[0]
        ax.plot(est_norm[:, 0], est_norm[:, 1], 'b-', linewidth=1.5, label='Estimated')
        ax.plot(gt_norm[:, 0], gt_norm[:, 1], 'r-', linewidth=1.5, label='Ground Truth')
        ax.scatter(est_norm[0, 0], est_norm[0, 1], c='green', s=100, zorder=5, label='Start')
        ax.scatter(est_norm[-1, 0], est_norm[-1, 1], c='blue', s=100, marker='*', zorder=5, label='Est End')
        ax.scatter(gt_norm[-1, 0], gt_norm[-1, 1], c='red', s=100, marker='*', zorder=5, label='GT End')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'{title} — XY Plane')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        # --- Right: ATE per frame ---
        ax2 = axes[1]
        errors = np.linalg.norm(est_norm[:, :2] - gt_norm[:, :2], axis=1)
        ax2.plot(errors, 'g-', linewidth=1.0, label='Position Error')
        ax2.axhline(y=np.mean(errors), color='r', linestyle='--', label=f'Mean: {np.mean(errors):.2f}m')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Error (m)')
        ax2.set_title('Per-Frame Position Error')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self._output_dir / filename
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
        plt.close()

        logger.info("[Visualizer] Saved: %s", output_path)
        print(f"  Plot saved: {output_path}")

    def plot_xyz_over_time(
        self,
        est_positions: np.ndarray,
        gt_positions: np.ndarray,
        filename: str = "xyz_over_time.png",
    ) -> None:
        """
        Plots X, Y, Z components over time separately.
        Useful for debugging which axis has drift.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise VisualizerError("matplotlib not installed.")

        if len(est_positions) == 0:
            return

        N = min(len(est_positions), len(gt_positions))
        est = est_positions[:N] - est_positions[0]
        gt = gt_positions[:N] - gt_positions[0]

        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        labels = ['X', 'Y', 'Z']
        colors_est = ['blue', 'green', 'red']

        for i, (label, color) in enumerate(zip(labels, colors_est)):
            ax = axes[i]
            ax.plot(est[:, i], color=color, linewidth=1.5, label=f'Est {label}')
            if i < 2:
                ax.plot(gt[:, i], color='black', linewidth=1.5, linestyle='--', label=f'GT {label}')
            ax.set_ylabel(f'{label} (m)')
            ax.legend()
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Frame')
        plt.suptitle('X, Y, Z Components Over Time')
        plt.tight_layout()

        output_path = self._output_dir / filename
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
        plt.close()

        logger.info("[Visualizer] Saved: %s", output_path)
        print(f"  Plot saved: {output_path}")

    def print_statistics(
        self,
        est_positions: np.ndarray,
        gt_positions: np.ndarray,
    ) -> None:
        """
        Prints trajectory comparison statistics to console.
        """
        if len(est_positions) == 0 or len(gt_positions) == 0:
            print("No valid positions to compare.")
            return

        N = min(len(est_positions), len(gt_positions))
        est = est_positions[:N] - est_positions[0]
        gt = gt_positions[:N] - gt_positions[0]

        errors = np.linalg.norm(est[:, :2] - gt[:, :2], axis=1)
        ate = float(np.sqrt(np.mean(errors ** 2)))

        print(f"\n{'='*50}")
        print(f"  TRAJECTORY STATISTICS ({N} frames)")
        print(f"{'='*50}")
        print(f"  ATE (RMSE)     : {ate:.4f} m")
        print(f"  Mean error     : {np.mean(errors):.4f} m")
        print(f"  Max error      : {np.max(errors):.4f} m")
        print(f"  Min error      : {np.min(errors):.4f} m")
        print(f"{'='*50}")
        print(f"  Est final pos  : X={est[-1,0]:.3f}, Y={est[-1,1]:.3f}")
        print(f"  GT  final pos  : X={gt[-1,0]:.3f},  Y={gt[-1,1]:.3f}")
        print(f"  X error        : {abs(est[-1,0] - gt[-1,0]):.3f} m")
        print(f"  Y error        : {abs(est[-1,1] - gt[-1,1]):.3f} m")
        print(f"{'='*50}")

    def __repr__(self) -> str:
        return f"Visualizer(output_dir={self._output_dir})"


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
        from core.pose_graph import PoseGraph

        loader = DataLoader(str(config_path))
        cam = CameraCalibration(loader)
        extractor = FeatureExtractor(loader, cam)
        matcher = Matcher(loader)
        estimator = MotionEstimator(loader, cam)
        scale_recovery = ScaleRecovery(loader, cam)
        pose_graph = PoseGraph(loader)
        visualizer = Visualizer(output_dir="data/viz")

        print("\n--- Full Pipeline Test (first 300 frames) ---")

        prev_features = None

        for idx, name, frame in loader.frame_generator():
            curr_features = extractor.extract(frame, name)

            if prev_features is not None:
                match_result = matcher.match(prev_features, curr_features)
                pose = estimator.estimate(match_result)
                scale_result = scale_recovery.recover(pose, name)
                pose_graph.update(pose, scale_result, name)

            prev_features = curr_features

            if idx >= 900:
                break

        est_positions = pose_graph.get_positions_array()
        gt_positions = pose_graph.get_gt_array(loader)

        visualizer.print_statistics(est_positions, gt_positions)

        visualizer.plot_trajectory_2d(
            est_positions, gt_positions,
            title="Drone SLAM",
            filename="trajectory_2d.png",
        )

        visualizer.plot_xyz_over_time(
            est_positions, gt_positions,
            filename="xyz_over_time.png",
        )

        pose_graph.save_trajectory("data/trajectory_output.csv")

    except Exception as e:
        logger.error("Error: %s", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)