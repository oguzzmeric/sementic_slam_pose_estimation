"""
main.py
========
Entry point for the Drone Semantic SLAM pipeline.
Runs the full pipeline on all frames in raw_frames/.

Usage:
    python main.py
    python main.py --config config.yaml
    python main.py --max-frames 300
    python main.py --no-viz
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Setup logging before imports
def setup_logging(level: str = "INFO", log_to_file: bool = False, log_file: str = "logs/slam.log") -> None:
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_to_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drone Semantic SLAM Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process (None = all)")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    parser.add_argument("--output-dir", type=str, default="data/viz", help="Visualization output directory")
    return parser.parse_args()


def run_pipeline(
    config_path: str,
    max_frames: int = None,
    visualize: bool = True,
    output_dir: str = "data/viz",
) -> dict:
    """
    Runs the full SLAM pipeline.

    Args:
        config_path : Path to config.yaml
        max_frames  : Maximum frames to process (None = all)
        visualize   : Whether to produce plots
        output_dir  : Directory for visualization outputs

    Returns:
        dict with pipeline results (ate, total_frames, valid_frames, runtime)
    """
    from utils.data_loader import DataLoader, DataLoaderError
    from utils.camera_calibration import CameraCalibration, CameraCalibrationError
    from core.feature_extractor import FeatureExtractor, FeatureExtractorError
    from core.matcher import Matcher, MatcherError
    from core.motion_estimator import MotionEstimator, MotionEstimatorError
    from core.scale_recovery import ScaleRecovery, ScaleRecoveryError
    from core.pose_graph import PoseGraph, PoseGraphError
    from utils.visualizer import Visualizer, VisualizerError

    # ------------------------------------------------------------------
    # Initialize modules
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("DRONE SEMANTIC SLAM PIPELINE")
    logger.info("=" * 60)

    try:
        loader       = DataLoader(config_path)
        cam          = CameraCalibration(loader)
        extractor    = FeatureExtractor(loader, cam)
        matcher      = Matcher(loader)
        estimator    = MotionEstimator(loader, cam)
        scale_rec    = ScaleRecovery(loader, cam)
        pose_graph   = PoseGraph(loader)
        visualizer   = Visualizer(output_dir=output_dir) if visualize else None

    except Exception as e:
        logger.error("Initialization failed: %s", e)
        raise

    total_frames = loader.total_frames
    process_limit = max_frames if max_frames is not None else total_frames
    logger.info("Total frames: %d | Processing: %d", total_frames, process_limit)

    # ------------------------------------------------------------------
    # Pipeline stats
    # ------------------------------------------------------------------
    stats = {
        "total"          : 0,
        "valid_pose"     : 0,
        "invalid_pose"   : 0,
        "h_selected"     : 0,
        "e_selected"     : 0,
        "warmup"         : 0,
        "autonomous"     : 0,
        "fallback"       : 0,
        "low_inlier"     : 0,
    }

    # ------------------------------------------------------------------
    # Main frame loop
    # ------------------------------------------------------------------
    prev_features = None
    start_time = time.time()
    last_log_time = start_time

    logger.info("Starting frame processing...")

    for idx, frame_name, frame in loader.frame_generator():

        # Progress log every 30 seconds
        now = time.time()
        if now - last_log_time > 30:
            elapsed = now - start_time
            fps = idx / elapsed if elapsed > 0 else 0
            eta = (process_limit - idx) / fps if fps > 0 else 0
            logger.info(
                "Progress: %d/%d frames (%.1f fps, ETA: %.0fs)",
                idx, process_limit, fps, eta,
            )
            last_log_time = now

        # Feature extraction
        try:
            curr_features = extractor.extract(frame, frame_name)
        except FeatureExtractorError as e:
            logger.warning("Feature extraction failed at %s: %s", frame_name, e)
            prev_features = None
            continue

        stats["total"] += 1

        if prev_features is not None:
            # Matching
            try:
                match_result = matcher.match(prev_features, curr_features)
            except MatcherError as e:
                logger.warning("Matching failed at %s: %s", frame_name, e)
                prev_features = curr_features
                continue

            # Motion estimation
            try:
                pose = estimator.estimate(match_result)
            except MotionEstimatorError as e:
                logger.warning("Motion estimation failed at %s: %s", frame_name, e)
                prev_features = curr_features
                continue

            # Scale recovery
            try:
                scale_result = scale_rec.recover(pose, frame_name)
            except ScaleRecoveryError as e:
                logger.warning("Scale recovery failed at %s: %s", frame_name, e)
                prev_features = curr_features
                continue

            # Pose graph update
            try:
                traj_point = pose_graph.update(pose, scale_result, frame_name)
            except PoseGraphError as e:
                logger.warning("Pose graph update failed at %s: %s", frame_name, e)
                prev_features = curr_features
                continue

            # Stats
            if pose.is_valid:
                stats["valid_pose"] += 1
                from core.motion_estimator import MatrixType
                if pose.matrix_type == MatrixType.HOMOGRAPHY:
                    stats["h_selected"] += 1
                elif pose.matrix_type == MatrixType.ESSENTIAL:
                    stats["e_selected"] += 1
                if pose.inlier_count < loader.get_hybrid_config()["min_inlier_count"]:
                    stats["low_inlier"] += 1
            else:
                stats["invalid_pose"] += 1

            if scale_result.mode == "warmup":
                stats["warmup"] += 1
            elif scale_result.mode == "autonomous":
                stats["autonomous"] += 1
            elif scale_result.mode == "autonomous_fallback":
                stats["fallback"] += 1

        prev_features = curr_features

        if idx + 1 >= process_limit:
            break

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    elapsed = time.time() - start_time
    fps = stats["total"] / elapsed if elapsed > 0 else 0

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info("Runtime       : %.1f seconds (%.1f fps)", elapsed, fps)
    logger.info("Total frames  : %d", stats["total"])
    logger.info("Valid poses   : %d", stats["valid_pose"])
    logger.info("Invalid poses : %d", stats["invalid_pose"])
    logger.info("H selected    : %d", stats["h_selected"])
    logger.info("E selected    : %d", stats["e_selected"])
    logger.info("Warmup scale  : %d", stats["warmup"])
    logger.info("Auto scale    : %d", stats["autonomous"])
    logger.info("Fallback scale: %d", stats["fallback"])
    logger.info("Low inlier    : %d", stats["low_inlier"])

    # ATE
    ate = pose_graph.compute_ate(loader)
    if ate is not None:
        logger.info("ATE (RMSE)    : %.4f m", ate)

    # Trajectory positions
    est_positions = pose_graph.get_positions_array()
    gt_positions  = pose_graph.get_gt_array(loader)

    if len(est_positions) > 0:
        logger.info("Final pos     : X=%.3f, Y=%.3f, Z=%.3f",
                    pose_graph.current_position[0],
                    pose_graph.current_position[1],
                    pose_graph.current_position[2])

    # Save trajectory CSV
    pose_graph.save_trajectory("data/trajectory_output.csv")

    # Visualization
    if visualize and visualizer is not None and len(est_positions) > 0:
        try:
            visualizer.print_statistics(est_positions, gt_positions)
            visualizer.plot_trajectory_2d(
                est_positions, gt_positions,
                title="Drone Semantic SLAM",
                filename="trajectory_2d.png",
            )
            visualizer.plot_xyz_over_time(
                est_positions, gt_positions,
                filename="xyz_over_time.png",
            )
            logger.info("Plots saved to: %s", output_dir)
        except VisualizerError as e:
            logger.warning("Visualization failed: %s", e)

    return {
        "ate"          : ate,
        "total_frames" : stats["total"],
        "valid_frames" : stats["valid_pose"],
        "runtime"      : elapsed,
        "fps"          : fps,
    }


def main() -> None:
    args = parse_args()

    # Load config for logging settings
    import yaml
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    log_cfg = config.get("logging", {})
    setup_logging(
        level=log_cfg.get("level", "INFO"),
        log_to_file=log_cfg.get("log_to_file", False),
        log_file=log_cfg.get("log_file", "logs/slam.log"),
    )

    try:
        results = run_pipeline(
            config_path=str(config_path),
            max_frames=args.max_frames,
            visualize=not args.no_viz,
            output_dir=args.output_dir,
        )

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  Frames processed : {results['total_frames']}")
        print(f"  Valid poses      : {results['valid_frames']}")
        print(f"  Runtime          : {results['runtime']:.1f}s ({results['fps']:.1f} fps)")
        if results['ate'] is not None:
            print(f"  ATE (RMSE)       : {results['ate']:.4f} m")
        print("=" * 60)

    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()