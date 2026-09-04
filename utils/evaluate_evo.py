"""
utils/evaluate_evo.py
======================
Trajectory evaluation with evo library.

Converts trajectory_output.csv and ground-truth.csv to TUM format,
then computes ATE with Sim(3) alignment (standard practice for monocular).

TUM format: timestamp tx ty tz qx qy qz qw

Why Sim(3) alignment matters:
    Monocular VO has inherent scale ambiguity. Comparing raw trajectories
    penalizes the system for a scale factor it cannot observe.
    Standard practice: align with optimal scale + rotation + translation
    before computing error.

Usage:
    python -m utils.evaluate_evo
    python -m utils.evaluate_evo --no-scale   (rigid alignment only)
"""

import argparse
import csv
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def read_trajectory_csv(path: str) -> List[dict]:
    """
    Reads trajectory_output.csv produced by pose_graph.save_trajectory().

    Expected columns: frame_name, x, y, z, scale, mode, is_valid
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["is_valid"].strip().lower() != "true":
                continue
            rows.append({
                "frame_name": row["frame_name"].strip(),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "z": float(row["z"]),
            })
    return rows


def read_ground_truth_csv(path: str) -> dict:
    """
    Columns: translation_x, translation_y, [translation_z], frame_numbers
    translation_z is optional; missing means 0.
    """
    index = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["frame_numbers"].strip()
            z_raw = row.get("translation_z", "")
            index[name] = (
                float(row["translation_x"]),
                float(row["translation_y"]),
                float(z_raw) if z_raw not in (None, "") else 0.0,
            )
    return index
    


def match_frame_name(name: str, gt_index: dict) -> Tuple[float, float]:
    """
    Ground truth CSV may store names without extension.
    Try both forms.
    """
    if name in gt_index:
        return gt_index[name]
    stem = Path(name).stem
    if stem in gt_index:
        return gt_index[stem]
    return None


def write_tum_file(
    output_path: str,
    positions: List[Tuple[float, float, float]],
) -> None:
    """
    Writes positions in TUM trajectory format.

    Format: timestamp tx ty tz qx qy qz qw

    We only have positions, no orientation. Identity quaternion used
    since ATE only evaluates translation.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for i, (x, y, z) in enumerate(positions):
            # timestamp as float index, identity quaternion
            f.write(f"{i:.6f} {x:.6f} {y:.6f} {z:.6f} 0.0 0.0 0.0 1.0\n")

    logger.info("[evo] Wrote %d poses -> %s", len(positions), path)


def build_tum_pair(
    traj_csv: str,
    gt_csv: str,
    out_dir: str,
) -> Tuple[str, str, int]:
    """
    Builds a matched pair of TUM files from our CSVs.

    Only frames present in both trajectory and ground truth are kept,
    in the same order, so timestamps align 1:1.

    Returns:
        (est_path, gt_path, n_matched)
    """
    traj_rows = read_trajectory_csv(traj_csv)
    gt_index = read_ground_truth_csv(gt_csv)

    est_positions = []
    gt_positions = []
    missing = 0

    for row in traj_rows:
        gt = match_frame_name(row["frame_name"], gt_index)
        if gt is None:
            missing += 1
            continue
        est_positions.append((row["x"], row["y"], row["z"]))
        gt_positions.append((gt[0], gt[1], gt[2]))

    if missing:
        logger.warning("[evo] %d frames had no ground truth, skipped.", missing)

    if len(est_positions) < 2:
        raise RuntimeError("Not enough matched poses to evaluate.")

    out = Path(out_dir)
    est_path = out / "est_tum.txt"
    gt_path  = out / "gt_tum.txt"

    write_tum_file(str(est_path), est_positions)
    write_tum_file(str(gt_path), gt_positions)

    return str(est_path), str(gt_path), len(est_positions)


def run_evo_ape(
    gt_path: str,
    est_path: str,
    align_scale: bool,
    plot_path: str = None,
) -> None:
    """
    Runs evo_ape (Absolute Pose Error) as a subprocess.

    Args:
        align_scale: if True uses Sim(3) alignment (-as),
                     else rigid Umeyama alignment (-a)
    """
    cmd = [
        "evo_ape", "tum",
        gt_path, est_path,
        "-as" if align_scale else "-a",
        "--no_warnings",
    ]

    if plot_path:
        cmd += ["--save_plot", plot_path]

    label = "Sim(3) — scale aligned" if align_scale else "SE(3) — rigid aligned"
    print("\n" + "=" * 62)
    print(f"  evo_ape :: {label}")
    print("=" * 62)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        print(result.stdout)
        if result.returncode != 0:
            print("--- stderr ---")
            print(result.stderr)
    except FileNotFoundError:
        print("evo_ape not found. Install with: pip install evo")
        sys.exit(1)


def manual_ate(
    est_path: str,
    gt_path: str,
    align_scale: bool,
) -> float:
    """
    Manual ATE computation with Umeyama alignment.
    Independent cross-check against evo output.

    Umeyama alignment finds optimal s, R, t minimizing:
        sum ||gt_i - (s * R * est_i + t)||^2
    """
    est = np.loadtxt(est_path)[:, 1:4]
    gt  = np.loadtxt(gt_path)[:, 1:4]

    N = min(len(est), len(gt))
    est = est[:N].T   # 3 x N
    gt  = gt[:N].T

    # Centroids
    mu_est = est.mean(axis=1, keepdims=True)
    mu_gt  = gt.mean(axis=1, keepdims=True)

    est_c = est - mu_est
    gt_c  = gt  - mu_gt

    # Cross-covariance
    W = gt_c @ est_c.T / N
    U, D, Vt = np.linalg.svd(W)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt

    if align_scale:
        var_est = (est_c ** 2).sum() / N
        s = np.trace(np.diag(D) @ S) / var_est if var_est > 1e-12 else 1.0
    else:
        s = 1.0

    t = mu_gt - s * R @ mu_est

    est_aligned = s * R @ est + t
    errors = np.linalg.norm(gt - est_aligned, axis=0)
    ate = float(np.sqrt(np.mean(errors ** 2)))

    return ate, s


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trajectory with evo",
        conflict_handler="resolve",
    )
    parser.add_argument("--traj", default="data/trajectory_output.csv")
    parser.add_argument("--gt",   default="data/ground-truth.csv")
    parser.add_argument("--out",  default="data/evo")
    parser.add_argument("--plot", action="store_true", help="Save evo plots")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not Path(args.traj).exists():
        print(f"Trajectory file not found: {args.traj}")
        print("Run 'python main.py' first to generate it.")
        sys.exit(1)

    est_path, gt_path, n = build_tum_pair(args.traj, args.gt, args.out)

    print("\n" + "=" * 62)
    print(f"  MATCHED POSES: {n}")
    print("=" * 62)

    # Manual cross-check first
    ate_rigid, s_rigid = manual_ate(est_path, gt_path, align_scale=False)
    ate_sim3,  s_sim3  = manual_ate(est_path, gt_path, align_scale=True)

    print("\n  Manual Umeyama cross-check")
    print("  " + "-" * 44)
    print(f"  SE(3)  rigid alignment    ATE = {ate_rigid:8.3f} m")
    print(f"  Sim(3) scale alignment    ATE = {ate_sim3:8.3f} m")
    print(f"  Recovered scale factor    s   = {s_sim3:8.4f}")
    print("  " + "-" * 44)

    if s_sim3 > 1.05:
        print(f"  -> Our trajectory is ~{s_sim3:.2f}x too SMALL")
    elif s_sim3 < 0.95:
        print(f"  -> Our trajectory is ~{1/s_sim3:.2f}x too LARGE")
    else:
        print("  -> Scale is close to correct")

    # evo runs
    plot_rigid = f"{args.out}/ape_rigid.pdf" if args.plot else None
    plot_sim3  = f"{args.out}/ape_sim3.pdf"  if args.plot else None

    run_evo_ape(gt_path, est_path, align_scale=False, plot_path=plot_rigid)
    run_evo_ape(gt_path, est_path, align_scale=True,  plot_path=plot_sim3)


if __name__ == "__main__":
    main()
    