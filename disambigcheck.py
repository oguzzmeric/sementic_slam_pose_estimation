"""
Essential matrix disambiguation -- oylama gercekten belirsiz oldugunda
(kazanan aday runner-up'a cok yakin oy aliyorsa), runner-up'i secmis
olsaydik yon hatasi daha mi iyi olurdu?

Uretim kodunu (core/motion_estimator.py) DEGISTIRMIYORUZ -- sadece
_decompose_essential'i monkeypatch'leyip, ayni E/mask'i yakalayip
KENDI 4-aday oylamamizi paralelde yapiyoruz. Uretimin kendi karari
(pose.R, pose.t) hic etkilenmiyor.

Yon hatasi tanimi hecheck.py/biascheck.py ile ayni: tahmini adim
yonu (dunya cercevesinde) ile GT adim yonu arasindaki aci farki.
"""

import csv
import numpy as np
import cv2

from utils.data_loader import DataLoader
from utils.camera_calibration import CameraCalibration
from core.feature_extractor import FeatureExtractor
from core.matcher import Matcher
from core.motion_estimator import MotionEstimator, MatrixType

CHEIRALITY_SAMPLE = 50
REPROJ_THRESHOLD = 4.0
AMBIGUOUS_MARGIN = 0.30  # (oy1-oy2)/oy1 < bu deger -> belirsiz say

loader = DataLoader("config.yaml")
cam = CameraCalibration(loader)
extractor = FeatureExtractor(loader, cam)
matcher = Matcher(loader)
estimator = MotionEstimator(loader, cam)

K = cam.K
K_inv = cam.K_inv
fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy

captured = []  # her _decompose_essential cagrisinda bir giris


def capturing_decompose(E, pts_prev, pts_curr, e_mask, e_inlier_mask):
    captured.append((E.copy(), pts_prev.copy(), pts_curr.copy(), e_inlier_mask.copy()))
    return orig_decompose(E, pts_prev, pts_curr, e_mask, e_inlier_mask)


orig_decompose = estimator._decompose_essential
estimator._decompose_essential = capturing_decompose


def triangulate_and_vote(R, t, pts_prev_in, pts_curr_in):
    """motion_estimator._decompose_homography ile ayni mantik (cheirality+reprojection)."""
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([R, t.reshape(3, 1)])
    n = min(CHEIRALITY_SAMPLE, len(pts_prev_in))
    reproj_th_sq = REPROJ_THRESHOLD ** 2
    votes = 0
    for j in range(n):
        u1, v1 = pts_prev_in[j]
        u2, v2 = pts_curr_in[j]
        p1 = K_inv @ np.array([u1, v1, 1.0])
        p2 = K_inv @ np.array([u2, v2, 1.0])
        A = np.array([
            p1[0] * P1[2] - P1[0], p1[1] * P1[2] - P1[1],
            p2[0] * P2[2] - P2[0], p2[1] * P2[2] - P2[1],
        ])
        _, _, Vt = np.linalg.svd(A)
        Xh = Vt[-1]
        if abs(Xh[3]) < 1e-9:
            continue
        X = Xh[:3] / Xh[3]
        Z1 = X[2]
        X_cam2 = R @ X + t.flatten()
        Z2 = X_cam2[2]
        if Z1 <= 0 or Z2 <= 0:
            continue
        if abs(Z1) < 1e-9 or abs(Z2) < 1e-9:
            continue
        u1_hat, v1_hat = fx * X[0] / Z1 + cx, fy * X[1] / Z1 + cy
        if (u1_hat - u1) ** 2 + (v1_hat - v1) ** 2 > reproj_th_sq:
            continue
        u2_hat, v2_hat = fx * X_cam2[0] / Z2 + cx, fy * X_cam2[1] / Z2 + cy
        if (u2_hat - u2) ** 2 + (v2_hat - v2) ** 2 > reproj_th_sq:
            continue
        votes += 1
    return votes, n


gt = {}
with open("data/ground-truth.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        z = r.get("translation_z", "") or 0.0
        gt[r["frame_numbers"].strip()] = np.array(
            [float(r["translation_x"]), float(r["translation_y"]), float(z)])


def heading(vec2):
    return np.degrees(np.arctan2(vec2[1], vec2[0]))


def angle_diff(a, b):
    d = (a - b + 180) % 360 - 180
    return abs(d)


records = []  # dict per E-selected step
R_world = np.eye(3)
prev_features, prev_name = None, None

for idx, name, frame in loader.frame_generator():
    curr = extractor.extract(frame, name)
    if prev_features is not None:
        n_before = len(captured)
        mr = matcher.match(prev_features, curr)
        pose = estimator.estimate(mr)

        if len(captured) > n_before and pose.is_valid:
            E, pts_prev, pts_curr, e_inlier_mask = captured[-1]
            m = e_inlier_mask.astype(bool)
            pts_prev_in, pts_curr_in = pts_prev[m], pts_curr[m]

            R1, R2, t_raw = cv2.decomposeEssentialMat(E)
            candidates = [(R1, t_raw), (R1, -t_raw), (R2, t_raw), (R2, -t_raw)]
            scored = []
            for R_c, t_c in candidates:
                votes, n_test = triangulate_and_vote(R_c, t_c, pts_prev_in, pts_curr_in)
                t_unit = t_c.flatten() / max(np.linalg.norm(t_c), 1e-12)
                scored.append((votes, R_c, t_unit))
            scored.sort(key=lambda s: -s[0])

            gt_curr, gt_prev = gt.get(name.rsplit(".", 1)[0]), gt.get(prev_name.rsplit(".", 1)[0])
            if gt_curr is not None and gt_prev is not None:
                gt_dir = (gt_curr[:2] - gt_prev[:2])
                if np.linalg.norm(gt_dir) > 1e-6:
                    h_gt = heading(gt_dir)
                    votes0, R0c, t0u = scored[0]
                    votes1, R1c, t1u = scored[1]
                    world_dir0 = (R_world @ t0u)[:2]
                    world_dir1 = (R_world @ t1u)[:2]
                    err0 = angle_diff(heading(world_dir0), h_gt)
                    err1 = angle_diff(heading(world_dir1), h_gt)
                    margin = (votes0 - votes1) / max(votes0, 1)
                    records.append(dict(
                        name=name, margin=margin, votes0=votes0, votes1=votes1,
                        err_winner=err0, err_runnerup=err1,
                        prod_matches_winner=np.allclose(pose.t.flatten(), t0u, atol=1e-6),
                    ))

        R_world = R_world @ pose.R if pose.is_valid else R_world

    prev_features, prev_name = curr, name

n = len(records)
print(f"E secilen ve GT'li adim sayisi = {n}")

margins = np.array([r["margin"] for r in records])
err_w = np.array([r["err_winner"] for r in records])
err_r = np.array([r["err_runnerup"] for r in records])
match = np.array([r["prod_matches_winner"] for r in records])

print(f"uretimin secimi bizim #1 adayimizla eslesiyor mu: {match.sum()}/{n}")
print()
print(f"TUM adimlar -- kazanan yon hatasi medyan={np.median(err_w):.2f}  "
      f"runner-up yon hatasi medyan={np.median(err_r):.2f}")
print(f"korelasyon(margin, kazanan_hata) = {np.corrcoef(margins, err_w)[0,1]:+.4f}")
print()

ambiguous = margins < AMBIGUOUS_MARGIN
n_amb = ambiguous.sum()
print(f"BELIRSIZ adimlar (margin<{AMBIGUOUS_MARGIN}): n={n_amb} ({100*n_amb/max(n,1):.1f}%)")
if n_amb > 0:
    w = err_w[ambiguous]
    r_ = err_r[ambiguous]
    print(f"  kazanan yon hatasi   medyan={np.median(w):.2f}  ortalama={np.mean(w):.2f}")
    print(f"  runner-up yon hatasi medyan={np.median(r_):.2f}  ortalama={np.mean(r_):.2f}")
    better = (r_ < w).sum()
    print(f"  runner-up KAZANANDAN daha iyi cikan adim sayisi: {better}/{n_amb} "
          f"({100*better/n_amb:.1f}%)")

not_ambiguous = ~ambiguous
if not_ambiguous.sum() > 0:
    w2 = err_w[not_ambiguous]
    print()
    print(f"NET (belirsiz olmayan) adimlar: n={not_ambiguous.sum()}  "
          f"kazanan yon hatasi medyan={np.median(w2):.2f}")
