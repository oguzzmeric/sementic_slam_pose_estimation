"""
Ucuz on-test: BA'ya yatirim yapmadan once, pencereli (3 kareli) rotasyon
yumusatmasi drift'i azaltiyor mu diye bakiyoruz. Eger azaltiyorsa BA'ya
yatirim mantikli; azaltmiyorsa sorun BA'nin cozemeyecegi bir sey demektir.

Yontem: pose.R'yi rotasyon vektorune (Rodrigues) cevirip, GECERLI
karelerin sirali dizisinde 3'lu medyan filtresi uyguluyoruz. Ceviri
(t_scaled) AYNI kaliyor -- tek degisken rotasyon yumusatmasi.

Bu gercek bir BA degil (sadece komsu R'lari ortalıyor, reprojection
hatasini optimize etmiyor) -- amac, coklu-kare tutarliligi zorlamanin
prensip olarak yardimci olup olmadigini ucuza gormek.
"""

import csv
import logging
import cv2
import numpy as np
from dataclasses import replace

logging.basicConfig(level=logging.WARNING)

from utils.data_loader import DataLoader
from utils.camera_calibration import CameraCalibration
from core.feature_extractor import FeatureExtractor
from core.matcher import Matcher
from core.motion_estimator import MotionEstimator
from core.scale_recovery import ScaleRecovery
from core.pose_graph import PoseGraph

loader = DataLoader("config.yaml")
cam = CameraCalibration(loader)
extractor = FeatureExtractor(loader, cam)
matcher = Matcher(loader)
estimator = MotionEstimator(loader, cam)
scale_rec = ScaleRecovery(loader, cam)

gt = {}
with open("data/ground-truth.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        z = r.get("translation_z", "") or 0.0
        gt[r["frame_numbers"].strip()] = np.array(
            [float(r["translation_x"]), float(r["translation_y"]), float(z)])

# --- Pass 1 (pahali): pipeline'i bir kez calistir, sonuclari sakla ---
cache = []  # (frame_name, pose, scale_result)
prev_features = None

for idx, name, frame in loader.frame_generator():
    curr = extractor.extract(frame, name)
    if prev_features is not None:
        mr = matcher.match(prev_features, curr)
        pose = estimator.estimate(mr)
        sr = scale_rec.recover(pose, name, match_result=mr)
        cache.append((name, pose, sr))
    prev_features = curr

# --- Pass 2 (ucuz): rotasyon yumusatma + coklu pencere boyutu taramasi ---
valid_idx = [i for i, (_, p, _) in enumerate(cache) if p.is_valid and p.R is not None]
rvecs = np.array([cv2.Rodrigues(cache[i][1].R)[0].flatten() for i in valid_idx])

gt_arr_cache = None
names_cache = None


def build_trajectory(window):
    smoothed_rvecs = np.copy(rvecs)
    if window > 1:
        half = window // 2
        for j in range(len(rvecs)):
            lo, hi = max(0, j - half), min(len(rvecs), j + half + 1)
            smoothed_rvecs[j] = np.median(rvecs[lo:hi], axis=0)

    smoothed_R = {}
    for k, j in enumerate(valid_idx):
        R_smooth, _ = cv2.Rodrigues(smoothed_rvecs[k])
        smoothed_R[j] = R_smooth

    pg = PoseGraph(loader)
    rows = []
    for i, (name, pose, sr) in enumerate(cache):
        pose_s = pose
        if i in smoothed_R:
            pose_s = replace(pose, R=smoothed_R[i])
        tp = pg.update(pose_s, sr, name)
        stem = name.rsplit(".", 1)[0]
        if stem in gt:
            rows.append((stem, tp.position.copy()))
    return rows


rows_prod = build_trajectory(window=1)
names = [r[0] for r in rows_prod]
est_prod = np.array([r[1] for r in rows_prod])
gt_arr = np.array([gt[n] for n in names])


def umeyama(P, Q):
    P = P.T; Q = Q.T
    n = P.shape[1]
    mu_p = P.mean(axis=1, keepdims=True)
    mu_q = Q.mean(axis=1, keepdims=True)
    Pc = P - mu_p; Qc = Q - mu_q
    W = Qc @ Pc.T / n
    U, D, Vt = np.linalg.svd(W)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_p = (Pc ** 2).sum() / n
    s = np.trace(np.diag(D) @ S) / var_p if var_p > 1e-12 else 1.0
    t = mu_q - s * R @ mu_p
    return s, R, t


def report(label, est):
    s, R, t = umeyama(est, gt_arr)
    est_al = (s * R @ est.T + t).T
    err_al = np.linalg.norm(gt_arr - est_al, axis=1)
    err_raw = np.linalg.norm(gt_arr - est, axis=1)

    d_est = np.diff(est_al, axis=0)
    d_gt = np.diff(gt_arr, axis=0)
    se = np.linalg.norm(d_est, axis=1); sg = np.linalg.norm(d_gt, axis=1)
    m_ok = (se > 1e-6) & (sg > 1e-6)
    h_est = np.arctan2(d_est[:, 1], d_est[:, 0])
    h_gt = np.arctan2(d_gt[:, 1], d_gt[:, 0])
    hd = np.degrees((h_gt - h_est + np.pi) % (2 * np.pi) - np.pi)[m_ok]

    print(f"--- {label} ---")
    print(f"  Sim(3) ATE (rmse)                   : {np.sqrt((err_al**2).mean()):7.2f} m   olcek={s:.4f}")
    print(f"  HIZALANMAMIS mean (yarisma metrigi) : {err_raw.mean():7.2f} m")
    print(f"  yon hatasi |medyan|                 : {np.median(np.abs(hd)):7.2f} deg")
    if len(hd) > 2:
        lag1 = np.corrcoef(hd[:-1], hd[1:])[0, 1]
        print(f"  lag-1 otokorelasyon                 : {lag1:+.4f}")
    print()


print("=" * 66)
print(f"  n={len(rows_prod)} kare  -- pencere buyuklugu taramasi")
print("=" * 66)
report("URETIM     (pencere=1, yumusatmasiz)", est_prod)
for w in (3, 7, 15, 31, 61):
    rows_w = build_trajectory(window=w)
    est_w = np.array([r[1] for r in rows_w])
    report(f"PENCERE={w:<3d}", est_w)
