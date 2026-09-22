"""
R_H esigi (0.45) -- mekanizma arastirmasi.

Not: 0.45 esigi ORB-SLAM'in H/F secim sezigisinden geliyor (bu datasete
ozel tune edilmemis, literatur sabiti). Soru: esige yakin kararlar neden
guvenilmez -- sahne gercekten belirsiz mi (az nokta, dusuk skor farki),
yoksa bu datasete ozel bir tesadufmu mu?

GT sadece DOGRULAMA icin kullaniliyor (yon hatasini olcmek) -- R_H'nin
kendisi uretimde GT'siz hesaplaniyor, bu script de sadece mevcut sinyalleri
(match_count, score_H, score_E) GT-bagimsiz olarak inceliyor.
"""

import csv
import logging
import numpy as np

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
pose_graph = PoseGraph(loader)

gt = {}
with open("data/ground-truth.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        z = r.get("translation_z", "") or 0.0
        gt[r["frame_numbers"].strip()] = np.array(
            [float(r["translation_x"]), float(r["translation_y"]), float(z)])

rows = []
prev_features = None

for idx, name, frame in loader.frame_generator():
    curr = extractor.extract(frame, name)
    if prev_features is not None:
        mr = matcher.match(prev_features, curr)
        pose = estimator.estimate(mr)
        sr = scale_rec.recover(pose, name, match_result=mr)
        tp = pose_graph.update(pose, sr, name)

        stem = name.rsplit(".", 1)[0]
        if pose.is_valid and stem in gt:
            total = pose.score_H + pose.score_E
            rh = pose.score_H / total if total > 1e-9 else 0.5
            rows.append((
                stem, rh, pose.score_H, pose.score_E,
                mr.match_count, pose.inlier_count,
                pose.matrix_type.name, tp.position.copy(),
            ))
    prev_features = curr

names = [r[0] for r in rows]
rh = np.array([r[1] for r in rows])
score_h = np.array([r[2] for r in rows])
score_e = np.array([r[3] for r in rows])
match_count = np.array([r[4] for r in rows], dtype=float)
inlier_count = np.array([r[5] for r in rows], dtype=float)
mtype = np.array([r[6] for r in rows])
est = np.array([r[7] for r in rows])
gt_arr = np.array([gt[n] for n in names])

THRESH = 0.45
dist_to_th = np.abs(rh - THRESH)


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


s, R, t = umeyama(est, gt_arr)
est_al = (s * R @ est.T + t).T
d_est = np.diff(est_al, axis=0)
d_gt = np.diff(gt_arr, axis=0)
se = np.linalg.norm(d_est, axis=1); sg = np.linalg.norm(d_gt, axis=1)
m_ok = (se > 1e-6) & (sg > 1e-6)
h_est = np.arctan2(d_est[:, 1], d_est[:, 0])
h_gt = np.arctan2(d_gt[:, 1], d_gt[:, 0])
hd = np.degrees((h_gt - h_est + np.pi) % (2 * np.pi) - np.pi)

n = min(len(hd), len(dist_to_th) - 1)
hd = hd[:n]
dist_v = dist_to_th[1:n+1]
mc_v = match_count[1:n+1]
ic_v = inlier_count[1:n+1]
rh_v = rh[1:n+1]
mt_v = mtype[1:n+1]
m_ok = m_ok[:n]
hd_abs = np.abs(hd)[m_ok]
dist_v = dist_v[m_ok]; mc_v = mc_v[m_ok]; ic_v = ic_v[m_ok]; rh_v = rh_v[m_ok]; mt_v = mt_v[m_ok]

print("=" * 66)
print(f"  n={len(hd_abs)}")
print("=" * 66)

print()
print("1) ESIGE UZAKLIK ILE YON HATASI")
print(f"   korelasyon(|R_H-0.45|, |yon_hatasi|) = {np.corrcoef(dist_v, hd_abs)[0,1]:+.4f}")
near = hd_abs[dist_v < 0.10]
far = hd_abs[dist_v >= 0.10]
print(f"   esige yakin  (|R_H-0.45|<0.10)  n={len(near):4d}  |yon_hata| medyan={np.median(near):6.2f}")
print(f"   esige uzak   (|R_H-0.45|>=0.10) n={len(far):4d}  |yon_hata| medyan={np.median(far):6.2f}")

print()
print("2) ESIGE YAKINKEN NOKTA SAYISI/GUVEN DUSUK MU? (mekanizma)")
print(f"   korelasyon(|R_H-0.45|, match_count)   = {np.corrcoef(dist_v, mc_v)[0,1]:+.4f}")
print(f"   korelasyon(|R_H-0.45|, inlier_count)  = {np.corrcoef(dist_v, ic_v)[0,1]:+.4f}")
print(f"   esige yakin  match_count medyan = {np.median(mc_v[dist_v<0.10]):7.1f}")
print(f"   esige uzak   match_count medyan = {np.median(mc_v[dist_v>=0.10]):7.1f}")
print(f"   esige yakin  inlier_count medyan = {np.median(ic_v[dist_v<0.10]):7.1f}")
print(f"   esige uzak   inlier_count medyan = {np.median(ic_v[dist_v>=0.10]):7.1f}")

print()
print("3) DILIM PROFILI (esige uzaklik siralamasi)")
order = np.argsort(dist_v)
d_s = dist_v[order]; hd_s = hd_abs[order]; mc_s = mc_v[order]
step = max(1, len(d_s)//10)
print(f"{'dilim':>6} {'|RH-.45|':>9} {'yon_hata':>9} {'match_cnt':>10}")
for i in range(0, len(d_s), step):
    sl = slice(i, min(i+step, len(d_s)))
    print(f"{i:6d} {np.median(d_s[sl]):9.3f} {np.median(hd_s[sl]):9.2f} {np.median(mc_s[sl]):10.1f}")

print()
print("4) H vs E -- match_count farkli mi (H genelde az noktada mi tetikleniyor)")
print(f"   H  match_count medyan = {np.median(mc_v[mt_v=='HOMOGRAPHY']):.1f}  n={int((mt_v=='HOMOGRAPHY').sum())}")
print(f"   E  match_count medyan = {np.median(mc_v[mt_v=='ESSENTIAL']):.1f}  n={int((mt_v=='ESSENTIAL').sum())}")
