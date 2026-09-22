"""
Hipotez C: taban yon hatasi (tam-konsensuslu E karelerinde bile ~27 derece)
rastgele/bagimsiz gurultu mu, yoksa sistematik/kalici bir sapma mi?

Ayirt edici testler:
  1. Isaretli (signed) hata medyani sifirdan farkli mi -- sistematik yanlilik.
  2. Ardisik hatalar arasindaki lag-1 otokorelasyon -- kalicilik/bulasma.
  3. |hata| ile inlier_count / R_H orani korelasyonu -- kalite-bagimliligi.

core/motion_estimator.py'ye dokunmuyor, sadece pipeline'i calistirip
PoseEstimate alanlarini (zaten var olan score_H, score_E, inlier_count)
okuyor.
"""

import csv
import logging
import numpy as np

logging.basicConfig(level=logging.WARNING)

from utils.data_loader import DataLoader
from utils.camera_calibration import CameraCalibration
from core.feature_extractor import FeatureExtractor
from core.matcher import Matcher
from core.motion_estimator import MotionEstimator, MatrixType
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
with open("data/ground-truth.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
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
            rh_ratio = pose.score_H / max(pose.score_H + pose.score_E, 1e-9)
            rows.append((
                stem, pose.matrix_type.name, pose.inlier_count, rh_ratio,
                tp.position.copy(),
            ))
    prev_features = curr

names = [r[0] for r in rows]
mtype = np.array([r[1] for r in rows])
inlier_count = np.array([r[2] for r in rows], dtype=float)
rh_ratio = np.array([r[3] for r in rows], dtype=float)
est = np.array([r[4] for r in rows])
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


s, R, t = umeyama(est, gt_arr)
est_al = (s * R @ est.T + t).T

d_est = np.diff(est_al, axis=0)
d_gt = np.diff(gt_arr, axis=0)
se = np.linalg.norm(d_est, axis=1)
sg = np.linalg.norm(d_gt, axis=1)
m_ok = (se > 1e-6) & (sg > 1e-6)

h_est = np.arctan2(d_est[:, 1], d_est[:, 0])
h_gt = np.arctan2(d_gt[:, 1], d_gt[:, 0])
hd = np.degrees((h_gt - h_est + np.pi) % (2 * np.pi) - np.pi)
hd = hd[m_ok]
ic = inlier_count[1:][m_ok]
rh = rh_ratio[1:][m_ok]
mt = mtype[1:][m_ok]

print("=" * 66)
print(f"  n={len(hd)} adim")
print("=" * 66)

print()
print("1) SISTEMATIK YANLILIK (isaretli hata, sifira yakin olmali)")
print(f"   medyan = {np.median(hd):+7.2f} deg")
print(f"   ortalama = {np.mean(hd):+7.2f} deg")
print(f"   std = {np.std(hd):7.2f} deg")
from scipy import stats as _stats  # noqa
try:
    tstat, pval = _stats.ttest_1samp(hd, 0.0)
    print(f"   t-test (H0: ortalama=0): t={tstat:.2f}  p={pval:.4g}")
except Exception:
    pass

print()
print("2) KALICILIK (ardisik adimlar arasi lag-1 otokorelasyon)")
if len(hd) > 2:
    lag1 = np.corrcoef(hd[:-1], hd[1:])[0, 1]
    print(f"   corr(hd[i], hd[i+1]) = {lag1:+.4f}")
    print("   (>0.3 ~ hatalar birkaç kare boyunca ayni yonde surukleniyor,")
    print("    ~0 ~ bagimsiz/rastgele, <-0.3 ~ salinim/overcorrection)")

print()
print("3) KALITE-BAGIMLILIGI (|hata| vs inlier_count, vs R_H orani)")
print(f"   corr(|hd|, inlier_count) = {np.corrcoef(np.abs(hd), ic)[0,1]:+.4f}")
print(f"   corr(|hd|, R_H orani)    = {np.corrcoef(np.abs(hd), rh)[0,1]:+.4f}")

lo_ic = np.abs(hd)[ic <= np.median(ic)]
hi_ic = np.abs(hd)[ic > np.median(ic)]
print(f"   dusuk inlier_count  |hata| medyan = {np.median(lo_ic):6.2f}  n={len(lo_ic)}")
print(f"   yuksek inlier_count |hata| medyan = {np.median(hi_ic):6.2f}  n={len(hi_ic)}")

print()
print("4) EN UZUN AYNI-YONLU HATA DIZISI (persistans)")
sign = np.sign(hd)
runs = []
cur = 1
for i in range(1, len(sign)):
    if sign[i] == sign[i-1] and sign[i] != 0:
        cur += 1
    else:
        runs.append(cur)
        cur = 1
runs.append(cur)
runs = np.array(runs)
print(f"   ortalama dizi uzunlugu = {runs.mean():.2f}  maksimum = {runs.max()}")
print(f"   (rastgele/bagimsiz isaretler icin beklenen ortalama ~2.0)")
