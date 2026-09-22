"""
E yolunda atilan cv2.recoverPose retval'ini (cheirality ile dogrulanan
nokta sayisi) yakalayip yon hatasiyla koreleyen teshis scripti.

core/motion_estimator.py'ye dokunmuyor -- calisma anda cv2.recoverPose'u
sarmalayip (monkeypatch) gercek pipeline'in urettigi degeri okuyor.
"""

import csv
import logging
import numpy as np
import cv2

logging.basicConfig(level=logging.WARNING)

from utils.data_loader import DataLoader
from utils.camera_calibration import CameraCalibration
from core.feature_extractor import FeatureExtractor
from core.matcher import Matcher
from core.motion_estimator import MotionEstimator, MatrixType
from core.scale_recovery import ScaleRecovery
from core.pose_graph import PoseGraph

# --- cv2.recoverPose'u sarmala: retval ve girdi maskesindeki nokta
#     sayisini kaydet, davranisi degistirme ---
_log = []
_orig_recoverPose = cv2.recoverPose


def _wrapped(*args, **kwargs):
    mask_in = kwargs.get("mask")
    n_in = int(np.sum(mask_in != 0)) if mask_in is not None else -1
    ret = _orig_recoverPose(*args, **kwargs)
    _log.append((ret[0], n_in))
    return ret


cv2.recoverPose = _wrapped

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
        n_before = len(_log)
        pose = estimator.estimate(mr)
        retval, n_in = _log[-1] if len(_log) > n_before else (None, None)
        sr = scale_rec.recover(pose, name, match_result=mr)
        tp = pose_graph.update(pose, sr, name)

        stem = name.rsplit(".", 1)[0]
        if pose.is_valid and stem in gt:
            rows.append((
                stem,
                pose.matrix_type.name,
                retval, n_in,
                pose.inlier_count,
                tp.position.copy(),
            ))
    prev_features = curr

cv2.recoverPose = _orig_recoverPose

names = [r[0] for r in rows]
mtype = np.array([r[1] for r in rows])
retval = np.array([r[2] if r[2] is not None else -1 for r in rows], dtype=float)
n_in = np.array([r[3] if r[3] is not None else -1 for r in rows], dtype=float)
inlier_count = np.array([r[4] for r in rows], dtype=float)
est = np.array([r[5] for r in rows])
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
hd_abs = np.abs(hd)

# hd[i] adim i-1 -> i icin. mtype/retval[i] ise kare i icin. i. adimin
# hatasi i. karenin decomposition kararina bagli -> hd[i-1] ile mtype[i]
# hizalanir (adim indeksleri 0..n-2, kare indeksleri 0..n-1; adim i,
# kare i ile kare i+1 arasinda -- karenin KENDI decomposition'i i+1.
# karede hesaplaniyor).
hd_a = hd_abs
mt = mtype[1:]
rv = retval[1:]
nin = n_in[1:]
ic = inlier_count[1:]

n = min(len(hd_a), len(mt))
hd_a = hd_a[:n]; mt = mt[:n]; rv = rv[:n]; nin = nin[:n]; ic = ic[:n]

print("=" * 66)
print(f"  n={n}  H={int((mt=='HOMOGRAPHY').sum())}  E={int((mt=='ESSENTIAL').sum())}")
print("=" * 66)

h_err = hd_a[mt == "HOMOGRAPHY"]
e_err = hd_a[mt == "ESSENTIAL"]
print()
print("H vs E -- YON HATASI")
print(f"  H  : medyan {np.median(h_err):6.2f}  ortalama {np.mean(h_err):6.2f}  n={len(h_err)}")
print(f"  E  : medyan {np.median(e_err):6.2f}  ortalama {np.mean(e_err):6.2f}  n={len(e_err)}")

e_mask = mt == "ESSENTIAL"
rv_e = rv[e_mask]; nin_e = nin[e_mask]; err_e = hd_a[e_mask]
valid_rv = (rv_e >= 0) & (nin_e > 0)
ratio = rv_e[valid_rv] / nin_e[valid_rv]
err_v = err_e[valid_rv]

print()
print("E YOLU -- recoverPose GUVEN ORANI (retval / girdi_inlier) vs YON HATASI")
print(f"  n={len(ratio)}  korelasyon(oran, |yon_hatasi|) = {np.corrcoef(ratio, err_v)[0,1]:+.4f}")
print(f"  oran medyan={np.median(ratio):.3f}  min={ratio.min():.3f}  maks={ratio.max():.3f}")

perfect = err_v[ratio >= 0.999]
imperfect = err_v[ratio < 0.999]
zero = err_v[ratio < 1e-9]
print()
print(f"  tam konsensus (oran=1.0)      |yon_hata| medyan={np.median(perfect):6.2f}  n={len(perfect)}")
print(f"  konsensus < 1.0               |yon_hata| medyan={np.median(imperfect) if len(imperfect) else float('nan'):6.2f}  n={len(imperfect)}")
print(f"  retval=0 (hic destek yok)     |yon_hata| medyan={np.median(zero) if len(zero) else float('nan'):6.2f}  n={len(zero)}")

# dagilim: oran esik gecisleri
for th in (1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.01):
    below = err_v[ratio < th]
    print(f"  oran < {th:<5.2f}  n={len(below):4d}  |yon_hata| medyan={np.median(below) if len(below) else float('nan'):7.2f}")

print()
print("  --- en dusuk guvenli 10 E-karesi ---")
order = np.argsort(ratio)
for i in order[:10]:
    idxs = np.where(e_mask)[0][valid_rv]
    j = idxs[i]
    print(f"  {names[j+1]:20s} oran={ratio[i]:.3f}  retval={rv_e[valid_rv][i]:.0f}/{nin_e[valid_rv][i]:.0f}  yon_hata={err_v[i]:6.2f}")
