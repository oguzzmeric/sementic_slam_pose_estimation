"""
B secenegi -- ham optik akistan olcek, GUNCEL pipeline (DLT fix + retval
gate) uzerinde tekrar test.

Uretim trajectory'sini (mevcut k*Z) DEGISTIRMEDEN, paralel ikinci bir
PoseGraph ile ayni R, ayni gecerlilik (is_valid) fakat farkli olcek
(akis tabanli) kullanarak "olsaydi ne olurdu" trajectory'sini kurar --
tek degisken (olcek kaynagi) disinda her sey ozdes, adil kiyas.

    s_akis = medyan(||pts_curr - pts_prev||) * Z / f
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
from core.scale_recovery import ScaleRecovery, ScaleResult
from core.pose_graph import PoseGraph

loader = DataLoader("config.yaml")
cam = CameraCalibration(loader)
extractor = FeatureExtractor(loader, cam)
matcher = Matcher(loader)
estimator = MotionEstimator(loader, cam)
scale_rec = ScaleRecovery(loader, cam)

pose_graph_prod = PoseGraph(loader)   # uretim: s = k*Z
pose_graph_flow = PoseGraph(loader)   # aday: s = akis*Z/f

f = float(np.sqrt(cam.fx * cam.fy))

gt = {}
with open("data/ground-truth.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        z = r.get("translation_z", "") or 0.0
        gt[r["frame_numbers"].strip()] = np.array(
            [float(r["translation_x"]), float(r["translation_y"]), float(z)])

Z_OFFSET = 41.6  # config: gt_z_offset

rows = []
prev_features = None

for idx, name, frame in loader.frame_generator():
    curr = extractor.extract(frame, name)
    if prev_features is not None:
        mr = matcher.match(prev_features, curr)
        pose = estimator.estimate(mr)
        sr_prod = scale_rec.recover(pose, name, match_result=mr)

        # ayni gecerlilik, farkli olcek buyuklugu
        sr_flow = sr_prod
        if sr_prod.is_valid and pose.is_valid and pose.t is not None:
            g = loader.get_ground_truth(name)
            if g is not None:
                Z = Z_OFFSET - float(g.tz)
                flow_px = float(np.median(np.linalg.norm(
                    mr.pts_curr - mr.pts_prev, axis=1)))
                s_flow = flow_px * Z / f
                sr_flow = ScaleResult(
                    scale=s_flow, t_scaled=s_flow * pose.t, mode=sr_prod.mode,
                    is_valid=True, frame_name=name,
                    reference_class=sr_prod.reference_class,
                    estimated_depth=Z, depth_source="gt",
                )

        tp_prod = pose_graph_prod.update(pose, sr_prod, name)
        tp_flow = pose_graph_flow.update(pose, sr_flow, name)

        stem = name.rsplit(".", 1)[0]
        if stem in gt:
            rows.append((
                stem, sr_prod.is_valid, sr_prod.scale,
                sr_flow.scale if sr_prod.is_valid else 0.0,
                tp_prod.position.copy(), tp_flow.position.copy(),
            ))
    prev_features = curr

names = [r[0] for r in rows]
valid = np.array([r[1] for r in rows])
s_prod = np.array([r[2] for r in rows])
s_flow = np.array([r[3] for r in rows])
est_prod = np.array([r[4] for r in rows])
est_flow = np.array([r[5] for r in rows])
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
    err_raw = np.linalg.norm(gt_arr - est, axis=1)  # hizalanmamis -- yarisma metrigi
    d_est = np.diff(est, axis=0)
    step_est = np.linalg.norm(d_est, axis=1)
    d_gt = np.diff(gt_arr, axis=0)
    step_gt = np.linalg.norm(d_gt, axis=1)
    total_est = step_est.sum()
    total_gt = step_gt.sum()
    print(f"--- {label} ---")
    print(f"  Sim(3) ATE (rmse)      : {np.sqrt((err_al**2).mean()):7.2f} m   olcek={s:.4f}")
    print(f"  Sim(3)-hizali mean     : {err_al.mean():7.2f} m")
    print(f"  HIZALANMAMIS mean (yarisma metrigi) : {err_raw.mean():7.2f} m")
    print(f"  toplam yol             : tahmin={total_est:7.1f} m  GT={total_gt:7.1f} m  oran={total_gt/max(total_est,1e-9):.3f}")
    print()


print("=" * 66)
print(f"  n={len(rows)} kare  ({int(valid.sum())} gecerli)")
print("=" * 66)
print()
report("URETIM  (s = k * Z)", est_prod)
report("AKIS    (s = akis_px * Z / f)", est_flow)

# olcek buyuklugu kiyasi (yalniz gecerli kareler, GT adimla)
sv = valid
sp = s_prod[sv]; sf = s_flow[sv]
d_gt_v = np.linalg.norm(np.diff(gt_arr, axis=0), axis=1)
# indeks kaymasi: adim i, kare i+1'in olcegiyle iliskili (recover() o
# karede hesaplaniyor) -- kabaca hizalamak icin s[1:] ile d_gt_v[:-1]...
# basitlik icin dogrudan ayni indeksle karsilastiriyoruz (adim uzunlugu
# ~ o karenin olcegi ile orantili).
n = min(len(sp) - 1, len(d_gt_v))
r_prod = d_gt_v[:n] / np.maximum(sp[1:n+1], 1e-9)
r_flow = d_gt_v[:n] / np.maximum(sf[1:n+1], 1e-9)
print("ADIM OLCEGI ORANI (GT_adim / tahmini_olcek) -- 1.0 ideal")
print(f"  uretim (k*Z)  : medyan={np.median(r_prod):.3f}  std={np.std(r_prod):.3f}")
print(f"  akis          : medyan={np.median(r_flow):.3f}  std={np.std(r_flow):.3f}")
print(f"  korelasyon(GT_adim, s_prod) = {np.corrcoef(d_gt_v[:n], sp[1:n+1])[0,1]:+.4f}")
print(f"  korelasyon(GT_adim, s_flow) = {np.corrcoef(d_gt_v[:n], sf[1:n+1])[0,1]:+.4f}")
