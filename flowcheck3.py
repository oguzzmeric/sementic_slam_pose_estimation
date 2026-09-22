"""
B (optik akis) + warmup-kalibreli duzeltme carpani.

    s = c * akis_px * Z / f,   c = warmup'ta medyan(s_ref / (akis_px*Z/f))

Warmup FAZINDA uc varyant da URETIMIN kendi olcegini kullanir (adil kiyas
-- gercek konuslanmada warmup zaten referans sinyali kullaniyor, o kisim
tartismali degil). Sadece AUTONOMOUS fazda ayrisiyorlar:

    uretim   : s = k * Z
    akis-ham : s = akis_px * Z / f            (onceki deney, referans)
    akis-kal : s = c * akis_px * Z / f        (bu deney)

core/*.py'ye dokunmuyor.
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

pg_prod = PoseGraph(loader)
pg_raw = PoseGraph(loader)
pg_cal = PoseGraph(loader)

f = float(np.sqrt(cam.fx * cam.fy))
Z_OFFSET = 41.6
WARMUP_END = scale_rec._warmup_end  # processed-frame index

c_history = []
rows = []
prev_features = None
prev_name = None
frame_idx = 0  # scale_rec._frame_count ile ayni sayim

gt = {}
with open("data/ground-truth.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        z = r.get("translation_z", "") or 0.0
        gt[r["frame_numbers"].strip()] = np.array(
            [float(r["translation_x"]), float(r["translation_y"]), float(z)])

for idx, name, frame in loader.frame_generator():
    curr = extractor.extract(frame, name)
    if prev_features is not None:
        frame_idx += 1
        mr = matcher.match(prev_features, curr)
        pose = estimator.estimate(mr)
        sr_prod = scale_rec.recover(pose, name, match_result=mr)

        sr_raw = sr_prod
        sr_cal = sr_prod

        if sr_prod.is_valid and pose.is_valid and pose.t is not None:
            g_curr = loader.get_ground_truth(name)
            g_prev = loader.get_ground_truth(prev_name)

            Z = None
            flow_px = None
            if g_curr is not None:
                Z = Z_OFFSET - float(g_curr.tz)
                flow_px = float(np.median(np.linalg.norm(
                    mr.pts_curr - mr.pts_prev, axis=1)))

            if frame_idx < WARMUP_END:
                # warmup: c ogren, ucu de uretimin olcegini kullansin
                if g_curr is not None and g_prev is not None and Z is not None and flow_px is not None:
                    s_ref = float(np.linalg.norm(g_curr.as_vector() - g_prev.as_vector()))
                    denom = flow_px * Z / f
                    if s_ref > 1e-6 and denom > 1e-9:
                        c_history.append(s_ref / denom)
                # sr_raw, sr_cal = sr_prod (zaten atandi)
            else:
                # autonomous: akis-ham ve akis-kal ayrisiyor
                if Z is not None and flow_px is not None:
                    s_flow = flow_px * Z / f
                    sr_raw = ScaleResult(
                        scale=s_flow, t_scaled=s_flow * pose.t, mode=sr_prod.mode,
                        is_valid=True, frame_name=name,
                        estimated_depth=Z, depth_source="gt",
                    )
                    c = float(np.median(c_history)) if c_history else 1.0
                    s_cal = c * s_flow
                    sr_cal = ScaleResult(
                        scale=s_cal, t_scaled=s_cal * pose.t, mode=sr_prod.mode,
                        is_valid=True, frame_name=name,
                        estimated_depth=Z, depth_source="gt",
                    )

        tp_prod = pg_prod.update(pose, sr_prod, name)
        tp_raw = pg_raw.update(pose, sr_raw, name)
        tp_cal = pg_cal.update(pose, sr_cal, name)

        stem = name.rsplit(".", 1)[0]
        if stem in gt:
            rows.append((stem, tp_prod.position.copy(), tp_raw.position.copy(),
                         tp_cal.position.copy()))
    prev_features = curr
    prev_name = name

c_final = float(np.median(c_history)) if c_history else float("nan")
print(f"ogrenilen c = {c_final:.4f}  (n={len(c_history)} warmup ornegi)")
print()

names = [r[0] for r in rows]
est_prod = np.array([r[1] for r in rows])
est_raw = np.array([r[2] for r in rows])
est_cal = np.array([r[3] for r in rows])
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
    step_est = np.linalg.norm(np.diff(est, axis=0), axis=1)
    step_gt = np.linalg.norm(np.diff(gt_arr, axis=0), axis=1)
    print(f"--- {label} ---")
    print(f"  Sim(3) ATE (rmse)                   : {np.sqrt((err_al**2).mean()):7.2f} m   olcek={s:.4f}")
    print(f"  HIZALANMAMIS mean (yarisma metrigi) : {err_raw.mean():7.2f} m")
    print(f"  toplam yol : tahmin={step_est.sum():7.1f} m  GT={step_gt.sum():7.1f} m  oran={step_gt.sum()/max(step_est.sum(),1e-9):.3f}")
    print()


print("=" * 66)
print(f"  n={len(rows)} kare")
print("=" * 66)
report("URETIM     (s = k*Z, hep)", est_prod)
report("AKIS-HAM   (autonomous'ta s = akis*Z/f)", est_raw)
report("AKIS-KAL   (autonomous'ta s = c*akis*Z/f)", est_cal)
