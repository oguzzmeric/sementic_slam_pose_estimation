"""
Optik akistan olcek kestirimi -- dogrulama.

Nadir bakista Z yukseklikteki zemin noktasi kameranin metrik hareketiyle
orantili kayar:

    piksel_kaymasi = f * s / Z    ->    s = piksel_kaymasi * Z / f

Kamera donerse noktalar oteleme olmadan da kayar. Rotasyon kaynakli
akis R ile hesaplanip cikarilir:

    p_rot = K R K^-1 p     (sonsuzdaki nokta yaklasimi)

Bu betik uc olcumu karsilastirir:
    ham akis        -> rotasyon cikarilmamis
    duzeltilmis akis-> rotasyon cikarilmis
    GT adim         -> gercek metrik yer degistirme
"""

import logging
import numpy as np

logging.basicConfig(level=logging.WARNING)

from utils.data_loader import DataLoader
from utils.camera_calibration import CameraCalibration
from core.feature_extractor import FeatureExtractor
from core.matcher import Matcher
from core.motion_estimator import MotionEstimator

Z_OFFSET = 41.6
LIMIT = 450

loader = DataLoader("config.yaml")
cam = CameraCalibration(loader)
ex = FeatureExtractor(loader, cam)
mt = Matcher(loader)
es = MotionEstimator(loader, cam)

K = cam.K
K_inv = cam.K_inv
f = float(np.sqrt(cam.fx * cam.fy))

rows = []
prev = None
prev_name = None

for idx, name, frame in loader.frame_generator():
    curr = ex.extract(frame, name)

    if prev is not None:
        mr = mt.match(prev, curr)
        if mr.match_count >= 8:
            pose = es.estimate(mr)

            g1 = loader.get_ground_truth(prev_name)
            g2 = loader.get_ground_truth(name)

            if g1 is not None and g2 is not None and pose.is_valid:
                gt_step = float(np.linalg.norm(g2.as_vector() - g1.as_vector()))
                Z = Z_OFFSET - float(g2.tz)

                p0 = mr.pts_prev
                p1 = mr.pts_curr

                # --- ham akis ---
                raw_flow = float(np.median(np.linalg.norm(p1 - p0, axis=1)))

                # --- rotasyon kaynakli akisi cikar ---
                # Sonsuzdaki nokta yaklasimi: p_rot ~ K R K^-1 p
                Hrot = K @ pose.R @ K_inv
                ph = np.hstack([p0, np.ones((len(p0), 1))])
                pr = (Hrot @ ph.T).T
                w = pr[:, 2:3]
                ok = np.abs(w[:, 0]) > 1e-9
                pr_xy = pr[ok, :2] / w[ok]
                corr_flow = float(np.median(np.linalg.norm(p1[ok] - pr_xy, axis=1)))

                rows.append((idx, gt_step, Z, raw_flow, corr_flow))

    prev = curr
    prev_name = name
    if idx >= LIMIT:
        break

a = np.array(rows)
idxs, gt_step, Z, raw_flow, corr_flow = a.T

s_raw = raw_flow * Z / f
s_corr = corr_flow * Z / f

print("=" * 70)
print(f"  {len(a)} kare   f = {f:.1f}")
print("=" * 70)
print()
print("KORELASYON  (GT adim ile)")
print(f"  ham akis          : {np.corrcoef(s_raw, gt_step)[0,1]:+.4f}")
print(f"  duzeltilmis akis  : {np.corrcoef(s_corr, gt_step)[0,1]:+.4f}")
print()

for nm, s in (("ham", s_raw), ("duzeltilmis", s_corr)):
    r = gt_step / np.maximum(s, 1e-9)
    print(f"{nm.upper()} AKIS")
    print(f"  kestirim medyan : {np.median(s):.3f} m")
    print(f"  GT medyan       : {np.median(gt_step):.3f} m")
    print(f"  oran (GT/kest)  : medyan {np.median(r):.3f}   std {np.std(r):.3f}")
    print()

print("DILIM PROFILI  (duzeltilmis)")
print(f"{'kare':>6} {'GT adim':>9} {'kestirim':>10} {'oran':>7} {'akis_px':>9} {'irtifa':>8}")
n = len(a); step = max(1, n // 10)
for i in range(0, n, step):
    sl = slice(i, min(i + step, n))
    print("%6d %9.3f %10.3f %7.3f %9.1f %8.1f" % (
        idxs[sl][0], np.median(gt_step[sl]), np.median(s_corr[sl]),
        np.median(gt_step[sl]) / max(np.median(s_corr[sl]), 1e-9),
        np.median(corr_flow[sl]), np.median(Z[sl])))

print()
print("KARSILASTIRMA  --  mevcut k sabiti yaklasimi")
k_fixed = 0.0427
s_k = k_fixed * Z
r_k = gt_step / np.maximum(s_k, 1e-9)
r_f = gt_step / np.maximum(s_corr, 1e-9)
print(f"  k sabit      : oran medyan {np.median(r_k):.3f}   std {np.std(r_k):.3f}")
print(f"  optik akis   : oran medyan {np.median(r_f):.3f}   std {np.std(r_f):.3f}")
print()
print("  (oran 1.0 ideal, std dusuk olmali)")