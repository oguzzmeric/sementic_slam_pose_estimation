"""
Olcek sicramasinin (Sim3 0.976 -> 1.144) kaynagini ayiklar:

  A) k_factor / warmup_scale_mean gercekten degisti mi (warmup ornek
     sayisi degisince kalibrasyon kaydi mi)?
  B) Tahmini toplam yol uzunlugu GT'ye kiyasla kisaldi mi (4 donmuş kare
     hareketi tamamen atladigi icin), ve bu kisalma tek basina Sim3
     olcek sicramasini aciklar mi?
  C) Bu 4 kare disinda, genel adim uzunlugu oraninda sistematik bir
     kisa/uzun egilim var mi (autonomous modda scale surekli dusuk mu
     tahmin ediliyor)?
"""

import csv
import logging
import numpy as np

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s [%(levelname)s] %(message)s")

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
        rows.append((stem, pose.is_valid, sr.mode, tp.position.copy()))
    prev_features = curr

print()
print("=" * 66)
print("A) WARMUP KALIBRASYONU")
print("=" * 66)
print(f"  warmup_scale_mean = {scale_rec.warmup_scale}")
print(f"  k_factor          = {scale_rec.k_factor}")

names = [r[0] for r in rows if r[0] in gt]
valid = np.array([r[1] for r in rows if r[0] in gt])
modes = [r[2] for r in rows if r[0] in gt]
est = np.array([r[3] for r in rows if r[0] in gt])
gt_arr = np.array([gt[n] for n in names])

d_gt_full = np.diff(gt_arr, axis=0)
step_gt_full = np.linalg.norm(d_gt_full, axis=1)
total_gt = step_gt_full.sum()

# tahmini yol uzunlugu: sadece GECERLI adimlar hareket ediyor, invalid
# olanlar T_world'u degistirmiyor (position ayni kaliyor) -> diff=0
d_est_full = np.diff(est, axis=0)
step_est_full = np.linalg.norm(d_est_full, axis=1)
total_est = step_est_full.sum()

print()
print("=" * 66)
print("B) TOPLAM YOL UZUNLUGU (ham, hizalanmamis)")
print("=" * 66)
print(f"  toplam GT yol   = {total_gt:9.2f} m")
print(f"  toplam tahmin   = {total_est:9.2f} m")
print(f"  oran (GT/tahmin)= {total_gt/max(total_est,1e-9):.4f}")
print(f"  (Sim3 olcek faktoru referans: 1.144 idi)")

n_invalid = int((~valid[1:]).sum())
missing_gt = step_gt_full[~valid[1:]].sum() if n_invalid else 0.0
print()
print(f"  invalid (donmus) adim sayisi   = {n_invalid}")
print(f"  bu adimlarda GT'nin kat ettigi mesafe = {missing_gt:.2f} m")
print(f"  bu, toplam GT yolunun %{100*missing_gt/total_gt:.2f}'i")

print()
print("=" * 66)
print("C) ADIM UZUNLUGU ORANI -- MOD BAZLI (yalniz gecerli adimlar)")
print("=" * 66)
modes_arr = np.array(modes[1:])
valid_step = valid[1:]
for m in ("warmup", "autonomous", "autonomous_fallback"):
    sel = valid_step & (modes_arr == m)
    if sel.sum() == 0:
        continue
    r = step_gt_full[sel] / np.maximum(step_est_full[sel], 1e-9)
    print(f"  {m:22s} n={sel.sum():4d}  GT/tahmin oran medyan={np.median(r):.3f}  ortalama={np.mean(r):.3f}")
