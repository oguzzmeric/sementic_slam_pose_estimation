"""
refine_trajectory.py
=====================
Uretim pipeline'ini calistirip, PoseGraph.refine_with_persistent_map ile
GTSAM tabanli kalici-harita duzeltmesini uygular, sonucu uretimle
karsilastirir.

SADECE WSL/Linux'ta calisir -- GTSAM'in Windows'ta PyPI wheel'i yok.
Windows'ta normal main.py akisi bu dosyayi hic import etmez, gtsam'a
bagimli degildir.

Kullanim (WSL):
    source venv/bin/activate
    python3 refine_trajectory.py
"""

import logging

import numpy as np

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")

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

prev_features = None
for idx, name, frame in loader.frame_generator():
    curr = extractor.extract(frame, name)
    if prev_features is not None:
        mr = matcher.match(prev_features, curr)
        pose = estimator.estimate(mr)
        sr = scale_rec.recover(pose, name, match_result=mr)
        pose_graph.update(pose, sr, name)
    prev_features = curr

print(f"uretim pipeline tamamlandi: {len(pose_graph.trajectory)} kare islendi")

refined = pose_graph.refine_with_persistent_map(cam, extractor)

pose_graph.save_trajectory("data/trajectory_output.csv")
pose_graph.save_trajectory("data/trajectory_output_persistent_ba.csv", trajectory=refined)
print("Kaydedildi: data/trajectory_output.csv, data/trajectory_output_persistent_ba.csv")


def mean_error(points):
    errs = []
    for p in points:
        if not p.is_valid:
            continue
        gt = loader.get_ground_truth(p.frame_name)
        if gt is None:
            continue
        errs.append(np.linalg.norm(p.position - gt.as_vector()))
    return (float(np.mean(errs)) if errs else float("nan")), len(errs)


prod_err, n1 = mean_error(pose_graph.trajectory)
ref_err, n2 = mean_error(refined)

print()
print("=" * 60)
print("  HIZALANMAMIS mean (yarisma metrigi) -- ikisi de PoseGraph'in")
print("  KENDI Sim(3) raporlama mekanizmasiyla hesaplandi")
print("=" * 60)
print(f"  URETIM (duzeltmesiz)           n={n1:4d}  : {prod_err:.2f} m")
print(f"  PERSISTENT-MAP BA              n={n2:4d}  : {ref_err:.2f} m")
