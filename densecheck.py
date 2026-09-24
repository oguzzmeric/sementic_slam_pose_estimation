"""
"Hizli hareket segmentinde frame_step dusurme" hipotezini izole test
eder. Tum pipeline'i yeniden kurmak (riskli, buyuk mimari degisiklik)
yerine, BILINEN COKEN segmentte (frame_001190 -> frame_001260, KLT'nin
kare arasi kayma 100-400px'e firladigi bolge) SADECE bu araligi:

  SEYREK (adim=5, su anki uretim)  vs  YOGUN (adim=1, HER ham kare)

olarak iki kere isleyip, yon hatasini (heading error -- tahmini adim
yonu ile GT adim yonu arasindaki aci) karsilastiriyoruz. Hipotez: kare
arasi hareket kucultulunce essential matrix / matching daha guvenilir
olur, yon hatasi kucuklur -- BU KISIMDA HENUZ BA YOK, sadece cig VO
kalitesini test ediyoruz (BA'nin ustune eklenip eklenmeyecegine karar
vermeden once).

core/*.py'ye dokunmuyor.
"""

import csv
import numpy as np

from utils.data_loader import DataLoader
from utils.camera_calibration import CameraCalibration
from core.feature_extractor import FeatureExtractor
from core.matcher import Matcher
from core.motion_estimator import MotionEstimator

SEGMENT_START = "frame_001190.webp"
SEGMENT_END = "frame_001260.webp"
SPARSE_STEP = 5

loader = DataLoader("config.yaml")
cam = CameraCalibration(loader)
extractor = FeatureExtractor(loader, cam)
matcher = Matcher(loader)
estimator = MotionEstimator(loader, cam)

# ham kareleri (adim=1) dogrudan frames_dir'den al -- loader.frame_list
# zaten frame_step=5 uygulanmis, ham/tam listeye ayrica erisiyoruz
all_raw = sorted(loader.frames_dir.iterdir(), key=lambda p: p.name)
i0 = next(i for i, p in enumerate(all_raw) if p.name == SEGMENT_START)
i1 = next(i for i, p in enumerate(all_raw) if p.name == SEGMENT_END)
dense_paths = all_raw[i0:i1 + 1]
print(f"yogun segment: {len(dense_paths)} ham kare ({SEGMENT_START} -> {SEGMENT_END})")

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


def run_chain(frame_paths):
    """Ardisik kareler uzerinde ORB+E/H ile rotasyon zincirler, her
    adimda (dunya-cercevesi tahmini yon acisi, GT yon acisi, |fark|)
    dondurur. Sadece YON -- olcek/metrik yok, gerekmiyor."""
    prev_features, prev_name = None, None
    R_world = np.eye(3)
    rows = []
    for p in frame_paths:
        name = p.name
        frame = loader.load_frame(p)
        curr = extractor.extract(frame, name)
        if prev_features is not None:
            mr = matcher.match(prev_features, curr)
            pose = estimator.estimate(mr)
            if pose.is_valid:
                gt_curr, gt_prev = gt.get(name.rsplit(".", 1)[0]), gt.get(prev_name.rsplit(".", 1)[0])
                if gt_curr is not None and gt_prev is not None:
                    gt_dir = gt_curr[:2] - gt_prev[:2]
                    if np.linalg.norm(gt_dir) > 1e-6:
                        world_dir = (R_world @ pose.t.flatten())[:2]
                        if np.linalg.norm(world_dir) > 1e-9:
                            err = angle_diff(heading(world_dir), heading(gt_dir))
                            rows.append((name, err))
                R_world = R_world @ pose.R
        prev_features, prev_name = curr, name
    return rows


print()
print("=== SEYREK (adim=5, suanki uretim) ===")
sparse_paths = dense_paths[::SPARSE_STEP]
if dense_paths[-1] not in sparse_paths:
    sparse_paths = sparse_paths + [dense_paths[-1]]
sparse_rows = run_chain(sparse_paths)
sparse_err = np.array([e for _, e in sparse_rows])
print(f"n={len(sparse_err)}  medyan yon hatasi={np.median(sparse_err):.2f} deg  "
      f"ortalama={np.mean(sparse_err):.2f} deg  max={np.max(sparse_err):.2f} deg")

print()
print("=== YOGUN (adim=1, HER ham kare) ===")
dense_rows = run_chain(dense_paths)
dense_err = np.array([e for _, e in dense_rows])
print(f"n={len(dense_err)}  medyan yon hatasi={np.median(dense_err):.2f} deg  "
      f"ortalama={np.mean(dense_err):.2f} deg  max={np.max(dense_err):.2f} deg")

