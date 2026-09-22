"""
Kalibrasyon dogrulama -- konuma bagli sapma testi.

Fikir: eger fx/fy/cx/cy ya da distortion katsayilari yanlissa, bbox-tabanli
derinlik tahmininin GT irtifaya orani goruntudeki KONUMA (merkeze uzaklik
VE yon) bagli olarak sistematik kayar. Sabit bir odak uzunlugu hatasi
(ornegin fy %2 yanlis) TUM derinlikleri ayni oranda kaydirir -- bunu zaten
biliyoruz (medyan oran 1.054). Ama eger cx/cy kaymissa ya da distortion
yanlissa, hata KONUMA gore degisir -- bu da her karede farkli feature
dagilimina gore farkli yonde R/t sapmasi yaratip "surekli ama kalici"
yon hatasi orintusunu acikliyor olabilir.

Testler:
  1) oran ~ merkeze radyal uzaklik (radial_px)      -- lens/odak testi
  2) oran ~ yatay ofset (u - cx), imza dahil         -- cx testi
  3) oran ~ dikey ofset (v - cy), imza dahil         -- cy testi

core/*.py'ye dokunmuyor, sadece ScaleRecovery'nin mevcut ozel metotlarini
(zaten warmup'ta ayni islemi yapiyor) okuyor.
"""

import logging
import numpy as np

logging.basicConfig(level=logging.WARNING)

from utils.data_loader import DataLoader
from utils.camera_calibration import CameraCalibration
from core.scale_recovery import ScaleRecovery

import sys
config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
print(f"config: {config_path}")

loader = DataLoader(config_path)
cam = CameraCalibration(loader)
scale_rec = ScaleRecovery(loader, cam)

Z_OFFSET = 41.6
cx, cy = cam.cx, cam.cy

rows = []
for fi in loader.frame_list:
    name = fi.name
    detections = loader.get_detections(name)
    if not detections:
        continue
    gt = loader.get_ground_truth(name)
    if gt is None:
        continue
    z_ref = Z_OFFSET - float(gt.tz)
    if not (1.0 <= z_ref <= 500.0):
        continue

    for det in detections:
        est = scale_rec._depth_from_detection(det)
        if est is None:
            continue
        x1, y1, x2, y2 = det.bbox
        u_c = (x1 + x2) / 2.0
        v_c = (y1 + y2) / 2.0
        ratio = est.depth / z_ref
        rows.append((est.radial_px, u_c - cx, v_c - cy, ratio, est.confidence))

radial = np.array([r[0] for r in rows])
du = np.array([r[1] for r in rows])
dv = np.array([r[2] for r in rows])
ratio = np.array([r[3] for r in rows])

print("=" * 66)
print(f"  n={len(rows)} tespit (GT ile eslesen)")
print("=" * 66)
print(f"  oran (bbox_derinlik/GT) genel medyan = {np.median(ratio):.4f}  std={np.std(ratio):.4f}")
print()

print("1) RADYAL TEST (lens/odak uzunlugu)")
print(f"   korelasyon(radial_px, oran) = {np.corrcoef(radial, ratio)[0,1]:+.4f}")
order = np.argsort(radial)
r_s = radial[order]; ratio_s = ratio[order]
n = len(r_s); step = max(1, n // 8)
print(f"   {'dilim':>6} {'radial_px':>10} {'oran_medyan':>12} {'n':>5}")
for i in range(0, n, step):
    sl = slice(i, min(i+step, n))
    print(f"   {i:6d} {np.median(r_s[sl]):10.1f} {np.median(ratio_s[sl]):12.4f} {len(r_s[sl]):5d}")

print()
print("2) YATAY TEST (cx dogrulugu) -- sol vs sag")
left = ratio[du < -50]
right = ratio[du > 50]
center = ratio[np.abs(du) <= 50]
print(f"   korelasyon(u-cx, oran) = {np.corrcoef(du, ratio)[0,1]:+.4f}")
print(f"   sol   (u-cx<-50)  n={len(left):4d}  oran medyan={np.median(left):.4f}")
print(f"   merkez(|u-cx|<50) n={len(center):4d}  oran medyan={np.median(center):.4f}")
print(f"   sag   (u-cx>50)   n={len(right):4d}  oran medyan={np.median(right):.4f}")

print()
print("3) DIKEY TEST (cy dogrulugu) -- ust vs alt")
top = ratio[dv < -50]
bottom = ratio[dv > 50]
midv = ratio[np.abs(dv) <= 50]
print(f"   korelasyon(v-cy, oran) = {np.corrcoef(dv, ratio)[0,1]:+.4f}")
print(f"   ust   (v-cy<-50)  n={len(top):4d}     oran medyan={np.median(top):.4f}")
print(f"   orta  (|v-cy|<50) n={len(midv):4d}     oran medyan={np.median(midv):.4f}")
print(f"   alt   (v-cy>50)   n={len(bottom):4d}     oran medyan={np.median(bottom):.4f}")
