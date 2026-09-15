import numpy as np
from utils.data_loader import DataLoader
from utils.camera_calibration import CameraCalibration

loader = DataLoader('config.yaml')
cam = CameraCalibration(loader)

print('K matrisi:')
print(cam.K)
print()

for px in [[1908.48, 1139.71], [2042, 1140], [3500, 2000]]:
    p = np.array(px, dtype=float)
    for Z in [20.0, 40.0]:
        P3 = cam.backproject_point(p, Z)
        p2 = cam.project_point(P3)
        print('piksel %-18s Z=%5.1f  ->  3B [%7.2f %7.2f %6.2f]  ->  %s  hata=%.2e' % (
            str(px), Z, P3[0], P3[1], P3[2], np.round(p2,2), np.linalg.norm(p-p2)))
