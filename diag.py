import numpy as np, logging
logging.basicConfig(level=logging.WARNING)
from utils.data_loader import DataLoader
from utils.camera_calibration import CameraCalibration
from core.feature_extractor import FeatureExtractor
from core.matcher import Matcher
from core.motion_estimator import MotionEstimator

loader = DataLoader('config.yaml')
cam = CameraCalibration(loader)
ex = FeatureExtractor(loader, cam)
mt = Matcher(loader)
es = MotionEstimator(loader, cam)

prev = None
rec = []
for idx, name, frame in loader.frame_generator():
    curr = ex.extract(frame, name)
    if prev is not None:
        pose = es.estimate(mt.match(prev, curr))
        if pose.is_valid:
            g1 = loader.get_ground_truth(pose.frame_name_prev)
            g2 = loader.get_ground_truth(name)
            if g1 and g2:
                gt = np.hypot(g2.tx-g1.tx, g2.ty-g1.ty)
                t = pose.t.flatten()
                rec.append((gt, np.linalg.norm(t[:2]), np.linalg.norm(t), abs(t[2])))
    prev = curr
    if idx >= 300: break

a = np.array(rec)
print(f'ornek sayisi        : {len(a)}')
print(f'GT yer degistirme   : medyan {np.median(a[:,0]):.4f} m')
print(f't_xy normu          : medyan {np.median(a[:,1]):.4f}')
print(f't tam norm          : medyan {np.median(a[:,2]):.4f}')
print(f'|t_z|               : medyan {np.median(a[:,3]):.4f}')
print()
print(f'olcek (XY paydali)  : medyan {np.median(a[:,0]/a[:,1]):.4f}')
print(f'olcek (tam paydali) : medyan {np.median(a[:,0]/a[:,2]):.4f}')
print(f'iki yontem orani    : {np.median(a[:,0]/a[:,2]) / np.median(a[:,0]/a[:,1]):.4f}')
