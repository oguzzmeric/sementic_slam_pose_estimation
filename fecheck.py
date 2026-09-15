import numpy as np, logging
logging.basicConfig(level=logging.WARNING)
from utils.data_loader import DataLoader
from utils.camera_calibration import CameraCalibration
from core.feature_extractor import FeatureExtractor

loader=DataLoader('config.yaml'); cam=CameraCalibration(loader)
ex=FeatureExtractor(loader,cam)
print('dynamic_classes tipi:', type(ex._dynamic_classes).__name__)
print()
for idx,name,frame in loader.frame_generator():
    f=ex.extract(frame,name)
    masked=100*int((f.mask==0).sum())/f.mask.size
    dets=loader.get_detections(name)
    dyn=[d for d in dets if d.class_id in ex._dynamic_classes]
    print('%-22s kp=%4d  maske=%5.2f%%  tespit=%2d  dinamik=%2d'%(
        name, f.keypoint_count, masked, len(dets), len(dyn)))
    if idx>=8: break
