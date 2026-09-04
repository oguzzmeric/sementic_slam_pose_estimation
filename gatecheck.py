import logging
logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logging.getLogger('utils').setLevel(logging.WARNING)
logging.getLogger('core.feature_extractor').setLevel(logging.WARNING)
logging.getLogger('core.matcher').setLevel(logging.WARNING)
logging.getLogger('core.scale_recovery').setLevel(logging.WARNING)

from utils.data_loader import DataLoader
from utils.camera_calibration import CameraCalibration
from core.feature_extractor import FeatureExtractor
from core.matcher import Matcher
from core.motion_estimator import MotionEstimator

loader=DataLoader('config.yaml'); cam=CameraCalibration(loader)
ex=FeatureExtractor(loader,cam); mt=Matcher(loader); es=MotionEstimator(loader,cam)

prev=None; n=0
for idx,name,frame in loader.frame_generator():
    curr=ex.extract(frame,name)
    if prev is not None:
        pose=es.estimate(mt.match(prev,curr))
        n+=1
        if n>=30: break
    prev=curr
