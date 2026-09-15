import numpy as np, cv2, logging
logging.basicConfig(level=logging.WARNING)
from utils.data_loader import DataLoader
from utils.camera_calibration import CameraCalibration
from core.feature_extractor import FeatureExtractor
from core.matcher import Matcher
from core.motion_estimator import MotionEstimator

loader=DataLoader('config.yaml'); cam=CameraCalibration(loader)
ex=FeatureExtractor(loader,cam); mt=Matcher(loader); es=MotionEstimator(loader,cam)

prev=None; n=0
print(' kare      S_H      S_E     R_H   H_inl   E_inl  secim')
for idx,name,frame in loader.frame_generator():
    curr=ex.extract(frame,name)
    if prev is not None:
        mr=mt.match(prev,curr)
        if mr.match_count>=8:
            H,_=cv2.findHomography(mr.pts_prev,mr.pts_curr,cv2.RANSAC,3.0)
            E,em=cv2.findEssentialMat(mr.pts_prev,mr.pts_curr,cameraMatrix=cam.K,
                                      method=cv2.RANSAC,prob=0.999,threshold=3.0)
            if H is not None and E is not None and E.shape==(3,3):
                sH,mH=es._compute_homography_score(H,mr.pts_prev,mr.pts_curr,3.0)
                sE,mE=es._compute_essential_score(E,mr.pts_prev,mr.pts_curr,3.0)
                rh=sH/(sH+sE) if (sH+sE)>1e-9 else 0
                print('%5d %8.1f %8.1f %7.3f %7d %7d  %s'%(
                    idx,sH,sE,rh,int(mH.sum()),int(mE.sum()),
                    'H' if rh>0.45 else 'E'))
                n+=1
        if n>=15: break
    prev=curr
