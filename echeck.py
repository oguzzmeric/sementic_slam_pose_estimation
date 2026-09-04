import numpy as np, cv2, logging
logging.basicConfig(level=logging.WARNING)
from utils.data_loader import DataLoader
from utils.camera_calibration import CameraCalibration
from core.feature_extractor import FeatureExtractor
from core.matcher import Matcher

loader=DataLoader('config.yaml'); cam=CameraCalibration(loader)
ex=FeatureExtractor(loader,cam); mt=Matcher(loader); K=cam.K

prev=None; done=0
for idx,name,frame in loader.frame_generator():
    curr=ex.extract(frame,name)
    if prev is not None:
        mr=mt.match(prev,curr)
        if mr.match_count>=8:
            H,_=cv2.findHomography(mr.pts_prev,mr.pts_curr,cv2.RANSAC,3.0)
            E,em=cv2.findEssentialMat(mr.pts_prev,mr.pts_curr,cameraMatrix=K,
                                      method=cv2.RANSAC,prob=0.999,threshold=3.0)
            nsH=cv2.decomposeHomographyMat(H,K)[0] if H is not None else 0
            line='kare %3d  eslesme=%4d  H_aday=%d' % (idx, mr.match_count, nsH)
            if E is not None and E.shape==(3,3):
                n,R,t,_=cv2.recoverPose(E,mr.pts_prev,mr.pts_curr,cameraMatrix=K,mask=em)
                ang=np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1)))
                line+='  E_inlier=%4d  t=[%+.3f %+.3f %+.3f]  aci=%.1f' % (
                    n, t[0,0], t[1,0], t[2,0], ang)
            else:
                line+='  E YOK'
            print(line)
            done+=1
        if done>=8: break
    prev=curr
