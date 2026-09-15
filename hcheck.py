import numpy as np, cv2, logging
logging.basicConfig(level=logging.WARNING)
from utils.data_loader import DataLoader
from utils.camera_calibration import CameraCalibration
from core.feature_extractor import FeatureExtractor
from core.matcher import Matcher

loader=DataLoader('config.yaml'); cam=CameraCalibration(loader)
ex=FeatureExtractor(loader,cam); mt=Matcher(loader)

prev=None; n=0
print(' kare  aday  t_norm_max  normal_z  aci')
for idx,name,frame in loader.frame_generator():
    curr=ex.extract(frame,name)
    if prev is not None:
        mr=mt.match(prev,curr)
        if mr.match_count>=8:
            H,_=cv2.findHomography(mr.pts_prev,mr.pts_curr,cv2.RANSAC,3.0)
            if H is not None:
                ns,Rs,ts,Ns=cv2.decomposeHomographyMat(H,cam.K)
                tn=[float(np.linalg.norm(t)) for t in ts]
                nz=[float(abs(nn.flatten()[2])) for nn in Ns]
                ang=[np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1))) for R in Rs]
                print('%5d %5d %11.5f %9.3f %5.1f'%(
                    idx,ns,max(tn),max(nz),np.median(ang)))
                n+=1
        if n>=15: break
    prev=curr
