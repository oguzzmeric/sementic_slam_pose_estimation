import numpy as np, cv2, logging
logging.basicConfig(level=logging.WARNING)
from utils.data_loader import DataLoader
from utils.camera_calibration import CameraCalibration
from core.feature_extractor import FeatureExtractor
from core.matcher import Matcher

loader=DataLoader('config.yaml'); cam=CameraCalibration(loader)
ex=FeatureExtractor(loader,cam); mt=Matcher(loader); K=cam.K

frames=loader.frame_list[:200]
cache={}
def feat(i):
    if i not in cache:
        p=frames[i]
        cache[i]=ex.extract(loader.load_frame(p),p.name)
    return cache[i]

print(' aralik  eslesme  H_aday  E_inlier  t_norm  aci')
for step in [1,2,3,5,8,12,20]:
    ms,ha,ei,tn,an=[],[],[],[],[]
    for a in range(0,120,10):
        b=a+step
        if b>=len(frames): break
        mr=mt.match(feat(a),feat(b))
        ms.append(mr.match_count)
        if mr.match_count<8: continue
        H,_=cv2.findHomography(mr.pts_prev,mr.pts_curr,cv2.RANSAC,3.0)
        if H is not None:
            ns,Rs,ts,_=cv2.decomposeHomographyMat(H,K)
            ha.append(ns)
            tn.append(max(float(np.linalg.norm(t)) for t in ts))
        E,em=cv2.findEssentialMat(mr.pts_prev,mr.pts_curr,cameraMatrix=K,
                                  method=cv2.RANSAC,prob=0.999,threshold=3.0)
        if E is not None and E.shape==(3,3):
            n,R,t,_=cv2.recoverPose(E,mr.pts_prev,mr.pts_curr,cameraMatrix=K,mask=em)
            ei.append(n)
            an.append(np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1))))
    print('%7d %8.0f %7.1f %9.0f %7.3f %5.1f'%(
        step, np.median(ms), np.mean(ha) if ha else 0,
        np.median(ei) if ei else 0, np.median(tn) if tn else 0,
        np.median(an) if an else 0))
