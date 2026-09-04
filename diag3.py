import numpy as np, logging
logging.basicConfig(level=logging.WARNING)
from utils.data_loader import DataLoader
from utils.camera_calibration import CameraCalibration
from core.feature_extractor import FeatureExtractor
from core.matcher import Matcher
from core.motion_estimator import MotionEstimator
from core.scale_recovery import ScaleRecovery

loader=DataLoader('config.yaml'); cam=CameraCalibration(loader)
ex=FeatureExtractor(loader,cam); mt=Matcher(loader)
es=MotionEstimator(loader,cam); sr=ScaleRecovery(loader,cam)

T=np.eye(4); prev=None; rows=[]
for idx,name,frame in loader.frame_generator():
    curr=ex.extract(frame,name)
    if prev is not None:
        pose=es.estimate(mt.match(prev,curr))
        res=sr.recover(pose,name)
        if pose.is_valid and res.is_valid:
            t=res.t_scaled.flatten().copy()
            n_before=np.linalg.norm(t)
            t[2]=0.0
            xy=np.linalg.norm(t)
            if xy>1e-9: t=t*(n_before/xy)
            L=np.eye(4); L[:3,:3]=pose.R; L[:3,3]=t
            p_old=T[:3,3].copy()
            T=T@L
            step3=np.linalg.norm(T[:3,3]-p_old)
            z_leak=abs(T[2,3])
            T[2,3]=0.0
            step2=np.linalg.norm(T[:3,3]-np.array([p_old[0],p_old[1],0.0]))
            rows.append((n_before,step3,step2,z_leak))
    prev=curr
    if idx>=300: break

a=np.array(rows)
print(f'olcek (t normu)      : medyan {np.median(a[:,0]):.4f}')
print(f'3B adim (Z kesmeden) : medyan {np.median(a[:,1]):.4f}')
print(f'2B adim (Z kesildi)  : medyan {np.median(a[:,2]):.4f}')
print(f'Z e sizan miktar     : medyan {np.median(a[:,3]):.4f}')
print()
print(f'3B/olcek : {np.median(a[:,1]/a[:,0]):.4f}   <- 1.0 olmali')
print(f'2B/olcek : {np.median(a[:,2]/a[:,0]):.4f}   <- kayip burada')
