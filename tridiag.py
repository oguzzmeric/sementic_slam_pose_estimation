import numpy as np, cv2, logging
logging.basicConfig(level=logging.WARNING)
from utils.data_loader import DataLoader
from utils.camera_calibration import CameraCalibration
from core.feature_extractor import FeatureExtractor
from core.matcher import Matcher

loader=DataLoader('config.yaml'); cam=CameraCalibration(loader)
ex=FeatureExtractor(loader,cam); mt=Matcher(loader)
K=cam.K; Ki=cam.K_inv
P1=K@np.hstack([np.eye(3),np.zeros((3,1))])

prev=None
for idx,name,frame in loader.frame_generator():
    curr=ex.extract(frame,name)
    if prev is not None:
        mr=mt.match(prev,curr)
        if mr.match_count>=8:
            H,_=cv2.findHomography(mr.pts_prev,mr.pts_curr,cv2.RANSAC,3.0)
            ns,Rs,ts,Ns=cv2.decomposeHomographyMat(H,K)
            for i in range(ns):
                R=Rs[i]; t=ts[i].reshape(3,1)
                tn=float(np.linalg.norm(t))
                if tn<1e-12:
                    print('aday %d: t_norm=0 ATLANDI'%i); continue
                for label,tt in (('ham',t),('birim',t/tn),('x40',t/tn*40.0)):
                    P2=K@np.hstack([R,tt])
                    Z1s=[]; Z2s=[]
                    for j in range(20):
                        u1,v1=mr.pts_prev[j]; u2,v2=mr.pts_curr[j]
                        p1=Ki@np.array([u1,v1,1.0]); p2=Ki@np.array([u2,v2,1.0])
                        A=np.array([p1[0]*P1[2]-P1[0],p1[1]*P1[2]-P1[1],
                                    p2[0]*P2[2]-P2[0],p2[1]*P2[2]-P2[1]])
                        _,_,Vt=np.linalg.svd(A); Xh=Vt[-1]
                        if abs(Xh[3])<1e-12: continue
                        X=Xh[:3]/Xh[3]
                        Z1s.append(X[2]); Z2s.append((R@X+tt.flatten())[2])
                    Z1s=np.array(Z1s); Z2s=np.array(Z2s)
                    print('aday %d %-6s t_norm=%.5f  Z1 med=%9.2f  Z2 med=%9.2f  ikisi_pozitif=%d/%d'%(
                        i,label,float(np.linalg.norm(tt)),np.median(Z1s),np.median(Z2s),
                        int(((Z1s>0)&(Z2s>0)).sum()),len(Z1s)))
                print()
            break
    prev=curr
