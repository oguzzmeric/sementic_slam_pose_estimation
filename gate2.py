import numpy as np, cv2, logging
logging.basicConfig(level=logging.WARNING)
from utils.data_loader import DataLoader
from utils.camera_calibration import CameraCalibration
from core.feature_extractor import FeatureExtractor
from core.matcher import Matcher

loader=DataLoader('config.yaml'); cam=CameraCalibration(loader)
ex=FeatureExtractor(loader,cam); mt=Matcher(loader)
K=cam.K; Ki=cam.K_inv; fx,fy,cx,cy=cam.fx,cam.fy,cam.cx,cam.cy
P1=K@np.hstack([np.eye(3),np.zeros((3,1))]); O1=np.zeros(3)

prev=None; done=0
print(' kare aday  onde  dusuk_par  reproj_fail  gecen  reproj_med')
for idx,name,frame in loader.frame_generator():
    curr=ex.extract(frame,name)
    if prev is not None:
        mr=mt.match(prev,curr)
        if mr.match_count>=8:
            H,_=cv2.findHomography(mr.pts_prev,mr.pts_curr,cv2.RANSAC,3.0)
            if H is not None:
                ns,Rs,ts,_=cv2.decomposeHomographyMat(H,K)
                best=(-1,0,0,0,0)
                for i in range(ns):
                    R=Rs[i]; t=ts[i].reshape(3,1)
                    if np.linalg.norm(t)<1e-12: continue
                    P2=K@np.hstack([R,t]); O2=(-R.T@t).flatten()
                    onde=0; lowp=0; rfail=0; gecen=0; errs=[]
                    for j in range(min(50,len(mr.pts_prev))):
                        u1,v1=mr.pts_prev[j]; u2,v2=mr.pts_curr[j]
                        p1=Ki@np.array([u1,v1,1.0]); p2=Ki@np.array([u2,v2,1.0])
                        A=np.array([p1[0]*P1[2]-P1[0],p1[1]*P1[2]-P1[1],
                                    p2[0]*P2[2]-P2[0],p2[1]*P2[2]-P2[1]])
                        _,_,Vt=np.linalg.svd(A); Xh=Vt[-1]
                        if abs(Xh[3])<1e-9: continue
                        X=Xh[:3]/Xh[3]
                        r1=X-O1; r2=X-O2
                        d1=np.linalg.norm(r1); d2=np.linalg.norm(r2)
                        if d1<1e-9 or d2<1e-9: continue
                        cp=np.dot(r1,r2)/(d1*d2)
                        Z1=X[2]; Xc2=R@X+t.flatten(); Z2=Xc2[2]
                        if Z1<=0 or Z2<=0:
                            if cp>0.99998: lowp+=1
                            continue
                        onde+=1
                        uh=fx*X[0]/Z1+cx; vh=fy*X[1]/Z1+cy
                        e=np.hypot(uh-u1,vh-v1); errs.append(e)
                        if e>4.0: rfail+=1; continue
                        uh2=fx*Xc2[0]/Z2+cx; vh2=fy*Xc2[1]/Z2+cy
                        if np.hypot(uh2-u2,vh2-v2)>4.0: rfail+=1; continue
                        gecen+=1
                    if gecen>best[0]:
                        best=(gecen,onde,lowp,rfail,np.median(errs) if errs else -1)
                print('%5d %4d %6d %10d %12d %6d %10.1f'%(
                    idx,ns,best[1],best[2],best[3],best[0],best[4]))
                done+=1
        if done>=10: break
    prev=curr
