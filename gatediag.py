import numpy as np, cv2, logging
logging.basicConfig(level=logging.WARNING)
from utils.data_loader import DataLoader
from utils.camera_calibration import CameraCalibration
from core.feature_extractor import FeatureExtractor
from core.matcher import Matcher

loader=DataLoader('config.yaml'); cam=CameraCalibration(loader)
ex=FeatureExtractor(loader,cam); mt=Matcher(loader)
K=cam.K; Ki=cam.K_inv
fx,fy,cx,cy=cam.fx,cam.fy,cam.cx,cam.cy

prev=None; done=0
for idx,name,frame in loader.frame_generator():
    curr=ex.extract(frame,name)
    if prev is not None:
        mr=mt.match(prev,curr)
        if mr.match_count>=8:
            H,_=cv2.findHomography(mr.pts_prev,mr.pts_curr,cv2.RANSAC,3.0)
            if H is not None:
                ns,Rs,ts,_=cv2.decomposeHomographyMat(H,K)
                P1=K@np.hstack([np.eye(3),np.zeros((3,1))]); O1=np.zeros(3)
                print('--- kare %d, %d aday ---'%(idx,ns))
                for i in range(ns):
                    R=Rs[i]; t=ts[i].reshape(3,1)
                    tn=np.linalg.norm(t)
                    if tn<1e-9:
                        print('  aday %d: t_norm=%.2e ATLANDI'%(i,tn)); continue
                    tu=t/tn; P2=K@np.hstack([R,tu]); O2=(-R.T@tu).flatten()
                    cosp=[]; errs=[]; nz=0; ntot=0
                    for j in range(min(50,len(mr.pts_prev))):
                        u1,v1=mr.pts_prev[j]; u2,v2=mr.pts_curr[j]
                        p1=Ki@np.array([u1,v1,1.0]); p2=Ki@np.array([u2,v2,1.0])
                        A=np.array([p1[0]*P1[2]-P1[0],p1[1]*P1[2]-P1[1],
                                    p2[0]*P2[2]-P2[0],p2[1]*P2[2]-P2[1]])
                        _,_,Vt=np.linalg.svd(A); Xh=Vt[-1]
                        if abs(Xh[3])<1e-9: continue
                        X=Xh[:3]/Xh[3]; ntot+=1
                        r1=X-O1; r2=X-O2
                        d1=np.linalg.norm(r1); d2=np.linalg.norm(r2)
                        if d1<1e-9 or d2<1e-9: continue
                        cosp.append(np.dot(r1,r2)/(d1*d2))
                        Z1=X[2]; Xc2=R@X+tu.flatten(); Z2=Xc2[2]
                        if Z1>0 and Z2>0:
                            nz+=1
                            uh=fx*X[0]/Z1+cx; vh=fy*X[1]/Z1+cy
                            errs.append(np.hypot(uh-u1,vh-v1))
                    cp=np.array(cosp); er=np.array(errs) if errs else np.array([np.nan])
                    print('  aday %d: onde=%d/%d  cos_par med=%.6f min=%.6f  reproj med=%.1f max=%.1f px'%(
                        i,nz,ntot,np.median(cp),cp.min(),np.nanmedian(er),np.nanmax(er)))
                done+=1
        if done>=4: break
    prev=curr
