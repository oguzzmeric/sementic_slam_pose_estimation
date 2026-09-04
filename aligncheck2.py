import csv, numpy as np
gt={}
with open('data/ground-truth.csv',encoding='utf-8') as f:
    for r in csv.DictReader(f):
        gt[r['frame_numbers'].strip()]=(float(r['translation_x']),float(r['translation_y']))
est=[]
with open('data/trajectory_output.csv',encoding='utf-8') as f:
    for r in csv.DictReader(f):
        if r['is_valid'].strip().lower()!='true': continue
        s=r['frame_name'].rsplit('.',1)[0]
        if s in gt: est.append((s,float(r['x']),float(r['y'])))
est.sort()
E=np.array([[a,b] for _,a,b in est]); G=np.array([gt[s] for s,_,_ in est])
dE=np.diff(E,axis=0); dG=np.diff(G,axis=0)
mE=np.linalg.norm(dE,axis=1); mG=np.linalg.norm(dG,axis=1)
ok=(mE>1e-6)&(mG>1e-6)
dE,dG=dE[ok],dG[ok]
hE=np.arctan2(dE[:,1],dE[:,0]); hG=np.arctan2(dG[:,1],dG[:,0])
d=np.degrees((hG-hE+np.pi)%(2*np.pi)-np.pi)
n=len(d)
print('ornek : %d'%n)
print('tum    : medyan %+7.1f  std %.1f'%(np.median(d),np.std(d)))
print('ilk150 : medyan %+7.1f  std %.1f'%(np.median(d[:150]),np.std(d[:150])))
print('son    : medyan %+7.1f  std %.1f'%(np.median(d[150:]),np.std(d[150:])))
print()
for i in range(0,n,n//8):
    sl=slice(i,min(i+n//8,n))
    print('  %4d-%4d : %+7.1f'%(i,min(i+n//8,n),np.median(d[sl])))
