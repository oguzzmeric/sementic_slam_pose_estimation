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
hE=np.degrees(np.arctan2(dE[:,1],dE[:,0])); hG=np.degrees(np.arctan2(dG[:,1],dG[:,0]))
diff=(hE-hG+180)%360-180
n=len(diff)//10
print('  kare    GT_yon   Est_yon      fark   GT_adim  Est_adim')
for i in range(10):
    sl=slice(i*n,(i+1)*n)
    print('%6d %9.1f %9.1f %9.1f %9.3f %9.3f' % (
        i*n, np.median(hG[sl]), np.median(hE[sl]), np.median(diff[sl]),
        np.median(np.linalg.norm(dG[sl],axis=1)),
        np.median(np.linalg.norm(dE[sl],axis=1))))
print()
print('yon farki  : medyan %+.1f deg   std %.1f' % (np.median(diff), np.std(diff)))
print('adim orani : %.3f' % (np.median(np.linalg.norm(dG,axis=1))/np.median(np.linalg.norm(dE,axis=1))))
