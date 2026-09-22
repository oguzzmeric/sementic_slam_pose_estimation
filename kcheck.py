import csv, numpy as np
gt={}
with open('data/ground-truth.csv',encoding='utf-8') as f:
    for r in csv.DictReader(f):
        z=r.get('translation_z','') or 0.0
        gt[r['frame_numbers'].strip()]=(float(r['translation_x']),
                                        float(r['translation_y']),float(z))
rows=[]
with open('data/trajectory_output.csv',encoding='utf-8') as f:
    for r in csv.DictReader(f):
        if r['is_valid'].strip().lower()!='true': continue
        s=r['frame_name'].rsplit('.',1)[0]
        if s in gt: rows.append((s,float(r['scale']),r['mode'].strip()))
rows.sort()
names=[r[0] for r in rows]; sc=np.array([r[1] for r in rows])
G=np.array([gt[n] for n in names])
dg=np.linalg.norm(np.diff(G,axis=0),axis=1)
Z=41.6-G[1:,2]
k_ideal=dg/np.maximum(Z,1e-9)
k_used=sc[1:]/np.maximum(Z,1e-9)
n=len(k_ideal); step=n//10
print('  kare   k_ideal   k_kullanilan   oran   GT_adim   irtifa')
for i in range(0,n,step):
    sl=slice(i,min(i+step,n))
    ki=np.median(k_ideal[sl]); ku=np.median(k_used[sl])
    print('%6d %9.5f %13.5f %7.3f %9.2f %8.1f'%(
        i,ki,ku,ki/max(ku,1e-9),np.median(dg[sl]),np.median(Z[sl])))
print()
print('k_ideal  : medyan %.5f   min %.5f   maks %.5f'%(
    np.median(k_ideal),k_ideal.min(),k_ideal.max()))
print('degisim orani (maks/min): %.2f'%(k_ideal.max()/max(k_ideal.min(),1e-9)))
