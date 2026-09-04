import numpy as np, csv
p=[]
with open('data/trajectory_output.csv',encoding='utf-8') as f:
    for r in csv.DictReader(f):
        if r['is_valid'].strip().lower()=='true':
            p.append((float(r['x']),float(r['y']),float(r['scale'])))
a=np.array(p)
step=np.linalg.norm(np.diff(a[:,:2],axis=0),axis=1)
sc=a[1:,2]
print(f'uygulanan olcek medyani : {np.median(sc):.4f}')
print(f'gerceklesen adim medyani: {np.median(step):.4f} m')
print(f'kayip orani             : {np.median(sc)/np.median(step):.4f}')
print()
m=step>1e-9
print(f'adim/olcek orani medyan : {np.median(step[m]/sc[m]):.4f}')
print(f'sifir adim sayisi       : {int((step<1e-9).sum())} / {len(step)}')
