import csv, json, numpy as np
gt={}
with open('data/ground-truth.csv',encoding='utf-8') as f:
    for r in csv.DictReader(f):
        gt[r['frame_numbers'].strip()]=float(r['translation_z'])

d=json.load(open('data/detections.json',encoding='utf-8'))
fy,cx,cy = 2683.39,1908.48,1139.71
f_=np.sqrt(2680.51*2683.39)
ref={2:1.5,3:4.5,4:7.0,5:12.0}

rows=[]
for name,dets in d.items():
    stem=name.rsplit('.',1)[0]
    if stem not in gt: continue
    best=None
    for x in dets:
        if x['class_id'] not in ref or x['confidence']<0.7: continue
        x1,y1,x2,y2=x['bbox']; w,h=abs(x2-x1),abs(y2-y1)
        if w<12 or h<12: continue
        dpx=max(w,h)
        rad=np.hypot((x1+x2)/2-cx,(y1+y2)/2-cy)
        c2=np.cos(np.arctan2(rad,f_))**2
        z=(fy*ref[x['class_id']])/dpx*c2
        if not (1<=z<=500): continue
        q=x['confidence']*c2
        if best is None or q>best[0]: best=(q,z,x['class_id'])
    if best: rows.append((int(stem.split('_')[1]), best[1], gt[stem], best[2]))

rows.sort()
a=np.array([(r[0],r[1],r[2]) for r in rows])
cls=np.array([r[3] for r in rows])

print(f'ornek : {len(a)}')
print(f'korelasyon  Z_bbox vs +Z_GT : {np.corrcoef(a[:,1], a[:,2])[0,1]:+.4f}')
print(f'korelasyon  Z_bbox vs -Z_GT : {np.corrcoef(a[:,1],-a[:,2])[0,1]:+.4f}')
print()
off_neg = a[:,1] + a[:,2]
print('=== NED HIPOTEZI:  Z_bbox = Z0 - Z_GT ===')
print(f'ofset : medyan {np.median(off_neg):7.2f}  std {np.std(off_neg):6.2f}')
print()
print('=== ZAMAN PROFILI (10 dilim) ===')
n=len(a)//10
print(f"{'kare':>7} {'Z_bbox':>9} {'Z_GT':>9} {'sinif':>7}")
for i in range(10):
    s=a[i*n:(i+1)*n]; c=cls[i*n:(i+1)*n]
    vals,cnts=np.unique(c,return_counts=True)
    print(f'{int(s[0,0]):7d} {np.median(s[:,1]):9.2f} {np.median(s[:,2]):9.2f} {int(vals[cnts.argmax()]):7d}')
print()
sc=np.array([(c==k).sum() for k in [2,3,4,5]])
print(f'sinif dagilimi 2/3/4/5 : {sc}')
