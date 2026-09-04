import csv, json, numpy as np
gt={}
with open('data/ground-truth.csv',encoding='utf-8') as f:
    for r in csv.DictReader(f):
        gt[r['frame_numbers'].strip()]=float(r['translation_z'])

d=json.load(open('data/detections.json',encoding='utf-8'))
fy, cx, cy, f_ = 2683.39, 1908.48, 1139.71, np.sqrt(2680.51*2683.39)
ref={2:1.5,3:4.5,4:7.0,5:12.0}

pairs=[]
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
        if best is None or q>best[0]: best=(q,z)
    if best: pairs.append((best[1], gt[stem]))

a=np.array(pairs)
off=a[:,0]-a[:,1]
print(f'ornek           : {len(a)}')
print(f'bbox Z          : medyan {np.median(a[:,0]):7.2f}  std {np.std(a[:,0]):6.2f}')
print(f'GT Z (bagil)    : medyan {np.median(a[:,1]):7.2f}  std {np.std(a[:,1]):6.2f}')
print()
print('=== OFSET HIPOTEZI:  Z_bbox = Z0 + Z_GT ===')
print(f'ofset Z0        : medyan {np.median(off):7.2f}  ortalama {np.mean(off):7.2f}  std {np.std(off):6.2f}')
print(f'korelasyon      : {np.corrcoef(a[:,0],a[:,1])[0,1]:+.4f}')
print()
print('std dusukse ve korelasyon pozitifse hipotez tutar.')
