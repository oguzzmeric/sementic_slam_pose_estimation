import csv, numpy as np
rows=[]
with open('data/ground-truth.csv',encoding='utf-8') as f:
    for r in csv.DictReader(f):
        rows.append((float(r['translation_x']),float(r['translation_y']),
                     float(r['translation_z'])))
a=np.array(rows)
d=np.linalg.norm(np.diff(a,axis=0),axis=1)

print('=== ARALIKLAR ===')
for i,ax in enumerate('XYZ'):
    print(f'{ax}: [{a[:,i].min():9.4f}, {a[:,i].max():9.4f}]  aralik {a[:,i].max()-a[:,i].min():8.4f}')
print()
print(f'kare arasi adim : medyan {np.median(d):.6f}  ortalama {np.mean(d):.6f}')
print(f'toplam yol      : {d.sum():.4f}')
print(f'bas-son mesafe  : {np.linalg.norm(a[-1]-a[0]):.4f}')
print()
print('=== BIRIM HIPOTEZLERI ===')
print('bbox irtifa ~47 m oldugunu varsayarsak:')
for name,ref in [('Z medyani', np.median(np.abs(a[:,2]))),
                 ('Z maks',    np.abs(a[:,2]).max())]:
    if ref>1e-9:
        print(f'  {name:10s} = {ref:8.4f}  ->  carpan {47.0/ref:9.2f}')
print()
print(f'ilk 5 Z : {a[:5,2]}')
print(f'son 5 Z : {a[-5:,2]}')
