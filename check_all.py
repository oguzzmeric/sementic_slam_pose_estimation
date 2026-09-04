import json, csv, numpy as np
from pathlib import Path

d = json.load(open('data/detections.json', encoding='utf-8'))
frames = list(d.keys())
n_det = [len(v) for v in d.values()]
ref = {2,3,4,5}
with_ref = sum(1 for v in d.values()
               if any(x['class_id'] in ref and x['confidence'] >= 0.7 for x in v))
cls = {}
for v in d.values():
    for x in v:
        cls[x['class_id']] = cls.get(x['class_id'], 0) + 1

print('=== TESPITLER ===')
print(f'toplam kare        : {len(frames)}')
print(f'ilk / son          : {frames[0]} .. {frames[-1]}')
print(f'tespit olan kare   : {sum(1 for n in n_det if n>0)}')
print(f'kare basina tespit : medyan {np.median(n_det):.1f}  max {max(n_det)}')
print(f'referans nesneli   : {with_ref}  (%{100*with_ref/len(frames):.1f})')
print('sinif dagilimi     :', {k: cls[k] for k in sorted(cls)})

gt = set()
with open('data/ground-truth.csv', encoding='utf-8') as f:
    rd = csv.DictReader(f)
    cols = rd.fieldnames
    for r in rd:
        gt.add(r['frame_numbers'].strip())

ext = {'.jpg','.jpeg','.png','.webp'}
fr = [p for p in Path('data/raw_frames').iterdir() if p.suffix.lower() in ext]
stems = {p.stem for p in fr}

print()
print('=== HIZA ===')
print(f'GT sutunlari : {cols}')
print(f'kare dosyasi : {len(fr)}')
print(f'GT kaydi     : {len(gt)}')
print(f'eslesen      : {len(stems & gt)}')
print(f'GT yok       : {len(stems - gt)}')
print(f'kare yok     : {len(gt - stems)}')
print(f'ilk 3 kare   : {sorted(p.name for p in fr)[:3]}')
print(f'ilk 3 GT     : {sorted(gt)[:3]}')
