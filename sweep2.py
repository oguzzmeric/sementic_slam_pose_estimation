import subprocess, yaml, re, sys, csv
import numpy as np

def raw_error():
    est, gt_idx = [], {}
    with open('data/ground-truth.csv', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            z = r.get('translation_z','') or 0.0
            gt_idx[r['frame_numbers'].strip()] = (
                float(r['translation_x']), float(r['translation_y']), float(z))
    with open('data/trajectory_output.csv', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['is_valid'].strip().lower() != 'true':
                continue
            stem = r['frame_name'].rsplit('.',1)[0]
            if stem not in gt_idx:
                continue
            est.append(((float(r['x']),float(r['y']),float(r['z'])), gt_idx[stem]))
    if len(est) < 2:
        return None
    e = np.array([a for a,b in est]); g = np.array([b for a,b in est])
    e = e - e[0]; g = g - g[0]
    return float(np.sqrt(np.mean(np.sum((e-g)**2, axis=1))))

for swap, flip in [(False,False),(True,False),(False,True),(True,True)]:
    cfg = yaml.safe_load(open('config.yaml', encoding='utf-8'))
    cfg['evaluation']['swap_xy'] = swap
    cfg['evaluation']['flip_y']  = flip
    yaml.safe_dump(cfg, open('config.yaml','w',encoding='utf-8'),
                   allow_unicode=True, sort_keys=False)
    subprocess.run([sys.executable,'main.py','--max-frames','400','--no-viz'],
                   capture_output=True)
    print(f'swap={swap!s:5s} flip={flip!s:5s}  hizalamasiz ATE = {raw_error():.2f} m')
