import subprocess, yaml, re, sys

combos = [(False,False),(True,False),(False,True),(True,True)]
results = []

for swap, flip in combos:
    cfg = yaml.safe_load(open('config.yaml', encoding='utf-8'))
    cfg['evaluation']['swap_xy'] = swap
    cfg['evaluation']['flip_y']  = flip
    yaml.safe_dump(cfg, open('config.yaml','w',encoding='utf-8'),
                   allow_unicode=True, sort_keys=False)

    subprocess.run([sys.executable,'main.py','--max-frames','400','--no-viz'],
                   capture_output=True)
    out = subprocess.run([sys.executable,'-m','utils.evaluate_evo'],
                         capture_output=True, text=True).stdout

    m = re.search(r'Sim\(3\).*?ATE\s*=\s*([\d.]+)', out)
    k = re.search(r'scale factor\s*s\s*=\s*([\d.]+)', out)
    ate = float(m.group(1)) if m else None
    sc  = float(k.group(1)) if k else None
    results.append((swap, flip, ate, sc))
    print(f'swap={swap!s:5s} flip={flip!s:5s}  ATE={ate}  scale={sc}')

print()
ok = [r for r in results if r[2]]
if ok:
    b = min(ok, key=lambda r: r[2])
    print(f'EN IYI: swap_xy={b[0]}  flip_y={b[1]}  ATE={b[2]:.2f}  scale={b[3]:.3f}')
