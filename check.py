with open('core/scale_recovery.py', encoding='utf-8') as f:
    c = f.read()
print('boyut       :', len(c))
print('k_factor    :', '_k_factor' in c)
print('cos2        :', '_bbox_geometry' in c)
print('max(w,h)    :', 'max(w, h)' in c)
print('DepthEstimate:', 'class DepthEstimate' in c)
