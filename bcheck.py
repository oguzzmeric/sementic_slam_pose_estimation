with open('core/motion_estimator.py', encoding='utf-8') as f:
    c = f.read()

i = c.find('findEssentialMat')
print('=== findEssentialMat cagrisi ===')
print(c[i-60:i+420])
