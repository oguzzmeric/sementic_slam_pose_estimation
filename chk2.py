with open('core/motion_estimator.py', encoding='utf-8') as f:
    c = f.read()
i = c.find('if best_R is None or best_votes')
print(c[i:i+500])
