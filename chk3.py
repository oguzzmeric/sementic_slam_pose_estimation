with open('core/motion_estimator.py', encoding='utf-8') as f:
    c = f.read()
i = c.find('X_cam2 = R_cand')
print('--- cheirality ---')
print(c[i-80:i+200])
j = c.find('best_t = t_')
print()
print('--- aday atama ---')
print(c[j-160:j+120])
