with open('core/motion_estimator.py', encoding='utf-8') as f:
    c = f.read()
i = c.find('for i in range(num_solutions)')
print(c[i:i+700])
