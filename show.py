with open('core/pose_graph.py', encoding='utf-8') as f:
    c=f.read()
i=c.find('t_vec = scale_result')
print(c[i:i+700])
