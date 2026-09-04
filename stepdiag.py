from utils.data_loader import DataLoader
loader = DataLoader('config.yaml')
cfg = loader.get_evaluation_config()
print('frame_step config :', cfg.get('frame_step', 'YOK'))
print('kare sayisi       :', loader.total_frames)
print('ilk 5 kare        :', [p.name for p in loader.frame_list[:5]])

with open('utils/data_loader.py', encoding='utf-8') as f:
    c = f.read()
print()
print('kodda frame_step  :', 'frame_step' in c)
i = c.find('frame_step')
if i > 0:
    print(c[i-300:i+250])
