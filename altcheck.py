from utils.data_loader import DataLoader
from utils.camera_calibration import CameraCalibration
from core.scale_recovery import ScaleRecovery

loader = DataLoader('config.yaml')
cfg = loader.get_evaluation_config()
print('=== CONFIG ===')
for k in ('altitude_source','gt_z_is_down','gt_z_offset','warmup_start','warmup_frames'):
    print('  ', k, ':', cfg.get(k, 'YOK'))

cam = CameraCalibration(loader)
sr = ScaleRecovery(loader, cam)
print()
print('=== NESNE ===')
print('   _altitude_source :', sr._altitude_source)
print('   _gt_z_down       :', getattr(sr, '_gt_z_down', 'ATTR YOK'))
print('   _gt_z_offset     :', getattr(sr, '_gt_z_offset', 'ATTR YOK'))
print()
print('=== ORNEK KARELER ===')
for name in ['frame_000000.webp','frame_000400.webp','frame_001000.webp']:
    g = loader.get_ground_truth(name)
    z = sr._gt_altitude(name)
    print('  ', name, ' tz=', g.tz if g else None, ' -> irtifa=', z)
