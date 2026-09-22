"""KLT'nin neden bazi pencerelerde tamamen coktugunu teshis eder --
her hop'ta kac nokta hayatta kaliyor, ilk karede kac kose bulunuyor."""
import numpy as np
import cv2
from utils.data_loader import DataLoader
from utils.camera_calibration import CameraCalibration
from core.feature_extractor import FeatureExtractor

loader = DataLoader("config.yaml")
cam = CameraCalibration(loader)
extractor = FeatureExtractor(loader, cam)

KLT_WIN = (63, 63)
KLT_MAX_LEVEL = 6
KLT_FB_THRESHOLD = 1.5


def clean_gray_and_mask(frame_bgr, frame_name):
    clean = cam.undistort(frame_bgr)
    gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    detections = loader.get_detections(frame_name)
    mask = extractor._build_semantic_mask(gray.shape, detections)
    return gray, mask


def debug_window(start, end):
    frame_paths = loader.frame_list[start:end + 1]
    frame_names = [p.name for p in frame_paths]
    print(f"pencere [{start}:{end}]  ilk kare={frame_names[0]}  son kare={frame_names[-1]}")

    gray0, mask0 = clean_gray_and_mask(loader.load_frame(frame_paths[0]), frame_names[0])
    mask_frac = float(np.mean(mask0 > 0))
    pts0 = cv2.goodFeaturesToTrack(gray0, maxCorners=400, qualityLevel=0.01, minDistance=12, mask=mask0)
    n0 = 0 if pts0 is None else len(pts0)
    print(f"  ilk kare: maske alani orani={mask_frac:.3f}  bulunan kose sayisi={n0}")
    if pts0 is None or n0 == 0:
        return

    lk_params = dict(winSize=KLT_WIN, maxLevel=KLT_MAX_LEVEL,
                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    tracks = pts0.reshape(-1, 1, 2)
    alive = np.ones(len(tracks), dtype=bool)
    gray_prev = gray0

    for i in range(1, len(frame_paths)):
        gray_curr, _ = clean_gray_and_mask(loader.load_frame(frame_paths[i]), frame_names[i])
        pts_curr, st_fwd, err_fwd = cv2.calcOpticalFlowPyrLK(gray_prev, gray_curr, tracks, None, **lk_params)
        pts_back, st_bwd, _ = cv2.calcOpticalFlowPyrLK(gray_curr, gray_prev, pts_curr, None, **lk_params)
        fb_err = np.linalg.norm((tracks - pts_back).reshape(-1, 2), axis=1)
        ok = (st_fwd.flatten() == 1) & (st_bwd.flatten() == 1) & (fb_err < KLT_FB_THRESHOLD)
        alive &= ok
        med_disp = float(np.median(np.linalg.norm((pts_curr - tracks).reshape(-1, 2), axis=1)))
        print(f"  hop {i:2d}: fwd_ok={int((st_fwd.flatten()==1).sum()):4d}  "
              f"bwd_ok={int((st_bwd.flatten()==1).sum()):4d}  "
              f"fb<thr={int((fb_err<KLT_FB_THRESHOLD).sum()):4d}  "
              f"kumulatif_hayatta={int(alive.sum()):4d}  medyan_kayma={med_disp:.1f}px")
        tracks = pts_curr
        gray_prev = gray_curr


print("=== pencere [7:31] -- longtrack_gtsam.py'de hep 4'te kesilen bolge ===")
debug_window(7, 31)
