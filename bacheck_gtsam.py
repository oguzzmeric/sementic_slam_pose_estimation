"""
Motion-only, pencereli (windowed) bundle adjustment -- GTSAM portu.

bacheck.py'deki scipy denemelerinin (v1-v4) aynisi ama optimizasyon
backend'i GTSAM. Track kurma + dusuk-paralaks gate MANTIGI degismedi --
sadece "hangi hesap makinesiyle cozuyoruz" degisti.

v4'te (scipy) bulunan somut sorun: regularizasyon (orijinal tahminden
uzaklasma cezasi) ve reprojection hatasi AYNI huber f_scale'i
paylasiyordu, mutlak olcekleri tutarsizdi. GTSAM'de her faktor tipi
KENDI gurultu modelini tasir (bilgi matrisi uzerinden birlesir), bu
sorunu yapisal olarak ortadan kaldirir:

  - ProjectionFactor      : piksel gurultusu (PIXEL_NOISE_SIGMA, sabit
                            fiziksel varsayim -- ORB eslesme tipik
                            piksel dogrulugu, datasete ozel degil)
  - PriorFactor (pose)    : "orijinal tahminden uzaklasma" cezasi,
                            sigma = PIXEL_NOISE_SIGMA / (f * sqrt(N))
                            radyan -- N = o adimin ORIJINAL inlier
                            sayisi. Ayni prensip (std hata ~ 1/sqrt(N))
                            ama artik kendi radyan biriminde, piksel
                            residual'larla paylasilan bir olcek yok.

Rotasyon matematigi de artik elle degil -- Pose3 GTSAM'in kendi SE(3)
manifold'unda dogru sekilde retract/compose yapiyor.
"""

import csv
import logging
import numpy as np
import gtsam
from gtsam import Pose3, Rot3, Point3, Cal3_S2
from gtsam.symbol_shorthand import X, L

logging.basicConfig(level=logging.WARNING)

from utils.data_loader import DataLoader
from utils.camera_calibration import CameraCalibration
from core.feature_extractor import FeatureExtractor
from core.matcher import Matcher
from core.motion_estimator import MotionEstimator
from core.scale_recovery import ScaleRecovery
from core.pose_graph import PoseGraph

WINDOW = 6
MAX_TRACKS_PER_WINDOW = 80
PARALLAX_COS_THRESHOLD = 0.99998
COORD_TOL_DECIMALS = 2
PIXEL_NOISE_SIGMA = 1.5   # tek fiziksel varsayim -- ORB eslesme tipik piksel gurultusu
ANCHOR_SIGMA = 1e-6       # pencerenin ilk karesi -- fiilen sabit

loader = DataLoader("config.yaml")
cam = CameraCalibration(loader)
extractor = FeatureExtractor(loader, cam)
matcher = Matcher(loader)
estimator = MotionEstimator(loader, cam)
scale_rec = ScaleRecovery(loader, cam)
pose_graph = PoseGraph(loader)

K = cam.K
F_FOCAL = float(np.sqrt(cam.fx * cam.fy))
gtsam_K = Cal3_S2(cam.fx, cam.fy, 0.0, cam.cx, cam.cy)

# --------------------------------------------------------------------
# 1) uretim pipeline'ini bir kere calistir, her adimi kaydet
# --------------------------------------------------------------------
steps = []
prev_features = None

for idx, name, frame in loader.frame_generator():
    curr = extractor.extract(frame, name)
    if prev_features is not None:
        mr = matcher.match(prev_features, curr)
        pose = estimator.estimate(mr)
        sr = scale_rec.recover(pose, name, match_result=mr)
        tp = pose_graph.update(pose, sr, name)

        if pose.is_valid and sr.is_valid and pose.inlier_mask is not None:
            m = pose.inlier_mask.astype(bool)
            steps.append(dict(
                valid=True, frame_name=name,
                R_local=pose.R.copy(), t_local=sr.t_scaled.flatten().copy(),
                pts_prev=mr.pts_prev[m], pts_curr=mr.pts_curr[m],
            ))
        else:
            steps.append(dict(valid=False, frame_name=name,
                               R_local=np.eye(3), t_local=np.zeros(3),
                               pts_prev=None, pts_curr=None))
    prev_features = curr

print(f"toplam adim = {len(steps)}  (gecerli = {sum(s['valid'] for s in steps)})")


def chain(local_rotations, local_translations):
    N = len(local_rotations)
    R_world = [np.eye(3)]
    t_world = [np.zeros(3)]
    for i in range(N):
        R_world.append(R_world[-1] @ local_rotations[i])
        t_world.append(t_world[-1] + R_world[-2] @ local_translations[i])
    return R_world, t_world


R_local_all = [s["R_local"] for s in steps]
t_local_all = [s["t_local"] for s in steps]
R_world_prod, t_world_prod = chain(R_local_all, t_local_all)


def triangulate(P1, P2, pt1, pt2):
    A = np.array([
        pt1[0] * P1[2] - P1[0], pt1[1] * P1[2] - P1[1],
        pt2[0] * P2[2] - P2[0], pt2[1] * P2[2] - P2[1],
    ])
    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[-1]
    if abs(Xh[3]) < 1e-9:
        return None
    return Xh[:3] / Xh[3]


def triangulate_multiview(Ps, pts):
    rows = []
    for P, pt in zip(Ps, pts):
        rows.append(pt[0] * P[2] - P[0])
        rows.append(pt[1] * P[2] - P[1])
    A = np.array(rows)
    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[-1]
    if abs(Xh[3]) < 1e-9:
        return None
    return Xh[:3] / Xh[3]


def projection_matrix(R_w, t_w):
    Rt = R_w.T
    return K @ np.hstack([Rt, -Rt @ t_w.reshape(3, 1)])


def build_tracks(start, end):
    n = end - start
    if n == 0 or steps[start]["pts_prev"] is None:
        return []
    tracks = [[steps[start]["pts_prev"][i], steps[start]["pts_curr"][i]]
              for i in range(len(steps[start]["pts_prev"]))]
    for i in range(1, n):
        pts_prev = steps[start + i]["pts_prev"]
        pts_curr = steps[start + i]["pts_curr"]
        if pts_prev is None or len(pts_prev) == 0:
            return []
        lookup = {}
        for j in range(len(pts_prev)):
            key = (round(float(pts_prev[j, 0]), COORD_TOL_DECIMALS),
                   round(float(pts_prev[j, 1]), COORD_TOL_DECIMALS))
            lookup[key] = j
        new_tracks = []
        for tr in tracks:
            last = tr[-1]
            key = (round(float(last[0]), COORD_TOL_DECIMALS),
                   round(float(last[1]), COORD_TOL_DECIMALS))
            j = lookup.get(key)
            if j is not None:
                new_tracks.append(tr + [pts_curr[j]])
        tracks = new_tracks
        if not tracks:
            break
    return [np.array(t) for t in tracks if len(t) == n + 1]


def parallax_ok(track, R_first, t_first, R_last, t_last):
    P_first = projection_matrix(R_first, t_first)
    P_last = projection_matrix(R_last, t_last)
    X = triangulate(P_first, P_last, track[0], track[-1])
    if X is None:
        return False
    ray1, ray2 = X - t_first, X - t_last
    d1, d2 = np.linalg.norm(ray1), np.linalg.norm(ray2)
    if d1 < 1e-9 or d2 < 1e-9:
        return False
    cos_parallax = float(np.dot(ray1, ray2) / (d1 * d2))
    return cos_parallax <= PARALLAX_COS_THRESHOLD


# --------------------------------------------------------------------
# 2) pencereli motion-only BA -- GTSAM factor graph
# --------------------------------------------------------------------
def solve_window(R_init, t_init, tracks, n_inliers_win):
    """
    R_init, t_init : pencerenin n+1 karesi icin BASLANGIC (uretim) global poz
                     tahmini -- index 0 sabit ankraj.
    tracks         : her biri (n+1, 2) piksel gozlemi.
    n_inliers_win  : n+1 kare icin degil, n LOKAL ADIM icin orijinal inlier
                     sayisi -- prior sigma'nin guven agirligi.
    """
    n = len(R_init) - 1
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    pixel_noise = gtsam.noiseModel.Isotropic.Sigma(2, PIXEL_NOISE_SIGMA)
    anchor_noise = gtsam.noiseModel.Isotropic.Sigma(6, ANCHOR_SIGMA)

    for i in range(n + 1):
        pose_i = Pose3(Rot3(R_init[i]), Point3(t_init[i]))
        initial.insert(X(i), pose_i)
        if i == 0:
            graph.add(gtsam.PriorFactorPose3(X(i), pose_i, anchor_noise))
        else:
            rot_sigma = PIXEL_NOISE_SIGMA / (F_FOCAL * np.sqrt(max(n_inliers_win[i - 1], 1)))
            step_len = float(np.linalg.norm(t_init[i] - t_init[i - 1]))
            trans_sigma = max(step_len * rot_sigma, 1e-4)
            sigmas = np.array([rot_sigma] * 3 + [trans_sigma] * 3)
            graph.add(gtsam.PriorFactorPose3(
                X(i), pose_i, gtsam.noiseModel.Diagonal.Sigmas(sigmas)))

    Ps_init = [projection_matrix(R_init[i], t_init[i]) for i in range(n + 1)]
    n_landmarks = 0
    for tr in tracks:
        Xw = triangulate_multiview(Ps_init, tr)
        if Xw is None:
            continue
        initial.insert(L(n_landmarks), Point3(Xw))
        for i, pt in enumerate(tr):
            graph.add(gtsam.GenericProjectionFactorCal3_S2(
                pt.astype(np.float64), pixel_noise, X(i), L(n_landmarks), gtsam_K))
        n_landmarks += 1

    if n_landmarks == 0:
        return R_init, t_init

    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(50)
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    result = optimizer.optimize()

    R_out, t_out = [], []
    for i in range(n + 1):
        p = result.atPose3(X(i))
        R_out.append(p.rotation().matrix())
        t_out.append(p.translation())
    return R_out, t_out


rng = np.random.default_rng(0)
R_local_corr = [r.copy() for r in R_local_all]
t_local_corr = [t.copy() for t in t_local_all]
N = len(steps)

n_windows_used = 0
n_tracks_total = 0
R0, t0 = np.eye(3), np.zeros(3)

for start in range(0, N, WINDOW - 1):
    end = min(start + WINDOW - 1, N)
    n = end - start
    if n <= 0:
        continue

    tracks = build_tracks(start, end)
    if not tracks:
        for i in range(n):
            t0 = t0 + R0 @ t_local_all[start + i]
            R0 = R0 @ R_local_all[start + i]
        continue

    R_last, t_last = R0.copy(), t0.copy()
    for i in range(n):
        t_last = t_last + R_last @ t_local_all[start + i]
        R_last = R_last @ R_local_all[start + i]

    tracks = [tr for tr in tracks if parallax_ok(tr, R0, t0, R_last, t_last)]
    if not tracks:
        R0, t0 = R_last, t_last
        continue

    if len(tracks) > MAX_TRACKS_PER_WINDOW:
        idx = rng.choice(len(tracks), MAX_TRACKS_PER_WINDOW, replace=False)
        tracks = [tracks[i] for i in idx]

    n_windows_used += 1
    n_tracks_total += len(tracks)

    # baslangic tahmini: ankrajdan orijinal lokal adimlarla ileri zincirle
    R_init = [R0.copy()]
    t_init = [t0.copy()]
    for i in range(n):
        t_init.append(t_init[-1] + R_init[-1] @ t_local_all[start + i])
        R_init.append(R_init[-1] @ R_local_all[start + i])

    n_inliers_win = [len(steps[start + i]["pts_prev"]) for i in range(n)]

    R_out, t_out = solve_window(R_init, t_init, tracks, n_inliers_win)

    # duzeltilmis global pozlardan LOKAL adimlari geri cikar
    for i in range(n):
        R_local_corr[start + i] = R_out[i].T @ R_out[i + 1]
        t_local_corr[start + i] = R_out[i].T @ (t_out[i + 1] - t_out[i])

    R0, t0 = R_out[-1], t_out[-1]

print(f"BA uygulanan pencere = {n_windows_used} / {(N + WINDOW - 2) // (WINDOW - 1)}   "
      f"toplam track = {n_tracks_total}   (ortalama {n_tracks_total / max(n_windows_used,1):.1f}/pencere)")

R_world_ba, t_world_ba = chain(R_local_corr, t_local_corr)

# --------------------------------------------------------------------
# 3) degerlendirme -- uretim vs BA, ayni GT karsilastirmasi
# --------------------------------------------------------------------
gt = {}
with open("data/ground-truth.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        z = r.get("translation_z", "") or 0.0
        gt[r["frame_numbers"].strip()] = np.array(
            [float(r["translation_x"]), float(r["translation_y"]), float(z)])

names = [s["frame_name"] for s in steps]
stems = [n.rsplit(".", 1)[0] for n in names]

prod_pos = np.array(t_world_prod[1:])
ba_pos = np.array(t_world_ba[1:])

mask = np.array([s in gt for s in stems])
gt_arr = np.array([gt[s] for s in np.array(stems)[mask]])
prod_arr = prod_pos[mask]
ba_arr = ba_pos[mask]


def umeyama(P, Q):
    P = P.T; Q = Q.T
    n = P.shape[1]
    mu_p = P.mean(axis=1, keepdims=True)
    mu_q = Q.mean(axis=1, keepdims=True)
    Pc = P - mu_p; Qc = Q - mu_q
    W = Qc @ Pc.T / n
    U, D, Vt = np.linalg.svd(W)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_p = (Pc ** 2).sum() / n
    s = np.trace(np.diag(D) @ S) / var_p if var_p > 1e-12 else 1.0
    t = mu_q - s * R @ mu_p
    return s, R, t


def report(label, est):
    s, R, t = umeyama(est, gt_arr)
    est_al = (s * R @ est.T + t).T
    err_al = np.linalg.norm(gt_arr - est_al, axis=1)
    err_raw = np.linalg.norm(gt_arr - est, axis=1)
    print(f"--- {label} ---")
    print(f"  Sim(3) ATE (rmse)                   : {np.sqrt((err_al**2).mean()):7.2f} m   olcek={s:.4f}")
    print(f"  HIZALANMAMIS mean (yarisma metrigi) : {err_raw.mean():7.2f} m")
    print()


print("=" * 66)
print(f"  n={len(gt_arr)} kare, pencere={WINDOW}, GTSAM")
print("=" * 66)
report("URETIM (duzeltmesiz)", prod_arr)
report("BA (GTSAM, coklu-kare track + guven-agirlikli prior)", ba_arr)
