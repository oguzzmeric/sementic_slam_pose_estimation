"""
Motion-only, pencereli (windowed) bundle adjustment -- v2.

v1 (ilk deneme) SADECE ardisik cift eslesmelerini kullaniyordu -- frame
i-1/i cifti ile i/i+1 cifti FARKLI noktalardi, yani gercek bir coklu-kare
capraz kontrol hic olusmuyordu; sonuc kotulesti (Sim3 ATE 24->68m).

v2: aynı fiziksel noktayi pencere boyunca (frame indeksi zincirleyerek --
her frame'in FeatureExtractor ciktisi bir kere hesaplanip iki ardisik
eslesmede de reuse edildigi icin pikel koordinatlari birebir ayni)
COK-KARELI "track" olarak takip eder. Bir track pencerenin BASINDAN
SONUNA kadar hayatta kalirsa (her ara adimda da inlier eslesme bulunursa)
kullanilir -- kismi track yok, tum-ya-da-hic.

Ayrica dusuk-paralaks (kameralar birbirine cok yakin/nokta cok uzak)
track'ler triangulation'a girmeden elenir (motion_estimator.py'deki
_PARALLAX_COS_THRESHOLD ile ayni esik) -- bunlar gurultulu/yanlis
sinyal veriyordu, v1'de bu gate yoktu.

Optimizasyon: SADECE lokal rotasyonlari (t_local sabit -- olcek/yon
kalibrasyonuna dokunmuyoruz) kucuk bir Lie-cebiri pertürbasyonuyla
(R_local' = R_local @ Exp(delta)) her track'in PENCEREDEKI TUM
karelerdeki reprojection hatasini minimize edecek sekilde duzeltir
(scipy.optimize.least_squares, robust huber loss). Bu, bir rotasyon
hatasinin ayni noktayi 3+ karede ayni anda yanlis yere dusurmesinden
gelen gercek capraz-kontrol sinyalini kullanir -- v1'de bu sinyal yoktu.

core/*.py'ye dokunmuyor -- once diagnostic olarak olculur.
"""

import csv
import logging
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

logging.basicConfig(level=logging.WARNING)

from utils.data_loader import DataLoader
from utils.camera_calibration import CameraCalibration
from core.feature_extractor import FeatureExtractor
from core.matcher import Matcher
from core.motion_estimator import MotionEstimator
from core.scale_recovery import ScaleRecovery
from core.pose_graph import PoseGraph

WINDOW = 6          # pencere buyuklugu (global poz sayisi; W-1 lokal donus optimize edilir)
MAX_TRACKS_PER_WINDOW = 80
REPROJ_F_SCALE = 4.0    # motion_estimator._REPROJ_THRESHOLD ile ayni piksel olcegi
PARALLAX_COS_THRESHOLD = 0.99998  # motion_estimator._PARALLAX_COS_THRESHOLD ile ayni
COORD_TOL_DECIMALS = 2  # pikel koordinat zincirleme icin yuvarlama hassasiyeti

loader = DataLoader("config.yaml")
cam = CameraCalibration(loader)
extractor = FeatureExtractor(loader, cam)
matcher = Matcher(loader)
estimator = MotionEstimator(loader, cam)
scale_rec = ScaleRecovery(loader, cam)
pose_graph = PoseGraph(loader)

K = cam.K

# --------------------------------------------------------------------
# 1) uretim pipeline'ini bir kere calistir, her adimi kaydet
# --------------------------------------------------------------------
steps = []  # dict: valid, R_local, t_local, pts_prev(inlier), pts_curr(inlier), frame_name
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

# --------------------------------------------------------------------
# 2) uretim (duzeltmesiz) global zincir -- referans
# --------------------------------------------------------------------
def chain(local_rotations, local_translations):
    """R_local[i]/t_local[i] listesinden global (R_world, t_world) listesi -- index 0 = kimlik baslangic."""
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


# --------------------------------------------------------------------
# 3) pencereli motion-only BA (v2 -- coklu-kare track + paralaks gate)
# --------------------------------------------------------------------
def triangulate(P1, P2, pt1, pt2):
    A = np.array([
        pt1[0] * P1[2] - P1[0],
        pt1[1] * P1[2] - P1[1],
        pt2[0] * P2[2] - P2[0],
        pt2[1] * P2[2] - P2[1],
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
    """Kamera (world->cam) projeksiyon matrisi K[R^T | -R^T t]."""
    Rt = R_w.T
    return K @ np.hstack([Rt, -Rt @ t_w.reshape(3, 1)])


def build_tracks(start, end):
    """
    steps[start..end-1] ardisik cift eslesmelerini zincirleyip pencere
    BASINDAN SONUNA (n+1 kare) hayatta kalan track'leri dondurur.
    Kismi track yok -- bir ara adimda kaybolan track tamamen dusurulur.
    """
    n = end - start
    if n == 0 or steps[start]["pts_prev"] is None:
        return []

    tracks = [[steps[start]["pts_prev"][i], steps[start]["pts_curr"][i]]
              for i in range(len(steps[start]["pts_prev"]))]

    for i in range(1, n):
        pts_prev = steps[start + i]["pts_prev"]
        pts_curr = steps[start + i]["pts_curr"]
        if pts_prev is None or len(pts_prev) == 0:
            return []  # pencere icinde gecersiz adim -- bu pencereyi atla

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
    """Ilk/son karenin (uretim pozlariyla) tahmini ucgenlemesi -- cok
    paralel isinlar (dusuk paralaks) triangulation'i gurultuye acar."""
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


def window_residuals(params, R_local_win, t_dir_win, t_mag_win, R0, t0, tracks):
    """
    params       : (6*(W-1),) her lokal adim icin [delta_R (3), delta_t_dir (3)]
    R_local_win  : pencere icindeki W-1 orijinal lokal rotasyon
    t_dir_win    : pencere icindeki W-1 lokal OTELEME YONU (birim vektor)
    t_mag_win    : pencere icindeki W-1 lokal ADIM UZUNLUGU -- SABIT, dokunulmuyor
                   (bu zaten k*Z / akis-tabanli kalibrasyondan geliyor)
    R0,t0        : pencere ankrajinin (ilk karenin) global pozu (SABIT,
                   onceki pencerenin DUZELTILMIS sonucu)
    tracks       : her biri (n+1, 2) -- pencerenin TUM karelerinde gorulen
                   ayni fiziksel noktanin piksel gozlemleri
    """
    n = len(R_local_win)
    params = params.reshape(n, 6)
    delta_R = params[:, :3]
    delta_t = params[:, 3:]

    R_world = [R0]
    t_world = [t0]
    for i in range(n):
        dR = Rotation.from_rotvec(delta_R[i]).as_matrix()
        R_corr = R_local_win[i] @ dR

        t_dir_corr = t_dir_win[i] + delta_t[i]
        norm = np.linalg.norm(t_dir_corr)
        if norm > 1e-9:
            t_dir_corr = t_dir_corr / norm
        t_corr = t_dir_corr * t_mag_win[i]

        R_world.append(R_world[-1] @ R_corr)
        t_world.append(t_world[-1] + R_world[-2] @ t_corr)

    Ps = [projection_matrix(R_world[i], t_world[i]) for i in range(n + 1)]

    residuals = []
    for tr in tracks:
        X = triangulate_multiview(Ps, tr)
        if X is None:
            continue
        Xh = np.append(X, 1.0)
        for P, pt in zip(Ps, tr):
            proj = P @ Xh
            if abs(proj[2]) < 1e-6:
                continue
            residuals.append(proj[0] / proj[2] - pt[0])
            residuals.append(proj[1] / proj[2] - pt[1])

    if not residuals:
        return np.zeros(6 * n)  # params'i degistirmeye gerek yok
    return np.array(residuals)


rng = np.random.default_rng(0)
R_local_corr = [r.copy() for r in R_local_all]
t_local_corr = [t.copy() for t in t_local_all]
N = len(steps)

n_windows_used = 0
n_tracks_total = 0

# Ankraj, ONCEKI pencerenin DUZELTILMIS sonucundan birikimli olarak
# tasinir -- ilk pencere kimlikten baslar, sonraki her pencere bir
# oncekinin gercekte cikmis global pozundan devam eder.
R0, t0 = np.eye(3), np.zeros(3)

for start in range(0, N, WINDOW - 1):
    end = min(start + WINDOW - 1, N)
    n = end - start
    if n <= 0:
        continue

    tracks = build_tracks(start, end)
    if not tracks:
        # duzeltme yapilamiyor -- ankraji orijinal (duzeltmesiz) lokal
        # adimlarla ileri tasi, sonraki pencere buradan devam etsin
        for i in range(n):
            t0 = t0 + R0 @ t_local_all[start + i]
            R0 = R0 @ R_local_all[start + i]
        continue

    # paralaks kapisi icin kaba bir son-kare tahmini -- orijinal (duzeltmesiz)
    # lokal adimlari suradaki ankrajdan ileri zincirleyerek
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

    R_local_win = R_local_all[start:end]
    t_mag_win = [float(np.linalg.norm(t)) for t in t_local_all[start:end]]
    t_dir_win = [t_local_all[start + i] / max(t_mag_win[i], 1e-12) for i in range(n)]

    x0 = np.zeros(6 * n)
    result = least_squares(
        window_residuals, x0, method="trf", loss="huber", f_scale=REPROJ_F_SCALE,
        args=(R_local_win, t_dir_win, t_mag_win, R0, t0, tracks),
        max_nfev=300,
    )

    params = result.x.reshape(n, 6)
    for i in range(n):
        dR = Rotation.from_rotvec(params[i, :3]).as_matrix()
        R_local_corr[start + i] = R_local_win[i] @ dR

        t_dir_corr = t_dir_win[i] + params[i, 3:]
        norm = np.linalg.norm(t_dir_corr)
        if norm > 1e-9:
            t_dir_corr = t_dir_corr / norm
        t_local_corr[start + i] = t_dir_corr * t_mag_win[i]

        # ankraji bu DUZELTILMIS adimla ileri tasi -- sonraki pencere buradan baslar
        t0 = t0 + R0 @ t_local_corr[start + i]
        R0 = R0 @ R_local_corr[start + i]

print(f"BA uygulanan pencere = {n_windows_used} / {(N + WINDOW - 2) // (WINDOW - 1)}   "
      f"toplam track = {n_tracks_total}   (ortalama {n_tracks_total / max(n_windows_used,1):.1f}/pencere)")

R_world_ba, t_world_ba = chain(R_local_corr, t_local_corr)


# --------------------------------------------------------------------
# 4) degerlendirme -- uretim vs BA, ayni GT karsilastirmasi
# --------------------------------------------------------------------
gt = {}
with open("data/ground-truth.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        z = r.get("translation_z", "") or 0.0
        gt[r["frame_numbers"].strip()] = np.array(
            [float(r["translation_x"]), float(r["translation_y"]), float(z)])

names = [s["frame_name"] for s in steps]
stems = [n.rsplit(".", 1)[0] for n in names]

# pose_graph zaten Sim3 hizalamasini/GT-raporlamasini uyguladi (tp.position);
# burada karsilastirma icin HAM (world-chain) pozisyonlari kendi Umeyama'imizla
# hizaliyoruz -- flowcheck3.py'deki yontemle ayni.
prod_pos = np.array(t_world_prod[1:])   # index 0 = kimlik baslangic, atla
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
print(f"  n={len(gt_arr)} kare, pencere={WINDOW}")
print("=" * 66)
report("URETIM (duzeltmesiz)", prod_arr)
report("BA (pencereli motion-only, rotasyon-sadece)", ba_arr)
