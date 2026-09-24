"""
core/pose_graph.py
===================
Pose graph and trajectory accumulation module.

Local VO chain:
    T_world = T_world * T_local
    T_local = [R | t_scaled]
              [0 |     1   ]
    raw_position = T_world[:3, 3]

The chain above lives in an arbitrary "local" frame -- its origin and axis
orientation come from the first camera pose, not from the GT/world frame.
Reporting raw_position directly (or guessing a fixed axis swap) only works
by luck. Instead, while GT is available (scale_result.mode == "warmup"),
every (raw_position, GT) pair is collected; the moment GT stops being
available a Umeyama similarity transform (scale + rotation + translation,
i.e. Sim(3)) is fit once from those pairs and reused for every subsequent
frame:

    position = s * R_align @ raw_position + t_align

This replaces the old swap_xy/flip_y flags (a hand-guessed 90-degree axis
swap, fit to one dataset) with a proper least-squares fit -- and gives a
natural hook for relocalization: whenever GT reappears later, the same
mechanism can refit and reset accumulated drift.

Coordinate frame flags (config.yaml evaluation section):
    force_2d : zero out Z accumulation (per-step, independent of the fit)

ATE:
    ATE = sqrt(1/N * sum(||t_GT_i - t_est_i||^2))
"""

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from core.motion_estimator import PoseEstimate
from core.scale_recovery import ScaleResult
from utils.data_loader import DataLoader

logger = logging.getLogger(__name__)


class PoseGraphError(Exception):
    pass


# ----------------------------------------------------------------------
# refine_with_persistent_map -- geometrik yardimcilar (self'e ihtiyaci yok)
# ----------------------------------------------------------------------

def _triangulate(P1: np.ndarray, P2: np.ndarray, pt1: np.ndarray, pt2: np.ndarray) -> Optional[np.ndarray]:
    A = np.array([
        pt1[0] * P1[2] - P1[0], pt1[1] * P1[2] - P1[1],
        pt2[0] * P2[2] - P2[0], pt2[1] * P2[2] - P2[1],
    ])
    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[-1]
    if abs(Xh[3]) < 1e-9:
        return None
    return Xh[:3] / Xh[3]


def _triangulate_multiview(Ps: list, pts: list) -> Optional[np.ndarray]:
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


def _projection_matrix(K: np.ndarray, R_w: np.ndarray, t_w: np.ndarray) -> np.ndarray:
    Rt = R_w.T
    return K @ np.hstack([Rt, -Rt @ t_w.reshape(3, 1)])


def _parallax_ok(K: np.ndarray, track: dict, R_init: list, t_init: list, cos_threshold: float) -> bool:
    """track kendi (start..start+len-1) araligindaki ilk/son gozlemle
    paralaksini kontrol eder -- pencerenin globalinden degil, kendi
    yasadigi araliktan."""
    s = track["start"]
    e = s + len(track["obs"]) - 1
    R_first, t_first = R_init[s], t_init[s]
    R_last, t_last = R_init[e], t_init[e]
    P_first = _projection_matrix(K, R_first, t_first)
    P_last = _projection_matrix(K, R_last, t_last)
    Xw = _triangulate(P_first, P_last, np.array(track["obs"][0]), np.array(track["obs"][-1]))
    if Xw is None:
        return False
    ray1, ray2 = Xw - t_first, Xw - t_last
    d1, d2 = np.linalg.norm(ray1), np.linalg.norm(ray2)
    if d1 < 1e-9 or d2 < 1e-9:
        return False
    cos_parallax = float(np.dot(ray1, ray2) / (d1 * d2))
    return cos_parallax <= cos_threshold


def _slice_track_to_window(track: dict, w_start: int, w_end: int, min_len: int) -> Optional[dict]:
    """Global (tum-ucus) track'in [w_start,w_end] penceresiyle kesisen
    kismini pencere-lokal indekslerle dondurur -- track'in KENDISI
    pencere sinirini asabilir, burada sadece bu blogun optimizasyonu
    icin gereken parca kesiliyor."""
    s = track["start"]
    e = s + len(track["obs"]) - 1
    lo, hi = max(s, w_start), min(e, w_end)
    if lo > hi or (hi - lo + 1) < min_len:
        return None
    obs = track["obs"][lo - s: hi - s + 1]
    return {"start": lo - w_start, "obs": obs}


def _build_full_flight_tracks(loader: DataLoader, cam, extractor, cfg: dict) -> list:
    """
    TUM UCUS boyunca SUREKLI KLT takibi -- pencere siniri yok. Aktif
    track sayisi azalinca besleme yapilir (yeni kose eklenir). Bir
    track sadece takip basarisiz oldugunda (KLT kaybettiginde)
    sonlanir, herhangi bir optimizasyon penceresi bittigi icin degil
    -- kalici harita mantigi budur.

    Doner: {'start': kare indeksi, 'obs': [(x,y), ...]} sozluk listesi.
    """
    replenish_threshold = int(cfg["replenish_threshold"])
    replenish_exclude_radius = int(cfg["replenish_exclude_radius"])
    klt_win = (int(cfg["klt_win_size"]), int(cfg["klt_win_size"]))
    klt_max_level = int(cfg["klt_max_level"])
    klt_fb_threshold = float(cfg["klt_fb_threshold"])

    def clean_gray_and_mask(frame_bgr, frame_name):
        clean = cam.undistort(frame_bgr)
        gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
        detections = loader.get_detections(frame_name)
        mask = extractor._build_semantic_mask(gray.shape, detections)
        return gray, mask

    frame_paths = loader.frame_list
    finished = []

    gray_prev, mask_prev = clean_gray_and_mask(loader.load_frame(frame_paths[0]), frame_paths[0].name)
    pts0 = cv2.goodFeaturesToTrack(gray_prev, maxCorners=400, qualityLevel=0.01, minDistance=12, mask=mask_prev)
    active = [{"start": 0, "obs": [tuple(p[0])]} for p in pts0] if pts0 is not None else []

    lk_params = dict(winSize=klt_win, maxLevel=klt_max_level,
                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    for i in range(1, len(frame_paths)):
        gray_curr, mask_curr = clean_gray_and_mask(loader.load_frame(frame_paths[i]), frame_paths[i].name)

        if active:
            pts_in = np.array([a["obs"][-1] for a in active], dtype=np.float32).reshape(-1, 1, 2)
            pts_curr, st_fwd, _ = cv2.calcOpticalFlowPyrLK(gray_prev, gray_curr, pts_in, None, **lk_params)
            pts_back, st_bwd, _ = cv2.calcOpticalFlowPyrLK(gray_curr, gray_prev, pts_curr, None, **lk_params)
            fb_err = np.linalg.norm((pts_in - pts_back).reshape(-1, 2), axis=1)
            ok = (st_fwd.flatten() == 1) & (st_bwd.flatten() == 1) & (fb_err < klt_fb_threshold)

            new_active = []
            for j, a in enumerate(active):
                if ok[j]:
                    a["obs"].append(tuple(pts_curr[j, 0, :]))
                    new_active.append(a)
                elif len(a["obs"]) >= 2:
                    finished.append(a)
            active = new_active

        if len(active) < replenish_threshold:
            excl_mask = mask_curr.copy()
            for a in active:
                x, y = a["obs"][-1]
                cv2.circle(excl_mask, (int(round(x)), int(round(y))), replenish_exclude_radius, 0, -1)
            n_needed = 400 - len(active)
            if n_needed > 0:
                new_pts = cv2.goodFeaturesToTrack(
                    gray_curr, maxCorners=n_needed, qualityLevel=0.01, minDistance=12, mask=excl_mask)
                if new_pts is not None:
                    for p in new_pts:
                        active.append({"start": i, "obs": [tuple(p[0])]})

        gray_prev = gray_curr
        if (i % 50) == 0:
            logger.info("[PoseGraph] persistent-map track: kare %d/%d, aktif=%d, bitmis=%d",
                        i, len(frame_paths) - 1, len(active), len(finished))

    for a in active:
        if len(a["obs"]) >= 2:
            finished.append(a)

    return finished


def _solve_window_gtsam(K: np.ndarray, f_focal: float, R_init: list, t_init: list,
                         tracks: list, confidence: int, cfg: dict):
    """Bir pencerenin (R_init/t_init baslangic tahminleri + track'ler)
    GTSAM factor graph'i ile duzeltilmis hali. GTSAM Windows'ta PyPI
    wheel'i olmadigi icin lazy import -- bu fonksiyon cagrilmadikca
    (yani refine_with_persistent_map hic cagrilmadikca) gtsam hic
    gerekmez, normal main.py/Windows akisi bundan etkilenmez."""
    try:
        import gtsam
        from gtsam import Pose3, Rot3, Point3, Cal3_S2
        from gtsam.symbol_shorthand import X, L
    except ImportError as e:
        raise PoseGraphError(
            "refine_with_persistent_map GTSAM gerektiriyor. GTSAM'in PyPI'da "
            "Windows wheel'i yok -- bu metodu WSL/Linux'ta calistirin "
            "(pip install gtsam manylinux wheel ile calisir)."
        ) from e

    gtsam_K = Cal3_S2(K[0, 0], K[1, 1], 0.0, K[0, 2], K[1, 2])
    pixel_noise_sigma = float(cfg["pixel_noise_sigma"])
    anchor_sigma = float(cfg["anchor_sigma"])

    n = len(R_init) - 1
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()
    pixel_noise = gtsam.noiseModel.Isotropic.Sigma(2, pixel_noise_sigma)
    anchor_noise = gtsam.noiseModel.Isotropic.Sigma(6, anchor_sigma)

    for i in range(n + 1):
        pose_i = Pose3(Rot3(R_init[i]), Point3(t_init[i]))
        initial.insert(X(i), pose_i)
        if i == 0:
            graph.add(gtsam.PriorFactorPose3(X(i), pose_i, anchor_noise))
        else:
            rot_sigma = pixel_noise_sigma / (f_focal * np.sqrt(max(confidence, 1)))
            step_len = float(np.linalg.norm(t_init[i] - t_init[i - 1]))
            trans_sigma = max(step_len * rot_sigma, 1e-4)
            sigmas = np.array([rot_sigma] * 3 + [trans_sigma] * 3)
            graph.add(gtsam.PriorFactorPose3(
                X(i), pose_i, gtsam.noiseModel.Diagonal.Sigmas(sigmas)))

    Ps_init = [_projection_matrix(K, R_init[i], t_init[i]) for i in range(n + 1)]
    n_landmarks = 0
    for tr in tracks:
        s = tr["start"]
        obs = [np.array(p) for p in tr["obs"]]
        Ps_sub = Ps_init[s:s + len(obs)]
        Xw = _triangulate_multiview(Ps_sub, obs)
        if Xw is None:
            continue
        initial.insert(L(n_landmarks), Point3(Xw))
        for k, pt in enumerate(obs):
            graph.add(gtsam.GenericProjectionFactorCal3_S2(
                pt.astype(np.float64), pixel_noise, X(s + k), L(n_landmarks), gtsam_K))
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


@dataclass
class TrajectoryPoint:
    frame_name: str
    frame_idx: int
    position: np.ndarray
    R_world: np.ndarray
    scale: float
    mode: str
    is_valid: bool

    def __repr__(self) -> str:
        pos = self.position
        return (
            f"TrajectoryPoint("
            f"frame={self.frame_name}, "
            f"pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}], "
            f"scale={self.scale:.4f}, "
            f"mode={self.mode})"
        )


class PoseGraph:
    def __init__(self, data_loader: DataLoader) -> None:
        logger.info("[PoseGraph] Initializing...")

        self._loader = data_loader
        self._eval_cfg = data_loader.get_evaluation_config()

        self._force_2d = bool(self._eval_cfg.get("force_2d", False))
        self._min_sim3_samples = int(self._eval_cfg.get("min_sim3_samples", 10))

        self._T_world = np.eye(4, dtype=np.float64)
        self._trajectory: List[TrajectoryPoint] = []
        self._frame_idx: int = 0

        # Sim(3) alignment: local-frame -> GT-frame. Fit once from
        # (raw_position, GT) pairs collected while GT is available.
        self._local_gt_pairs: List[tuple] = []
        self._sim3_fit: Optional[tuple] = None  # (s, R, t) once fit

        # Per-step local (R, t) history -- kept so refine_with_persistent_map
        # can re-chain and re-optimize after the fact without re-running the
        # front-end. Identity/zero for invalid steps, matching update()'s own
        # no-op behavior for those steps.
        self._local_steps: List[dict] = []

        logger.info(
            "[PoseGraph] Ready -- force_2d=%s, min_sim3_samples=%d",
            self._force_2d, self._min_sim3_samples,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_local_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        return T

    @staticmethod
    def _umeyama(P: np.ndarray, Q: np.ndarray) -> Optional[tuple]:
        """
        Least-squares similarity transform P -> Q: finds (s, R, t) minimizing
        sum ||s*R@p_i + t - q_i||^2. Standard Umeyama/Kabsch solution via SVD.

        Returns None if P is degenerate (near-zero spread -- SVD would be
        numerically meaningless).
        """
        Pt = P.T
        Qt = Q.T
        n = Pt.shape[1]
        mu_p = Pt.mean(axis=1, keepdims=True)
        mu_q = Qt.mean(axis=1, keepdims=True)
        Pc = Pt - mu_p
        Qc = Qt - mu_q

        var_p = (Pc ** 2).sum() / n
        if var_p < 1e-9:
            return None

        W = Qc @ Pc.T / n
        U, D, Vt = np.linalg.svd(W)
        S = np.eye(3)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            S[2, 2] = -1
        R = U @ S @ Vt
        s = float(np.trace(np.diag(D) @ S) / var_p)
        t = (mu_q - s * R @ mu_p).flatten()
        return s, R, t

    def _fit_sim3(self) -> Optional[tuple]:
        if len(self._local_gt_pairs) < self._min_sim3_samples:
            return None
        P = np.array([p[0] for p in self._local_gt_pairs])
        Q = np.array([p[1] for p in self._local_gt_pairs])
        return self._umeyama(P, Q)

    def _report_position(self, raw_position: np.ndarray) -> np.ndarray:
        """Maps a local-frame position into the GT/world frame using the
        fitted Sim(3) transform. Before a fit exists (not enough GT samples
        collected yet) the raw local position is returned as-is."""
        if self._sim3_fit is None:
            return raw_position.copy()
        s, R, t = self._sim3_fit
        return s * (R @ raw_position) + t

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        pose: PoseEstimate,
        scale_result: ScaleResult,
        frame_name: str,
    ) -> TrajectoryPoint:
        self._frame_idx += 1

        if not pose.is_valid or not scale_result.is_valid:
            position = self._report_position(self._T_world[:3, 3])
            R_world = self._T_world[:3, :3].copy()
            point = TrajectoryPoint(
                frame_name=frame_name,
                frame_idx=self._frame_idx,
                position=position,
                R_world=R_world,
                scale=0.0,
                mode="invalid",
                is_valid=False,
            )
            self._trajectory.append(point)
            self._local_steps.append(dict(
                frame_name=frame_name, R_local=np.eye(3), t_local=np.zeros(3), mode="invalid",
            ))
            return point


        t_vec = scale_result.t_scaled.flatten().copy()

        if self._force_2d:
            # Zeroing Z shortens the XY norm. Rescale so the planar step
            # keeps the metric length that scale recovery assigned to it.
            n_before = float(np.linalg.norm(t_vec))
            t_vec[2] = 0.0
            xy = float(np.linalg.norm(t_vec))
            if xy > 1e-9 and n_before > 1e-9:
                t_vec *= (n_before / xy)

        T_local = self._build_local_transform(pose.R, t_vec)
        self._T_world = self._T_world @ T_local

        if self._force_2d:
            self._T_world[2, 3] = 0.0

        raw_position = self._T_world[:3, 3].copy()

        gt = self._loader.get_ground_truth(frame_name)
        if scale_result.mode == "warmup" and gt is not None:
            # GT is available -- collect a correspondence sample and report
            # GT directly (no need to estimate what we already know).
            self._local_gt_pairs.append((raw_position.copy(), gt.as_vector()))
            position = gt.as_vector().copy()
        else:
            if self._sim3_fit is None:
                self._sim3_fit = self._fit_sim3()
            position = self._report_position(raw_position)

        R_world = self._T_world[:3, :3].copy()

        point = TrajectoryPoint(
            frame_name=frame_name,
            frame_idx=self._frame_idx,
            position=position,
            R_world=R_world,
            scale=scale_result.scale,
            mode=scale_result.mode,
            is_valid=True,
        )
        self._trajectory.append(point)
        self._local_steps.append(dict(
            frame_name=frame_name, R_local=pose.R.copy(), t_local=t_vec.copy(),
            mode=scale_result.mode,
        ))

        logger.debug(
            "[PoseGraph] %s pos=[%.3f, %.3f, %.3f]",
            frame_name, position[0], position[1], position[2],
        )
        return point

    # ------------------------------------------------------------------
    # Offline refinement -- persistent-map windowed BA (GTSAM, WSL/Linux only)
    # ------------------------------------------------------------------

    def refine_with_persistent_map(self, camera_calibration, feature_extractor) -> List[TrajectoryPoint]:
        """
        update() ile zaten toplanmis self._local_steps'i (her adimin
        R_local/t_local'i) alip, TUM UCUS boyunca SUREKLI bir KLT takibi
        (pencere siniri yok, besleme ile) yapar; ortaya cikan (pencere
        sinirini asabilen) track'leri, GTSAM windowed BA'ya verirken
        config.yaml:persistent_ba.window kadar bloklara kirpar.

        Nicin: normal update() akisinda her pencere sifirdan track
        kuruyordu -- bir pencerenin sonunda hayatta olan bir nokta,
        sonraki pencerede "hic yasamamis" gibi kaybediliyordu. Bu metot
        o sinirlamayi kaldirir. Olculen etki (22-24 Eylul, tanı.md):
        hizalanmamis (yarisma) hata 147.94m -> 141.29m (%4.5 iyilesme),
        sabit-15-beslemesiz'den (%2.7-3) daha iyi.

        GEREKSINIM: GTSAM. Windows'ta PyPI wheel'i yok -- bu metot
        SADECE WSL/Linux'ta calisir (main.py/Windows akisi bunu hic
        cagirmaz, dolayisiyla gtsam'a bagimli degil).

        Returns:
            Duzeltilmis TrajectoryPoint listesi (self._trajectory
            DEGISTIRILMEZ -- cagiran taraf save_trajectory'ye bu listeyi
            ayrica verebilir).
        """
        cfg = self._loader.get_persistent_ba_config()
        window = int(cfg["window"])
        min_track_len = int(cfg["min_track_len_in_window"])
        max_tracks = int(cfg["max_tracks_per_window"])
        parallax_cos_threshold = float(cfg["parallax_cos_threshold"])

        K = camera_calibration.K
        f_focal = float(np.sqrt(camera_calibration.fx * camera_calibration.fy))

        R_local_all = [s["R_local"] for s in self._local_steps]
        t_local_all = [s["t_local"] for s in self._local_steps]
        N = len(self._local_steps)

        logger.info("[PoseGraph] refine_with_persistent_map: surekli KLT takibi baslatiliyor (%d adim)...", N)
        global_tracks = _build_full_flight_tracks(self._loader, camera_calibration, feature_extractor, cfg)
        logger.info("[PoseGraph] refine_with_persistent_map: %d track bulundu (medyan uzunluk=%.1f)",
                     len(global_tracks),
                     float(np.median([len(t["obs"]) for t in global_tracks])) if global_tracks else 0.0)

        rng = np.random.default_rng(0)
        R_local_corr = [r.copy() for r in R_local_all]
        t_local_corr = [t.copy() for t in t_local_all]

        R0, t0 = np.eye(3), np.zeros(3)
        start = 0
        while start < N - 1:
            n = min(window - 1, N - 1 - start)
            end = start + n

            R_init = [R0.copy()]
            t_init = [t0.copy()]
            for i in range(n):
                t_init.append(t_init[-1] + R_init[-1] @ t_local_all[start + i])
                R_init.append(R_init[-1] @ R_local_all[start + i])

            win_tracks = []
            for tr in global_tracks:
                sliced = _slice_track_to_window(tr, start, end, min_track_len)
                if sliced is not None and _parallax_ok(K, sliced, R_init, t_init, parallax_cos_threshold):
                    win_tracks.append(sliced)

            if not win_tracks:
                R0, t0 = R_init[-1], t_init[-1]
                start = end
                continue

            if len(win_tracks) > max_tracks:
                idx = rng.choice(len(win_tracks), max_tracks, replace=False)
                win_tracks = [win_tracks[i] for i in idx]

            R_out, t_out = _solve_window_gtsam(K, f_focal, R_init, t_init, win_tracks, len(win_tracks), cfg)

            for i in range(n):
                R_local_corr[start + i] = R_out[i].T @ R_out[i + 1]
                t_local_corr[start + i] = R_out[i].T @ (t_out[i + 1] - t_out[i])

            R0, t0 = R_out[-1], t_out[-1]
            start = end

        # yeniden zincirle, warmup adimlarindaki (duzeltilmis raw, GT) ciftleriyle
        # Sim(3)'u YENIDEN uydur -- duzeltilmis zincir orijinalinden farkli.
        R_world = np.eye(3)
        t_world = np.zeros(3)
        gt_pairs = []
        raw_positions = []
        for i, step in enumerate(self._local_steps):
            t_world = t_world + R_world @ t_local_corr[i]
            R_world = R_world @ R_local_corr[i]
            raw_positions.append(t_world.copy())
            if step["mode"] == "warmup":
                gt = self._loader.get_ground_truth(step["frame_name"])
                if gt is not None:
                    gt_pairs.append((t_world.copy(), gt.as_vector()))

        sim3 = None
        if len(gt_pairs) >= self._min_sim3_samples:
            P = np.array([p[0] for p in gt_pairs])
            Q = np.array([p[1] for p in gt_pairs])
            sim3 = self._umeyama(P, Q)

        refined: List[TrajectoryPoint] = []
        for i, step in enumerate(self._local_steps):
            gt = self._loader.get_ground_truth(step["frame_name"])
            if step["mode"] == "warmup" and gt is not None:
                position = gt.as_vector().copy()
            elif sim3 is not None:
                s, R, t = sim3
                position = s * (R @ raw_positions[i]) + t
            else:
                position = raw_positions[i].copy()
            refined.append(TrajectoryPoint(
                frame_name=step["frame_name"], frame_idx=i + 1, position=position,
                R_world=np.eye(3), scale=0.0, mode=step["mode"],
                is_valid=(step["mode"] != "invalid"),
            ))

        return refined

    # ------------------------------------------------------------------
    # Trajectory access
    # ------------------------------------------------------------------

    @property
    def trajectory(self) -> List[TrajectoryPoint]:
        return self._trajectory

    @property
    def valid_trajectory(self) -> List[TrajectoryPoint]:
        return [p for p in self._trajectory if p.is_valid]

    @property
    def current_position(self) -> np.ndarray:
        raw = self._T_world[:3, 3].copy()
        return self._report_position(raw)

    @property
    def current_rotation(self) -> np.ndarray:
        return self._T_world[:3, :3].copy()

    def get_positions_array(self) -> np.ndarray:
        valid = self.valid_trajectory
        if not valid:
            return np.empty((0, 3))
        return np.array([p.position for p in valid])

    def get_gt_array(self, data_loader: DataLoader) -> np.ndarray:
        gt_positions = []
        for point in self.valid_trajectory:
            gt = data_loader.get_ground_truth(point.frame_name)
            if gt is not None:
                gt_positions.append([gt.tx, gt.ty, gt.tz])   # tz artik gercek
            else:
                gt_positions.append([0.0, 0.0, 0.0])
        return np.array(gt_positions)

    def compute_ate(self, data_loader: DataLoader, use_3d: Optional[bool] = None) -> Optional[float]:
        """
        ATE = sqrt(1/N * sum(||t_GT_i - t_est_i||^2))

        use_3d None ise config'deki ate_3d degeri kullanilir.
        Iki veri setini karsilastirirken ayni metrigi kullan --
        yeni setin 3B ATE'si ile eski setin 2B ATE'si yan yana
        konulmaz.
        """
        if use_3d is None:
            use_3d = bool(self._eval_cfg.get("ate_3d", False))

        est_positions = self.get_positions_array()
        gt_positions = self.get_gt_array(data_loader)

        if len(est_positions) < 2 or len(gt_positions) < 2:
            logger.warning("[PoseGraph] Not enough points for ATE.")
            return None

        N = min(len(est_positions), len(gt_positions))
        dims = 3 if use_3d else 2
        est = est_positions[:N, :dims]
        gt = gt_positions[:N, :dims]

        est = est - est[0]
        gt = gt - gt[0]

        errors = np.linalg.norm(est - gt, axis=1)
        ate = float(np.sqrt(np.mean(errors ** 2)))

        logger.info("[PoseGraph] ATE(%dD) = %.4f m (%d points)", dims, ate, N)
        return ate

    def save_trajectory(self, output_path: str, trajectory: Optional[List[TrajectoryPoint]] = None) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        points = trajectory if trajectory is not None else self._trajectory

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_name", "x", "y", "z", "scale", "mode", "is_valid"])
            for point in points:
                writer.writerow([
                    point.frame_name,
                    f"{point.position[0]:.6f}",
                    f"{point.position[1]:.6f}",
                    f"{point.position[2]:.6f}",
                    f"{point.scale:.6f}",
                    point.mode,
                    point.is_valid,
                ])

        logger.info("[PoseGraph] Saved: %s (%d points)", path, len(points))

    def __repr__(self) -> str:
        fit = "pending" if self._sim3_fit is None else f"s={self._sim3_fit[0]:.4f}"
        return (
            f"PoseGraph("
            f"total={len(self._trajectory)}, "
            f"valid={len(self.valid_trajectory)}, "
            f"force_2d={self._force_2d}, "
            f"sim3={fit})"
        )


# ------------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config_path = Path(__file__).resolve().parent.parent / "config.yaml"

    try:
        from utils.data_loader import DataLoader
        from utils.camera_calibration import CameraCalibration
        from core.feature_extractor import FeatureExtractor
        from core.matcher import Matcher
        from core.motion_estimator import MotionEstimator
        from core.scale_recovery import ScaleRecovery

        loader = DataLoader(str(config_path))
        cam = CameraCalibration(loader)
        extractor = FeatureExtractor(loader, cam)
        matcher = Matcher(loader)
        estimator = MotionEstimator(loader, cam)
        scale_recovery = ScaleRecovery(loader, cam)
        pose_graph = PoseGraph(loader)

        print(f"\n{pose_graph}")
        print("\n--- Pose Graph Test (first 300 frames) ---")

        prev_features = None

        for idx, name, frame in loader.frame_generator():
            curr_features = extractor.extract(frame, name)

            if prev_features is not None:
                match_result = matcher.match(prev_features, curr_features)
                pose = estimator.estimate(match_result)
                scale_result = scale_recovery.recover(pose, name)
                traj_point = pose_graph.update(pose, scale_result, name)

                if idx % 50 == 0:
                    print(f"\n[{idx}] {traj_point}")

            prev_features = curr_features

            if idx >= 300:
                break

        positions = pose_graph.get_positions_array()

        print(f"\n--- Trajectory Statistics ---")
        print(f"  Total frames  : {len(pose_graph.trajectory)}")
        print(f"  Valid frames  : {len(pose_graph.valid_trajectory)}")
        if len(positions) > 0:
            print(f"  X range       : [{positions[:,0].min():.3f}, {positions[:,0].max():.3f}] m")
            print(f"  Y range       : [{positions[:,1].min():.3f}, {positions[:,1].max():.3f}] m")
            print(f"  Z range       : [{positions[:,2].min():.3f}, {positions[:,2].max():.3f}] m")
            print(f"  Final pos     : {pose_graph.current_position}")

        ate = pose_graph.compute_ate(loader)
        if ate is not None:
            print(f"\n  ATE           : {ate:.4f} m")

        pose_graph.save_trajectory("data/trajectory_output.csv")
        print(f"\n  Saved: data/trajectory_output.csv")

    except Exception as e:
        logger.error("Error: %s", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)