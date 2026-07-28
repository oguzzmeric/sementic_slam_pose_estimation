"""
core/motion_estimator.py
=========================
Motion estimation modülü.
RANSAC ile outlier eleme, Hybrid H/E seçimi ve SVD decomposition ile
iki frame arasındaki R (rotation) ve t (translation unit vector) çıkarır.

Matematiksel temel:
    H = K(R + t*n^T/d)K^-1  → Planar sahneler için
    E = K^T * F * K          → Genel 3D sahneler için
    E = [t]x * R             → Essential matrix tanımı

Hybrid seçim kriteri (ORB-SLAM2):
    R_H = S_H / (S_H + S_E)
    R_H > 0.45 → H seç
    R_H <= 0.45 → E seç
"""

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple

import cv2
import numpy as np

from core.matcher import MatchResult
from utils.camera_calibration import CameraCalibration
from utils.data_loader import DataLoader

logger = logging.getLogger(__name__)


class MotionEstimatorError(Exception):
    """MotionEstimator'a özgü hata sınıfı."""
    pass


class MatrixType(Enum):
    """Seçilen motion estimation matrisini belirtir."""
    HOMOGRAPHY = auto()
    ESSENTIAL = auto()
    NONE = auto()


@dataclass
class PoseEstimate:
    """
    Tek bir frame çiftinin pose estimation çıktısını temsil eder.

    Attributes:
        frame_name_prev : Önceki frame adı
        frame_name_curr : Mevcut frame adı
        R               : Rotation matrisi (3x3, float64)
        t               : Translation unit vektörü (3x1, float64) — henüz scale yok
        inlier_mask     : RANSAC inlier maskesi (N x 1, uint8)
        inlier_count    : RANSAC inlier sayısı
        matrix_type     : Hangi matrisin kullanıldığı (H veya E)
        score_H         : Homography RANSAC skoru
        score_E         : Essential Matrix RANSAC skoru
        is_valid        : Pose estimation başarılı mı
    """
    frame_name_prev: str
    frame_name_curr: str
    R: Optional[np.ndarray] = None
    t: Optional[np.ndarray] = None
    inlier_mask: Optional[np.ndarray] = None
    inlier_count: int = 0
    matrix_type: MatrixType = MatrixType.NONE
    score_H: float = 0.0
    score_E: float = 0.0
    is_valid: bool = False

    @property
    def inlier_ratio(self) -> float:
        """İnlier oranı — toplam eşleşmeye göre."""
        if self.inlier_mask is None or len(self.inlier_mask) == 0:
            return 0.0
        return self.inlier_count / len(self.inlier_mask)

    def __repr__(self) -> str:
        return (
            f"PoseEstimate("
            f"prev={self.frame_name_prev}, "
            f"curr={self.frame_name_curr}, "
            f"matrix={self.matrix_type.name}, "
            f"inliers={self.inlier_count}, "
            f"inlier_ratio={self.inlier_ratio:.2f}, "
            f"valid={self.is_valid})"
        )


class MotionEstimator:
    """
    Hybrid H/E motion estimator.

    Pipeline:
        1. H ve E'yi RANSAC ile paralel hesapla
        2. Her ikisi için RANSAC skoru hesapla
        3. R_H = S_H / (S_H + S_E) → threshold ile seç
        4. Seçilen matris decompose et → R, t
        5. inlier_count < min_inlier_count → deep features uyarısı

    Decomposition:
        H → decomposeHomographyMat → 4 çözüm → cheirality testi ile 1 seç
        E → recoverPose → direkt 1 çözüm (cheirality check dahili)
    """

    # Hybrid seçim eşiği — ORB-SLAM2'den alındı
    _HOMOGRAPHY_SCORE_RATIO_THRESHOLD = 0.45

    # RANSAC parametreleri
    _H_RANSAC_CONFIDENCE = 0.999
    _E_RANSAC_CONFIDENCE = 0.999
    _H_RANSAC_MAX_ITER = 2000
    _E_RANSAC_MAX_ITER = 2000

    def __init__(
        self,
        data_loader: DataLoader,
        camera_calibration: CameraCalibration,
    ) -> None:
        logger.info("[MotionEstimator] Başlatılıyor...")

        self._K = camera_calibration.K
        self._cam = camera_calibration

        feat_cfg = data_loader.get_feature_config()
        hybrid_cfg = data_loader.get_hybrid_config()

        self._ransac_threshold = self._parse_ransac_threshold(feat_cfg)
        self._min_inlier_count = self._parse_min_inlier_count(hybrid_cfg)

        logger.info(
            "[MotionEstimator] Başlatıldı — ransac_threshold=%.1f, min_inlier_count=%d",
            self._ransac_threshold,
            self._min_inlier_count,
        )

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_ransac_threshold(feat_cfg: dict) -> float:
        if "ransac_threshold" not in feat_cfg:
            raise MotionEstimatorError(
                "features config'inde 'ransac_threshold' bulunamadı."
            )
        return float(feat_cfg["ransac_threshold"])

    @staticmethod
    def _parse_min_inlier_count(hybrid_cfg: dict) -> int:
        if "min_inlier_count" not in hybrid_cfg:
            raise MotionEstimatorError(
                "hybrid config'inde 'min_inlier_count' bulunamadı."
            )
        return int(hybrid_cfg["min_inlier_count"])

    # ------------------------------------------------------------------
    # RANSAC Skorlama
    # ------------------------------------------------------------------

    def _compute_homography_score(
        self,
        H: np.ndarray,
        pts_prev: np.ndarray,
        pts_curr: np.ndarray,
        threshold: float,
    ) -> Tuple[float, np.ndarray]:
        """
        Homography için RANSAC skoru hesaplar.

        Skor = H ve H^-1 transfer hatalarının simetrik toplamı.

        Matematiksel olarak:
            e_forward  = ||p' - H*p||^2
            e_backward = ||p - H^-1*p'||^2
            score += (threshold - sqrt(e)) eğer e < threshold^2
        """
        N = len(pts_prev)
        inlier_mask = np.zeros(N, dtype=np.uint8)
        score = 0.0

        pts_prev_h = np.hstack([pts_prev, np.ones((N, 1))])
        pts_curr_h = np.hstack([pts_curr, np.ones((N, 1))])

        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return 0.0, inlier_mask

        th_squared = threshold ** 2

        for i in range(N):
            p_fwd = H @ pts_prev_h[i]
            if abs(p_fwd[2]) < 1e-9:
                continue
            p_fwd /= p_fwd[2]
            e_fwd = (pts_curr[i, 0] - p_fwd[0]) ** 2 + (pts_curr[i, 1] - p_fwd[1]) ** 2

            p_bwd = H_inv @ pts_curr_h[i]
            if abs(p_bwd[2]) < 1e-9:
                continue
            p_bwd /= p_bwd[2]
            e_bwd = (pts_prev[i, 0] - p_bwd[0]) ** 2 + (pts_prev[i, 1] - p_bwd[1]) ** 2

            if e_fwd < th_squared and e_bwd < th_squared:
                inlier_mask[i] = 1
                score += (threshold - np.sqrt(e_fwd)) + (threshold - np.sqrt(e_bwd))

        return score, inlier_mask

    def _compute_essential_score(
        self,
        E: np.ndarray,
        pts_prev: np.ndarray,
        pts_curr: np.ndarray,
        threshold: float,
    ) -> Tuple[float, np.ndarray]:
        """
        Essential Matrix için RANSAC skoru hesaplar.

        Sampson distance:
            d = (p'^T E p)^2 / (||Ep||_1^2 + ||Ep||_2^2 + ||E^Tp'||_1^2 + ||E^Tp'||_2^2)
        """
        N = len(pts_prev)
        inlier_mask = np.zeros(N, dtype=np.uint8)
        score = 0.0
        th_squared = threshold ** 2

        K_inv = self._cam.K_inv
        pts_prev_n = (K_inv @ np.hstack([pts_prev, np.ones((N, 1))]).T).T
        pts_curr_n = (K_inv @ np.hstack([pts_curr, np.ones((N, 1))]).T).T

        for i in range(N):
            p  = pts_prev_n[i]
            pp = pts_curr_n[i]

            Ep  = E @ p
            Etp = E.T @ pp
            pEp = pp @ Ep

            denom = Ep[0]**2 + Ep[1]**2 + Etp[0]**2 + Etp[1]**2

            if denom < 1e-9:
                continue

            d_sampson = (pEp ** 2) / denom

            if d_sampson < th_squared:
                inlier_mask[i] = 1
                score += threshold - np.sqrt(d_sampson)

        return score, inlier_mask

    # ------------------------------------------------------------------
    # H Decomposition
    # ------------------------------------------------------------------

    def _decompose_homography(
        self,
        H: np.ndarray,
        pts_prev: np.ndarray,
        pts_curr: np.ndarray,
        inlier_mask: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """t
        Homography matrisini R ve t'ye decompose eder.

        cv2.decomposeHomographyMat 4 çözüm döndürür.
        Doğru çözümü cheirality testi ile seçer:
            Her çözüm için inlier noktaları triangulate et,
            her iki kameradan da Z > 0 olan nokta sayısını say,
            en fazla pozitif derinlik veren çözümü seç.
        """
        num_solutions, Rs, ts, normals = cv2.decomposeHomographyMat(H, self._K)

        if num_solutions == 0:
            logger.warning("[MotionEstimator] H decomposition çözüm bulunamadı.")
            return None, None

        inlier_idx = np.where(inlier_mask == 1)[0]
        if len(inlier_idx) == 0:
            t0_norm = np.linalg.norm(ts[0])
            if t0_norm < 1e-9:
                return Rs[0], np.zeros((3, 1))
            return Rs[0], ts[0].reshape(3, 1) / t0_norm

        pts_prev_in = pts_prev[inlier_idx]
        pts_curr_in = pts_curr[inlier_idx]

        test_count = min(50, len(pts_prev_in))
        pts_prev_test = pts_prev_in[:test_count]
        pts_curr_test = pts_curr_in[:test_count]

        best_R = None
        best_t = None
        best_positive_count = -1

        K_inv = self._cam.K_inv
        P1 = self._K @ np.hstack([np.eye(3), np.zeros((3, 1))])

        for i in range(num_solutions):
            R_cand = Rs[i]
            t_cand = ts[i].reshape(3, 1)

            t_norm = np.linalg.norm(t_cand)
            if t_norm < 1e-9:
                continue

            t_unit = t_cand / t_norm
            P2 = self._K @ np.hstack([R_cand, t_unit])

            positive_count = 0

            for j in range(test_count):
                p1 = K_inv @ np.array([pts_prev_test[j, 0], pts_prev_test[j, 1], 1.0])
                p2 = K_inv @ np.array([pts_curr_test[j, 0], pts_curr_test[j, 1], 1.0])

                A = np.array([
                    p1[0] * P1[2] - P1[0],
                    p1[1] * P1[2] - P1[1],
                    p2[0] * P2[2] - P2[0],
                    p2[1] * P2[2] - P2[1],
                ])

                _, _, Vt = np.linalg.svd(A)
                X = Vt[-1]

                if abs(X[3]) < 1e-9:
                    continue

                X = X / X[3]
                Z1 = X[2]
                X_cam2 = R_cand @ X[:3] + t_unit.flatten()
                Z2 = X_cam2[2]

                if Z1 > 0 and Z2 > 0:
                    positive_count += 1

            if positive_count > best_positive_count:
                best_positive_count = positive_count
                best_R = R_cand
                best_t = t_unit

        if best_R is None:
            logger.warning(
                "[MotionEstimator] Cheirality testi başarısız, "
                "ilk geçerli çözüm alınıyor."
            )
            for i in range(num_solutions):
                t_norm = np.linalg.norm(ts[i])
                if t_norm > 1e-9:
                    best_R = Rs[i]
                    best_t = ts[i].reshape(3, 1) / t_norm
                    break
            if best_R is None:
                return None, None

        logger.debug(
            "[MotionEstimator] H decomposition: pozitif derinlik=%d/%d",
            best_positive_count, test_count,
        )

        return best_R, best_t.reshape(3, 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(self, match_result: MatchResult) -> PoseEstimate:
        """
        MatchResult'tan R ve t tahmin eder.

        Adımlar:
            1. Minimum nokta kontrolü
            2. H hesapla + skor
            3. E hesapla + skor
            4. Hybrid seçim: R_H = S_H / (S_H + S_E)
            5. Seçilen matrisi decompose et → R, t
            6. inlier_count < min_inlier_count → uyarı
        """
        pose = PoseEstimate(
            frame_name_prev=match_result.frame_name_prev,
            frame_name_curr=match_result.frame_name_curr,
        )

        if not match_result.has_enough_matches:
            logger.warning(
                "[MotionEstimator] Yetersiz eşleşme (%d), pose estimation atlanıyor.",
                match_result.match_count,
            )
            return pose

        pts_prev = match_result.pts_prev
        pts_curr = match_result.pts_curr

        try:
            # 1. Homography — RANSAC
            H, h_mask = cv2.findHomography(
                pts_prev,
                pts_curr,
                method=cv2.RANSAC,
                ransacReprojThreshold=self._ransac_threshold,
                confidence=self._H_RANSAC_CONFIDENCE,
                maxIters=self._H_RANSAC_MAX_ITER,
            )

            # 2. Essential Matrix — RANSAC
            E, e_mask = cv2.findEssentialMat(
                pts_prev,
                pts_curr,
                cameraMatrix=self._K,
                method=cv2.RANSAC,
                prob=self._E_RANSAC_CONFIDENCE,
                threshold=self._ransac_threshold,
                maxIters=self._E_RANSAC_MAX_ITER,
            )

            if H is None or E is None:
                logger.warning(
                    "[MotionEstimator] H veya E hesaplanamadı: %s → %s",
                    match_result.frame_name_prev,
                    match_result.frame_name_curr,
                )
                return pose

            # 3. RANSAC Skorları
            score_H, h_inlier_mask = self._compute_homography_score(
                H, pts_prev, pts_curr, self._ransac_threshold
            )
            score_E, e_inlier_mask = self._compute_essential_score(
                E, pts_prev, pts_curr, self._ransac_threshold
            )

            pose.score_H = score_H
            pose.score_E = score_E

            total_score = score_H + score_E
            if total_score < 1e-9:
                logger.warning(
                    "[MotionEstimator] H ve E skorları sıfır: %s → %s",
                    match_result.frame_name_prev,
                    match_result.frame_name_curr,
                )
                return pose

            # 4. Hybrid Seçim
            R_H = score_H / total_score

            logger.debug(
                "[MotionEstimator] S_H=%.2f, S_E=%.2f, R_H=%.3f → %s seçildi.",
                score_H, score_E, R_H,
                "H" if R_H > self._HOMOGRAPHY_SCORE_RATIO_THRESHOLD else "E",
            )

            # 5. Decomposition
            if R_H > self._HOMOGRAPHY_SCORE_RATIO_THRESHOLD:
                R, t = self._decompose_homography(
                    H, pts_prev, pts_curr, h_inlier_mask
                )
                selected_mask = h_inlier_mask
                pose.matrix_type = MatrixType.HOMOGRAPHY
            else:
                _, R, t, _ = cv2.recoverPose(
                    E,
                    pts_prev,
                    pts_curr,
                    cameraMatrix=self._K,
                    mask=e_mask,
                )
                t_norm = np.linalg.norm(t)
                if t_norm > 1e-9:
                    t = t / t_norm
                selected_mask = e_inlier_mask
                pose.matrix_type = MatrixType.ESSENTIAL

            if R is None or t is None:
                logger.warning(
                    "[MotionEstimator] Decomposition başarısız: %s → %s",
                    match_result.frame_name_prev,
                    match_result.frame_name_curr,
                )
                return pose
            
            if abs(np.linalg.det(R) - 1.0) > 0.01:
                logger.warning(
                "[MotionEstimator] Dejenere R matrisi (det=%.4f): %s → %s",
                np.linalg.det(R),
                match_result.frame_name_prev,
                match_result.frame_name_curr,
                            )
                return pose

            if np.linalg.norm(R.T @ R - np.eye(3)) > 0.01:
                logger.warning(
                "[MotionEstimator] R ortogonal değil: %s → %s",
                match_result.frame_name_prev,
                match_result.frame_name_curr,
                )
                return pose
            
            trace_val = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
            rotation_angle_deg = np.degrees(np.arccos(trace_val))
            if rotation_angle_deg > 30.0:
                logger.warning(
                    "[MotionEstimator] Aşırı rotation açısı (%.1f°): %s → %s",
                    rotation_angle_deg,
                    match_result.frame_name_prev,
                    match_result.frame_name_curr,
                    )
                return pose
            
            # 6. Pose güncelle
            pose.R = R
            pose.t = t
            pose.inlier_mask = selected_mask
            pose.inlier_count = int(np.sum(selected_mask))
            pose.is_valid = True

            if pose.inlier_count < self._min_inlier_count:
                logger.warning(
                    "[MotionEstimator] Düşük inlier sayısı: %d < %d. "
                    "Deep features fallback tetiklenmeli.",
                    pose.inlier_count,
                    self._min_inlier_count,
                )

            logger.debug("[MotionEstimator] %s", pose)
            return pose

        except cv2.error as e:
            raise MotionEstimatorError(
                f"OpenCV hatası ({match_result.frame_name_prev} → "
                f"{match_result.frame_name_curr}): {e}"
            ) from e

    def __repr__(self) -> str:
        return (
            f"MotionEstimator("
            f"ransac_threshold={self._ransac_threshold}, "
            f"min_inlier_count={self._min_inlier_count})"
        )


# ------------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config_path = Path(__file__).resolve().parent.parent / "config.yaml"

    try:
        from utils.data_loader import DataLoader, DataLoaderError
        from utils.camera_calibration import CameraCalibration, CameraCalibrationError
        from core.feature_extractor import FeatureExtractor, FeatureExtractorError
        from core.matcher import Matcher

        loader = DataLoader(str(config_path))
        cam = CameraCalibration(loader)
        extractor = FeatureExtractor(loader, cam)
        matcher = Matcher(loader)
        estimator = MotionEstimator(loader, cam)

        print(f"\n{estimator}")
        print("\n--- Motion Estimation Testi (ilk 5 frame çifti) ---")

        prev_features = None
        results = []

        for idx, name, frame in loader.frame_generator():
            curr_features = extractor.extract(frame, name)

            if prev_features is not None:
                match_result = matcher.match(prev_features, curr_features)
                pose = estimator.estimate(match_result)
                results.append(pose)

                print(f"\n[{idx}] {pose}")
                if pose.is_valid:
                    print(f"     R:\n{pose.R}")
                    print(f"     t: {pose.t.T}")
                    print(f"     S_H={pose.score_H:.2f}, S_E={pose.score_E:.2f}")

            prev_features = curr_features

            if idx >= 100:
                break

        valid = [r for r in results if r.is_valid]
        if valid:
            print(f"\n--- İstatistikler ---")
            print(f"  Geçerli pose  : {len(valid)}/{len(results)}")
            print(f"  Ort. inlier   : {np.mean([r.inlier_count for r in valid]):.1f}")
            print(f"  H seçilen     : {sum(1 for r in valid if r.matrix_type == MatrixType.HOMOGRAPHY)}")
            print(f"  E seçilen     : {sum(1 for r in valid if r.matrix_type == MatrixType.ESSENTIAL)}")

    except Exception as e:
        logger.error("Hata: %s", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)