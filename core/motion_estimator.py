"""
core/motion_estimator.py
=========================
Motion estimation module.
RANSAC outlier removal, Hybrid H/E selection, SVD decomposition.

Math:
    H = K(R + t*n^T/d)K^-1  -> Planar scenes
    E = K^T * F * K          -> General 3D scenes

Hybrid selection (ORB-SLAM2):
    R_H = S_H / (S_H + S_E)
    R_H > 0.45 -> H selected
    R_H <= 0.45 -> E selected
    H decomposition fails -> fallback to E automatically
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
    pass


class MatrixType(Enum):
    HOMOGRAPHY = auto()
    ESSENTIAL = auto()
    NONE = auto()


@dataclass
class PoseEstimate:
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
        1. Compute H and E with RANSAC in parallel
        2. Compute RANSAC score for each
        3. R_H = S_H / (S_H + S_E) -> threshold selection
        4. Decompose primary matrix -> R, t
           H decomposition fails -> fallback to E
        5. Validate R (determinant, orthogonality, rotation angle)
        6. inlier_count < min_inlier_count -> deep features warning
    """

    _HOMOGRAPHY_SCORE_RATIO_THRESHOLD = 0.45
    _H_RANSAC_CONFIDENCE = 0.999
    _E_RANSAC_CONFIDENCE = 0.999
    _H_RANSAC_MAX_ITER = 2000
    _E_RANSAC_MAX_ITER = 2000
    _MAX_ROTATION_DEG = 30.0
    # Cheirality voting gates
    _CHEIRALITY_SAMPLE = 50          # kac nokta oylamaya girsin
    _PARALLAX_COS_THRESHOLD = 0.99998  # ~0.36 derece; altinda cekimser kal
    _REPROJ_THRESHOLD = 4.0 

    def __init__(
        self,
        data_loader: DataLoader,
        camera_calibration: CameraCalibration,
    ) -> None:
        logger.info("[MotionEstimator] Initializing...")

        self._K = camera_calibration.K
        self._cam = camera_calibration

        feat_cfg = data_loader.get_feature_config()
        hybrid_cfg = data_loader.get_hybrid_config()

        self._ransac_threshold = self._parse_ransac_threshold(feat_cfg)
        self._min_inlier_count = self._parse_min_inlier_count(hybrid_cfg)

        logger.info(
            "[MotionEstimator] Ready -- ransac_threshold=%.1f, min_inlier_count=%d",
            self._ransac_threshold,
            self._min_inlier_count,
        )

    @staticmethod
    def _parse_ransac_threshold(feat_cfg: dict) -> float:
        if "ransac_threshold" not in feat_cfg:
            raise MotionEstimatorError("features config missing 'ransac_threshold'.")
        return float(feat_cfg["ransac_threshold"])

    @staticmethod
    def _parse_min_inlier_count(hybrid_cfg: dict) -> int:
        if "min_inlier_count" not in hybrid_cfg:
            raise MotionEstimatorError("hybrid config missing 'min_inlier_count'.")
        return int(hybrid_cfg["min_inlier_count"])

    # ------------------------------------------------------------------
    # R Validation
    # ------------------------------------------------------------------

    def _validate_rotation(
        self,
        R: np.ndarray,
        frame_prev: str,
        frame_curr: str,
    ) -> bool:
        det = np.linalg.det(R)
        if abs(det - 1.0) > 0.01:
            logger.warning(
                "[MotionEstimator] Invalid det(R)=%.4f: %s -> %s",
                det, frame_prev, frame_curr,
            )
            return False

        orth_error = np.linalg.norm(R.T @ R - np.eye(3))
        if orth_error > 0.01:
            logger.warning(
                "[MotionEstimator] R not orthogonal (err=%.4f): %s -> %s",
                orth_error, frame_prev, frame_curr,
            )
            return False

        trace_val = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(trace_val))
        logger.debug("[MotionEstimator] Rotation angle: %.2f deg", angle_deg)

        return True

    # ------------------------------------------------------------------
    # RANSAC Scoring
    # ------------------------------------------------------------------

    def _compute_homography_score(
        self,
        H: np.ndarray,
        pts_prev: np.ndarray,
        pts_curr: np.ndarray,
        threshold: float,
    ) -> Tuple[float, np.ndarray]:
        N = len(pts_prev)
        inlier_mask = np.zeros(N, dtype=np.uint8)
        score = 0.0

        pts_prev_h = np.hstack([pts_prev, np.ones((N, 1))])
        pts_curr_h = np.hstack([pts_curr, np.ones((N, 1))])

        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return 0.0, inlier_mask

        th_sq = threshold ** 2

        for i in range(N):
            p_fwd = H @ pts_prev_h[i]
            if abs(p_fwd[2]) < 1e-9:
                continue
            p_fwd /= p_fwd[2]
            e_fwd = (pts_curr[i, 0] - p_fwd[0])**2 + (pts_curr[i, 1] - p_fwd[1])**2

            p_bwd = H_inv @ pts_curr_h[i]
            if abs(p_bwd[2]) < 1e-9:
                continue
            p_bwd /= p_bwd[2]
            e_bwd = (pts_prev[i, 0] - p_bwd[0])**2 + (pts_prev[i, 1] - p_bwd[1])**2

            if e_fwd < th_sq and e_bwd < th_sq:
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
        N = len(pts_prev)
        inlier_mask = np.zeros(N, dtype=np.uint8)
        score = 0.0
        th_sq = threshold ** 2

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
            d = (pEp**2) / denom
            if d < th_sq:
                inlier_mask[i] = 1
                score += threshold - np.sqrt(d)

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
        """
        Decomposes Homography into R and t.
 
        decomposeHomographyMat returns four candidate solutions. Only one is
        physically possible; the rest place points behind a camera or are
        mirror images. The correct one is selected by voting.
 
        Each triangulated point votes for a candidate only if it passes
        three tests:
 
          1. Cheirality -- the point must lie in front of both cameras.
             A point behind the camera cannot have been imaged.
 
          2. Parallax gate -- when the two camera centres are close, the
             viewing rays are nearly parallel and triangulation becomes
             numerically unstable. Distant points can then land at negative
             depth even though they are genuinely in front. ORB-SLAM2
             handles this by exempting low-parallax points from the
             cheirality test rather than letting them cast a false vote.
 
          3. Reprojection error -- cheirality only asks whether the point is
             in front, not whether it is in the right place. Projecting the
             triangulated point back into both images and comparing against
             the observed pixel catches solutions that satisfy cheirality
             but are geometrically wrong.
 
        Without gates 2 and 3 the vote is noisy and the wrong candidate can
        win, which shows up downstream as heading error.
        """
        num_solutions, Rs, ts, normals = cv2.decomposeHomographyMat(H, self._K)
 
        if num_solutions == 0:
            logger.debug("[MotionEstimator] H decomposition: no solutions.")
            return None, None
 
        inlier_idx = np.where(inlier_mask == 1)[0]
        if len(inlier_idx) == 0:
            logger.debug("[MotionEstimator] H decomposition: no inlier points.")
            return None, None
 
        pts_prev_in = pts_prev[inlier_idx]
        pts_curr_in = pts_curr[inlier_idx]
        test_count = min(self._CHEIRALITY_SAMPLE, len(pts_prev_in))
        pts_prev_test = pts_prev_in[:test_count]
        pts_curr_test = pts_curr_in[:test_count]
 
        K_inv = self._cam.K_inv
        fx, fy = self._cam.fx, self._cam.fy
        cx, cy = self._cam.cx, self._cam.cy
 
        # camera 1 sits at the world origin -- points are already normalized
        # (K_inv applied below), so P1/P2 must NOT reapply K here.
        P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
        O1 = np.zeros(3)
 
        reproj_th_sq = self._REPROJ_THRESHOLD ** 2
 
        best_R = None
        best_t = None
        best_votes = -1
        best_stats = None
 
        for i in range(num_solutions):
            R_cand = Rs[i]
            t_cand = ts[i].reshape(3, 1)
 
            t_norm = np.linalg.norm(t_cand)
            if t_norm < 1e-12:
                continue
 
            t_unit = t_cand / t_norm
            P2 = np.hstack([R_cand, t_cand])
 
            # camera 2 centre in world coordinates
            O2 = (-R_cand.T @ t_cand).flatten()
 
            votes = 0
            n_low_parallax = 0
            n_behind = 0
            n_reproj_fail = 0
 
            for j in range(test_count):
                u1, v1 = pts_prev_test[j]
                u2, v2 = pts_curr_test[j]
 
                p1 = K_inv @ np.array([u1, v1, 1.0])
                p2 = K_inv @ np.array([u2, v2, 1.0])
                # --- DLT triangulation ---
                A = np.array([
                    p1[0] * P1[2] - P1[0],
                    p1[1] * P1[2] - P1[1],
                    p2[0] * P2[2] - P2[0],
                    p2[1] * P2[2] - P2[1],
                ])
 
                _, _, Vt = np.linalg.svd(A)
                X_h = Vt[-1]
 
                if abs(X_h[3]) < 1e-9:
                    continue
 
                X = X_h[:3] / X_h[3]
 
                if not np.all(np.isfinite(X)):
                    continue
 
                # --- gate 2: parallax ---
                ray1 = X - O1
                ray2 = X - O2
                d1 = np.linalg.norm(ray1)
                d2 = np.linalg.norm(ray2)
                if d1 < 1e-9 or d2 < 1e-9:
                    continue
                cos_parallax = float(np.dot(ray1, ray2) / (d1 * d2))
 
                low_parallax = cos_parallax > self._PARALLAX_COS_THRESHOLD
 
                # --- gate 1: cheirality ---
                Z1 = X[2]
                X_cam2 = R_cand @ X + t_cand.flatten()
                Z2 = X_cam2[2]
 
                if (Z1 <= 0 or Z2 <= 0):
                    if low_parallax:
                        # Triangulation is unreliable here; abstain rather
                        # than cast a vote either way.
                        n_low_parallax += 1
                        continue
                    n_behind += 1
                    continue
 
                # --- gate 3: reprojection error ---
                if abs(Z1) < 1e-9 or abs(Z2) < 1e-9:
                    continue
 
                u1_hat = fx * X[0] / Z1 + cx
                v1_hat = fy * X[1] / Z1 + cy
                err1 = (u1_hat - u1) ** 2 + (v1_hat - v1) ** 2
                if err1 > reproj_th_sq:
                    n_reproj_fail += 1
                    continue
 
                u2_hat = fx * X_cam2[0] / Z2 + cx
                v2_hat = fy * X_cam2[1] / Z2 + cy
                err2 = (u2_hat - u2) ** 2 + (v2_hat - v2) ** 2
                if err2 > reproj_th_sq:
                    n_reproj_fail += 1
                    continue
 
                votes += 1
 
            if votes > best_votes:
                best_votes = votes
                best_R = R_cand
                best_t = t_cand
                best_stats = (n_low_parallax, n_behind, n_reproj_fail)
 
        if best_R is None or best_votes <= 0:
            logger.debug(
                "[MotionEstimator] Cheirality vote inconclusive "
                "(best=%d of %d samples).", best_votes, test_count,
            )
            return None, None
 
        lp, bh, rf = best_stats
        logger.debug(
            "[MotionEstimator] H decomposition: %d/%d votes "
            "(low_parallax=%d, behind=%d, reproj_fail=%d)",
            best_votes, test_count, lp, bh, rf,
        )
        best_t = best_t / np.linalg.norm(best_t)  # normalize translation
 
        return best_R, best_t.reshape(3, 1)

    # ------------------------------------------------------------------
    # E Decomposition helper
    # ------------------------------------------------------------------

    def _decompose_essential(
        self,
        E: np.ndarray,
        pts_prev: np.ndarray,
        pts_curr: np.ndarray,
        e_mask: np.ndarray,
        e_inlier_mask: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        try:
            retval, R, t, _ = cv2.recoverPose(
                E, pts_prev, pts_curr,
                cameraMatrix=self._K,
                mask=e_mask,
            )
            if retval <= 0:
                logger.debug(
                    "[MotionEstimator] recoverPose: no candidate supported "
                    "by cheirality (retval=%d).", retval,
                )
                return None, None, e_inlier_mask
            t_norm = np.linalg.norm(t)
            if t_norm > 1e-9:
                t = t / t_norm
            return R, t, e_inlier_mask
        except cv2.error as e:
            logger.warning("[MotionEstimator] recoverPose failed: %s", e)
            return None, None, e_inlier_mask

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(self, match_result: MatchResult) -> PoseEstimate:
        pose = PoseEstimate(
            frame_name_prev=match_result.frame_name_prev,
            frame_name_curr=match_result.frame_name_curr,
        )

        if not match_result.has_enough_matches:
            logger.warning(
                "[MotionEstimator] Not enough matches (%d), skipping.",
                match_result.match_count,
            )
            return pose

        pts_prev = match_result.pts_prev
        pts_curr = match_result.pts_curr

        try:
            # 1. H + E parallel RANSAC
            H, h_mask = cv2.findHomography(
                pts_prev, pts_curr,
                method=cv2.RANSAC,
                ransacReprojThreshold=self._ransac_threshold,
                confidence=self._H_RANSAC_CONFIDENCE,
                maxIters=self._H_RANSAC_MAX_ITER,
            )
            E, e_mask = cv2.findEssentialMat(
                pts_prev, pts_curr,
                cameraMatrix=self._K,
                method=cv2.RANSAC,
                prob=self._E_RANSAC_CONFIDENCE,
                threshold=self._ransac_threshold,
                maxIters=self._E_RANSAC_MAX_ITER,
            )

            if H is None or E is None:
                logger.warning(
                    "[MotionEstimator] H or E failed: %s -> %s",
                    match_result.frame_name_prev,
                    match_result.frame_name_curr,
                )
                return pose

            # 2. RANSAC Scores
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
                return pose

            # 3. Hybrid Selection
            R_H_ratio = score_H / total_score

            logger.debug(
                "[MotionEstimator] S_H=%.2f, S_E=%.2f, R_H=%.3f -> %s selected.",
                score_H, score_E, R_H_ratio,
                "H" if R_H_ratio > self._HOMOGRAPHY_SCORE_RATIO_THRESHOLD else "E",
            )

            # 4. Decomposition with E fallback
            R, t, selected_mask = None, None, h_inlier_mask

            if R_H_ratio > self._HOMOGRAPHY_SCORE_RATIO_THRESHOLD:
                R, t = self._decompose_homography(
                    H, pts_prev, pts_curr, h_inlier_mask
                )
                selected_mask = h_inlier_mask
                pose.matrix_type = MatrixType.HOMOGRAPHY

                if R is None or t is None:
                    logger.debug("[MotionEstimator] H failed, no fallback.")
            else:
                R, t, selected_mask = self._decompose_essential(
                    E, pts_prev, pts_curr, e_mask, e_inlier_mask
                )
                pose.matrix_type = MatrixType.ESSENTIAL

            if R is None or t is None:
                logger.warning(
                    "[MotionEstimator] Decomposition failed: %s -> %s",
                    match_result.frame_name_prev,
                    match_result.frame_name_curr,
                )
                return pose

            # 5. R Validation
            if not self._validate_rotation(
                R,
                match_result.frame_name_prev,
                match_result.frame_name_curr,
            ):
                return pose

            # 6. Update pose
            pose.R = R
            pose.t = t
            pose.inlier_mask = selected_mask
            pose.inlier_count = int(np.sum(selected_mask))
            pose.is_valid = True

            if pose.inlier_count < self._min_inlier_count:
                logger.warning(
                    "[MotionEstimator] Low inlier count: %d < %d.",
                    pose.inlier_count,
                    self._min_inlier_count,
                )

            logger.debug("[MotionEstimator] %s", pose)
            return pose

        except cv2.error as e:
            raise MotionEstimatorError(
                f"OpenCV error ({match_result.frame_name_prev} -> "
                f"{match_result.frame_name_curr}): {e}"
            ) from e

    def __repr__(self) -> str:
        return (
            f"MotionEstimator("
            f"ransac_threshold={self._ransac_threshold}, "
            f"min_inlier_count={self._min_inlier_count})"
        )


if __name__ == "__main__":
    import sys
    from pathlib import Path

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config_path = Path(__file__).resolve().parent.parent / "config.yaml"

    try:
        from utils.data_loader import DataLoader
        from utils.camera_calibration import CameraCalibration
        from core.feature_extractor import FeatureExtractor
        from core.matcher import Matcher

        loader = DataLoader(str(config_path))
        cam = CameraCalibration(loader)
        extractor = FeatureExtractor(loader, cam)
        matcher = Matcher(loader)
        estimator = MotionEstimator(loader, cam)

        print(f"\n{estimator}")
        print("\n--- Motion Estimation Test (first 30 frame pairs) ---")

        prev_features = None
        results = []

        for idx, name, frame in loader.frame_generator():
            curr_features = extractor.extract(frame, name)

            if prev_features is not None:
                match_result = matcher.match(prev_features, curr_features)
                pose = estimator.estimate(match_result)
                results.append(pose)

                if pose.is_valid:
                    trace_val = np.clip((np.trace(pose.R) - 1.0) / 2.0, -1.0, 1.0)
                    angle = np.degrees(np.arccos(trace_val))
                    print(
                        f"[{idx}] {pose.matrix_type.name} "
                        f"angle={angle:.2f}deg "
                        f"t={pose.t.T}"
                    )

            prev_features = curr_features

            if idx >= 30:
                break

        valid = [r for r in results if r.is_valid]
        print(f"\n--- Statistics ---")
        print(f"  Valid poses : {len(valid)}/{len(results)}")
        print(f"  H selected  : {sum(1 for r in valid if r.matrix_type == MatrixType.HOMOGRAPHY)}")
        print(f"  E selected  : {sum(1 for r in valid if r.matrix_type == MatrixType.ESSENTIAL)}")

    except Exception as e:
        logger.error("Error: %s", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)