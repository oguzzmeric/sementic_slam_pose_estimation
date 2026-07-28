"""
core/feature_extractor.py
==========================
Feature extraction ve semantic maskeleme modülü.
ORB ile keypoint tespiti ve descriptor hesabı yapar.
Dinamik objeler semantic mask ile engellenir.

Phase 2'de bu modül SuperPoint ile swap edilir.
Interface değişmez — odometry.py bu değişimden habersiz kalır.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from utils.camera_calibration import CameraCalibration, CameraCalibrationError
from utils.data_loader import DataLoader, DataLoaderError, Detection

logger = logging.getLogger(__name__)


class FeatureExtractorError(Exception):
    """FeatureExtractor'a özgü hata sınıfı."""
    pass


@dataclass
class FrameFeatures:
    """
    Tek bir frame'in feature extraction çıktısını temsil eder.

    Attributes:
        frame_name : Frame dosya adı
        keypoints  : ORB keypoint listesi
        descriptors: ORB binary descriptor matrisi (N x 32, uint8)
        mask       : Semantic mask (H x W, uint8) — 255: statik, 0: dinamik
        clean_frame: Undistorted BGR frame
    """
    frame_name: str
    keypoints: List[cv2.KeyPoint]
    descriptors: Optional[np.ndarray]
    mask: np.ndarray
    clean_frame: np.ndarray

    @property
    def keypoint_count(self) -> int:
        return len(self.keypoints)

    @property
    def has_descriptors(self) -> bool:
        return self.descriptors is not None and len(self.descriptors) > 0

    def __repr__(self) -> str:
        return (
            f"FrameFeatures(frame={self.frame_name}, "
            f"keypoints={self.keypoint_count}, "
            f"has_descriptors={self.has_descriptors})"
        )


class FeatureExtractor:
    """
    ORB tabanlı feature extraction ve semantic maskeleme.

    Pipeline'daki yeri:
        1. Frame undistort  → CameraCalibration
        2. Semantic mask    → YOLO detections → dinamik objeler engellenir
        3. ORB detect       → FAST keypoints
        4. ORB describe     → rBRIEF descriptors

    Phase 2 swap notu:
        SuperPoint ile değiştirildiğinde bu sınıf kaldırılır,
        aynı extract() interface'ini sunan SuperPointExtractor yazılır.
        odometry.py'da hiçbir değişiklik gerekmez.
    """

    def __init__(
        self,
        data_loader: DataLoader,
        camera_calibration: CameraCalibration,
    ) -> None:
        """
        FeatureExtractor'ı başlatır.

        Args:
            data_loader        : Başlatılmış DataLoader instance'ı.
            camera_calibration : Başlatılmış CameraCalibration instance'ı.

        Raises:
            FeatureExtractorError: Config'de eksik parametre varsa.
        """
        logger.info("[FeatureExtractor] Başlatılıyor...")

        self._loader = data_loader
        self._cam = camera_calibration

        feat_cfg = data_loader.get_feature_config()
        sem_cfg = data_loader.get_semantic_config()

        self._orb = self._build_orb(feat_cfg)
        self._dynamic_classes = self._parse_dynamic_classes(sem_cfg)

        logger.info(
            "[FeatureExtractor] ORB kuruldu — max_features=%d, dynamic_classes=%s",
            feat_cfg["max_features"],
            self._dynamic_classes,
        )

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_orb(feat_cfg: dict) -> cv2.ORB:
        """
        Config'den ORB detector oluşturur.

        Args:
            feat_cfg: config.yaml features bloğu.

        Returns:
            cv2.ORB instance'ı.

        Raises:
            FeatureExtractorError: Eksik veya geçersiz parametre varsa.
        """
        required = ["max_features", "scale_factor", "n_levels"]
        missing = [k for k in required if k not in feat_cfg]
        if missing:
            raise FeatureExtractorError(
                f"features config'inde eksik anahtarlar: {missing}"
            )

        try:
            return cv2.ORB_create(
                nfeatures=int(feat_cfg["max_features"]),
                scaleFactor=float(feat_cfg["scale_factor"]),
                nlevels=int(feat_cfg["n_levels"]),
            )
        except Exception as e:
            raise FeatureExtractorError(f"ORB oluşturulamadı: {e}") from e

    @staticmethod
    def _parse_dynamic_classes(sem_cfg: dict) -> set:
        """
        Config'den dinamik class ID setini parse eder.

        Args:
            sem_cfg: config.yaml semantic bloğu.

        Returns:
            int set — maskelenecek class ID'leri.

        Raises:
            FeatureExtractorError: dynamic_classes eksikse.
        """
        if "dynamic_classes" not in sem_cfg:
            raise FeatureExtractorError(
                "semantic config'inde 'dynamic_classes' anahtarı bulunamadı."
            )
        return set(int(c) for c in sem_cfg["dynamic_classes"])

    # ------------------------------------------------------------------
    # Semantic Mask
    # ------------------------------------------------------------------

    def _build_semantic_mask(
        self,
        frame_shape: Tuple[int, int],
        detections: List[Detection],
    ) -> np.ndarray:
        """
        YOLO tespitlerinden semantic mask üretir.

        Mask mantığı:
            255 (beyaz) → statik bölge — ORB buradan feature çıkarır
            0   (siyah) → dinamik obje  — ORB bu bölgeyi yok sayar

        Sadece config'deki dynamic_classes'a ait bbox'lar maskelenir.
        Diğer class'lar (UAP, UAI, MAKİNE) maskelenmez.

        Args:
            frame_shape: (height, width) tuple.
            detections : Bu frame'e ait Detection listesi.

        Returns:
            uint8 mask array, shape=(H, W).
        """
        mask = np.ones(frame_shape, dtype=np.uint8) * 255

        masked_count = 0
        for det in detections:
            if det.class_id not in self._dynamic_classes:
                continue

            bbox = det.bbox
            if len(bbox) != 4:
                logger.warning(
                    "[FeatureExtractor] Geçersiz bbox formatı, atlanıyor: %s", bbox
                )
                continue

            x1, y1, x2, y2 = map(int, bbox)

            # Sınır kontrolü — bbox frame dışına taşabilir
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame_shape[1], x2)
            y2 = min(frame_shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                logger.warning(
                    "[FeatureExtractor] Dejenere bbox, atlanıyor: [%d,%d,%d,%d]",
                    x1, y1, x2, y2,
                )
                continue

            cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)
            masked_count += 1

        logger.debug(
            "[FeatureExtractor] Semantic mask: %d/%d tespit maskelendi.",
            masked_count, len(detections),
        )
        return mask

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, frame: np.ndarray, frame_name: str) -> FrameFeatures:
        """
        Tek bir frame üzerinde tam feature extraction pipeline'ını çalıştırır.

        Adımlar:
            1. Undistort
            2. Grayscale dönüşümü
            3. Semantic mask üretimi
            4. ORB detect + compute (mask uygulanmış)

        Args:
            frame      : Raw BGR frame (diskten okunmuş).
            frame_name : Frame dosya adı (detections lookup için).

        Returns:
            FrameFeatures dataclass instance'ı.

        Raises:
            FeatureExtractorError: Kritik bir hata oluşursa.
        """
        try:
            # 1. Undistort
            clean_frame = self._cam.undistort(frame)

            # 2. Grayscale
            gray = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2GRAY)

            # 3. Semantic mask
            detections = self._loader.get_detections(frame_name)
            mask = self._build_semantic_mask(gray.shape, detections)

            # 4. ORB detect + compute
            keypoints, descriptors = self._orb.detectAndCompute(gray, mask)

            if keypoints is None:
                keypoints = []

            logger.debug(
                "[FeatureExtractor] %s → %d keypoint bulundu.",
                frame_name, len(keypoints),
            )

            return FrameFeatures(
                frame_name=frame_name,
                keypoints=keypoints,
                descriptors=descriptors,
                mask=mask,
                clean_frame=clean_frame,
            )

        except CameraCalibrationError as e:
            raise FeatureExtractorError(
                f"Undistort hatası ({frame_name}): {e}"
            ) from e
        except cv2.error as e:
            raise FeatureExtractorError(
                f"OpenCV hatası ({frame_name}): {e}"
            ) from e

    def extract_from_loader(self, frame_name: str) -> FrameFeatures:
        """
        Frame adından otomatik olarak frame'i yükler ve extract() çağırır.

        Args:
            frame_name: Frame dosya adı.

        Returns:
            FrameFeatures instance'ı.
        """
        frame_path = next(
            (p for p in self._loader.frame_list if p.name == frame_name), None
        )
        if frame_path is None:
            raise FeatureExtractorError(
                f"Frame bulunamadı: {frame_name}"
            )
        frame = self._loader.load_frame(frame_path)
        return self.extract(frame, frame_name)

    def __repr__(self) -> str:
        return (
            f"FeatureExtractor("
            f"dynamic_classes={self._dynamic_classes})"
        )


# ------------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config_path = __import__("pathlib").Path(__file__).resolve().parent.parent / "config.yaml"

    try:
        loader = DataLoader(str(config_path))
        cam = CameraCalibration(loader)
        extractor = FeatureExtractor(loader, cam)

        # İlk 3 frame üzerinde test
        print("\n--- Feature Extraction Testi ---")
        for idx, name, frame in loader.frame_generator():
            features = extractor.extract(frame, name)
            print(f"[{idx}] {features}")

            # Mask istatistikleri
            total_pixels = features.mask.size
            masked_pixels = int(np.sum(features.mask == 0))
            mask_ratio = masked_pixels / total_pixels * 100
            print(f"     Maskelenen alan: {mask_ratio:.2f}%")

            if idx >= 2:
                break

        # Debug görsel — ilk frame
        print("\n--- Debug Görsel Üretiliyor ---")
        first_name = loader.frame_list[0].name
        features = extractor.extract_from_loader(first_name)

        debug_img = cv2.drawKeypoints(
            features.clean_frame,
            features.keypoints,
            None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )

        # Mask'i görsel olarak üst üste koy
        mask_colored = cv2.cvtColor(features.mask, cv2.COLOR_GRAY2BGR)
        mask_colored[features.mask == 0] = [0, 0, 80]  # Maskelenen alan koyu kırmızı

        cv2.imwrite("debug_feature_extraction.jpg", debug_img)
        cv2.imwrite("debug_semantic_mask.jpg", mask_colored)
        print(f"  debug_feature_extraction.jpg → {features.keypoint_count} keypoint")
        print(f"  debug_semantic_mask.jpg → maskeleme görsel")

    except (DataLoaderError, CameraCalibrationError, FeatureExtractorError) as e:
        logger.error("Hata: %s", e)
        sys.exit(1)