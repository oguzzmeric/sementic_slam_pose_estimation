"""
utils/camera_calibration.py
============================
Kamera kalibrasyon modülü.
K matrisi, distorsiyon katsayıları ve undistort işlemlerini yönetir.
Hiçbir modül K matrisini kendisi oluşturmaz, bu modülden alır.
"""

import logging
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from utils.data_loader import DataLoader, DataLoaderError

logger = logging.getLogger(__name__)


class CameraCalibrationError(Exception):
    """CameraCalibration'a özgü hata sınıfı."""
    pass


class CameraCalibration:
    """
    Kamera kalibrasyon parametrelerini yönetir.

    Sorumluluklar:
    - config.yaml'dan K matrisi ve distorsiyon katsayılarını okur
    - Undistort map'lerini önceden hesaplar (performans optimizasyonu)
    - Frame undistort işlemi yapar
    - K matrisini ve türevlerini diğer modüllere sağlar

    Hiçbir zaman:
    - Feature extraction yapmaz
    - Görüntü analizi yapmaz
    - Dosya sistemine doğrudan erişmez
    """

    def __init__(self, data_loader: DataLoader) -> None:
        """
        CameraCalibration'ı başlatır.

        Args:
            data_loader: Başlatılmış DataLoader instance'ı.

        Raises:
            CameraCalibrationError: Config'de eksik veya geçersiz parametre varsa.
        """
        logger.info("[CameraCalibration] Başlatılıyor...")

        cam_cfg = data_loader.get_camera_config()
        self._parse_and_validate(cam_cfg)

        # Undistort map'lerini önceden hesapla — her frame'de tekrar hesaplanmaz
        self._undistort_map1, self._undistort_map2 = self._build_undistort_maps(
            cam_cfg["original_width"],
            cam_cfg["original_height"],
        )

        logger.info("[CameraCalibration] Başlatma tamamlandı.")
        logger.debug("[CameraCalibration] K matrisi:\n%s", self.K)
        logger.debug("[CameraCalibration] Distorsiyon: %s", self.dist_coeffs)

    # ------------------------------------------------------------------
    # Parse & Validate
    # ------------------------------------------------------------------

    def _parse_and_validate(self, cam_cfg: dict) -> None:
        """
        Config bloğunu parse eder ve K matrisini oluşturur.

        Args:
            cam_cfg: config.yaml'dan gelen camera_rgb bloğu.

        Raises:
            CameraCalibrationError: Eksik alan veya geçersiz değer varsa.
        """
        required_keys = ["fx", "fy", "cx", "cy", "distortion_coefficients",
                         "original_width", "original_height"]
        missing = [k for k in required_keys if k not in cam_cfg]
        if missing:
            raise CameraCalibrationError(
                f"camera_rgb config'inde eksik anahtarlar: {missing}"
            )

        try:
            fx = float(cam_cfg["fx"])
            fy = float(cam_cfg["fy"])
            cx = float(cam_cfg["cx"])
            cy = float(cam_cfg["cy"])
            dist = cam_cfg["distortion_coefficients"]
        except (TypeError, ValueError) as e:
            raise CameraCalibrationError(
                f"Kalibrasyon parametresi dönüşüm hatası: {e}"
            ) from e

        # Pozitiflik kontrolü
        if fx <= 0 or fy <= 0:
            raise CameraCalibrationError(
                f"Focal length pozitif olmalı. fx={fx}, fy={fy}"
            )

        if not isinstance(dist, (list, tuple)) or len(dist) != 5:
            raise CameraCalibrationError(
                f"distortion_coefficients 5 elemanlı liste olmalı, alınan: {dist}"
            )

        # K matrisi — 3x3 intrinsic matrix
        # [fx  0  cx]
        # [ 0 fy  cy]
        # [ 0  0   1]
        self.K = np.array([
            [fx,  0.0, cx],
            [0.0, fy,  cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        self.dist_coeffs = np.array(dist, dtype=np.float64)

        # K'nın inverse'i — back-projection için (semantic_scale.py kullanacak)
        self.K_inv = np.linalg.inv(self.K)

        # Scalar değerler — diğer modüllerin doğrudan erişimi için
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        self.image_width = int(cam_cfg["original_width"])
        self.image_height = int(cam_cfg["original_height"])

    # ------------------------------------------------------------------
    # Undistort map — önceden hesaplanır
    # ------------------------------------------------------------------

    def _build_undistort_maps(
        self,
        width: int,
        height: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Undistort remap map'lerini önceden hesaplar.

        cv2.undistort() her frame'de K ve dist'i tekrar işler.
        initUndistortRectifyMap + remap kombinasyonu map'leri bir kez hesaplar,
        sonraki her frame için sadece remap çağrılır — ~3x daha hızlı.

        Args:
            width: Görüntü genişliği (piksel).
            height: Görüntü yüksekliği (piksel).

        Returns:
            (map1, map2): cv2.remap için float32 map çifti.
        """
        map1, map2 = cv2.initUndistortRectifyMap(
            self.K,
            self.dist_coeffs,
            R=None,          # Rectification yok — monocular
            newCameraMatrix=self.K,
            size=(width, height),
            m1type=cv2.CV_32FC1,
        )
        logger.debug(
            "[CameraCalibration] Undistort map'leri hesaplandı: %dx%d", width, height
        )
        return map1, map2

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        """
        Verilen frame'e lens distorsiyon düzeltmesi uygular.

        initUndistortRectifyMap ile önceden hesaplanan map'leri kullanır.
        Her frame çağrısında K ve dist tekrar işlenmez.

        Args:
            frame: BGR formatında distorted görüntü.

        Returns:
            BGR formatında undistorted görüntü.

        Raises:
            CameraCalibrationError: Frame boyutu map boyutuyla uyuşmuyorsa.
        """
        h, w = frame.shape[:2]
        if w != self.image_width or h != self.image_height:
            raise CameraCalibrationError(
                f"Frame boyutu uyuşmuyor. "
                f"Beklenen: {self.image_width}x{self.image_height}, "
                f"Alınan: {w}x{h}"
            )

        return cv2.remap(
            frame,
            self._undistort_map1,
            self._undistort_map2,
            interpolation=cv2.INTER_LINEAR,
        )

    def project_point(self, point_3d: np.ndarray) -> np.ndarray:
        """
        3D kamera koordinatındaki bir noktayı 2D piksel koordinatına projekte eder.

        Projeksiyon denklemi:
            [u]   [fx  0  cx] [X/Z]
            [v] = [ 0 fy  cy] [Y/Z]
            [1]   [ 0  0   1] [ 1 ]

        Args:
            point_3d: [X, Y, Z] kamera koordinatı.

        Returns:
            [u, v] piksel koordinatı.
        """
        X, Y, Z = point_3d
        if abs(Z) < 1e-9:
            raise CameraCalibrationError("Z sıfıra yakın, projeksiyon tanımsız.")

        u = self.fx * (X / Z) + self.cx
        v = self.fy * (Y / Z) + self.cy
        return np.array([u, v], dtype=np.float64)

    def backproject_point(self, pixel: np.ndarray, depth: float) -> np.ndarray:
        """
        2D piksel koordinatını verilen derinlikle 3D kamera koordinatına geri projekte eder.

        Back-projection denklemi:
            P_c = Z * K_inv * [u, v, 1]^T

        Args:
            pixel: [u, v] piksel koordinatı.
            depth: Metrik derinlik (Z), metre cinsinden.

        Returns:
            [X, Y, Z] kamera koordinatı.
        """
        u, v = pixel
        p_hom = np.array([u, v, 1.0], dtype=np.float64)
        p_cam = depth * (self.K_inv @ p_hom)
        return p_cam

    def summary(self) -> str:
        """Kalibrasyon parametrelerini özetleyen string döndürür."""
        return (
            f"CameraCalibration Summary\n"
            f"  fx={self.fx:.4f}, fy={self.fy:.4f}\n"
            f"  cx={self.cx:.4f}, cy={self.cy:.4f}\n"
            f"  dist={self.dist_coeffs.tolist()}\n"
            f"  image size={self.image_width}x{self.image_height}\n"
            f"  K:\n{self.K}\n"
            f"  K_inv:\n{self.K_inv}\n"
        )

    def __repr__(self) -> str:
        return (
            f"CameraCalibration("
            f"fx={self.fx:.2f}, fy={self.fy:.2f}, "
            f"cx={self.cx:.2f}, cy={self.cy:.2f})"
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

    config_path = Path(__file__).resolve().parent.parent / "config.yaml"

    try:
        loader = DataLoader(str(config_path))
        cam = CameraCalibration(loader)

        print(cam.summary())

        # Undistort testi — ilk frame
        first_frame_path = loader.frame_list[0]
        raw_frame = loader.load_frame(first_frame_path)
        undistorted = cam.undistort(raw_frame)
        print(f"Undistort testi: {first_frame_path.name}")
        print(f"  Input shape : {raw_frame.shape}")
        print(f"  Output shape: {undistorted.shape}")

        # Back-projection testi
        pixel = np.array([cam.cx, cam.cy])  # görüntü merkezi
        depth = 50.0                         # 50 metre
        p_cam = cam.backproject_point(pixel, depth)
        print(f"\nBack-projection testi:")
        print(f"  Piksel: {pixel} → 3D: {p_cam}")

        # Projeksiyon testi — back-projection'ın tersini al
        p_proj = cam.project_point(p_cam)
        print(f"  3D: {p_cam} → Piksel: {p_proj}")
        print(f"  Round-trip hata: {np.linalg.norm(pixel - p_proj):.6f} piksel")

    except (DataLoaderError, CameraCalibrationError) as e:
        logger.error("Hata: %s", e)
        sys.exit(1)