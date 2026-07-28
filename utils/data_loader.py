"""
utils/data_loader.py
====================
Merkezi veri yönetim modülü.
Tüm I/O operasyonları buradan yönetilir — frame okuma, GT okuma, detections okuma.
Hiçbir modül doğrudan dosya sistemine erişmez, DataLoader üzerinden erişir.
"""

import cv2
import csv
import json
import logging
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)


class DataLoaderError(Exception):
    """DataLoader'a özgü hata sınıfı."""
    pass


class GroundTruthRecord:
    """Tek bir GT kaydını temsil eder."""
    __slots__ = ("frame_name", "tx", "ty")

    def __init__(self, frame_name: str, tx: float, ty: float) -> None:
        self.frame_name = frame_name
        self.tx = tx
        self.ty = ty

    def as_vector(self) -> np.ndarray:
        """[tx, ty] vektörünü döndürür."""
        return np.array([self.tx, self.ty], dtype=np.float64)

    def __repr__(self) -> str:
        return f"GroundTruthRecord(frame={self.frame_name}, tx={self.tx:.6f}, ty={self.ty:.6f})"


class Detection:
    """Tek bir YOLO tespitini temsil eder."""
    __slots__ = ("class_id", "confidence", "bbox")

    def __init__(self, class_id: int, confidence: float, bbox: List[int]) -> None:
        self.class_id = class_id
        self.confidence = confidence
        self.bbox = bbox  # [x1, y1, x2, y2]

    def __repr__(self) -> str:
        return f"Detection(class_id={self.class_id}, conf={self.confidence:.3f}, bbox={self.bbox})"


class DataLoader:
    """
    Merkezi veri yönetim sınıfı.

    Sorumluluklar:
    - raw_frames/ klasöründen frame'leri lexicographic sırayla okur
    - ground-truth.csv'yi parse eder ve frame adına göre index'ler
    - detections.json'ı parse eder ve frame adına göre index'ler
    - config.yaml'ı okur ve ilgili modüllere dağıtır

    Hiçbir zaman:
    - Görüntü işleme yapmaz
    - Matris hesabı yapmaz
    - Harici bir modülün iç state'ini değiştirmez
    """

    def __init__(self, config_path: str) -> None:
        """
        DataLoader'ı başlatır.

        Args:
            config_path: config.yaml dosyasının tam yolu.

        Raises:
            DataLoaderError: Config dosyası bulunamazsa veya eksik alan varsa.
        """
        self.config_path = Path(config_path)
        self._validate_path(self.config_path, "Config dosyası")

        logger.info("[DataLoader] Başlatılıyor: %s", self.config_path)

        self.config = self._load_config()
        self.base_dir = self.config_path.parent

        # Kritik yolları config'den türet
        self.frames_dir = self.base_dir / "data" / "raw_frames"
        self.gt_path = self.base_dir / self.config["evaluation"]["ground_truth_file"]
        self.detections_path = self.base_dir / "data" / "detections.json"

        # Yol validasyonları
        self._validate_path(self.frames_dir, "raw_frames klasörü")
        self._validate_path(self.gt_path, "Ground-truth CSV")
        self._validate_path(self.detections_path, "detections.json")

        # Veri index'leri — lazy load, ilk erişimde yüklenir
        self._gt_index: Optional[Dict[str, GroundTruthRecord]] = None
        self._detections_index: Optional[Dict[str, List[Detection]]] = None
        self._frame_list: Optional[List[Path]] = None

        logger.info("[DataLoader] Başlatma tamamlandı.")

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def _load_config(self) -> dict:
        """config.yaml'ı okur ve döndürür."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise DataLoaderError(f"Config parse hatası: {e}") from e

        required_keys = ["camera_rgb", "features", "semantic", "evaluation", "hybrid"]
        missing = [k for k in required_keys if k not in config]
        if missing:
            raise DataLoaderError(f"Config'de eksik anahtarlar: {missing}")

        logger.debug("[DataLoader] Config yüklendi: %s", list(config.keys()))
        return config

    # ------------------------------------------------------------------
    # Frame yönetimi
    # ------------------------------------------------------------------

    def _build_frame_list(self) -> List[Path]:
        """
        raw_frames/ klasöründeki tüm geçerli görüntü dosyalarını
        lexicographic sırayla listeler.

        Returns:
            Sıralı Path listesi.

        Raises:
            DataLoaderError: Klasörde hiç geçerli frame yoksa.
        """
        valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
        frames = [
            f for f in self.frames_dir.iterdir()
            if f.suffix.lower() in valid_extensions
        ]

        if not frames:
            raise DataLoaderError(
                f"raw_frames klasöründe geçerli görüntü bulunamadı: {self.frames_dir}"
            )

        # Lexicographic sort — isimlendirme formatından bağımsız çalışır
        frames.sort(key=lambda p: p.name)
        logger.info("[DataLoader] %d frame bulundu.", len(frames))
        return frames

    @property
    def frame_list(self) -> List[Path]:
        """Lazy-loaded frame listesi."""
        if self._frame_list is None:
            self._frame_list = self._build_frame_list()
        return self._frame_list

    @property
    def total_frames(self) -> int:
        """Toplam frame sayısı."""
        return len(self.frame_list)

    def load_frame(self, frame_path: Path) -> np.ndarray:
        """
        Tek bir frame'i diskten okur.

        Args:
            frame_path: Okunacak frame'in Path nesnesi.

        Returns:
            BGR formatında numpy array.

        Raises:
            DataLoaderError: Dosya okunamazsa.
        """
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise DataLoaderError(f"Frame okunamadı: {frame_path}")
        return frame

    def frame_generator(self) -> Generator[Tuple[int, str, np.ndarray], None, None]:
        """
        Tüm frame'leri sırayla yield eden generator.

        Yields:
            (frame_index, frame_name, frame_bgr) tuple'ı.

        Kullanım:
            for idx, name, frame in loader.frame_generator():
                ...
        """
        for idx, frame_path in enumerate(self.frame_list):
            frame = self.load_frame(frame_path)
            logger.debug("[DataLoader] Frame yüklendi: %s (%d/%d)", frame_path.name, idx + 1, self.total_frames)
            yield idx, frame_path.name, frame

    # ------------------------------------------------------------------
    # Ground Truth
    # ------------------------------------------------------------------

    def _load_ground_truth(self) -> Dict[str, GroundTruthRecord]:
        """
        ground-truth.csv'yi parse eder ve frame adına göre index'ler.

        CSV formatı: translation_x, translation_y, frame_numbers

        Returns:
            {frame_name: GroundTruthRecord} dict'i.

        Raises:
            DataLoaderError: CSV parse hatası veya eksik sütun.
        """
        index: Dict[str, GroundTruthRecord] = {}
        required_columns = {"translation_x", "translation_y", "frame_numbers"}

        try:
            with open(self.gt_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)

                if reader.fieldnames is None:
                    raise DataLoaderError("Ground-truth CSV boş veya başlık satırı yok.")

                actual_columns = set(reader.fieldnames)
                missing_cols = required_columns - actual_columns
                if missing_cols:
                    raise DataLoaderError(
                        f"Ground-truth CSV'de eksik sütunlar: {missing_cols}"
                    )

                for line_num, row in enumerate(reader, start=2):
                    try:
                        frame_name = row["frame_numbers"].strip()
                        tx = float(row["translation_x"])
                        ty = float(row["translation_y"])
                        index[frame_name] = GroundTruthRecord(frame_name, tx, ty)
                    except (ValueError, KeyError) as e:
                        logger.warning(
                            "[DataLoader] GT satır %d parse hatası, atlanıyor: %s", line_num, e
                        )
                        continue

        except OSError as e:
            raise DataLoaderError(f"Ground-truth CSV okunamadı: {e}") from e

        logger.info("[DataLoader] Ground-truth yüklendi: %d kayıt.", len(index))
        return index

    @property
    def gt_index(self) -> Dict[str, GroundTruthRecord]:
        """Lazy-loaded GT index'i."""
        if self._gt_index is None:
            self._gt_index = self._load_ground_truth()
        return self._gt_index

    def get_ground_truth(self, frame_name: str) -> Optional[GroundTruthRecord]:
        """
        Belirli bir frame için GT kaydını döndürür.

        Args:
            frame_name: Frame dosya adı (örn. "frame_000000.jpg").

        Returns:
            GroundTruthRecord veya None (kayıt yoksa).
        """
        # CSV'de uzantısız isim olabilir, her iki formatta da ara
        record = self.gt_index.get(frame_name)
        if record is None:
            stem = Path(frame_name).stem
            record = self.gt_index.get(stem)
        return record

    # ------------------------------------------------------------------
    # Detections
    # ------------------------------------------------------------------

    def _load_detections(self) -> Dict[str, List[Detection]]:
        """
        detections.json'ı parse eder ve frame adına göre index'ler.

        Returns:
            {frame_name: [Detection, ...]} dict'i.

        Raises:
            DataLoaderError: JSON parse hatası.
        """
        try:
            with open(self.detections_path, "r", encoding="utf-8") as f:
                raw: Dict = json.load(f)
        except json.JSONDecodeError as e:
            raise DataLoaderError(f"detections.json parse hatası: {e}") from e
        except OSError as e:
            raise DataLoaderError(f"detections.json okunamadı: {e}") from e

        index: Dict[str, List[Detection]] = {}
        for frame_name, det_list in raw.items():
            detections = []
            for det in det_list:
                try:
                    detections.append(Detection(
                        class_id=int(det["class_id"]),
                        confidence=float(det["confidence"]),
                        bbox=[int(v) for v in det["bbox"]],
                    ))
                except (KeyError, ValueError) as e:
                    logger.warning(
                        "[DataLoader] Frame %s tespit parse hatası, atlanıyor: %s",
                        frame_name, e,
                    )
                    continue
            index[frame_name] = detections

        logger.info("[DataLoader] Detections yüklendi: %d frame.", len(index))
        return index

    @property
    def detections_index(self) -> Dict[str, List[Detection]]:
        """Lazy-loaded detections index'i."""
        if self._detections_index is None:
            self._detections_index = self._load_detections()
        return self._detections_index

    def get_detections(self, frame_name: str) -> List[Detection]:
        """
        Belirli bir frame için tespit listesini döndürür.

        Args:
            frame_name: Frame dosya adı.

        Returns:
            Detection listesi. Frame'de tespit yoksa boş liste.
        """
        return self.detections_index.get(frame_name, [])

    # ------------------------------------------------------------------
    # Config erişim yardımcıları
    # ------------------------------------------------------------------

    def get_camera_config(self) -> dict:
        """Kamera kalibrasyon parametrelerini döndürür."""
        return self.config["camera_rgb"]

    def get_feature_config(self) -> dict:
        """Feature extraction parametrelerini döndürür."""
        return self.config["features"]

    def get_semantic_config(self) -> dict:
        """Semantic ölçeklendirme parametrelerini döndürür."""
        return self.config["semantic"]

    def get_hybrid_config(self) -> dict:
        """Hybrid AI geçiş parametrelerini döndürür."""
        return self.config["hybrid"]

    def get_evaluation_config(self) -> dict:
        """Değerlendirme parametrelerini döndürür."""
        return self.config["evaluation"]

    # ------------------------------------------------------------------
    # Yardımcı metodlar
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_path(path: Path, label: str) -> None:
        """
        Bir Path'in var olup olmadığını kontrol eder.

        Args:
            path: Kontrol edilecek Path.
            label: Hata mesajında kullanılacak etiket.

        Raises:
            DataLoaderError: Path mevcut değilse.
        """
        if not path.exists():
            raise DataLoaderError(f"{label} bulunamadı: {path}")

    def summary(self) -> str:
        """DataLoader durumunu özetleyen string döndürür."""
        return (
            f"DataLoader Summary\n"
            f"  Config     : {self.config_path}\n"
            f"  Frames dir : {self.frames_dir}\n"
            f"  GT path    : {self.gt_path}\n"
            f"  Detections : {self.detections_path}\n"
            f"  Total frames (lazy): {'yüklenmedi' if self._frame_list is None else len(self._frame_list)}\n"
            f"  GT records (lazy)  : {'yüklenmedi' if self._gt_index is None else len(self._gt_index)}\n"
            f"  Detections (lazy)  : {'yüklenmedi' if self._detections_index is None else len(self._detections_index)}\n"
        )

    def __repr__(self) -> str:
        return f"DataLoader(config={self.config_path})"


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
        print(loader.summary())

        # Frame listesi testi
        print(f"\nİlk 3 frame: {[f.name for f in loader.frame_list[:3]]}")
        print(f"Son 3 frame : {[f.name for f in loader.frame_list[-3:]]}")

        # GT testi
        first_frame = loader.frame_list[0].name
        gt = loader.get_ground_truth(first_frame)
        print(f"\nGT ({first_frame}): {gt}")

        # Detections testi
        dets = loader.get_detections(first_frame)
        print(f"Detections ({first_frame}): {len(dets)} tespit")
        for d in dets[:3]:
            print(f"  {d}")

        # Frame generator testi — sadece ilk 3 frame
        print("\nFrame generator testi (ilk 3):")
        for idx, name, frame in loader.frame_generator():
            print(f"  [{idx}] {name} — shape: {frame.shape}")
            if idx >= 2:
                break

    except DataLoaderError as e:
        logger.error("DataLoader hatası: %s", e)
        sys.exit(1)