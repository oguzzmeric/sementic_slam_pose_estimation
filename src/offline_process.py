import cv2
import json
from ultralytics import YOLO
import os
from tqdm import tqdm # İlerleme çubuğu için (Eğer yüklü değilse: pip install tqdm)

# --- AYARLAR VE YOLLAR ---
MODEL_PATH = "models/best.pt"
INPUT_VIDEO = "data/test4.mp4"
OUTPUT_VIDEO = "data/masked_video4.mp4"
OUTPUT_JSON = "data/dynamic_objects4.json"

TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080
TARGET_FPS = 30 # Sistemi bu FPS değerine sabitleyeceğiz

# Maskelenecek Dinamik Sınıflar
DYNAMIC_CLASSES = [1, 2, 3, 4, 5, 8, 9]

def process_video():
    print("[BİLGİ] YOLO11l Modeli Yükleniyor...")
    model = YOLO(MODEL_PATH)
    
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print("[HATA] Video dosyası bulunamadı!")
        return

    # Orijinal videonun metadatalarını oku
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[BİLGİ] Orijinal Video Analizi: {total_frames} Kare | {input_fps:.2f} FPS")

    # 30 FPS Sabitleme (Kare Atlama) Çarpanı
    # Örn: Video 60 FPS ise skip_ratio = 2 olur (Her 2 kareden 1'ini işler)
    skip_ratio = max(1, round(input_fps / TARGET_FPS))
    if skip_ratio > 1:
        print(f"[BİLGİ] Kare Atlama Aktif! Her {skip_ratio}. kare işlenecek (Hedef: {TARGET_FPS} FPS).")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Çıktı videosunu KESİNLİKLE sabit 30 FPS olarak ayarlıyoruz (Hızlanma olmasın diye)
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, TARGET_FPS, (TARGET_WIDTH, TARGET_HEIGHT))
    
    frame_id = 0
    processed_frame_count = 0
    intelligence_log = []
    
    print("[BİLGİ] İşlem Başladı...")
    
    # Körlemesine beklememek için döngüyü tqdm ilerleme çubuğuyla sarıyoruz
    with tqdm(total=total_frames, desc="Video İşleniyor") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_id += 1
            pbar.update(1) # İlerleme çubuğunu 1 adım kaydır
            
            # --- 30 FPS SABİTLEME FİLTRESİ ---
            # Eğer oran 2 ise ve frame_id tek sayıysa bu kareyi pas geç (İşlem yükünü %50 azaltır)
            if frame_id % skip_ratio != 0:
                continue
                
            processed_frame_count += 1
            
            # 1. Standartlaştırma (1080p)
            frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
            
            # 2. YOLO Tespiti
            results = model(frame, verbose=False)
            frame_objects = []
            
            # 3. Maskeleme
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                if class_id in DYNAMIC_CLASSES:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    class_name = model.names[class_id]
                    
                    obj_data = {
                        "class": class_name,
                        "confidence": round(conf, 2),
                        "bbox": [x1, y1, x2, y2],
                        "pixel_width": x2 - x1,
                        "pixel_height": y2 - y1
                    }
                    frame_objects.append(obj_data)
                    
                    # Karartma
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
            
            # Zaman damgasını artık sabit 30 FPS üzerinden hatasız hesaplıyoruz
            if frame_objects:
                intelligence_log.append({
                    "frame_id": processed_frame_count,
                    "timestamp_sec": round(processed_frame_count / TARGET_FPS, 2),
                    "objects": frame_objects
                })
                
            out.write(frame)

    cap.release()
    out.release()
    
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(intelligence_log, f, ensure_ascii=False, indent=4)
        
    print(f"\n[BAŞARILI] İşlem bitti! Toplam {total_frames} kareden {processed_frame_count} adedi seçilerek 30 FPS videoya dönüştürüldü.")

if __name__ == "__main__":
    process_video()