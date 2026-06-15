import cv2
from ultralytics import YOLO

# --- AYARLAR ---
# Dosya isimlerini kendi koyduğun isimlere göre güncelle!
MODEL_PATH = "models/best.pt" 
VIDEO_PATH = "data/test.mp4"

print("[BİLGİ] Model yükleniyor...")
model = YOLO(MODEL_PATH)

print("[BİLGİ] Video açılıyor...")
cap = cv2.VideoCapture(VIDEO_PATH)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("[BİLGİ] Video bitti veya okunamadı.")
        break
        
    # YOLO ile inference (çıkarım) yap
    results = model(frame, conf=0.5) # Güven skoru %50 üzeri olanları göster
    
    # Tespit edilen kutuları çizin
    annotated_frame = results[0].plot()
    
    # Görüntüyü ekrana bas
    cv2.imshow("Drone SLAM - Model Kesifi", annotated_frame)
    
    # Çıkmak için 'q' tuşuna bas
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()