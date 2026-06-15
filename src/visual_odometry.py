import cv2

# --- YOLLAR ---
VIDEO_PATH = "data/masked_video.mp4"

def test_orb_features():
    print("[BİLGİ] Haritacı (ORB) Sahaya İniyor...")
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[HATA] Maskelenmiş video bulunamadı! Lütfen yolu kontrol et: {VIDEO_PATH}")
        return

    # ORB Algoritmasını Başlat (Maksimum 1000 nokta yakalayacak)
    orb = cv2.ORB_create(nfeatures=1000)

    print("[BİLGİ] Taktiksel Görüş Ekranı Açılıyor... (Çıkmak için ekrandayken 'q' tuşuna bas)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[BİLGİ] Video akışı bitti.")
            break

        # İşlemciyi yormamak için görüntüyü gri tonlamaya çeviriyoruz (Algoritmalar gri matris sever)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, feature_mask = cv2.threshold(gray_frame, 10,255, cv2.THRESH_BINARY)


        # Özellik Çıkarımı: Koordinatları (Keypoints) ve kimlikleri (Descriptors) bul
        keypoints, descriptors = orb.detectAndCompute(gray_frame, mask=feature_mask)
        
        # Bulunan noktaları orijinal renkli kare üzerine yemyeşil çiz
        frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0), flags=0)

        # Şov Vakti: Ekrana Bas
        cv2.imshow("ORB Taktiksel Gorus", frame_with_keypoints)

        # 'q' tuşuna basarsan çık, yoksa videoyu normal hızında (~30 FPS) oynat
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_orb_features()