import cv2
import numpy as np
# --- YOLLAR ---
VIDEO_PATH = "data/masked_video2.mp4"

def test_orb_features():
    print("[BİLGİ] Haritacı (ORB) Sahaya İniyor...")
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[HATA] Maskelenmiş video bulunamadı! Lütfen yolu kontrol et: {VIDEO_PATH}")
        return

    # ORB Algoritmasını Başlat (Maksimum 1000 nokta yakalayacak)
    orb = cv2.ORB_create(nfeatures=1000)

    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=1
    )
    search_param = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_param)
    print("[BİLGİ] : AÇILIYOR FLANN")


    print("[BİLGİ] Taktiksel Görüş Ekranı Açılıyor... (Çıkmak için ekrandayken 'q' tuşuna bas)")

    prev_keypoints = None
    prev_descriptors = None
    prev_frame = None

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
        
        if prev_descriptors is not None and descriptors is not None:
            matches = flann.knnMatch(prev_descriptors, descriptors,k=2) #k en benzeyen iki aday
            good_matches = []

            #matches'ları eleriz ki en yüksek doğruluk oranı olanla eşleşsin
            for match_set in matches:
                if len(match_set) ==2:
                    m, n =match_set
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            
            match_frame = cv2.drawMatches(
                prev_frame, prev_keypoints, 
                frame, keypoints, 
                good_matches, None, 
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                matchColor=(0, 255, 0) # Eşleşen çizgiler fıstık yeşili olacak
            )
            h, w = match_frame.shape[:2]
            resized_match_frame = cv2.resize(match_frame,(w//3, h//3))
            cv2.imshow("FLAN EŞLEŞTİRME ",resized_match_frame)

        prev_keypoints = keypoints
        prev_descriptors = descriptors
        prev_frame = frame
        
        # 'q' tuşuna basarsan çık, yoksa videoyu normal hızında (~30 FPS) oynat
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_orb_features()