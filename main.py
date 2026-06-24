# main.py
import cv2
import numpy as np
import json
import os
from core.odometry import PlanarOdometry
from utils.height_estimator import HeightEstimator

# --- AYARLAR VE YOLLAR ---
VIDEO_PATH = "data/masked_video4.mp4"
INPUT_LOG_PATH = "data/dynamic_objects4.json"
OUTPUT_MEMORY_PATH = "data/agent_memory4.json"

def load_visual_memory():
    
    if not os.path.exists(INPUT_LOG_PATH):
        print(f"[UYARI] Görsel log bulunamadı: {INPUT_LOG_PATH}")
        return {}
    
    with open(INPUT_LOG_PATH, 'r', encoding='utf-8') as f:
        raw_logs = json.load(f)
        
    
    memory_dict = {item["frame_id"]: item for item in raw_logs}
    print(f"[BİLGİ] {len(memory_dict)} karelik görsel hafıza yüklendi.")
    return memory_dict

def main():
    print("[BİLGİ] Modüler SLAM Motoru ve Veri Füzyonu Başlatılıyor...")
    
    visual_memory = load_visual_memory()
    agent_memory_log = [] # LLM için zenginleştirilmiş son liste
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    odom = PlanarOdometry(focal=1000.0)
    estimator = HeightEstimator(focal=1000.0)
    
    traj_canvas = np.zeros((600, 600, 3), dtype=np.uint8) # bu ne işe yarar
    prev_draw_x, prev_draw_y = 300, 300
    ret, prev_frame = cap.read()

    frame_id = 0 # Kareleri senkronize etmek için sayaç

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_id += 1 # Her döngüde 1 artır 

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, feature_mask = cv2.threshold(gray_frame, 10, 255, cv2.THRESH_BINARY) # feature mask ne işe yarar

        current_height = estimator.get_height(frame,feature_mask)
        cur_x, cur_y, keypoints, good_matches = odom.track(gray_frame, feature_mask, current_height)

        #data fusion
        if frame_id in visual_memory:
            enriched_data = visual_memory[frame_id]
            enriched_data["odometry"] = {
                "x_meter": round(cur_x , 2),
                "y_meter": round(cur_y , 2),
                "z_altitude": round(current_height, 2)  #neden 2'yi virgül ile ayırıyoruz 
            }
            agent_memory_log.append(enriched_data)
        
        print(f"[KARE {frame_id}] X: {cur_x:.2f}m | Y: {-cur_y:.2f}m | İRTİFA: {current_height:.2f}m")

        draw_x = int(cur_x * 2) + 300
        draw_y = int(cur_y * 2) + 300

        if 0 <= draw_x < 600 and 0 <= draw_y < 600:
            cv2.line(traj_canvas, (prev_draw_x, prev_draw_y), (draw_x, draw_y), (0, 255, 0), 2)
            prev_draw_x, prev_draw_y = draw_x, draw_y
            
            temp_canvas = traj_canvas.copy()
            cv2.circle(temp_canvas, (draw_x, draw_y), 5, (0, 0, 255), -1)
            cv2.imshow("Kusbakisi Rota", temp_canvas)

        if odom.prev_keypoints is not None and keypoints is not None and len(good_matches) > 0:
            try:
                match_frame = cv2.drawMatches(prev_frame, odom.prev_keypoints, frame, keypoints, good_matches, None, flags=2, matchColor=(0, 255, 0))            
            except cv2.error:
                pass
        
        prev_frame = frame.copy()
        if cv2.waitKey(33) & 0xFF == ord('q'): break

    cap.release()    
    cv2.destroyAllWindows()

    with open(OUTPUT_MEMORY_PATH, 'w', encoding='utf-8') as f:
        json.dump(agent_memory_log, f, ensure_ascii=False, indent=4)
    print(f"\n[BAŞARILI] Ajan Hafızası Kaydedildi: {OUTPUT_MEMORY_PATH}")

if __name__ == "__main__":
    main()

