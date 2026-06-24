# main.py
import cv2
import numpy as np
from core.odometry import PlanarOdometry
from utils.height_estimator import HeightEstimator

VIDEO_PATH = "data/masked_video.mp4"

def main():
    print("[BİLGİ] Modüler SLAM Motoru Başlatılıyor...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    # Motorları Başlat
    odom = PlanarOdometry(focal=1000.0)
    estimator = HeightEstimator(focal=1000.0)
    
    traj_canvas = np.zeros((600, 600, 3), dtype=np.uint8)
    prev_draw_x, prev_draw_y = 300, 300
    ret, prev_frame = cap.read()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, feature_mask = cv2.threshold(gray_frame, 10, 255, cv2.THRESH_BINARY)

        # 1. Aşama: İrtifayı Al (Kalman içerde çalışır)
        current_height = estimator.get_height(frame, feature_mask)

        # 2. Aşama: Hareketi Hesapla (ORB içerde çalışır)
        cur_x, cur_y, keypoints, good_matches = odom.track(gray_frame, feature_mask, current_height)

        # 3. Aşama: Terminal Raporu
        print(f"[MOTOR] X: {cur_x:.2f}m | Y: {-cur_y:.2f}m | İRTİFA: {current_height:.2f}m")

        # --- ÇİZİM İŞLEMLERİ ---
        draw_x = int(cur_x * 20) + 300
        draw_y = int(cur_y * 20) + 300

        if 0 <= draw_x < 600 and 0 <= draw_y < 600:
            cv2.line(traj_canvas, (prev_draw_x, prev_draw_y), (draw_x, draw_y), (0, 255, 0), 2)
            prev_draw_x, prev_draw_y = draw_x, draw_y
            
            temp_canvas = traj_canvas.copy()
            cv2.circle(temp_canvas, (draw_x, draw_y), 5, (0, 0, 255), -1)
            cv2.imshow("Kusbakisi Rota", temp_canvas)

        if odom.prev_keypoints is not None and keypoints is not None:
            match_frame = cv2.drawMatches(prev_frame, odom.prev_keypoints, frame, keypoints, good_matches, None, flags=2, matchColor=(0, 255, 0))
            cv2.imshow("ORB Eslesmeleri", cv2.resize(match_frame, (frame.shape[1]//3, frame.shape[0]//3)))

        prev_frame = frame.copy()
        if cv2.waitKey(33) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()