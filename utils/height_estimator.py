# utils/height_estimator.py
import cv2
import numpy as np
from utils.kalman import KalmanFilter

class HeightEstimator:
    def __init__(self, focal=1000.0):
        self.focal = focal
        # Kalman'ı burada başlatıyoruz
        self.kf = KalmanFilter(dt=0.033, process_noise=0.05, measurement_noise=2.0)

    def get_height(self, frame, feature_mask):
        inverted_mask = cv2.bitwise_not(feature_mask)
        contours, _ = cv2.findContours(inverted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_heights = []
        for c in contours:
            area = cv2.contourArea(c)
            if 1500 < area < 30000:
                x_b, y_b, w_box, h_box = cv2.boundingRect(c)
                car_pixel_width = min(w_box, h_box)
                
                if car_pixel_width > 0:
                    estimated_h = (1.8 * self.focal) / car_pixel_width
                    detected_heights.append(estimated_h)
                    # Sadece tespit edilen arabaları çiz
                    cv2.rectangle(frame, (x_b, y_b), (x_b + w_box, y_b + h_box), (255, 0, 0), 2)

        # Kalman Güncellemesi
        if len(detected_heights) > 0:
            current_instant_height = np.mean(detected_heights)
            self.kf.predict()
            return self.kf.update(current_instant_height)
        else:
            return self.kf.predict()