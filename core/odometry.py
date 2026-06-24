# core/odometry.py
import cv2
import numpy as np

class PlanarOdometry:
    def __init__(self, focal=1000.0):
        self.focal = focal
        self.orb = cv2.ORB_create(nfeatures=1000)
        
        index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
        search_param = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_param)
        
        self.cur_x = 0.0
        self.cur_y = 0.0
        self.prev_keypoints = None
        self.prev_descriptors = None

    def track(self, gray_frame, feature_mask, current_height):
        keypoints, descriptors = self.orb.detectAndCompute(gray_frame, mask=feature_mask)
        good_matches = []
        
        if self.prev_descriptors is not None and descriptors is not None:
            matches = self.flann.knnMatch(self.prev_descriptors, descriptors, k=2)
            
            for match_set in matches:
                if len(match_set) == 2:
                    m, n = match_set
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) > 8:
                pts1 = np.float32([self.prev_keypoints[m.queryIdx].pt for m in good_matches])
                pts2 = np.float32([keypoints[m.trainIdx].pt for m in good_matches])

                # Medyan Kayma
                dx_pixel = np.median(pts1[:, 0] - pts2[:, 0]) 
                dy_pixel = np.median(pts1[:, 1] - pts2[:, 1]) 

                # Metreye Çevir ve Ekle
                self.cur_x += (dx_pixel * current_height) / self.focal
                self.cur_y += (dy_pixel * current_height) / self.focal

        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        
        return self.cur_x, self.cur_y, keypoints, good_matches