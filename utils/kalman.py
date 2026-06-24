import numpy as np

class KalmanFilter:
    def __init__(self, dt=0.033, process_noise=0.01, measurement_noise=1.0):
        self.dt = dt
        self.x = np.array([[25.0], [0.0]]) # [Height, Velocity]
        self.F = np.array([[1.0, self.dt], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        self.P = np.eye(2) * 1.0
        self.Q = np.eye(2) * process_noise
        self.R = np.array([[measurement_noise]])

    def predict(self):
        self.x = self.F.dot(self.x)
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q
        return self.x[0][0]

    def update(self, z):
        y = z - self.H.dot(self.x)
        S = self.H.dot(self.P).dot(self.H.T) + self.R
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))
        self.x = self.x + K.dot(y)
        self.P = (np.eye(2) - K.dot(self.H)).dot(self.P)
        return self.x[0][0]