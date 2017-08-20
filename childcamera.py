import cv2
import numpy as np

class ChildCamera:
    def __init__(self, cameraID, calibration_file):
        self.cap = cv2.VideoCapture(cameraID)

        f = np.load(calibration_file)
        self.camera_matrix = f['camera_matrix']
        self.dist_coefficients = f['distortion_coefficients']

    def get_frame(self):
        unused, frame = self.cap.read()
        return frame

    def get_frame_undistorted(self):
        distorted = self.get_frame()
        h, w = distorted[:2]

        newcameratx, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coefficients, (w, h), 1, (w, h))

        undistorted = cv2.undistort(distorted, self.camera_matrix, self.dist_coefficients, None, newcameratx)

        x, y, w, h = roi
        undistorted = undistorted[y:y+h, x:x+w]

        return undistorted