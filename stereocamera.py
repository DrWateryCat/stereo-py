import cv2
from childcamera import ChildCamera

class StereoCamera:
    def __init__(self, left_camera, right_camera):
        self.left_camera = left_camera
        self.right_camera = right_camera
        self.stereo_kernel = cv2.StereoBM_Create()

    def get_depth_map(self):
        left_img = self.left_camera.get_frame_undistorted()
        right_img = self.right_camera.get_frame_undistorted()

        disparity = self.stereo_kernel.compute(left_img, right_img)
        return disparity