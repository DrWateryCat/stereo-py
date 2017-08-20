from childcamera import ChildCamera
from stereocamera import StereoCamera

import cv2

from matplotlib import pyplot as plt

def main():
    left_camera_calibration_file = "left_calib.npz"
    right_camera_calibration_file = "right_calib.npz"

    left_camera = ChildCamera(0, left_camera_calibration_file)
    right_camera = ChildCamera(1, right_camera_calibration_file)

    stereo = StereoCamera(left_camera, right_camera)

    disparity = stereo.get_depth_map()

    plt.imshow(disparity, 'gray')
    plt.show()

if __name__ == '__main__':
    main()