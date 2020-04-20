#!/usr/bin/python2
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import os.path as path
import pickle
import time


def determine_calibration(func):
    """
    The result parameters of camera calibration will be stored in \
    cam_param_py2.pickle, if camera has been calibrated.
    """
    calibrate_file = path.abspath(path.join(path.dirname(__file__), "./cam_param_py2.pickle"))
    # print (calibrate_file,"--------------------")
    def calibrate_judger(*args, **kwargs):
        if path.exists(calibrate_file):
            print('Camera has been calibrated, loading parameter file ... ')
            with open(calibrate_file) as cal_file:
                calibration = pickle.load(cal_file)
        else:
            print('Computing camera calibration ... ')
            calibration = func(*args, **kwargs)
            with open(calibrate_file, 'w') as cal_file:
                pickle.dump(calibration, cal_file, protocol=2)
        print('Camera calibration complete.')
        # print(calibration)
        return calibration
    return calibrate_judger


@determine_calibration
def calibrate_camera(calib_path, display=True):
    """
    Specify the path of calibration chessboard
    :param calib_path: The path of checkerboard
    :param display: if True, show lines and corners in chessboard
    :return: camera matrix and distortion
    """
    assert path.exists(calib_path), '"{}" calibration images must be exist.'.format(calib_path)
    # prepare world points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    # Set initial chessboard parameter matrix
    wdp = np.zeros((6 * 9, 3), np.float32)
    wdp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    # Arrays to store object points and pixel points from all the images.
    worldpoints = []  # 3d points in real world space
    pixelpoints = []  # 2d points in pixel plane.
    # loads imgs into a list
    imgs = glob.glob(path.join(calib_path, 'calibration*.jpg'))
    # Step through the list and search for chessboard corners
    for filename in imgs:
        img = cv2.imread(filename)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        pattern_flag, corners = cv2.findChessboardCorners(img_gray, (9, 6), None)
        if pattern_flag is True:
            worldpoints.append(wdp)
            pixelpoints.append(corners)
            # for display result of calibration
            if display:
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (9, 6), corners, pattern_flag)
                cv2.imshow('img',img)
                print("Picture: " + filename + ' calibrating ...')
                cv2.waitKey(300)
    if display:
        cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(worldpoints, pixelpoints, img_gray.shape[::-1], None, None)
    # print(mtx,dist)
    return mtx, dist


def undistort_image(frame, mtx, dist, display=True):
    """
    Undistort a frame given camera matrix and distortion coefficients.
    :param frame: input frame
    :param mtx: camera matrix
    :param dist: distortion coefficients
    :param display: if True, show frame before/after distortion correction
    :return: undistorted frame
    """
    frame_undistorted = cv2.undistort(frame, mtx, dist, newCameraMatrix=mtx)

    if display:
        fig, ax = plt.subplots(nrows=1, ncols=2)
        # fig.suptitle('Undistort Image Before & After')
        ax[0].set_title('Before calibration')
        ax[1].set_title('After calibration')
        ax[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax[1].imshow(cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2RGB))
        
        #for comparing camera undistorted
        plt.savefig('../output_images/undistort_image_before_to_after.jpg',dpi=300)
        plt.show()

    return frame_undistorted


if __name__ == '__main__':
    
    # for calibrate cam
    calib_path='../camera_cal'
    mtx, dist = calibrate_camera(calib_path)
    
    
    imgs_raw = glob.glob('../test_images/*.jpg')
    for img_raw in imgs_raw:
        load_img = cv2.imread(img_raw)
        img_undistorted = undistort_image(load_img, mtx, dist, False)
        cv2.imwrite("../output_images/calibrated_images/" + img_raw[15:-4] + "_calibration_after.jpg", img_undistorted)
    print("Undistorting imgs have finshed")
        
