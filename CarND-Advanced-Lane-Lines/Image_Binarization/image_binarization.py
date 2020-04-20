import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import time
import os
import sys
# For loading other py code
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from Camera_Calibration.camera_calibration import calibrate_camera, undistort_image


# selected threshold to highlight yellow lines
YELLOW_HSV_TH_MIN = np.array([0, 70, 70])
YELLOW_HSV_TH_MAX = np.array([50, 255, 255])


def thresh_frame_hsv(frame, min_values, max_values, display=False):
    """
    Threshold a color frame in HSV space
    """
    image_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    min_th = np.all(image_hsv > min_values, axis=2)
    max_th = np.all(image_hsv < max_values, axis=2)

    image_out = np.logical_and(min_th, max_th)

    if display:
        plt.imshow(image_out, cmap='gray')
        plt.show()

    return image_out


def thresh_frame_sobel(frame, kernel_size):
    """
    Apply Sobel edge detection to an input frame, then threshold the result
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size)

    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_mag = np.uint8(sobel_mag / np.max(sobel_mag) * 255)

    _, sobel_mag = cv2.threshold(sobel_mag, 50, 1, cv2.THRESH_BINARY)

    return sobel_mag.astype(bool)


def get_binary_from_equalized_grayscale(frame):
    """
    Apply histogram equalization to an input frame, threshold it and return the (binary) result.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    eq_global = cv2.equalizeHist(gray)

    _, th = cv2.threshold(eq_global, thresh=250, maxval=255, type=cv2.THRESH_BINARY)

    return th


def binarize_image(img, img_name='', display=False):
    """
    Convert an input frame to a binary image which highlight as most as possible the lane-lines.

    :param img: input color frame
    :param display: if True, show intermediate results
    :return: binarized frame
    """
    h, w = img.shape[:2]

    binary = np.zeros(shape=(h, w), dtype=np.uint8)

    # highlight yellow lines by threshold in HSV color space
    hsv_yellow_mask = thresh_frame_hsv(img, YELLOW_HSV_TH_MIN, YELLOW_HSV_TH_MAX, display=False)
    binary = np.logical_or(binary, hsv_yellow_mask)

    # highlight white lines by thresholding the equalized frame
    eq_white_mask = get_binary_from_equalized_grayscale(img)
    binary = np.logical_or(binary, eq_white_mask)

    # get Sobel binary mask (thresholded gradients)
    sobel_mask = thresh_frame_sobel(img, kernel_size=9)
    binary = np.logical_or(binary, sobel_mask)

    # apply a light morphology to "fill the gaps" in the binary image
    kernel = np.ones((5, 5), np.uint8)
    # closing = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    closing = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    if display:
        f, ax = plt.subplots(2, 3)
        f.set_facecolor('white')
        ax[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax[0, 0].set_title(img_name[15:-4])
        ax[0, 0].set_axis_off()
        ax[0, 0].set_axis_bgcolor('red')
        ax[0, 1].imshow(eq_white_mask, cmap='gray')
        ax[0, 1].set_title('white mask')
        ax[0, 1].set_axis_off()

        ax[0, 2].imshow(hsv_yellow_mask, cmap='gray')
        ax[0, 2].set_title('yellow mask')
        ax[0, 2].set_axis_off()

        ax[1, 0].imshow(sobel_mask, cmap='gray')
        ax[1, 0].set_title('sobel mask')
        ax[1, 0].set_axis_off()

        ax[1, 1].imshow(binary, cmap='gray')
        ax[1, 1].set_title('before closure')
        ax[1, 1].set_axis_off()

        ax[1, 2].imshow(closing, cmap='gray')
        ax[1, 2].set_title('after closure')
        ax[1, 2].set_axis_off()
        
        plt.savefig("../output_images/binrized_images/" + img_name[15:-4] + "_binarization_after.jpg",dpi=300)
        # plt.show()
        
        # plt.close()
        # plt.pause(0.3)

    return closing


if __name__ == '__main__':
    
    # load dir
    imgs_raw = glob.glob('../test_images/*.jpg')
    
    # for calibrating and undistorting
    calib_path='../camera_cal'
    mtx, dist = calibrate_camera(calib_path)
        
    for img_test in imgs_raw:
        load_img = cv2.imread(img_test)
        # undistort_image
        img_undistorted = undistort_image(load_img, mtx, dist, False)
        # undistort_image
        img_binarization = binarize_image(img=img_undistorted, img_name=img_test, display=True)
        # restore [0-1] to [0-255]
        cv2.imshow('closing', 255*img_binarization)   
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
    print("Binarizing imgs have finshed")
