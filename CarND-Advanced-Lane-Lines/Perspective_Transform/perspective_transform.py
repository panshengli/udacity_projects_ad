import os
import sys
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
# For loading other py code
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from Camera_Calibration.camera_calibration import calibrate_camera, undistort_image
from Image_Binarization.image_binarization import binarize_image

def transforming_image(img, img_name='', display=False):
    """
    Apply perspective transform to input frame to get the bird's eye view.
    :param img: input color frame
    :param display: if True, show the transformation result
    :return: warped_img image, and both forward and backward transformation matrices
    """
    h, w = img.shape[:2]
    

    # src = np.float32([[w, h-10],    # br
    #                   [0, h-10],    # bl
    #                   [546, 465],   # tl
    #                   [732, 465]])  # tr
    # dst = np.float32([[w, h],       # br
    #                   [0, h],       # bl
    #                   [0, 0],       # tl
    #                   [w, 0]])      # tr
    src = np.float32([[1080,710],     # br
                      [200,710],        # bl
                      [590,452],      # tl
                      [729,452]])     # tr
    dst = np.float32([[980, 720],       # br
                      [300, 720],       # bl
                      [300, 0],       # tl
                      [980, 0]])      # tr

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # print (h,w,src,dst)
    warped_img = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)

    if display:
        f, axarray = plt.subplots(1, 2)
        f.set_facecolor('white')
        axarray[0].set_title('transforming before')
        axarray[0].imshow(img, cmap='gray')
        for point in src:
            axarray[0].plot(*point)
        axarray[1].set_title('transforming after')
        axarray[1].imshow(warped_img, cmap='gray')
        for point in dst:
            axarray[1].plot(*point)
        for axis in axarray:
            axis.set_axis_off()
        plt.savefig("../output_images/transformed_images/" + img_name[15:-4] + "_transformation_after.jpg",dpi=300)
        # plt.show()

    return warped_img, M, Minv


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
        img_binarization = binarize_image(img=img_undistorted, img_name=img_test, display=False)
        # transform_image
        transforming_img, M, Minv = transforming_image(img_binarization, img_name=img_test, display=True)
        cv2.imshow("transforming_img",transforming_img*255)
        cv2.waitKey(500)
        cv2.destroyAllWindows()
    print("Transforming imgs have finshed")


