import os
import sys
import cv2
import os
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import numpy as np
import yaml
import glob
# For loading other py code
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from Camera_Calibration.camera_calibration import calibrate_camera, undistort_image
from Image_Binarization.image_binarization import binarize_image
from Perspective_Transform.perspective_transform import transforming_image
from Params_Config.params_config import xm_per_pix, ym_per_pix, time_window, processed_frames
from Lane_Fit.lane_fit import get_fits_by_sliding_windows, draw_back_onto_the_road, Line, get_fits_by_previous_fits



line_lt = Line(buffer_len=time_window)  # line on the left of the lane
line_rt = Line(buffer_len=time_window)  # line on the right of the lane


def compute_offset_from_center(line_lt, line_rt, frame_width):
    """
    Compute offset from center of the inferred lane.
    The offset from the lane center can be computed under the hypothesis that the camera is fixed
    and mounted in the midpoint of the car roof. In this case, we can approximate the car's deviation
    from the lane center as the distance between the center of the image and the midpoint at the bottom
    of the image of the two lane-lines detected.

    :param line_lt: detected left lane-line
    :param line_rt: detected right lane-line
    :param frame_width: width of the undistorted frame
    :return: inferred offset
    """
    if line_lt.detected and line_rt.detected:
        line_lt_bottom = np.mean(line_lt.all_x[line_lt.all_y > 0.95 * line_lt.all_y.max()])
        line_rt_bottom = np.mean(line_rt.all_x[line_rt.all_y > 0.95 * line_rt.all_y.max()])
        lane_width = line_rt_bottom - line_lt_bottom
        midpoint = frame_width / 2
        offset_pix = abs((line_lt_bottom + lane_width / 2) - midpoint)
        offset_meter = xm_per_pix * offset_pix
    else:
        offset_meter = -1

    return offset_meter


def main_process(frame, img_name='', keep_state=True):
    """
    Apply whole lane detection pipeline to an input color frame.
    :param frame: input color frame
    :param keep_state: if True, lane-line state is conserved (this permits to average results)
    :return: output blend with detected lane overlaid
    """

    global line_lt, line_rt, processed_frames
    
    # undistort the image using coefficients found in calibration
    img_undistorted = undistort_image(frame, mtx, dist, display=False)

    # binarize the frame s.t. lane lines are highlighted as much as possible
    img_binarization = binarize_image(img_undistorted, img_name, display=False)

    # compute perspective transform to obtain bird's eye view
    transforming_img, M, Minv = transforming_image(img_binarization, img_name, display=False)

    # fit 2-degree polynomial curve onto lane lines found
    if processed_frames > 0 and keep_state and line_lt.detected and line_rt.detected:
        line_lt, line_rt, img_fit = get_fits_by_previous_fits(transforming_img, line_lt, line_rt, img_name, display=False)
    else:
        line_lt, line_rt, img_fit = get_fits_by_sliding_windows(transforming_img, line_lt, line_rt, img_name, n_windows=9, display=False)

    # compute offset in meter from center of the lane
    offset_meter = compute_offset_from_center(line_lt, line_rt, frame_width=frame.shape[1])

    # draw the surface enclosed by lane lines back onto the original frame
    blend_on_road = draw_back_onto_the_road(img_undistorted, Minv, line_lt, line_rt, keep_state)
    
    # add 
    mean_curvature_meter = np.mean([line_lt.curvature_meter, line_rt.curvature_meter])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blend_on_road, 'Radius of curvature: {:.02f}m'.format(mean_curvature_meter), (20, 50), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(blend_on_road, 'Offset of center: {:.02f}m'.format(offset_meter), (20, 100), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    processed_frames += 1

    return blend_on_road


if __name__ == '__main__':

    # for calibrating and undistorting
    calib_path='./camera_cal'
    mtx, dist = calibrate_camera(calib_path)

    select_mode = 'video'

    if select_mode == 'video':
        selector = 'project'
        clip = VideoFileClip('{}_video.mp4'.format(selector)).fl_image(main_process)
        clip.write_videofile('{}_{}_output.mp4'.format(selector, time_window), audio=False)

    elif select_mode == 'images':
        # load dir
        imgs_raw = glob.glob('./test_images/*.jpg')
        
        # set lane buffer
        line_lt, line_rt = Line(buffer_len=time_window), Line(buffer_len=time_window)   

        for img_test in imgs_raw:
            load_img = cv2.imread(img_test)
            blend_img = main_process(load_img, img_name=img_test, keep_state=False)
            cv2.imwrite('output_images/final_image/{}'.format(img_test[14:-4]+"_blend_image.jpg"), blend_img)
            cv2.imshow("blend_img",blend_img)
            cv2.waitKey(500)
            cv2.destroyAllWindows()
        print("All Images Done")   
    
      
    
       
