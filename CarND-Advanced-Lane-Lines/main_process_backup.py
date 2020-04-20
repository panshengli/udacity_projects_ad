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


def prepare_out_blend_frame(blend_on_road, img_binary, img_birdeye, img_fit, line_lt, line_rt, offset_meter):
    """
    Prepare the final pretty pretty output blend, given all intermediate pipeline images

    :param blend_on_road: color image of lane blend onto the road
    :param img_binary: thresholded binary image
    :param img_birdeye: bird's eye view of the thresholded binary image
    :param img_fit: bird's eye view with detected lane-lines highlighted
    :param line_lt: detected left lane-line
    :param line_rt: detected right lane-line
    :param offset_meter: offset from the center of the lane
    :return: pretty blend with all images and stuff stitched
    """
    h, w = blend_on_road.shape[:2]

    thumb_ratio = 0.2
    thumb_h, thumb_w = int(thumb_ratio * h), int(thumb_ratio * w)

    off_x, off_y = 20, 15

    # add a gray rectangle to highlight the upper area
    mask = blend_on_road.copy()
    mask = cv2.rectangle(mask, pt1=(0, 0), pt2=(w, thumb_h+2*off_y), color=(0, 0, 0), thickness=cv2.FILLED)
    blend_on_road = cv2.addWeighted(src1=mask, alpha=0.2, src2=blend_on_road, beta=0.8, gamma=0)

    # add thumbnail of binary image
    thumb_binary = cv2.resize(img_binary, dsize=(thumb_w, thumb_h))
    thumb_binary = np.dstack([thumb_binary, thumb_binary, thumb_binary]) * 255
    blend_on_road[off_y:thumb_h+off_y, off_x:off_x+thumb_w, :] = thumb_binary

    # add thumbnail of bird's eye view
    thumb_birdeye = cv2.resize(img_birdeye, dsize=(thumb_w, thumb_h))
    thumb_birdeye = np.dstack([thumb_birdeye, thumb_birdeye, thumb_birdeye]) * 255
    blend_on_road[off_y:thumb_h+off_y, 2*off_x+thumb_w:2*(off_x+thumb_w), :] = thumb_birdeye

    # add thumbnail of bird's eye view (lane-line highlighted)
    thumb_img_fit = cv2.resize(img_fit, dsize=(thumb_w, thumb_h))
    blend_on_road[off_y:thumb_h+off_y, 3*off_x+2*thumb_w:3*(off_x+thumb_w), :] = thumb_img_fit

    # add text (curvature and offset info) on the upper right of the blend
    mean_curvature_meter = np.mean([line_lt.curvature_meter, line_rt.curvature_meter])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blend_on_road, 'Curvature radius: {:.02f}m'.format(mean_curvature_meter), (860, 60), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(blend_on_road, 'Offset from center: {:.02f}m'.format(offset_meter), (860, 130), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    return blend_on_road


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
    
    # mean_curvature_meter = np.mean([line_lt.curvature_meter, line_rt.curvature_meter])
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(blend_on_road, 'Curvature radius: {:.02f}m'.format(mean_curvature_meter), (860, 60), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    # cv2.putText(blend_on_road, 'Offset from center: {:.02f}m'.format(offset_meter), (860, 130), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    
    

    # stitch on the top of final output images from different steps of the pipeline
    blend_output = prepare_out_blend_frame(blend_on_road, img_binarization, transforming_img, img_fit, line_lt, line_rt, offset_meter)

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
    
      
    
       
