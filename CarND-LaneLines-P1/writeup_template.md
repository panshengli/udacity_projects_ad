# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

  The name of project fucntion is image_process_pipeline, it includes 6 steps. which are listed in the fllowing content:

  First, I converted the images to grayscale with the function of cv2.cvtColor(), and debug shown with plt.imshow(image_grayscale, cmap='gray')
  
  Second，To make pixels smoother, I used the gaussian_blur() function and plot by plt.imshow(image_gaussianBlur, cmap='gray')
  
  Next, Using algorithm of canny() to extract edge pixel，which thresholds is 55 and 130, respectively.
  
  The next step is to use mask to find the region of interest(ROI), which is really hard to find, and in special environment may fail, such as in the last challenge video.
  
  The fllowing step is detect Hough Lines and draw lines on blank image, and I refer to the previous example of tutorial settings for parameter settings.
  
  Drawing lines on original image comes to the end.
  
  In order to show the final detection of the pictures, I put the result pictures in the folder of pipeline_imgs.


### 2. Identify potential shortcomings with your current pipeline

 
As for the setting of mask, it can not meet the detection of lane line under all conditions. The detection results of lane line will be very different with different fov.

The conditions of Hough extraction are complex or Canny operator lines are messy, a large number of fake lane lines will be extracted, such as, in the challenge video i find the messy lines.


### 3. Suggest possible improvements to your pipeline

Firstly， On Canny operator, i can extract the contour edge by certain direction, for example, only the vertical lane line can be extracted, and ignore horizontal lines, like beam.

Secondly，The fitting of lane line can be done in the form of cubic equation, and the parameters c0 to c4 are given finally.
