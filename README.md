## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


## Overview

In this project, our goal was to write a software pipeline to identify the lane boundaries in several video streams with increasing difficult.


The suggested steps of this project were the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


## Proposed Solution

### Camera calibration

The code for this step can be found on image_functions.py, starting at row 48.

We start defining the early stopping criteria for the refining step that will be done at the end of the calibration. These steps were recommended by OpenCV documentation as a way to increase the accuracy of the calibration.

We then create a meshgrid of object points and lists of points in the 3D world and it's conversion to 2D coordinates.

At this point we iterate through every calibration image, converting them to grayscale, and using the OpenCV function to find chessboard corners in each image, storing the 3D and 2D points to pass on to the calibration function.

Once we find our camera parameters we can save them to a pickle file so we don't need to do this step every time we call our pipeline. It's important to note that this parameters are specific to each camera, so if you change your camera you would need to find new camera parameters.

More detailed information about camera calibration can be found at [[1]](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html).

#### Examples

##### 1. Raw calibration images
![alt text][image1]

![alt text][image2]

##### 2. Calibrated image

![alt text][image3]

### Pipeline

#### 1. Correct camera images

Using the camera parameters we found during our calibration step and OpenCV undistort function we can correct the images from the car camera.

![alt text][image4]

We can see that our correction removed a lot of the "bending" that occurs on the edges of the images, it is more evident on the left and right corners of the hood.

#### 2. Changing color space and creating binary

RGB is usually not the best color space to use in computer vision applications as it focuses only on color when creating the channels. Based on knowledge from [P1 - Finding lane lines](https://github.com/bguisard/CarND-LaneLines-P1) I decided to use [HSV](https://en.wikipedia.org/wiki/HSL_and_HSV) as it showed that it is a lot more useful to capture different light conditions.

After converting the image to HSV I combined a white filter and a yellow filter into an image with a single channel and also created a binary image (1 where there is any yellow or white and 0 otherwise).

The code for this step can be found on image_functions.py at row 136.

##### HSV masking

![alt text][image5]

##### Binary

![alt text][image6]

#### 3. Sobel

We will apply the Sobel operator to the color masked image rather than the raw image. This will enhance the performance of this function since we have already removed a lot of unwanted information on the previous step.

I started with the recommended parameters from the lesson and didn't change it. The suggested parameters were enough to get great results on the test images, and when implementing the pipeline for the videos I noticed that I was getting better results without using Sobel at all.

The code for this step can be found on image_functions.py at row 173. I have also implemented a more complex version of Sobel, combining different thresholds but ended up removing them from the final solution (rows 207 to 284 on image_functions.py).

![alt text][image7]


#### 4. Perspective transform

Now that we removed most of the "noise" from our image we will create a bird's eye view of the lane in front of us. This was achieved by selecting a Region of Interest (ROI) that was optimized for the configuration of this specific test car - e.g. center mounted camera, image size 720 x 1280, fixed height and angle.

Although I have turned the fixed coordinates into percentages of the provided image, I didn't have the opportunity to test the pipeline on different video streams.

We then map the four corners of our ROI into the four corners of our image and use OpenCV functions to get the image from a new perspective.

You can find the core details below, and more information on image_functions.py

```python
# Source points
src[0] = [np.uint(img_w / 2) - 55, np.uint(img_h / 2) + 100]
src[1] = [np.uint(img_w / 2 + 55), np.uint(img_h / 2) + 100]
src[2] = [np.uint(img_w * 5 / 6) + 60, img_h]
src[3] = [(np.uint(img_w / 6) - 10), img_h]

# Calculate the destination points
dst = np.zeros((4, 2), dtype=np.float32)
dst[0] = [np.uint(img_w / 4), 0]
dst[1] = [np.uint(img_w * 3 / 4), 0]
dst[2] = [np.uint(img_w * 3 / 4), img_h]
dst[3] = [np.uint(img_w / 4), img_h]
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 585, 460      | 320, 0       |
| 695, 460      | 960, 0       |
| 1126, 720     | 960, 720     |
| 203, 720      | 320, 720     |

And the image below shows the lane lines after the transformation.

![alt text][image8]

#### 5. Fitting a polynomial to lane lines

Following the suggestions on the project I wrote two different functions to find lane lines, a more robust one that relies on histogram of image binaries to find X and Y points for each lane line and a "recursive" version that uses the line found on the previous frame as the baseline to where to look for lines in the current frame.

The code for this two functions can be found on pipeline.py at rows 7 and 121.

The images below show the differences between the two methods, while the first uses multiple moving windows to look for lane lines, the second one only searches around the lane found on the previous frame.

##### Robust

![alt text][image9]

##### Recursive

![alt text][image10]

#### 6. Lane curvature and car position in the lane

The last thing we needed to calculate before plotting our lane lines on top of the original image was lane curvature and the vehicle position with respect to the center of the lane.

The two snippets below show how this can be easily achieved:

```python
def get_curvature(poly, y_eval=0):

    A, B, C = poly
    R = ((1 + (2 * A * y_eval + B) ** 2) ** 1.5) / np.absolute(2 * A)

    return R

def get_lane_offset(left_fit, right_fit, imsize=(720, 1280), x_mpp=3.7 / 700):

    img_h = imsize[0]

    img_center = imsize[1] / 2
    left_lane_start = left_fit[0] * img_h ** 2 + left_fit[1] * img_h + left_fit[2]
    right_lane_start = right_fit[0] * img_h ** 2 + right_fit[1] * img_h + right_fit[2]
    lane_center = np.mean((left_lane_start, right_lane_start))

    dist = (img_center - lane_center) * x_mpp

    if dist == 0:
        side = ' '
    elif dist > 0:
        side = 'right'
    else:
        side = 'left'

    return np.absolute(dist), side
```

The full implementation with all references can be found at pipeline.py.

### Putting it all together

You can see below the output of the pipeline on our test image.

![alt text][image11]

### Laneline class

As we can see the tools we have created so far are enough to find lines in several individual test images, but if we try to run them on a video stream the lines won't have continuity and our lane lines will be jittering from frame to frame.

To solve that we need a Class to wrap these lines and control how they change from frame to frame.

Our class have several parameters that can be optimized to ensure a smooth transition of lines from frame to frame, without compromising the detection capability of the pipeline.

Some examples are:

- current polynomial fit
- curvature
- last time a good fit was detected
- best_fit (as controlled by several parameters) vs current_fit

We also need a new function to find our lane lines that can use our wrapper class. The definition of the function can be seen below, while the full code can be found at row 343 on pipeline.py while the full code for the Line class can be found at laneline.py.

```python
def find_lane_lines(leftline, rightline, warped, nwindows=9, margin=100,
                return_img=False, plot_boxes=False, plot_line=False, verbose=0):

    """
    This function finds lane lines on images using two different approaches.

    1 - If is the first iteration of leftlane and rightlane it will use a slower
        algorithm to find lane lines looking through all nonzero points of the
        warped binary.

    2 - If it's not the first iteration of the lane class objects it will use
        a window to search the points based on the best fit that the lane class
        objects have.

    Inputs:
    leftlane - object of Lane class
    rightlane - object of Lane class
    warped - Warped binary of image that went trough our pipeline.

    Outputs:
    Updates - leftlane and rightlane
    out_img - [Optional] - Returns a top view of image with pixels found and
                           boxes of search [optional] and lane lines [optional]
    """
```   

### Results

The pipeline performed very well on the easy and challenge videos, just requiring some fine-tuning of the hyper parameters of our laneline class. Unfortunately the results were very poor on the harder challenge video.

Fine-tuning of these parameters proved to be quite a painful process. The lack of any metric that can be processed by our pipeline makes the optimization process a manual and tedious process. The results on the individual images and on the first two videos are encouraging, but the parameters of the lines had to be fine tuned for each set of videos and what we see is the output of two overfitted models, rather than a model than can truly generalize and be delivered to production.

Although I am certain we could achieve a model that generalizes well, I believe that if we had a set of annotated images with lane lines we could approach this problem from a different angle, using modern deep learning techniques to train a network to identify the lines, as suggested by [[3]](http://www.cv-foundation.org/openaccess/content_cvpr_2016_workshops/w3/papers/Gurghian_DeepLanes_End-To-End_Lane_CVPR_2016_paper.pdf).

The video results can be found on the links below:

[Project video](./video_output/project_video.mp4)

[Challenge](./video_output/challenge.mp4)

[Harder challenge](./video_output/hard.mp4)

## References
[1] [OpenCV Camera Calibration](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html)

[2] [Finding Lane Lines on the Road](https://github.com/bguisard/CarND-LaneLines-P1)

[3] [DeepLanes: End-To-End Lane Position Estimation using Deep Neural Networks](http://www.cv-foundation.org/openaccess/content_cvpr_2016_workshops/w3/papers/Gurghian_DeepLanes_End-To-End_Lane_CVPR_2016_paper.pdf)

[//]: # (Image References)

[image1]: ./examples/calibration_1.png "Distorted"
[image2]: ./examples/calibration_2.png "Another set of distorted images"
[image3]: ./examples/calibrated.png "Calibration comparison"
[image4]: ./examples/undistorted.png "Undistorted"
[image5]: ./examples/hsv_mask.png "HSV Mask"
[image6]: ./examples/binary.png "Binary from color mask"
[image7]: ./examples/sobel.png "Sobel"
[image8]: ./examples/birdseye.png "Top down perspective"
[image9]: ./examples/find_lane_robust.png "Robust search"
[image10]: ./examples/find_lane_recursive.png "Recursive search"
[image11]: ./examples/result.png "Recursive search"
