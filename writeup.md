## Writeup Template

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_hog.png
[image2]: ./output_images/not_car_hog.png
[image3]: ./output_images/sliding_windows.png
[image4]: ./output_images/bboxes_heat1.png
[image5]: ./output_images/bboxes_heat2.png
[image6]: ./output_images/bboxes_heat3.png
[image7]: ./output_images/example1.png
[image8]: ./output_images/example2.png
[image9]: ./output_images/example3.png
[image10]: ./output_images/example4.png
[image11]: ./output_images/example5.png
[image12]: ./output_images/example6.png

[video1]: ./project_output5.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. 
You're reading it! Many parts the code is based on Udacity's lectures.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 2nd code cell of the IPython notebook.

I started by reading in all the `vehicle` and `non-vehicle` images.   

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

I found out that the best color space was `YCrCb`.
Below there is an example of 2 random images using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`. These 2 different images are one of the class "vehicles" and one of the class "non-vehicles". We can easily see the difference:

Vehicles class:   
![alt text][image1]

Non-vehicle class:   
![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.
 I experimented with a variety of values for the HOG parameters. However, apart from tweaking the color space and using all the channels, I was not able to see a significant difference in accurancy when tuning the other parameters. The biggest difference was in the processing time, which dramatically increased as the values of the parameters increased, especially the pixels per cell.. Therefore I decided to leave the rest of the options in default values, since the processing time was acceptable and I could not see any other benefits.

In the end, I use binary spacial binning of (32,32), 32 histogram bins, orientation of 9 with 8 HOG pixels per cell and 2 HOG cells per block. The total features length was 8460 features.
 
#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using 20% of my samples as a test set. The code is in the 8th code cell of the IPython notebook. The training process was very fast (less than 12 seconds) and very accurate (approximately 99% accurancy). The code is based on the Udacity's lectures.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used the slide_window function provided in the Udacity class (code cell 5),  to draw the area of the image that will be scanned. I chose to discard the upper top of the picture, since the flying vehicles are not yet in production and therefore probably not present. The area where I scan is visible below:   
![alt text][image3]

However in the end, I used the Hog Sub-sampling Window Search method, also provided in the class material.This is included within the function find_cars (code cell 3). With this method a big performance increase is achieved.   

Since I used time related filtering of false positives with the use of multiple frames, I decided to decrease a little the scaling which produced more boxes, and increased the accuracy of detection. The overlapping was also based on the proposed parameters of the class and even if it produced a lot of scanning bounding boxes, it is important for increasing the accuracy.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

In my find_cars function, using hog sub-sampling I extracted the features of the objects. I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. I also saved the lists of bounding boxes, to transfer information between the frames and avoid false positive items (see details below).  Here are some example images:

![alt text][image4]   
![alt text][image5]    
![alt text][image6]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_output5.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

The pipeline for the video that contains the also the false positive filtering is available in cell 4 of the notebook.   

I would like to note that I also used a frame-based filters. This filter combined the bounding boxes of the last 10 frames produces a combined heatmap, which was practically averaged across 10 frames. This averaged result, was refreshed every 5 frames (i.e. Every 5 frames, a new bounding box is created, according to the averaged heatmap of 10 frames). This is turn produces excellent results.

### Here are six frames with their bounding boxes drawn, their corresponting heatmaps and the final bounding boxes: 
![alt text][image7]   
![alt text][image8]   
![alt text][image9]   
![alt text][image10]   
![alt text][image11]   
![alt text][image12]   





---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The biggest problem that can be found in the video is when the 2 cars overlap in the image. When this happens, the algorithm things that it is only one car, and not 2 different cars. This could be solved if also kept other parameters of the objects, apart from the heatmap, such as the color (the 2 cars have the same color). However, the problem would still remain, if the cars had the same color.

Although the "minimizing" of processing in 4/5 of the frames increases dramatically the performance of the algorithm, it is still only 12.27it/s in a desktop machine. The processing time would be even bigger, if deep learning is added in the algorithm to undestand better the movement of the cars (so that It can better distinguish multiple cars overlapping) and to change the parameters in realtime, when needed.



