# Lane Regression

## Submission format
We focus on the cars left, current, and right lane by regressing the lane borders l1, l0, r0, r1 from left to right.
l0 is our left lane border closest to us, l1 is the left lane border one further to the left and so on.
The dataset contains more than those lane borders, but those should be the cleanest. 
They are auto generated on a frame by frame basis, so they won't be perfect.

Format:
* The results have to be stored jointly for all images within a single json file.
* Each image is stored as a dict based on its base path (video_name/image_name.png)
* Each image needs to contain for lanes as keys, l1, l0, r0, r1
* The result for each lane is stored as x-value for each y-value.
* * OpenCV's coordinate system has y-values start from the top of the image and increase as you go down
* * The first (upper) 300 pixels of the image are not evaluated. There barely are any labels for those.
* * You may submit x values for all y values across the image (717) or ignore the first 300 (417)

Metric:
Mean absolute distance horizontally for each vertical pixel for each lane

## Simple mean baseline evaluations:
  On the training set
    Overall mean absolute error 36.52 pixels
    Invidiaul lanes {'l1': 34.62, 'l0': 35.30, 'r0': 37.89, 'r1': 38.42}

  On the validation set
    Overall mean absolute error 33.34
    Invidiaul lanes {'l1': 33.47, 'l0': 33.48, 'r0': 32.88, 'r1': 33.68}

  On the test set
    Overall mean absolute error: 31.00 pixels
    Individual lanes {'l1': 33.78, 'l0': 26.34, 'r0': 30.24, 'r1': 34.75}

I am currently not listing the paper baselines, because the dataset splits, and evaluation changed since publication.
I may or may not re-train a model close to the paper baseline. Feel free to submit your results.

All results will be listed as part of the [lane approximation benchmark](https://unsupervised-llamas.com/llamas/benchmark_splines).
