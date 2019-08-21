# Unsupervised LLAMAS
Code for the Unsupervised Labeled LAne MArkerS (LLAMAS) dataset. The dataset and more information is available at https://unsupervised-llamas.com. 
The leaderboard is available [here](https://unsupervised-llamas.com/llamas/benchmarks). Since lane markers and lane detection are evaluated based on multiple metrics, new metrics can be added to the benchmarks as well.

All contributions are welcome, e.g. sample scripts in different machine learning frameworks.  You can even change the [website's code](https://github.com/karstenBehrendt/benchmarks_website/tree/master/benchmarks/llamas)

## Errors and Suggestions
In case you encounter any issues with the data or scripts, please create an issue ticket, create a PR, or send me an email.
For questions about training deep learning approaches for lane marker detection or segmentation in the different frameworks, please checkout Stackoverflow.
You can reach me at "llamas" + the at sign since this is an email + kbehrendt.com.

## Starter Code
Make sure to check the label_scripts/label_file_scripts.py for loading and using the annotations. There exist a few sample use-cases and implementations.

The simple_baseline folder contains a simplistic baseline approach in Tensorflow which is supposed to be easy to understand.

**ENet-SAD-Simple** folder contains the **ENet-SAD** model which achieves state-of-the-art performance in TuSimple, CULane and BDD100K datasets. It also achieves appealing performance in LLAMAS dataset. Details can be found in README in [ENet-SAD-Simple](/ENet-SAD-Simple) and [this repo](https://github.com/cardwing/Codes-for-Lane-Detection).

The deeplab folder offers some scripts to train deeplab models for the unsupervised LLAMAS dataset.

All results for the leaderboards are calculated based on scripts in the evaluation folder.

## Video
Make sure to checkout the [Youtube video](https://youtu.be/kp0qz8PuXxA) with samples from the dataset and baseline approaches.

## Sample

![Sample Image Gray](https://github.com/karstenbehrendt/unsupervised_llamas/blob/master/samples/sample_gray.jpg) ![Sample Image Color](https://github.com/karstenbehrendt/unsupervised_llamas/blob/master/samples/sample_color.jpg)
![Sample Image Labeled](https://github.com/karstenbehrendt/unsupervised_llamas/blob/master/samples/sample_labeled.jpg)
3D points are available and spline interpolation on labels is possible.
