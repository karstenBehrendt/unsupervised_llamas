# Simple Baseline Segmenting Lane Markers
Code is provided for a simple training of a fully convolutional network in tensorflow.
The tfrecords should be created using the functionality of the deeplab folder (or your own implementation).

## Training
Once the tfrecords are created, train_binary allows to start training a binary classifier using only the paths to the tfrecords for training and validation. It is only meant to be a starting point but does train an already useful classifier.

## Results
The model is trained on crops of the original images without any data augmentation or explicitly training on the validation set. Results are available on the official leadboard for [binary segmentation](https://unsupervised-llamas.com/llamas/benchmark_binary) and [lane dependent segmentation](https://unsupervised-llamas.com/llamas/benchmark_multi).

The binary segmentation was trained on grayscale inputs while the multi-class segmentation additionally included gradient images for location information since the simplistic network only has a small region of view.

## Video
The output of the baseline approaches are visualized after the dataset samples as part of this [Youtube video](https://youtu.be/kp0qz8PuXxA).
