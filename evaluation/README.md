# Evaluation
This folder contains the evaluation scripts for the leaderboard.
Additional scripts and metrics may be added.

## Submission Format
For segmentatation approaches, the results have to be submitted as png images for each image in the test set.
```
The script expects all images to be named according to the label files, i.e.,
recording_folder/label_file.json + '_' + {class integer} + '.png'

The class integers / enums are:
    0: background
    1: l1
    2: l0
    3: r0
    4: r1
In the binary case 1 is enough for the evaluation.

An example image path for r0 (first marker to the right) is:
/PATH_TO_FOLDER/llamas/trained_nets/2019_03_03__17_53_39_multi_marker_net_gradients/
markers-1456725_test/images-2014-12-22-13-22-35_mapping_280S_2nd_lane/
1419283521_0744236000.json_3.png
```

Make sure to see evaluate_segmentation.py and test your submission format for the validation set before submitting data.

## Leaderboards
Benchmark results are displayed on the unsupervised LLAMAS website [here](https://unsupervised-llamas.com/llamas/benchmarks).
