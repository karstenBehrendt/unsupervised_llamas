#!/usr/bin/env python3
"""
Calculates
  true positives (tp)
  false positives (fp)
  true negatives (tn)
  false negatives (fn)
  precision
  recall
  average precision / AUC / PR curves

Additional metrics are welcome
One problem with lane marker segmentation is that the absolute number of correctly
classified pixels often is not helpful because background pixels far outweigh
the lane marker pixels. In absolute terms, marking all pixels as background likely
is the best solution but not helpful for the problem at hand.

Notes
-----
Don't use Python2. There may be integer divisions that I missed.

Options for calculating AUC / Precision Recall curve
1)
It may be faster to sort (prediction, label) pixels by probability and
go through those. O(n log n) in the amount of pixels per image.
Sorting takes about .36 seconds on my current system.
Expected speedup should be about 50%

2)
Bucket sort is possible as well. O(n) to put probabilities into k buckets.
o(n) to calculate the poc / auc. May be faster than using sort().
Sort however may be implemented in C. Still an approximation, as 3).

3) * current implementation. It was easy and can be replaced any time.
O(k * n), k being the amount of threshold steps,
which is not as accurate but may leverage the c/c++ numpy backend.
tp/tn/fp/fn take about one second to calculate
"""
# NOTE There should be tests

import numpy


def _debug_view(prediction, label):
    """ Shows prediction and label for visual debugging """
    prediction = (prediction * 255).astype(numpy.uint8)
    label = (label * 255).astype(numpy.uint8)
    c = numpy.zeros((717, 1276), dtype=numpy.uint8)

    debug_image = numpy.stack((prediction, label, c), axis=-1)
    import cv2   # Not forcing cv2 dependency for metrics
    cv2.imshow('debug_image', debug_image)
    cv2.waitKey(1000)


def thresholded_binary(prediction, threshold):
    """ Thresholds prediction to 0 and 1 according to threshold """
    return (prediction >= threshold).astype(int)


def true_positive(prediction, label):
    """ Calculates number of correctly classified foreground pixels """
    num_tp = numpy.sum(numpy.logical_and(label != 0, prediction == label))
    return num_tp


def false_positive(prediction, label):
    """ Calculates number of incorrectly predicted foreground pixels """
    num_fp = numpy.sum(numpy.logical_and(label == 0, prediction != 0))
    return num_fp


def true_negative(prediction, label):
    """ Calculates number of correctly identified background pixels """
    num_tn = numpy.sum(numpy.logical_and(label == 0, prediction == label))
    return num_tn


def false_negative(prediction, label):
    """ Calculates number of missed foreground pixels """
    num_fn = numpy.sum(numpy.logical_and(label != 0, prediction == 0))
    return num_fn


def binary_approx_auc(prediction, label):
    """ Calculates approximated auc and best precision-recall combination

    Parameters
    ----------
    prediction : numpy.ndarray
                 raw prediction output in [0, 1]
    label : numpy.ndarray
            target / label, values are either 0 or 1

    Returns
    -------
    Dict of approximate AUC, "corner" precision, "corner" recall
    {'precision', 'recall', 'auc'}

    Notes
    -----
    See docstring for alternative implementation options
    Approximated by 100 uniform thresholds between 0 and 1
    """
    # NOTE May achieve speedup by checking if label is all zeros
    num_steps = 100
    auc_value = 0

    # Most upper right precision, recall point
    corner_precision = 0
    corner_recall = 0
    corner_auc = 0
    corner_threshold = 0

    precisions = [1]
    recalls = [0]

    # Individual precision recall evaluation for those steps
    for i in range(num_steps + 1):
        threshold = (num_steps - i) / num_steps
        thresholded_prediction = thresholded_binary(prediction, threshold)

        # tn = true_negative(thresholded_prediction, label)
        tp = true_positive(thresholded_prediction, label)
        fn = false_negative(thresholded_prediction, label)
        fp = false_positive(thresholded_prediction, label)

        precision = 0 if (tp + fp) == 0 else tp / (tp + fp)
        recall = 0 if (tp + fn) == 0 else tp / (tp + fn)

        if (precision * recall) > corner_auc:
            corner_auc = precision * recall
            corner_precision = precision
            corner_recall = recall
            corner_threshold = threshold

        precisions.append(precision)
        recalls.append(recall)

        auc_value += (recalls[-1] - recalls[-2]) * precisions[-2]

    return {'recall': corner_recall, 'precision': corner_precision,
            'threshold': corner_threshold, 'auc': auc_value}
