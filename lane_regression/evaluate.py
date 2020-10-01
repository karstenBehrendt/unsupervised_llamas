"""Evaluates lane regression results

Submission format (as json file):
{
   "label_base": {
       'l1': [x0, x1, x2, x3, x4, ..., x717],
       'l0': [x0, x1, x2, x3, x4, ..., x717],
       'r0': [x0, x1, x2, x3, x4, ..., x717],
       'r1': [x0, x1, x2, x3, x4, ..., x717],

   },  # or since the upper part isn't evaluated
   "label_base": {
       'l1': [x300, ..., x717],
       'l0': [x300, ..., x717],
       'r0': [x300, ..., x717],
       'r1': [x300, ..., x717],

   },
   ... (one entry for each label / image within a set
}

Markers from left to right:
l1, l0, car / camera, r0, r1

The main metric for evaluation is mean abs distance in pixels
between regressed markers and reference markers.
"""

import argparse
import json
import math

import numpy

from unsupervised_llamas.label_scripts import dataset_constants
from unsupervised_llamas.common import helper_scripts
from unsupervised_llamas.label_scripts import spline_creator


def compare_lane(reference_lane, detected_lane, vertical_cutoff=300):
    """Mean deviation in pixels"""
    assert len(reference_lane) == 717, "Reference lane is too short"
    assert len(detected_lane) >= 717 - vertical_cutoff, "Need at least 417 pixels per lane"

    # Reference lanes go from 0 to 717. If a horizontal entry is not
    # defined, it is stored as -1. We have to filter for that.

    reference_lane = reference_lane[vertical_cutoff:]
    if len(detected_lane) == 717:  # lane regressed across complete image
        detected_lane = detected_lane[vertical_cutoff:]
    elif len(detected_lane) == 417:  # lane regress across part of image that is relevant
        pass
    else:
        raise NotImplementedError(f"Evaluations not implemented for length of detected lane: {len(detected_lane)}")

    reference_lane = [x if x != -1 else float('nan') for x in reference_lane]
    # Results are only allowed to be nan where the labels also are invalid.
    # Just don't add nans to your submissions within the relevant sections of the image.
    assert all([not math.isnan(x) or math.isnan(x_ref) for x, x_ref in zip(detected_lane, reference_lane)]), "NaNs not allowe within lower part of image"

    lane_diff = numpy.subtract(reference_lane, detected_lane)
    abs_lane_diff = numpy.abs(lane_diff)
    mean_abs_diff = numpy.nanmean(abs_lane_diff)
    return mean_abs_diff


def evaluate(eval_file: str, split: str):

    assert eval_file.endswith(".json"), "Detections need to be in json file"
    with open(eval_file) as efh:
        regressions = json.load(efh)

    labels = helper_scripts.get_labels(split=split)
    results = {"l1": [], "l0": [], "r0": [], "r1": []}
    for label in labels:
        spline_labels = spline_creator.get_horizontal_values_for_four_lanes(label)
        assert len(spline_labels) == 4, "Incorrect number of lanes"
        key = helper_scripts.get_label_base(label)
        regression_lanes = regressions[key]
        for lane, lane_key in zip(spline_labels, ["l1", "l0", "r0", "r1"]):
            result = compare_lane(lane, regression_lanes[lane_key])
            results[lane_key].append(result)

    # Overall mean
    all_distances = []
    for value in results.values():
        all_distances.extend(value)
    mean_distance = numpy.nanmean(all_distances)
    print("Overall mean absolute error", mean_distance )

    # Invididual lanes
    for key, value in results.items():
        results[key] = numpy.nanmean(value)
    print("Invidiaul lanes", results)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", help="file to be evaluated", required=True)
    parser.add_argument("--split", help="train, valid, or test", default="valid")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    assert args.split in ["train", "valid", "test"]
    evaluate(eval_file=args.eval_file, split=args.split)
