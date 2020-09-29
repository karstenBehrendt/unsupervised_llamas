"""Evaluates lane regression results

Submission format (as json file):
{
   "label_base": {
       'l1': [x0,x1, x2, x3, x4, ..., x399],
       'l0': [x0,x1, x2, x3, x4, ..., x399],
       'r0': [x0,x1, x2, x3, x4, ..., x399],
       'r1': [x0,x1, x2, x3, x4, ..., x399],

   },
   ... (one entry for each label / image within a set
}

Markers from left to right:
l1, l0, car / camera, r0, r1

The main metric for evaluation is mean abs distance in pixels
between regressed markers and reference markers.
"""

from unsupervised_llamas.label_scripts import dataset_constants
from unsupervised_llamas.label_scripts import helper_scripts
from unsupervised_llamas.label_scripts import spline_creator

import argparse
import json

import numpy


def compare_lane(reference_lane, detected_lane, vertical_cutoff=400):
    """Mean deviation in pixels"""
    assert len(reference_lane) > 300, "Reference lane is too short"
    assert len(detected_lane) > 300, "Detected lane is too short"

    # Reference lanes go from 0 to 717. If a horizontal entry is not
    # defined, it is stored as -1. We have to filter for that.

    reference_lane = reference_lane[:vertical_cutoff]
    detected_lane = detected_lane[:vertical_cutoff]

    reference_lane = [x if x != -1 else float('nan') for x in reference_lane]

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

    for key, value in results.items():
        results[key] = numpy.nanmean(value)
    print(results)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", help="file to be evaluated", required=True)
    parser.add_argument("--split", help="train, valid, or test", default="valid")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    assert args.split in ["train", "valid", "test"]
    evaluate(eval_file=args.eval_file, split=args.split)
