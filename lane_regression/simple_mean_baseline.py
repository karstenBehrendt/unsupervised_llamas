"""Very simple baseline by just taking the mean lane locations

This script is not meant to be pretty. It's just supposed to give
a very quick baseline. It also helps us test our evaluation pipeline.
"""

import argparse
from collections import defaultdict
import json

import numpy
import os
import tqdm

from unsupervised_llamas.label_scripts import dataset_constants
from unsupervised_llamas.common import helper_scripts
from unsupervised_llamas.label_scripts import spline_creator


def calculate_means():
    # We store everything in memory, make sure to have >4GB to spare
    # NOTE The individual blocks should be separate functions

    if not os.path.exists("mean_label_pre.json"):  # Load all data from labels
        train_labels = helper_scripts.get_labels(split="train")
        valid_labels = helper_scripts.get_labels(split="valid")
        labels = train_labels + valid_labels
        lanes = {
            "l1": defaultdict(list),
            "l0": defaultdict(list),
            "r0": defaultdict(list),
            "r1": defaultdict(list)
        }

        # Iterate over all lane labels and store them
        for label in tqdm.tqdm(labels, desc="Going through labels"):
            spline_labels = spline_creator.get_horizontal_values_for_four_lanes(label)
            for lane, lane_key in zip(spline_labels, ["l1", "l0", "r0", "r1"]):
                for y_value, x_value in enumerate(lane):
                    lanes[lane_key][y_value].append(x_value)
        # Writes about 2 GB. Technically, this doesn't need to be stored. It's just for debugging
        json.dump(lanes, open("mean_label_pre.json", "w"))  # use with, was lazy
    else:
        lanes = json.load(open("mean_label_pre.json"))  # use with, was lazy

    if not os.path.exists("mean_label.json"):  # Calculate averages
        for key, lane in lanes.items():
            for y_value, x_values in lane.items():
                clean_x_values = [x if x!=-1 else float('nan') for x in x_values]
                lanes[key][y_value] = numpy.nanmean(clean_x_values)
        json.dump(lanes, open("mean_label.json", "w"))  # use with, was lazy
    else:
        lanes = json.load(open("mean_label.json"))  # use with, was lazy

    # Clean: Move dict to list, remove first 300 entries per image
    for key, lane in lanes.items():
        lanes[key] = [x_value for x_value in lane.values()]
        # Remove upper part of image with really sparse information
        lanes[key] = lanes[key][300:]  # 40% less storage use

    # Write results for the three splits for evaluations
    for split in ["train", "valid", "test"]:
        print(f"Writing {split} set. This may take a couple of minutes")
        split_results = {}
        labels = helper_scripts.get_labels(split)
        for label in labels:
            split_results[helper_scripts.get_label_base(label)] = lanes
        with open(f"{split}_mean_results.json", "w") as results_handle:
            json.dump(split_results, results_handle)


if __name__ == "__main__":
    calculate_means()
