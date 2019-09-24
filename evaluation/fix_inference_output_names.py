#!/usr/bin/env python
"""
A quick script to adapt to the naming schema of the evaluation scripts.
Not needed if files are named according to the evaluation scripts.
"""
import argparse
import os

import tqdm

from unsupervised_llamas.common import helper_scripts


def fix_names(input_folder, input_string, output_string):
    """ Changes all names within folder according to parameters

    Parameters
    ----------
    input_folder : str
                   folder containing inference images
    input_string : str
                   substring to be replace within each image
    output_string : str
                    what the input_string should be

    Notes
    -----
    This function is only needed if the scripts don't follow the
    expected naming conventions in the first place.
    """
    segmentation_images = helper_scripts.get_files_from_folder(input_folder, '.png')
    for segmentation_image in tqdm.tqdm(segmentation_images, desc='renaming images'):
        output_path = segmentation_image.replace(input_string, output_string)
        os.rename(segmentation_image, output_path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_folder', type=str, required=True)
    parser.add_argument('--input_string', type=str, required=True)
    parser.add_argument('--output_string', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    fix_names(args.input_folder, args.input_string, args.output_string)
