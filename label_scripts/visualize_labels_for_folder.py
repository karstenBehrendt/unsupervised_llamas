"""
Functions to draw all labels for a given folder
"""
# NOTE There are a few TODOs that would make it a lot nicer

import os

import cv2
import tqdm

from unsupervised_llamas.common import helper_scripts
from unsupervised_llamas.label_scripts import visualize_labels


def segmentation_for_folder(input_folder, output_folder, color=None):
    """ Draws segmentation images for a given folder of labels

    Parameters
    ----------
    input_folder: str
                  path with json files / labels
    output_folder: str
                   folder to store segmentation images, cannot exist
    color: int, gray color value
           (int, int, int), BGR values
           None for default colors

    Returns nothing
    """
    # TODO Add color image option
    # TODO keep input name and folders
    if os.path.exists(output_folder):
        raise IOError('Output folder already exists, stopping to not mess things up')
    os.makedirs(output_folder)

    input_labels = helper_scripts.get_files_from_folder(input_folder, '.json')

    for i, label_path in tqdm.tqdm(enumerate(input_labels)):
        segmentation_image = visualize_labels.create_segmentation_image(label_path, image='gray', color=color)
        cv2.imwrite(os.path.join(output_folder, str(i) + '.png'), segmentation_image)


def splines_for_folder(input_folder, output_folder):
    """ Draws segmentation images for a given folder of labels

    Parameters
    ----------
    input_folder: str
                  path with json files / labels
    output_folder: str
                   folder to store segmentation images, cannot exist

    Returns nothing
    """
    # TODO Add color image option
    # TODO keep input name and folders
    if os.path.exists(output_folder):
        raise IOError('Output folder already exists, stopping to not mess things up')
    os.makedirs(output_folder)

    input_labels = helper_scripts.get_files_from_folder(input_folder, '.json')

    for i, label_path in tqdm.tqdm(enumerate(input_labels)):
        spline_image = visualize_labels.create_spline_image(label_path, 'gray')
        cv2.imwrite(os.path.join(output_folder, str(i) + '.png'), spline_image)
