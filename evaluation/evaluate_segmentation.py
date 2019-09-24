#!/usr/bin/env python3
""" The evaluation script for the segmentation part of the unsupervised
llamas dataset.

It calculates AUC, and best precision-recall combinations for each class.

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

Use png files for lossless compression.
Files are stored for individual channels because it's easy. Four channel images
would not be an issue but after that it may not be too straightforward.

Make sure to scale predictions from 0 to 255 when storing as image.
cv2.imwrite may write zeros and ones only for a given float as dtype with values
between 0 and one, even though cv2.imshow visualizes it correctly.

Usage:
    python3 evaluate_segmentation.py \
        --inference_folder folder_with_stored_inference_images
        --multi_class (optional if it is not binary)
"""
# TODO Needs to be tested
# TODO The binary and multi_class evaluation can probably be combined
#      by just checking which files exist
# TODO The multithreading call can be implemented in a cleaner way

import argparse
import concurrent.futures
import os
import pprint

import cv2
import tqdm

from unsupervised_llamas.common import helper_scripts
from unsupervised_llamas.evaluation import segmentation_metrics
from unsupervised_llamas.label_scripts import dataset_constants
from unsupervised_llamas.label_scripts import segmentation_labels


def binary_eval_single_image(inputs):
    # Single argument call for the threaded function.
    # This can probably be implemented in a cleaner way.
    return single_threaded_binary_eval_single_image(inputs[0], inputs[1])


def multi_eval_single_image(inputs):
    # Single argument call for the threaded function.
    # This can probably be implemented in a cleaner way.
    return single_threaded_multi_eval_single_image(inputs[0], inputs[1])


def single_threaded_multi_eval_single_image(label_path, segmentation_folder):
    target = segmentation_labels.create_multi_class_segmentation_label(label_path)

    results = {}
    for i in range(5):
        # TODO Needs to be adapted for more cases farther lanes
        # Currently (in order) background, l1, l0, r0, r1
        segmentation_path = os.path.join(
            segmentation_folder,
            helper_scripts.get_label_base(label_path)) + '_{}.png'.format(i)

        segmentation = cv2.imread(segmentation_path, cv2.IMREAD_GRAYSCALE).astype(float) / 255
        if segmentation is None:
            raise IOError('Could not read image. Is this label path correct?', label_path)
        results[i] = segmentation_metrics.binary_approx_auc(segmentation, target[:, :, i])

    return results


def single_threaded_binary_eval_single_image(label_path, segmentation_folder):
    target = segmentation_labels.create_binary_segmentation_label(label_path)

    segmentation_path = os.path.join(
        segmentation_folder,
        helper_scripts.get_label_base(label_path)) + '_1.png'
    segmentation = cv2.imread(segmentation_path, cv2.IMREAD_GRAYSCALE).astype(float) / 255

    results = {}
    results[1] = segmentation_metrics.binary_approx_auc(segmentation, target)
    return results


def evaluate_set(segmentation_folder, eval_function, dataset_split='test', max_workers=8):
    """ Runs evaluation for a given image folder

    Parameters
    ----------
    segmentation_folder : str
                          folder with predictions / inference images according to docstring
    eval_function : function
                    Currently the binary or multi-class evaluation function
    dataset_split : str
                    'train', 'valid', or 'test'. Calculates metrics for that split.
    max_workers : int
                  Number of threads to use

    Returns
    -------
    Dictionary with AP for each class and best precision-recall combination

    Raises
    ------
    IOError if inference image does not exist for a sample in the defined split

    Notes
    -----
    Use max_workers=1 for single threaded call. This makes debugging a lot easier.
    """
    label_folder = os.path.join(dataset_constants.LABELS, dataset_split)
    if not os.path.isdir(label_folder):
        raise IOError('Could not find labels for split {} at {}'.format(
            dataset_split, label_folder))
    label_paths = helper_scripts.get_labels(dataset_split)

    if not os.path.isdir(segmentation_folder):
        raise IOError('Could not find segmentation folder at', segmentation_folder)

    # This still takes a couple of hours.
    eval_dicts = {}
    if max_workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            for label_path, single_eval in tqdm.tqdm(
                    zip(label_paths, executor.map(
                        eval_function, zip(label_paths, [segmentation_folder] * len(label_paths)))),
                    desc='Scoring test samples', total=len(label_paths)):
                eval_dicts[label_path] = single_eval
    else:  # mainly for debugging
        for label_path in tqdm.tqdm(
                label_paths, desc='Scoring test samples', total=len(label_paths)):
            eval_dicts[label_path] = eval_function((label_path, segmentation_folder))

    # The reduce step. Calculates averages
    label_paths = list(eval_dicts.keys())
    lanes = list(eval_dicts[label_paths[0]].keys())
    metrics = list(eval_dicts[label_paths[0]][lanes[0]].keys())

    mean_results = {}
    for lane in lanes:
        mean_results[lane] = {}
        for metric in metrics:
            mean = 0
            for label_path in label_paths:
                mean += eval_dicts[label_path][lane][metric]
            mean /= len(label_paths)
            mean_results[lane][metric] = mean

    pprint.pprint(segmentation_folder)
    pprint.pprint(mean_results)
    return mean_results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inference_folder', type=str, required=True,
                        help='Folder of inference images, see docstring')
    parser.add_argument('--multi_class', action='store_true')
    parser.add_argument('--max_workers', type=int, default=8)
    parser.add_argument(
        '--split', type=str, required=False, default='test',
        help="('train' | 'valid' | 'test') to select the split to evaluate")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    eval_function = multi_eval_single_image if args.multi_class else binary_eval_single_image
    evaluate_set(args.inference_folder, eval_function, dataset_split=args.split,
                 max_workers=args.max_workers)
