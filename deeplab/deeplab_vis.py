#!/usr/bin/env python3
"""
Visualize deeplab models on the unsupervised lamas dataset.
See training file with changes needed in models/.../deeplab

Usage:
    python3 deeplab_vis.py
    --help for arguments
    Needs to have existing tfrecords from unsupervised Lamas dataset

This is only an example. Not a good one either. It is better to implement your own scripts.
"""

import argparse
import os
import subprocess

from unsupervised_llamas.deeplab import deeplab_common


def vis_deeplab(settings):
    """ Draws segmentations based on trained deeplab models """
    checkpoint_dir = deeplab_common.checkpoint_dir(settings)

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(settings['gpu'])
    dataset = deeplab_common.segmentation_set_name(settings)
    dataset_dir = deeplab_common.tfrecords_dir(settings)
    vis_call = [
        'python3',
        os.path.join(deeplab_common.DEEPLAB_DIR, 'vis.py'),

        '--logtostderr',
        '--vis_split=valid',
        '--dataset={}'.format(dataset),
        # '--dataset_dir={}'.format(settings['dataset_dir']),
        '--model_variant=xception_65',
        '--atrous_rates=6',
        '--atrous_rates=12',
        '--atrous_rates=18',
        '--output_stride=16',
        '--decoder_output_stride=4',

        '--vis_crop_size=1276,717',  # May need to be changed # 513,513
        '--checkpoint_dir={}'.format(checkpoint_dir),
        '--vis_logdir={}_vis'.format(checkpoint_dir),
        '--dataset_dir={}'.format(dataset_dir),
        '--also_save_raw_predictions']

    subprocess.call(vis_call, env=env)


def parse_args():
    """ Defines defaults and command line parser """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num_iterations', type=int, default=10**6, help='Number of iterations')
    parser.add_argument('--input_type', type=str, default='gray', help='gray, location, or color')
    parser.add_argument('--problem', type=str, default='multi', help='binary or multi')
    parser.add_argument('--gpu', type=int, default=0, help='0 to n, n being your number of GPUs')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Training directory with checkpoint')  # Define single checkpoint?

    return vars(parser.parse_args())


if __name__ == '__main__':
    vis_deeplab(parse_args())
