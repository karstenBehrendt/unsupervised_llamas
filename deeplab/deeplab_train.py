#!/usr/bin/env python3
"""
Train deeplab models on the unsupervised llamas dataset

Before using, add:
# -----------------------------------------------------------------------
_UNSUPERVISED_LLAMAS_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 58269,
        'valid': 20844,
        'test': 20929,
    },
    num_classes=5,
    ignore_label=255,
)

_BINARY_UNSUPERVISED_LLAMAS_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 58269,
        'valid': 20844,
        'test': 20929,
    },
    num_classes=2,
    ignore_labe=l255,
)
# -----------------------------------------------------------------------
into tensorflow/models/research/deeplab/datasets/data_generator.py


Replace:
-------------------------------------------------------------------------
# Set to False if one does not want to re-use the trained classifier weights.
flags.DEFINE_boolean('initialize_last_layer', False,
                     'Initialize the last layer.')

flags.DEFINE_boolean('last_layers_contain_logits_only', True,
                     'Only consider logits as last layers or not.')
-------------------------------------------------------------------------

and add the datasets to the dataset information
_DATASETS_INFORMATION = {
    'cityscapes': _CITYSCAPES_INFORMATION,
    'pascal_voc_seg': _PASCAL_VOC_SEG_INFORMATION,
    'ade20k': _ADE20K_INFORMATION,
    'unsupervised_llamas': _UNSUPERVISED_LLAMAS_INFORMATION,
    'binary_unsupervised_llamas': _BINARY_UNSUPERVISED_LLAMAS_INFORMATION,
}


in deeplab/train.py

Usage:
    python3 deeplab_train.py
    --help for arguments
    Needs to have existing tfrecords from unsupervised Llamas dataset

This is only an example. I don't recommend training based on this or even using this.
"""

import argparse
import os
import subprocess

from unsupervised_llamas.label_scripts import dataset_constants as dc
from unsupervised_llamas.deeplab import deeplab_common


def train_deeplab(settings):
    """ Prepares variables to call tensorflow/models/research/deeplab's training function """
    train_dir = '{}_deeplab_{}_{}'.format(dc.TRAIN_DIRECTORY,
                                          settings['input_type'],
                                          settings['problem'])
    os.makedirs(train_dir, exist_ok=True)

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(settings['gpu'])
    train_call = [
        'python',
        os.path.join(deeplab_common.DEEPLAB_DIR, 'train.py'),

        # TODO FIXME There is an issue here and those are set manually in deeplab/train.py
        # '--noinitialize_last_layer'
        # '--last_layers_contain_logits_only',

        '--logtostderr',
        '--train_split=train',
        # NOTE Add dataset into deeplab/datasets/segmentation_dataset
        '--dataset={}'.format(deeplab_common.segmentation_set_name(settings)),
        '--model_variant=xception_65',
        '--atrous_rates=6',
        '--atrous_rates=12',
        '--atrous_rates=18',
        '--output_stride=16',
        '--decoder_output_stride=4',
        '--train_crop_size=513',
        '--train_crop_size=513',
        '--save_interval_secs=3600',
        '--train_batch_size=4',
        '--training_number_of_steps={}'.format(settings['num_iterations']),
        '--fine_tune_batch_norm=true',
        '--tf_initial_checkpoint={}'.format(deeplab_common.PRETRAINED_PATH),
        '--train_logdir={}'.format(train_dir),
        '--dataset_dir={}'.format(deeplab_common.tfrecords_dir(settings))]

    subprocess.call(train_call, env=env)


def parse_args():
    """ Defines defaults and command line parser """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num_iterations', type=int, default=10**6, help='Number of iterations')
    parser.add_argument('--input_type', type=str, default='gray', help='gray, location or color')
    parser.add_argument('--problem', type=str, default='multi', help='binary or multi')
    parser.add_argument('--gpu', type=int, default=0, help='0 to n, n being your number of GPUs')

    return vars(parser.parse_args())


if __name__ == '__main__':
    train_deeplab(parse_args())
