#!/usr/bin/env python3
"""
Segmentation dataset specific to the tf models deeplab implementation
https://github.com/tensorflow/models/tree/master/research/deeplab

Needs paths configured in label_scripts/dataset_constants.py

Usage:
    python deeplab_tfrecords.py

Paramaters / Flags:
    --color_input switches input images to color images intead of grayscale
                  The driver assistance camera focuses on specific colors with an RCCB pattern.
                  Color images may not look like your typical video. The grayscale
                  images may look nicer. The dataset contains both.
    --multi_class switches pixel-level annotations from binary to lane specific annotations
                 The binary problem marks every pixel as either 0 for not being part of a marker
                 or 1 for marker pixels. The multi-class problem adds the information which lane,
                 relative to the vehicle, this marker pixel belongs to.
                 The annotation files allow you to add even more information!
    --location_gradients Adds location information as extra channels to the input channels.
                         Lane markers are location dependent. Location information can be
                         useful because of limited receptive fields and crops.
                         Implemented by scaling pixel location as relativ coordinate in the
                         respective coordinate. --> x: 0, 0.03, 0.06, ...., 0.97, 1.0
                         (There are not gradients stored in the image, they are just called
                          gradient images because their values increase linearly. There
                          could be better names.)
"""
# NOTE Could use a few sanity checks
# NOTE Could use some output on the current flags

import os
from random import shuffle
import sys

import cv2
from deeplab.datasets import build_data  # requires models/research in PYTHONPATH
import numpy
import tensorflow as tf
import tqdm

from unsupervised_llamas.common import constants
from unsupervised_llamas.label_scripts import dataset_constants as dc
from unsupervised_llamas.label_scripts import helper_scripts
from unsupervised_llamas.label_scripts import label_file_scripts
from unsupervised_llamas.label_scripts import segmentation_labels
from unsupervised_llamas.label_scripts import visualize_labels


# NOTE Also using tf.app.flags because build_data complains about other args
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('multi_class', False, 'Multi class segmentation, otherwise binary')
tf.app.flags.DEFINE_boolean('color_input', False, 'Color input, otherwise gray images')
tf.app.flags.DEFINE_boolean('location_gradients', False, 'Add gradient images to grayscale image')


def create_deeplab_tfrecords(input_folder, tfrecord_file):
    """Creates a tfrecord file for a given folder

    Parameters:
        input_folder: str, path to samples for a given dataset
        tfrecord_file: str, path to tfrecord that will be created

    Flags:
        See docstring for more information
        color_input: whether to use gray or color images
        multi_class: binary or multi-class segmentation
        location_gradients: location information as extra channels
    """
    label_paths = helper_scripts.get_files_from_folder(input_folder, '.json')
    shuffle(label_paths)
    print('{} label files in {}'.format(len(label_paths), input_folder))

    loc_grad_x = list(map(lambda z: z / constants.IMAGE_WIDTH * 255, range(constants.IMAGE_WIDTH)))
    loc_grad_y = list(map(lambda z: z / constants.IMAGE_HEIGHT * 255, range(constants.IMAGE_HEIGHT)))
    loc_grad_x = numpy.asarray([loc_grad_x] * constants.IMAGE_HEIGHT)
    loc_grad_y = numpy.asarray([loc_grad_y] * constants.IMAGE_WIDTH).transpose()
    loc_grad_x = numpy.round(loc_grad_x).astype(numpy.uint8)
    loc_grad_y = numpy.round(loc_grad_y).astype(numpy.uint8)

    os.makedirs(os.path.dirname(tfrecord_file), exist_ok=True)
    with tf.python_io.TFRecordWriter(tfrecord_file) as writer:
        for label_path in tqdm.tqdm(label_paths, total=len(label_paths),
                                    desc='Creating ' + tfrecord_file):

            image_name = os.path.basename(label_path).replace('.json', '')
            if FLAGS.color_input:
                image_data = label_file_scripts.read_image(label_path, image_type='color')
            else:
                image_data = label_file_scripts.read_image(label_path, image_type='gray')
                if FLAGS.location_gradients:
                    image_data = numpy.stack([image_data, loc_grad_x, loc_grad_y], -1)
            image_data = cv2.imencode('.png', image_data)[1].tostring()

            if FLAGS.multi_class:
                segmentation_label = segmentation_labels.create_multi_class_segmentation_label(
                    label_path)
                segmentation = numpy.zeros(segmentation_label.shape[0:2], numpy.uint8)
                for class_index in range(1, 5):
                    segmentation[segmentation_label[:, :, class_index] > 0] = class_index
            else:
                segmentation = visualize_labels.create_segmentation_image(
                    label_path, image='blank')
                segmentation = cv2.cvtColor(segmentation, cv2.COLOR_BGR2GRAY)
                segmentation = segmentation > 0
                segmentation = segmentation.astype(numpy.uint8)

            segmentation = cv2.imencode('.png', segmentation)[1].tostring()

            example = build_data.image_seg_to_tfexample(
                image_data, image_name, constants.IMAGE_HEIGHT,
                constants.IMAGE_WIDTH, segmentation)

            writer.write(example.SerializeToString())


def create_deeplab_sets():
    """ Create the individual sets based on command line arguments
    Sets output names based on flags and calls tfrecord creation for the individual datasets
    """
    if FLAGS.color_input and FLAGS.location_gradients:
        print('Cannot add location gradients to color image')
        sys.exit()

    prefix1 = 'color' if FLAGS.color_input else 'gray'
    prefix1 = prefix1 + '_grads' if FLAGS.location_gradients else prefix1
    prefix2 = 'multi' if FLAGS.multi_class else 'binary'
    tfrecords_folder = os.path.join(dc.TFRECORDS_FOLDER, '{}_{}'.format(prefix1, prefix2))
    os.makedirs(tfrecords_folder, exist_ok=True)

    # TODO test folder not available
    # TODO Adapt test to handle missing annotations or remove test
    osj = os.path.join
    # NOTE the dash after 'train', 'valid', and 'test' terminates set names in deeplab
    for dataset in [(osj(dc.LABELS, 'train'), osj(tfrecords_folder, 'train-set.tfrecords')),
                    (osj(dc.LABELS, 'valid'), osj(tfrecords_folder, 'valid-set.tfrecords')),
                    (osj(dc.LABELS, 'test'), osj(tfrecords_folder, 'test-set.tfrecords'))]:
        # not multithreaded anymore. Call without loop possible
        create_deeplab_tfrecords(*dataset)


if __name__ == '__main__':
    # READ THE DOCSTRING BEFORE RUNNING THIS
    create_deeplab_sets()
