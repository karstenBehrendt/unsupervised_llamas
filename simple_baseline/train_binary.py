"""
Simple training script for segmenting lane markers.
It should be straightforward to replace network architectures.

This is a simple implementation, not a particularly clean one.
"""

import argparse
import os

import cv2
import numpy
import numpy.random
import tensorflow as tf
import tqdm

from unsupervised_llamas.common import constants
from unsupervised_llamas.label_scripts import dataset_constants
from unsupervised_llamas.simple_baseline import segmentation_batch_reader
from unsupervised_llamas.simple_baseline import simple_net
from unsupervised_llamas.simple_baseline import utils

DEBUG = True
IMAGE_WIDTH = 600
IMAGE_HEIGHT = 600
BATCH_SIZE = 5

SUFFIX = 'marker_net'


def normalize_image(tf_tensor, name=None):
    return tf.subtract(tf_tensor / (255.0 / 2.0), 1, name=name)  # using function to name it


def scale_image_values(tf_tensor, name=None):
    return tf.divide(tf_tensor, 255.0, name=name)


def random_crop_params(crop_height=400, crop_width=400, image_height=717, image_width=1276):
    x = numpy.random.random_integers(0, image_width - crop_width)
    y = numpy.random.random_integers(0, image_height - crop_height)
    return {'x1': x, 'y1': y, 'x2': x + crop_width, 'y2': y + crop_height}


def segmentation_functions(input_batch, is_training, segmentation):
    """
    """
    segmentation = tf.cast(tf.equal(segmentation, 0), tf.float32)
    with tf.name_scope('inference_values'):
        logits = simple_net.lane_marker_net_2rt(input_batch, is_training)
        prediction = tf.identity(logits, name='prediction')  # named for restoring

    with tf.name_scope('losses'):
        abs_loss = tf.reduce_sum(tf.abs(prediction - segmentation), name='abs_loss')
        misclassifications = tf.reduce_sum(tf.round(tf.abs(prediction - segmentation)), name='misclassifications')
        train_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=segmentation, logits=prediction, name='train_loss')

    return {'prediction': prediction, 'abs_loss': abs_loss, 'train_loss': train_loss,
            'misclassifications': misclassifications}


def train(train_tfrecords, valid_tfrecords, checkpoint=None):

    train_directory = dataset_constants.TRAIN_DIRECTORY + SUFFIX
    os.makedirs(train_directory)

    print('Working with ', constants.NUM_TRAIN_IMAGES, 'for training and ',
          constants.NUM_VALID_IMAGES, 'for validation')

    def current_sample(current_epoch, current_minibatch):
        return current_epoch * constants.NUM_TRAIN_IMAGES + current_minibatch * BATCH_SIZE

    with tf.Graph().as_default() as default_graph:

        # All placeholders
        is_training = tf.placeholder(tf.bool, name='is_training')
        segmentation_input = tf.placeholder(tf.float32, shape=(None, None, None, 1), name='segmentation_input')
        image_input = tf.placeholder(tf.float32, shape=(None, None, None, 1), name='image_input')
        segmentation_batch = tf.identity(segmentation_input, name='segmentation_batch')

        image_batch = normalize_image(image_input, name='image_batch')

        funcs = segmentation_functions(image_batch, is_training, segmentation_batch)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # for batch_norm, assigns averages

        train_step = tf.train.AdamOptimizer().minimize(funcs['train_loss'])

        train_batch = segmentation_batch_reader.batch_reader(
            train_tfrecords, batch_size=BATCH_SIZE, name='train_batch')
        valid_batch = segmentation_batch_reader.batch_reader(
            valid_tfrecords, batch_size=BATCH_SIZE, name='valid_batch')

        merged_tf_train_summaries = tf.summary.merge_all()

        writer = tf.summary.FileWriter(train_directory)   # NOTE not another with
        writer.add_graph(default_graph)  # not given per default

        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=5, keep_checkpoint_every_n_hours=1)
        init_op = tf.global_variables_initializer()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            sess.run(init_op)

            if checkpoint is not None:
                print('Restoring last checkpoint from', checkpoint)
                actual_checkpoint = utils.get_checkpoint(checkpoint)
                print('Restoring', actual_checkpoint)
                saver.restore(sess, actual_checkpoint)
                print('Checkpoint restored')

            for epoch in range(500):  # number of training epochs

                #######################################################################################################
                # Training

                mean_abs_loss = 0
                mean_miss_loss = 0
                mean_train_loss = 0
                for minibatch in tqdm.tqdm(range(constants.NUM_TRAIN_IMAGES // BATCH_SIZE), desc='Training epoch ' + str(epoch)):
                    train_batch_numpy = sess.run(train_batch)
                    rcp = random_crop_params(crop_height=IMAGE_HEIGHT, crop_width=IMAGE_WIDTH)
                    gray_crop = train_batch_numpy['camera_image'][:, rcp['y1']:rcp['y2'], rcp['x1']:rcp['x2'], :]
                    seg_crop = train_batch_numpy['segmentation_image'][:, rcp['y1']:rcp['y2'], rcp['x1']:rcp['x2'], :]
                    feed_dict = {
                        'is_training:0': True,
                        image_input: gray_crop,
                        segmentation_input: seg_crop}

                    iteration = sess.run(
                        {
                            'update_ops': update_ops,
                            'train_step': train_step,
                            'prediction': funcs['prediction'],
                            'abs_loss': funcs['abs_loss'],
                            'miss_loss': funcs['misclassifications'],
                            'train_loss': funcs['train_loss'],
                            'input_batch': image_batch,
                            'segmentation_batch': segmentation_batch,
                            'train_summaries': merged_tf_train_summaries,
                        },
                        feed_dict=feed_dict)
                    mean_abs_loss += numpy.mean(iteration['abs_loss'])
                    mean_miss_loss += numpy.mean(iteration['miss_loss'])
                    mean_train_loss += numpy.mean(iteration['train_loss'])

                    if minibatch % 25 == 0 and DEBUG:
                        debug_image = (iteration['input_batch'][0, :, :, 0] + 1.0) / 2.0
                        debug_image = cv2.cvtColor(debug_image, cv2.COLOR_GRAY2BGR)
                        debug_label = iteration['segmentation_batch'][0, :, :, 0]
                        debug_prediction = iteration['prediction'][0, :, :, 0]
                        debug_prediction = 1 / (1 + numpy.exp(debug_prediction))

                        debug_image[:, :, 1] = debug_label
                        debug_image[:, :, 2] = debug_prediction
                        cv2.imshow('debug_image', debug_image)
                        cv2.imshow('debug_prediction', debug_prediction)
                        cv2.waitKey(5)

                mean_train_train_loss = mean_train_loss / float(minibatch + 1)
                mean_train_miss_loss = mean_miss_loss / float(minibatch + 1)
                mean_train_abs_loss = mean_abs_loss / float(minibatch + 1)
                print('mean train loss', mean_train_loss, epoch)

                #######################################################################################################
                # Validation

                for minibatch in tqdm.tqdm(range(constants.NUM_VALID_IMAGES // BATCH_SIZE)):

                    valid_batch_numpy = sess.run(valid_batch)
                    rcp = random_crop_params(crop_height=IMAGE_HEIGHT, crop_width=IMAGE_WIDTH)
                    gray_crop = valid_batch_numpy['camera_image'][:, rcp['y1']:rcp['y2'], rcp['x1']:rcp['x2'], :]
                    seg_crop = valid_batch_numpy['segmentation_image'][:, rcp['y1']:rcp['y2'], rcp['x1']:rcp['x2'], :]

                    feed_dict = {'is_training:0': False,
                                 image_input: gray_crop,
                                 segmentation_input: seg_crop}

                    iteration = sess.run(
                        {
                            'abs_loss': funcs['abs_loss'],
                            'train_loss': funcs['train_loss'],
                            'miss_loss': funcs['misclassifications'],
                        },
                        feed_dict=feed_dict)
                    mean_abs_loss += numpy.mean(iteration['abs_loss'])
                    mean_train_loss += numpy.mean(iteration['train_loss'])
                    mean_miss_loss += numpy.mean(iteration['miss_loss'])
                mean_valid_train_loss = mean_train_loss / float(minibatch + 1)
                mean_valid_miss_loss = mean_miss_loss / float(minibatch + 1)
                mean_valid_abs_loss = mean_abs_loss / float(minibatch + 1)
                print('mean_valid_loss', mean_valid_train_loss, epoch)

                #######################################################################################################

                print('Writing checkpoint')
                saver.save(sess, os.path.join(train_directory, 'markers'), global_step=current_sample(epoch + 2, 0))
                print('Done writing checkpoint')

                mean_abs = tf.Summary()
                mean_abs.value.add(tag='mean_train_abs_loss', simple_value=mean_train_abs_loss)
                mean_abs.value.add(tag='mean_valid_abs_loss', simple_value=mean_valid_abs_loss)
                mean_abs.value.add(tag='mean_train_miss_loss', simple_value=mean_train_miss_loss)
                mean_abs.value.add(tag='mean_valid_miss_loss', simple_value=mean_valid_miss_loss)
                mean_abs.value.add(tag='mean_train_train_loss', simple_value=mean_train_train_loss)
                mean_abs.value.add(tag='mean_valid_train_loss', simple_value=mean_valid_train_loss)
                writer.add_summary(mean_abs, current_sample(epoch + 1, 0))

            coord.request_stop()
            coord.join(threads)
            writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Train some regression')
    parser.add_argument('--train_tfrecords', type=str, required=True,
                        help='Tfrecords file for training')
    parser.add_argument('--valid_tfrecords', type=str, required=True,
                        help='Tfrecords file for validation')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='If provided, continues training for a given checkpoint')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args.train_tfrecords, args.valid_tfrecords, args.checkpoint)
