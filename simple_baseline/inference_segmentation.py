"""
A very simple inference script which stores segmentation images
to file for a given image folder and a checkpoint of a trained model
"""

import argparse
import os
import pdb  # noqa
import time

import cv2
import numpy
from scipy.special import softmax, expit
import tensorflow as tf
import tqdm

from unsupervised_llamas.label_scripts import dataset_constants
from unsupervised_llamas.common import helper_scripts
from unsupervised_llamas.simple_baseline import utils


def gradient_images():
    x = list(map(lambda z: z / 1276.0 * 255, range(1276)))
    y = list(map(lambda z: z / 717.0 * 255, range(717)))
    grad_x = numpy.asarray([x] * 717)
    grad_y = numpy.asarray([y] * 1276).transpose()
    return grad_x, grad_y


class NetModel():
    def __init__(self, checkpoint):
        checkpoint = utils.get_checkpoint(checkpoint)

        tf.Graph().as_default()
        tf.reset_default_graph()  # if multiple evaluations are used within one script

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self._sess = tf.Session(config=config)

        new_saver = tf.train.import_meta_graph(checkpoint + '.meta')
        init_op = tf.global_variables_initializer()
        self._sess.run(init_op)
        new_saver.restore(self._sess, checkpoint)

        print('All setup for inference')

    def __del__(self):
        self._sess.close()

    def single_batch_inference(self, input_dict):
        input_dict['is_training:0'] = False
        pf = tf.get_default_graph().get_tensor_by_name('inference_values/prediction:0')
        return self._sess.run(pf, feed_dict=input_dict)


def model_speed(checkpoint_file, num_samples, num_channels):
    """Crude method to measure network speeed without optimization"""
    nm = NetModel(checkpoint=checkpoint_file)
    images = numpy.random.random_integers(0, 255, (num_samples, 1, 1216, 717, num_channels))
    with tf.Session():
        start = time.time()
        for image in tqdm.tqdm(images):
            nm.single_batch_inference({'image_input:0': image})

        end = time.time()
    duration = end - start
    print('Inference duration per sample', duration / num_samples, 'based on', num_samples)


def folder_inference(checkpoint_file, image_folder, gray=True, binary=True, location=False, suffix='_test'):
    """
    checkpoint_file: str, path to checkpoint, can also be folder
    tfrecord_file: str, path to file
    """
    out_folder = checkpoint_file + suffix

    input_images = helper_scripts.get_files_from_folder(image_folder, '.png')
    if suffix == '_test':
        assert len(input_images) == dataset_constants.NUM_TEST_IMAGES

    nm = NetModel(checkpoint=checkpoint_file)

    if location:
        # grad_x, grad_y = gradient_images()
        # grad_x_batch = numpy.expand_dims(numpy.asarray([grad_x]), -1)
        # grad_y_batch = numpy.expand_dims(numpy.asarray([grad_y]), -1)
        pass

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config):
        for image_path in tqdm.tqdm(input_images):

            base_path = image_path.replace(image_folder, '')
            base_path = base_path[1:] if base_path.startswith('/') else base_path

            camera_image = cv2.imread(image_path, 0 if gray else 1)
            camera_image = numpy.expand_dims(camera_image, axis=0)

            if binary:
                camera_image = numpy.expand_dims(camera_image, axis=-1)
            if location:
                # Currently, there are images stored to file like this
                # So, this one needs to be implemented, should be < 5 lines
                raise NotImplementedError('Add gradient images to inference input')
            feed_dict = {'image_input:0': camera_image}
            prediction = nm.single_batch_inference(feed_dict)  # A bit bigger. Upsampling, padding

            # Multiply by 255 to get an actual image and stuff
            os.makedirs(os.path.dirname(os.path.join(out_folder, base_path)), exist_ok=True)
            if binary:
                prediction = (expit(prediction)[0, :717, :, 0] - 1.0) * -1  # == softmax
                output_file = os.path.splitext(os.path.join(out_folder, base_path))[0] + '.json_1.png'
                cv2.imwrite(output_file, (prediction * 255).astype(numpy.uint8))
            else:
                prediction = prediction[0, :717, :, :]
                prediction = softmax(prediction, axis=2)
                for i in range(prediction.shape[-1]):
                    output_file = os.path.splitext(os.path.join(out_folder, base_path))[0] + '.json_' + str(i) + '.png'
                    prediction_image = prediction[:, :, i]
                    cv2.imwrite(output_file, (prediction_image * 255).astype(numpy.uint8))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Stores segmentaiton images to file for a given folder and checkpoint')
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Directory or checkpoint file to use for inference / segmenting markers')
    parser.add_argument(
        '--image_folder', type=str, required=True,
        help='Folder with input images to be used for inference')
    parser.add_argument(
        '--gray', action='store_true',
        help='If the input images are used in grayscale instead of color images')
    parser.add_argument(
        '--location', action='store_true',
        help='Add gradient images as two additional channels onto input images')
    parser.add_argument(
        '--binary', action='store_true',
        help='Binary segmentation only, i.e., 0 or 1 instead of segmenting lanes also')
    parser.add_argument(
        '--suffix', type=str, default='_test',
        help='Name for inference run. Will be added to output folder. _test will verify split size.')
    return parser.parse_args()


if __name__ == '__main__':
    args = vars(parse_args())
    folder_inference(checkpoint_file=args['checkpoint'],
                     image_folder=args['image_folder'],
                     gray=args['gray'],
                     binary=args['binary'],
                     location=args['location'],
                     suffix=args['suffix'])
