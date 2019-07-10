#!/usr/bin/env python3
"""
A couple of iterators for older tfrecords files.
This one should be adapted to work with deeplab tfrecords
so that there aren't too many different formats

Includes different versions of batch_readers. Most of them
can be removed.
"""

# TODO Not up to date and should not be used

import cv2
import numpy
import tensorflow as tf
import tqdm


def dataset_iterator_python(tfrecords_file, debug_print=False):
    """ Quick and dirty dataset iterator """
    # TODO Needs to be adapted to work with deeplab tfrecord format
    # NOTE incomplete!
    example = tf.train.Example()
    with tf.Session():
        for record in tqdm.tqdm(tf.python_io.tf_record_iterator(tfrecords_file)):
            example.ParseFromString(record)

            camera_image = example.features.feature['camera_image'].bytes_list.value[0]
            segmentation = example.features.feature['segmentation'].bytes_list.value[0]
            camera_image = tf.image.decode_png(camera_image, channels=1).eval()
            segmentation = tf.image.decode_png(segmentation, channels=1).eval()
            multi_class_segmentation = example.features.feature['multi_class_segmentation'].\
                bytes_list.value[0]
            multi_class_segmentation = numpy.fromstring(multi_class_segmentation,
                                                        dtype=numpy.uint8)
            multi_class_segmentation = numpy.reshape(multi_class_segmentation, [717, 1276, 5])

            if debug_print:
                print('#################')
                cv2.imshow('camera_image', camera_image)
                cv2.imshow('segmentation', segmentation)
                cv2.waitKey(0)

            yield {'camera_image': numpy.expand_dims(camera_image, 0),
                   'segmentation_image': numpy.expand_dims(segmentation, 0),
                   'multi_class_segmentation': numpy.expand_dims(multi_class_segmentation, 0)}


def dataset_iterator(tfrecords_file):
    """ Yields single samples from tfrecord file for debugging """
    # TODO Needs to be adapted to work with deeplab tfrecords
    num_samples = sum(1 for _ in tqdm.tqdm(tf.python_io.tf_record_iterator(tfrecords_file),
                                           desc='Getting number of samples. May take a bit.'))
    print('Number of samples', num_samples)
    with tf.Session() as data_sess:
        batch = batch_reader(tfrecords_file, batch_size=1)

        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord)

        for _ in tqdm.tqdm(range(num_samples)):
            numpy_batch = data_sess.run(batch)
            yield numpy_batch


def _parse_function(example):
    # TODO Needs to be adapted to work with deeplab tfrecords
    features = {
        'camera_image': tf.FixedLenFeature([], tf.string),
        'segmentation': tf.FixedLenFeature([], tf.string),
        'multi_class_segmentation': tf.FixedLenFeature([], tf.string),
    }
    example = tf.parse_single_example(example, features)
    camera_image = tf.image.decode_png(example['camera_image'], channels=1, dtype=tf.uint8)
    camera_image.set_shape([717, 1276, 1])
    segmentation = tf.image.decode_png(example['segmentation'], channels=1, dtype=tf.uint8)
    segmentation.set_shape([717, 1276, 1])
    multi_class_segmentation = tf.decode_raw(example['multi_class_segmentation'], tf.uint8)
    multi_class_segmentation = tf.reshape(multi_class_segmentation, [717, 1276, 5])

    batch_entries = {'camera_image': camera_image, 'segmentation_image': segmentation,
                     'multi_class_segmentation': multi_class_segmentation}
    return batch_entries


def batch_reader(dataset_file, batch_size=10, name=None):
    """ tf.train.batch for dataset"""
    # TODO Needs to be adapted to work with deeplab tfrecords
    paths = [dataset_file]
    tfrecord_file_queue = tf.train.string_input_producer(paths, name='itfrecord_queue')

    reader = tf.TFRecordReader()
    _, batch = reader.read(tfrecord_file_queue)
    batch_entries = _parse_function(batch)
    batch = tf.train.batch(batch_entries, batch_size=batch_size,
                           num_threads=4, capacity=50, name=name)

    return batch


def dataset_reader(batch_size=1):
    # TODO Needs to be adapted to work with deeplab tfrecords
    """ Another batch reader, this time tf.data.TFRecordDataset """
    filenames = tf.placeholder(tf.string, shape=[None])
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_function, num_parallel_calls=4)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size * 10)
    iterator = dataset.make_initializable_iterator()
    return iterator, filenames
