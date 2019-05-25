"""
Collection of small helper functions
"""


import os
from unsupervised_llamas.label_scripts import dataset_constants as dc

# Only needed for deeplab training:
# TODO set paths
DEEPLAB_DIR = '/TODO/TODO/tensorflow/models/research/deeplab/'
# TODO NOTE This one is very specific to the base model you are training
PRETRAINED_PATH = ('TODO/TODO/'
                   'deeplabv3_pascal_trainval/model.ckpt.index')


def tfrecords_dir(settings):
    """ Creates names for tfrecord folders according to settings """
    prefix1 = settings['input_type']
    prefix1 = prefix1.replace('location', 'gray_grads')
    prefix2 = settings['problem']
    tfrecords_folder = os.path.join(dc.TFRECORDS_FOLDER, '{}_{}'.format(prefix1, prefix2))
    return tfrecords_folder


def segmentation_set_name(settings):
    """ Deeplab sdataset name """
    return 'unsupervised_llamas' if settings['problem'] == 'multi'\
        else 'binary_unsupervised_llamas'


def checkpoint_dir(settings):
    checkpoint = settings['checkpoint_dir']
    if checkpoint.endswith('/'):
        checkpoint = checkpoint[:-1]
    return checkpoint
