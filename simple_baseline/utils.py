""" Small collection of util functions """
import os
import tensorflow as tf


def get_checkpoint(checkpoint):
    """ Returns last checkpoint from directory, checkpoint file, or 'checkpoint' file """
    if checkpoint is None:
        return None
    if os.path.basename(checkpoint) == 'checkpoint':
        checkpoint = os.path.dirname(checkpoint)
    if os.path.isdir(checkpoint):
        last_checkpoint = tf.train.latest_checkpoint(checkpoint)
        checkpoint = last_checkpoint
    print('Loading checkpoint:', checkpoint)
    return checkpoint
