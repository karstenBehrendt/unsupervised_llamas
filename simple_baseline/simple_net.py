"""
This should only be used as a baseline comparison.
It could be added to some results table
"""

import tensorflow as tf

DEBUG = True


def lane_marker_net_2rt(inputs, is_training, out_classes=1):
    """ Super simple, very old network """
    with tf.name_scope('marker_net'):
        conv1 = tf.contrib.slim.conv2d(inputs, num_outputs=32, kernel_size=(3, 3), stride=1, padding='SAME', scope='conv1')
        conv2 = tf.contrib.slim.conv2d(conv1, num_outputs=32, kernel_size=(3, 3), stride=1, padding='SAME', scope='conv2')
        pool1 = tf.contrib.slim.max_pool2d(conv2, kernel_size=(2, 2), stride=2, padding='SAME', scope='pool1')
        conv3 = tf.contrib.slim.conv2d(pool1, num_outputs=64, kernel_size=(5, 5), stride=1, padding='SAME', scope='conv3')
        pool2 = tf.contrib.slim.max_pool2d(conv3, kernel_size=(2, 2), stride=2, padding='SAME', scope='pool2')
        conv4 = tf.contrib.slim.conv2d(pool2, num_outputs=96, kernel_size=(5, 5), stride=1, padding='SAME', activation_fn=None, scope='conv4')
        batch_norm1 = tf.contrib.slim.batch_norm(conv4, activation_fn=tf.nn.relu, is_training=is_training, scope='batch_norm1')
        deconv1 = tf.contrib.slim.conv2d_transpose(batch_norm1, num_outputs=96, kernel_size=(2, 2), stride=(2, 2), padding='SAME', activation_fn=tf.nn.relu, scope='deconv1')
        conv5 = tf.contrib.slim.conv2d(deconv1, num_outputs=64, kernel_size=(3, 3), stride=1, padding='SAME', scope='conv5')
        deconv2 = tf.contrib.slim.conv2d_transpose(conv5, num_outputs=64, kernel_size=(2, 2), stride=(2, 2), padding='SAME', activation_fn=tf.nn.relu, scope='deconv2')
        conv6 = tf.contrib.slim.conv2d(deconv2, num_outputs=128, kernel_size=(3, 3), stride=1, padding='SAME', scope='conv6')
        out = tf.contrib.slim.conv2d(conv6, num_outputs=out_classes, kernel_size=(1, 1), stride=1, padding='SAME', scope='logits', activation_fn=None)

    return out


if __name__ == '__main__':
    some_inputs = tf.placeholder(tf.float32, (10, 460, 640, 1), name='some_inputs')
    prediction = lane_marker_net_2rt(some_inputs, True)
    print(prediction.shape)
    print('Made it through the net')
