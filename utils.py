import numpy as np
import tensorflow as tf
import scipy.misc

def batch_norm(x, scope):
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope=scope)

def conv2d(input, output_dim, k_h=4, k_w=4, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', [k_h, k_w, input.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        bias = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(tf.nn.conv2d(input, weight, strides=[1, d_h, d_w, 1], padding='SAME'), bias)
        return conv

def deconv2d(input, output_shape, k_h=4, k_w=4, d_h=2, d_w=2, stddev=0.02, name="deconv2d"):
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', [k_h, k_w, output_shape[-1], input.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable('bias', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(tf.nn.conv2d_transpose(input, weight, output_shape=output_shape, strides=[1, d_h, d_w, 1]), bias)
        return deconv

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

def linear(input, output_size, scope=None, stddev=0.02, bias_start=0.0):
    shape = input.get_shape().as_list()
    with tf.variable_scope(scope or "Linear"):
        weight = tf.get_variable("weight", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        return tf.matmul(input, weight) + bias

def imread(path):
    return scipy.misc.imread(path)

def imsave(image, path):
    return scipy.misc.imsave(path, image)
