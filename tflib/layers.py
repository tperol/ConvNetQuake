import tensorflow as tf
import numpy as np

def conv(inputs,
         nfilters,
         ksize,
         stride=1,
         padding='SAME',
         use_bias=True,
         activation_fn=tf.nn.relu,
         initializer=tf.contrib.layers.variance_scaling_initializer(),
         regularizer=None,
         scope=None,
         reuse=None):
  with tf.variable_scope(scope, reuse=reuse):
    n_in = inputs.get_shape().as_list()[-1]
    weights = tf.get_variable(
      'weights',
      shape=[ksize, ksize, n_in, nfilters],
      dtype=inputs.dtype.base_dtype,
      initializer=initializer,
      collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.VARIABLES],
      regularizer=regularizer)

    strides = [1, stride, stride, 1]
    current_layer = tf.nn.conv2d(inputs, weights, strides, padding=padding)

    if use_bias:
      biases = tf.get_variable(
        'biases',
        shape=[nfilters,],
        dtype=inputs.dtype.base_dtype,
        initializer=tf.constant_initializer(0.0),
        collections=[tf.GraphKeys.BIASES, tf.GraphKeys.VARIABLES])
      current_layer = tf.nn.bias_add(current_layer, biases)

    if activation_fn is not None:
      current_layer = activation_fn(current_layer)

    return current_layer

def transpose_conv(inputs,
         nfilters,
         ksize,
         stride=1,
         padding='SAME',
         use_bias=True,
         activation_fn=tf.nn.relu,
         initializer=tf.contrib.layers.variance_scaling_initializer(),
         regularizer=None,
         scope=None,
         reuse=None):
  with tf.variable_scope(scope, reuse=reuse):
    n_in = inputs.get_shape().as_list()[-1]
    weights = tf.get_variable(
      'weights',
      shape=[ksize, ksize, n_in, nfilters],
      dtype=inputs.dtype.base_dtype,
      initializer=initializer,
      collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.VARIABLES],
      regularizer=regularizer)

    bs, h, w, c = inputs.get_shape().as_list()
    strides = [1, stride, stride, 1]
    out_shape = [bs, stride*h, stride*w, c]
    current_layer = tf.nn.conv2d_transpose(inputs, weights, out_shape, strides, padding=padding)

    if use_bias:
      biases = tf.get_variable(
        'biases',
        shape=[nfilters,],
        dtype=inputs.dtype.base_dtype,
        initializer=tf.constant_initializer(0.0),
        collections=[tf.GraphKeys.BIASES, tf.GraphKeys.VARIABLES])
      current_layer = tf.nn.bias_add(current_layer, biases)

    if activation_fn is not None:
      current_layer = activation_fn(current_layer)

    return current_layer

def conv1(inputs,
         nfilters,
         ksize,
         stride=1,
         padding='SAME',
         use_bias=True,
         activation_fn=tf.nn.relu,
         initializer=tf.contrib.layers.variance_scaling_initializer(),
         regularizer=None,
         scope=None,
         reuse=None):
  with tf.variable_scope(scope, reuse=reuse):
    n_in = inputs.get_shape().as_list()[-1]
    weights = tf.get_variable(
      'weights',
      shape=[ksize, n_in, nfilters],
      dtype=inputs.dtype.base_dtype,
      initializer=initializer,
      collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.VARIABLES],
      regularizer=regularizer)

    current_layer = tf.nn.conv1d(inputs, weights, stride, padding=padding)

    if use_bias:
      biases = tf.get_variable(
        'biases',
        shape=[nfilters,],
        dtype=inputs.dtype.base_dtype,
        initializer=tf.constant_initializer(0.0),
        collections=[tf.GraphKeys.BIASES, tf.GraphKeys.VARIABLES])
      current_layer = tf.nn.bias_add(current_layer, biases)

    if activation_fn is not None:
      current_layer = activation_fn(current_layer)

    return current_layer

def atrous_conv1d(inputs,
                nfilters,
                ksize,
                rate=1,
                padding='SAME',
                use_bias=True,
                activation_fn=tf.nn.relu,
                initializer=tf.contrib.layers.variance_scaling_initializer(),
                regularizer=None,
                scope=None,
                reuse=None):
  """ Use tf.nn.atrous_conv2d and adapt to 1d"""
  with tf.variable_scope(scope, reuse=reuse):
    # from (bs,width,c) to (bs,width,1,c)
    inputs = tf.expand_dims(inputs,2)

    n_in = inputs.get_shape().as_list()[-1]
    weights = tf.get_variable(
      'weights',
      shape=[ksize, 1, n_in, nfilters],
      dtype=inputs.dtype.base_dtype,
      initializer=initializer,
      collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.VARIABLES],
      regularizer=regularizer)

    current_layer = tf.nn.atrous_conv2d(inputs,weights, rate, padding=padding)

    # Resize into (bs,width,c)
    current_layer = tf.squeeze(current_layer,squeeze_dims=[2])

    if use_bias:
      biases = tf.get_variable(
        'biases',
        shape=[nfilters,],
        dtype=inputs.dtype.base_dtype,
        initializer=tf.constant_initializer(0.0),
        collections=[tf.GraphKeys.BIASES, tf.GraphKeys.VARIABLES])
      current_layer = tf.nn.bias_add(current_layer, biases)

    if activation_fn is not None:
      current_layer = activation_fn(current_layer)

    return current_layer



def conv3(inputs,
         nfilters,
         ksize,
         stride=1,
         padding='SAME',
         use_bias=True,
         activation_fn=tf.nn.relu,
         initializer=tf.contrib.layers.variance_scaling_initializer(),
         regularizer=None,
         scope=None,
         reuse=None):
  with tf.variable_scope(scope, reuse=reuse):
    n_in = inputs.get_shape().as_list()[-1]
    weights = tf.get_variable(
      'weights',
      shape=[ksize, ksize, ksize, n_in, nfilters],
      dtype=inputs.dtype.base_dtype,
      initializer=initializer,
      collections=[tf.GraphKeys.WEIGHTS, tf.GraphKeys.VARIABLES],
      regularizer=regularizer)

    strides = [1, stride, stride, stride, 1]
    current_layer = tf.nn.conv3d(inputs, weights, strides, padding=padding)

    if use_bias:
      biases = tf.get_variable(
        'biases',
        shape=[nfilters,],
        dtype=inputs.dtype.base_dtype,
        initializer=tf.constant_initializer(0.0),
        collections=[tf.GraphKeys.BIASES, tf.GraphKeys.VARIABLES])
      current_layer = tf.nn.bias_add(current_layer, biases)

    if activation_fn is not None:
      current_layer = activation_fn(current_layer)

    return current_layer


def fc(inputs, nfilters, use_bias=True, activation_fn=tf.nn.relu,
       initializer=tf.contrib.layers.variance_scaling_initializer(),
       regularizer=None, scope=None, reuse=None):
  with tf.variable_scope(scope, reuse=reuse):
    n_in = inputs.get_shape().as_list()[-1]
    weights = tf.get_variable(
      'weights',
      shape=[n_in, nfilters],
      dtype=inputs.dtype.base_dtype,
      initializer=initializer,
      regularizer=regularizer)

    current_layer = tf.matmul(inputs, weights)

    if use_bias:
      biases = tf.get_variable(
        'biases',
        shape=[nfilters,],
        dtype=inputs.dtype.base_dtype,
        initializer=tf.constant_initializer(0))
      current_layer = tf.nn.bias_add(current_layer, biases)

    if activation_fn is not None:
      current_layer = activation_fn(current_layer)

  return current_layer

def batch_norm(inputs, center=False, scale=False,
               decay=0.999, epsilon=0.001, reuse=None,
               scope=None, is_training=False):
  return tf.contrib.layers.batch_norm(
    inputs, center=center, scale=scale,
    decay=decay, epsilon=epsilon, activation_fn=None,
    reuse=reuse,trainable=False, scope=scope, is_training=is_training)

relu = tf.nn.relu

def crop_like(inputs, like, name=None):
  with tf.name_scope(name):
    _, h, w, _ = inputs.get_shape().as_list()
    _, new_h, new_w, _ = like.get_shape().as_list()
    crop_h = (h-new_h)/2
    crop_w = (w-new_w)/2
    cropped = inputs[:, crop_h:crop_h+new_h, crop_w:crop_w+new_w, :]
    return cropped
