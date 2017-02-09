"""
build a simple compression model like
google compression model did. the refere-
ence paper is Full Resolution Image Comp-
ression with Recurrent Neural Networks 
"""

# pylint: disable=missing-docstring
# pylint: disable=W0311

import gzip
import os
import re
import sys
import tarfile
import math
import collections
import copy
import numpy as np
#import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope as vs
import tensorflow as tf
import sg_input

FLAGS = tf.app.flags.FLAGS

#basic model param
#
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', 'cifar-10-batches-bin',
                           """path to train data.""")
tf.app.flags.DEFINE_boolean('use_fp16', False, """train the model using fp16.""")
tf.app.flags.DEFINE_integer('compress_iteration', 16,
                            """number of iterations to compress""")
tf.app.flags.DEFINE_float('hidden_rconnect_weight', 0.1,
                          """weight for hidden residual connection""")
tf.app.flags.DEFINE_float('input_rconnect_weight', 0.1,
                          """weight fro input residual connection""")
#constants decribing the data set

#constants decribing the training process
MOVING_AVERAGE_DECAY = 0.9999
NUM_EPOCHS_PER_DECAY = 350.0
LEARNING_RATE_DECAY_FACTOR = 0.1
INITIAL_LEARNING_RATE = 0.1
#stored on CPU or GPU
def _variable_with_weight_decay(name, shape, stddev, w_decay):
    """ create an intialized with a trancated normal distribution."""
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var_initializer = tf.truncated_normal_initializer(stddev=stddev, dtype=dtype)
    var = tf.get_variable(name, shape, dtype, var_initializer)
    if w_decay is not None: #weight-decay
        weight_decay = tf.mul(tf.nn.l2_loss(var), w_decay, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def residual_RNNCell(input, hidden_state, hidden_conv_k, input_conv_k,
                     hidden_conv_s, input_conv_s,
                     input_resi_conv_k, input_resi_conv_s):
    """GRU rnn cell with two residual connection
    args:
    """
    #input_dim = tf.shape(input)[3]
    input_dim = input.get_shape()[3]
    #out_dim = tf.shape(hidden_state)[3]
    out_dim = hidden_state.get_shape()[3]
    #print input_dim.value
    #input  conv
    input_conv_k_shape = input_conv_k + [input_dim.value] + [out_dim.value*3]
    kernel_input_conv = _variable_with_weight_decay('weights_input_conv',
                                                    shape=input_conv_k_shape,
                                                    stddev=5e-2,
                                                    w_decay=0.0)
    input_conv = tf.nn.conv2d(input, kernel_input_conv, input_conv_s,
                              padding='SAME')
    z_resi, r_resi, o_resi = array_ops.split(3, 3, input_conv)
    #hidden conv
    hidden_conv_k_shape = hidden_conv_k + [out_dim.value] + [out_dim.value*3]
    kernel_hid_conv = _variable_with_weight_decay('weights_hidden_conv',
                                                  shape=hidden_conv_k_shape,
                                                  stddev=5e-2,
                                                  w_decay=0.0)
    hid_conv = tf.nn.conv2d(hidden_state, kernel_hid_conv, hidden_conv_s,
                            padding='SAME')
    z_hid, r_hid, rc_hid = array_ops.split(3, 3, hid_conv)
    #calc z r
    z_gate = math_ops.add(z_resi, z_hid)
    z_gate = tf.nn.sigmoid(z_gate)
    r_gate = math_ops.add(r_resi, r_hid)
    r_gate = tf.nn.sigmoid(r_gate)
    #calc new_hidden
    zopp_mul_hidden = math_ops.sub(hidden_state, math_ops.mul(z_gate, hidden_state))
    #reset
    r_mul_hidden = math_ops.mul(r_gate, hidden_state)
    hidden_reset_conv_shape = [1, 1] + [out_dim.value, out_dim.value]
    kernel_reset_conv = _variable_with_weight_decay('weights_reset_conv',
                                                    shape=hidden_reset_conv_shape,
                                                    stddev=5e-2,
                                                    w_decay=0.0)
    r_mul_hid_conv = tf.nn.conv2d(r_mul_hidden, kernel_reset_conv, hidden_conv_s,
                                  padding='SAME')
    r_mul_hid_conv_tanh = tf.nn.tanh(math_ops.add(o_resi, r_mul_hid_conv))
    zr_mul_hid_conv_tanh = math_ops.mul(z_hid, r_mul_hid_conv_tanh)
    hid_rconnect_conv = math_ops.scalar_mul(FLAGS.hidden_rconnect_weight, rc_hid)
    #new hidden state
    hidden_state = math_ops.add(math_ops.add(zopp_mul_hidden, zr_mul_hid_conv_tanh),
                                hid_rconnect_conv)
    #input residual connection 1x1 kernel
    input_resi_conv_kshape = input_resi_conv_k + [input_dim.value, out_dim.value]
    kernel_input_rc_conv = _variable_with_weight_decay('weights_input_rc_conv',
                                                       shape=input_resi_conv_kshape,
                                                       stddev=5e-2,
                                                       w_decay=0.0)
    input_resi_conv = tf.nn.conv2d(input, kernel_input_rc_conv, input_resi_conv_s,
                                   padding='SAME')
    input_resi = math_ops.scalar_mul(FLAGS.input_rconnect_weight, input_resi_conv)
    rnn_out = math_ops.add(hidden_state, input_resi)
    return rnn_out, hidden_state

def inference(images):
    """build the compression model.
    args:
        images: images from imagenet
    returns:
        loss: ssd
    """
    #input 32x32x3
    #use a state
    #zero state for hidden state
    hidden_state_enc = array_ops.zeros([FLAGS.batch_size, 8, 8, 256])
    hidden_state_dec_rnn1 = array_ops.zeros([FLAGS.batch_size, 8, 8, 256])
    hidden_state_dec_rnn2 = array_ops.zeros([FLAGS.batch_size, 16, 16, 256])
    #hidden_rconnect_w = math_ops.m
    #residual_images = images
    #every iteration output is store in a ndarray 
    out_resi_images = []
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    for i in range(FLAGS.compress_iteration):
    #conv1 3x3x64
        reuse = True if i > 0 else False
        with tf.variable_scope('encoder_conv1', reuse=reuse) as scope:
            if i > 0: tf.get_variable_scope().reuse_variables()
            kernel = _variable_with_weight_decay('weights',
                                                 shape=[3, 3, 3, 64],
                                                 stddev=5e-2,
                                                 w_decay=0.0)
            encoder_conv1 = tf.nn.conv2d(images, kernel, [1, 2, 2, 1], padding='SAME')
            encoder_conv1_biases = tf.get_variable('biases', [64], dtype,
                                                   tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(encoder_conv1, encoder_conv1_biases)
            encoder_conv1_out = tf.nn.relu(pre_activation, name=scope.name)
    #rnn
    #encoder
        with tf.variable_scope('encoder_rnn1', reuse=reuse) as scope:
            if i > 0: tf.get_variable_scope().reuse_variables()
            encoder_rnn1_out, hidden_state_enc = residual_RNNCell(encoder_conv1_out,
                                                                  hidden_state_enc,
                                                                  [1, 1],
                                                                  [3, 3],
                                                                  [1, 1, 1, 1],
                                                                  [1, 2, 2, 1],
                                                                  [3, 3],
                                                                  [1, 2, 2, 1])
    #binarizer
    #binarizer is stateless
        with tf.variable_scope('binarizer', reuse=reuse) as scope:
            kernel_binarizer_conv = _variable_with_weight_decay('weights_binarizer_conv',
                                                                shape=[1, 1, 256, 32],
                                                                stddev=5e-2,
                                                                w_decay=0.0)
            binarize_conv = tf.nn.conv2d(encoder_rnn1_out, kernel_binarizer_conv,
                                         [1, 1, 1, 1], padding='SAME')
            #sign
            #binarize_conv = tf.nn.softsign(binarize_conv)
            binarize_conv_sign = math_ops.sign(binarize_conv)
        with tf.variable_scope('decoder_conv1', reuse=reuse) as scope:
            if i > 0: tf.get_variable_scope().reuse_variables()
            #conv 1 1 256
            kernel = _variable_with_weight_decay('weights_decode_conv1',
                                                 shape=[1, 1, 32, 256],
                                                 stddev=5e-2,
                                                 w_decay=0.0)
            decoder_conv1 = tf.nn.conv2d(binarize_conv_sign, kernel,
                                         [1, 1, 1, 1], padding='SAME')
            decoder_conv1_biases = tf.get_variable('biases', [256], dtype,
                                                   tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(decoder_conv1, decoder_conv1_biases)
            decoder_conv1_out = tf.nn.relu(pre_activation, name=scope.name)
        #decode rnn1 8x8x256
        with tf.variable_scope('decoder_rnn1', reuse=reuse) as scope:
            if i > 0: tf.get_variable_scope().reuse_variables()
            decoder_rnn1_out, hidden_state_dec_rnn1 = residual_RNNCell(decoder_conv1_out,
                                                                       hidden_state_dec_rnn1,
                                                                       [1, 1],
                                                                       [2, 2],
                                                                       [1, 1, 1, 1],
                                                                       [1, 1, 1, 1],
                                                                       [2, 2],
                                                                       [1, 1, 1, 1])
        #depth to width
        decoder_rnn1_out = tf.reshape(decoder_rnn1_out, [-1, 16, 16, 64])
        #decode rnn2
        with tf.variable_scope('decoder_rnn2', reuse=reuse) as scope:
            if i > 0: tf.get_variable_scope().reuse_variables()
            decoder_rnn2_out, hidden_state_dec_rnn2 = residual_RNNCell(decoder_rnn1_out,
                                                                       hidden_state_dec_rnn2,
                                                                       [1, 1],
                                                                       [2, 2],
                                                                       [1, 1, 1, 1],
                                                                       [1, 1, 1, 1],
                                                                       [2, 2],
                                                                       [1, 1, 1, 1])
        #depth to width
        decoder_rnn2_out = tf.reshape(decoder_rnn2_out, [-1, 32, 32, 64])
        #decode conv2
        with tf.variable_scope('decoder_conv2', reuse=reuse) as scope:
            if i > 0: tf.get_variable_scope().reuse_variables()
            #conv 1 1 256
            kernel = _variable_with_weight_decay('weights_decode_conv2',
                                                 shape=[1, 1, 64, 3],
                                                 stddev=5e-2,
                                                 w_decay=0.0)
            decoder_conv2 = tf.nn.conv2d(decoder_rnn2_out, kernel,
                                         [1, 1, 1, 1], padding='SAME')
            decoder_conv2_biases = tf.get_variable('biases', [3], dtype,
                                                   tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(decoder_conv2, decoder_conv2_biases)
            decoder_conv2_out = tf.nn.relu(pre_activation, name=scope.name)
        #calc residual images
        #array_ops.copy_host
        if i == 0:
            tmp_first_out = array_ops.zeros([FLAGS.batch_size, 32, 32, 3])
            tmp_first_out = math_ops.add(decoder_conv2_out, tmp_first_out)
            out_resi_images.append(tmp_first_out)
        tmp_resi_out = array_ops.zeros([FLAGS.batch_size, 32, 32, 3])
        images = math_ops.sub(images, decoder_conv2_out)
        tmp_resi_out = math_ops.add(tmp_resi_out, images)
        out_resi_images.append(tmp_resi_out)
    return out_resi_images

#define the loss function
def loss(out_resi_images):
    """
    loss function for this network
    loss = beta*sigma(out_reisi_images)
    beta = 1/(s*n)
    s = B*H*W*C
    n = iteration
    """
    loss_l1 = 0
    beta = 1.0/(FLAGS.batch_size*32*32*3*FLAGS.compress_iteration)
    for i, resi_img in enumerate(out_resi_images):
        if i == 0:
            continue
        resi_img = tf.reshape(resi_img, [-1])
        #resi_img.reshape(-1)
        resi_img_abs = math_ops.abs(resi_img)
        #resi_img_abs = np.fabs(resi_img)
        #loss_l1 += sum(resi_img_abs)
        loss_l1 += math_ops.reduce_sum(resi_img_abs)
        loss_l1 = beta*loss_l1
        tf.add_to_collection('losses', loss_l1)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')
#use a class

def inputs(eval_data, batch_size):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    batch_size: batch_size
  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  #data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  data_dir = FLAGS.data_dir
  images, labels = sg_input.inputs(eval_data=eval_data,
                                   data_dir=data_dir,
                                   batch_size=batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels
  