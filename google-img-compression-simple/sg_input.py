"""
read cifar-10 as the train data
"""

import os

from six.moves import xrange
import tensorflow as tf
import numpy as np

#global variable decribing the CIFIAR-10 data set
IMAGE_SIZE = 32
NUM_CLASS = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

def read_cifar10(filename_queue):
    """
    read and parse examples from cifar-10
    """
    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()
    label_bytes = 1
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height*result.width*result.depth

    record_bytes = label_bytes + image_bytes
    #read a record, getting filenames from the filename_queue
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)
    record_bytes = tf.decode_raw(value, tf.uint8)#record_bytes represent numbers
    result.label = tf.cast(
        tf.strided_slice(record_bytes, [0], [label_bytes], [1]), tf.int32 #strided_slice error
    )
    result.label = tf.reshape(result.label, [1])
    #print result.label.get_shape()
    #print result.label.get_shape()
    #reshape from [depth*height*width] to [depth, height, width]
    #tmp_major = tf.strided_slice(record_bytes, [label_bytes], [label_bytes+image_bytes], [1])
    #print tmp_major.get_shape()
    depth_major = tf.reshape(
        tf.strided_slice(record_bytes, [label_bytes],
                         [label_bytes+image_bytes], [1]),
        [result.depth, result.height, result.width]
    )
    #convert from [depth, height, width] to [height, width, depth]
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])
    return result

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    """construct a queued batch of images and labels"""
    num_preprocess_threads = 2#running queue threads num
    #image_shape = image.get_shape()
    #print image_shape
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3*batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3*batch_size)
    #display the training images with a summary
    #tf.image_summary('images', images)
    return images, tf.reshape(label_batch, [batch_size])

def inputs(eval_data, data_dir, batch_size):
    """construct input for evaluation and training using reader ops"""
    if not eval_data:
        filenames = [os.path.join(data_dir, 'data_batch_%d' %i)
                     for i in xrange(1, 6)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filenames = [os.path.join(data_dir, 'test_batch.bin')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
    #create a queue taht prodeuces the filenames to read
    filename_queue = tf.train.string_input_producer(filenames)
    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    #image cropped size
    height = IMAGE_SIZE
    width = IMAGE_SIZE

    #crop the central, for me the total 32x32 img is used
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
                                                           width, height)
    float_image = tf.image.per_image_standardization(resized_image)
    #random shuffle param
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = (num_examples_per_epoch*min_fraction_of_examples_in_queue)
    #generate a batch of images and labels by building up a queue of examples
    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, batch_size,
                                           shuffle=False)
