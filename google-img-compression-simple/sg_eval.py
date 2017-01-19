"""
Evaluation for img compression.
Model is from the checkpoint file
now this code only support to evalate
one sample per step
"""
#pylint: disable=C0301

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import os

import tensorflow as tf
from tensorflow.python.ops import math_ops

import sg_model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', './compression_eval',
                           """Directory for even logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """data for evaluation.""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'compress_train_log/',
                           """Directory for checkpoint file""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60*5,
                            """how often to run evaluation""")
tf.app.flags.DEFINE_integer('num_examples', 1,#for debug use 1
                            """number of examples to run""")
tf.app.flags.DEFINE_bool('run_onece', False, """whether to run eval only onece""")
#initialize all variable
def eval_once(saver, loss_tensor, encode_imgs_tensors, decode_imgs_tensors,
              encode_png_tensor, encode_png_input_tensor):
    """run eval once
        input is tensor with shape(batch_size, height, width, channels)
    """
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return
        #start queue runners
        coord = tf.train.Coordinator()#a coordinator like a threadpool
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))
            num_iter = int(math.ceil(FLAGS.num_examples/1))#orig FLAGS.batch_size, now use 1 for debug
            step = 0
            while step < num_iter and not coord.should_stop():
                #encode encode_imgs
                step += 1
                #print(encode_imgs[0].get_shape())
                #encode_img = np.uint8(np.clip(encode_imgs[0]+0.5, 0, 255))
                #encode_img = encode_img.squeeze()
                #encode_png_img = sess.run(encode_png_tensor,
                #                          feed_dict={encode_png_input_tensor: encode_img})
                #with tf.gfile.FastGFile(os.path.join(FLAGS.eval_dir,
                #                                     'image_{0:05d}.png'.format(step)),
                #                        'w') as output_enc_image:
                #    output_enc_image.write(encode_png_img)
                loss, encode_imgs, decode_imgs = sess.run([loss_tensor, encode_imgs_tensors, decode_imgs_tensors])
                print(loss)
                #store decode_img 4-d tensor
                for index, decode_img in enumerate(decode_imgs):
                    #decode_img_3d = tf.reshape([32, 32, 3])
                    #print(decode_img.shape)
                    img = np.uint8(np.clip(decode_img+0.5, 0, 255))
                    img = img.squeeze()
                    png_img = sess.run(encode_png_tensor,
                                       feed_dict={encode_png_input_tensor: img})
                    with tf.gfile.FastGFile(os.path.join(FLAGS.eval_dir,
                                                         'image_{0:05d}_{1:02d}.png'.format(step, index+1)),
                                            'w') as output_image:
                        output_image.write(png_img)
                    #output enc imgs
                encode_img = np.uint8(np.clip(encode_imgs+0.5, 0, 255))
                encode_img = encode_img.squeeze()
                encode_png_img = sess.run(encode_png_tensor,
                                          feed_dict={encode_png_input_tensor: encode_img})
                with tf.gfile.FastGFile(os.path.join(FLAGS.eval_dir,
                                                     'image_{0:05d}.png'.format(step)),
                                        'w') as output_enc_image:
                    output_enc_image.write(encode_png_img)
        except Exception as e:
            coord.request_stop(e)
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

def evaluate():
    """Evaluate the compression model,
       print loss and the decoded img"""
    with tf.Graph().as_default() as g:
        #get images and labels from cifar-10 test
        eval_data = True
        images, labels = sg_model.inputs(eval_data=eval_data, batch_size=1)
        #inference
        resi_images = sg_model.inference(images)
        #recontruct images
        decode_images = []
        #decode_images.append(resi_images[0])
        for i, resi_image in enumerate(resi_images):
            if i == 0:
                decode_images.append(resi_image)
            elif i == 1:
                continue
            else:
                decode_images.append(math_ops.sub(math_ops.add(resi_images[0], resi_images[1]),
                                                  resi_image))
                #decode_images.append(math_ops.add(decode_images[i-1], resi_image))
        #build a graph, and a subgraph for png image
        loss_l1 = sg_model.loss(resi_images)
        encode_png_input_tensor = tf.placeholder(tf.uint8)
        encode_png_tensor = tf.image.encode_png(encode_png_input_tensor)
        #restore the
        variables_to_restore = tf.trainable_variables()
        saver = tf.train.Saver(variables_to_restore)
        eval_once(saver, loss_l1, images, decode_images,
                  encode_png_tensor, encode_png_input_tensor)
        #tf.train.write_graph()#may select whether to write the variable

def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()

if __name__ == '__main__':
    tf.app.run()
