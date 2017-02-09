"""
train a simple compression model
"""
from datetime import datetime
import time

import tensorflow as tf
import sg_model
#pylint: disable=W0201

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'compress_train_log/',
                           """Directory where to write event logs and checkpoint""")
tf.app.flags.DEFINE_integer('max_steps', 1,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whethre to log device placement.""")

def train():
    """train my simple compress model."""
    with tf.Graph().as_default():
        #global_step for learning-rate decay
        global_step = tf.Variable(0, name='global_step', trainable=False)
        #global_step = tf.Variable(0, trainable=False, name='global_step')
        #global_step should be a tensor with name 'global_step')
        #get images from input
        images, labels = sg_model.inputs(eval_data=False, batch_size=FLAGS.batch_size)
        #inference model
        resi_images = sg_model.inference(images)
        #calculate loss
        loss_l1 = sg_model.loss(resi_images)
        #learning decay
        starter_learning_rate = 0.1
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   150000, 0.1, staircase=True)
        #train op, back propagation
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_l1,
                                                                  global_step=global_step)
        init = tf.initialize_all_variables()
        init_local = tf.initialize_local_variables()
        class _LoggerHook(tf.train.SessionRunHook):
            """logs loss and runtime"""
            def begin(self):
                self._step = -3#there is two step for initialize

            def before_run(self, run_context):
                self._step += 1
                self._start_time = time.time()
                return tf.train.SessionRunArgs(loss_l1)

            def after_run(self, run_context, run_values):
                duration = time.time() - self._start_time
                loss_value = run_values.results
                if self._step % 10 == 0:
                    num_examples_per_step = FLAGS.batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    format_str = ('%s: step %d, loss_l1 = %.2f (%.1f examples/sec; %.3f'
                                  'sec/batch)')
                    print (format_str % (datetime.now(), self._step, loss_value,
                                         examples_per_sec, sec_per_batch))
        with tf.train.MonitoredTrainingSession(checkpoint_dir=FLAGS.train_dir,
                                               hooks=[tf.train.StopAtStepHook(
                                                   last_step=FLAGS.max_steps),
                                                      tf.train.NanTensorHook(loss_l1),
                                                      _LoggerHook()],
                                               config=tf.ConfigProto(
                                                   log_device_placement=
                                                   FLAGS.log_device_placement)) as mon_sess:
            summary_writer = tf.summary.FileWriter('./log', mon_sess.graph)
            mon_sess.run(init)
            mon_sess.run(init_local)
            while not mon_sess.should_stop():
                mon_sess.run(train_op)
            summary_writer.close()

def main(argv=None):
    """
    tensorflow main function
    """
    train()

if __name__ == '__main__':
    """
    main func
    """
    tf.app.run()
