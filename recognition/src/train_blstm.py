from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
import model_blstm


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', '../data/',
                           "data directory")
tf.app.flags.DEFINE_string('checkpoints_dir', '../checkpoints/',
                           "training checkpoints directory")
tf.app.flags.DEFINE_string('log_dir', '../train_log/',
                           "summary directory")
tf.app.flags.DEFINE_integer('batch_size', 512,
                            "mini-batch size")
tf.app.flags.DEFINE_integer('total_epoches', 100,
                            "total training epoches")
tf.app.flags.DEFINE_integer('hidden_size', 100,
                            "size of LSTM hidden memory")
tf.app.flags.DEFINE_integer('num_layers', 1,
                            "number of stacked blstm")
tf.app.flags.DEFINE_integer("input_dims", 6,
                            "input dimensions")
tf.app.flags.DEFINE_integer("num_classes", 100 + 1,
                            "num_labels + 1(blank)")
tf.app.flags.DEFINE_integer('num_steps', 12,
                            "total steps of time")
tf.app.flags.DEFINE_float('learning_rate', 0.0001,
                          "learning rate of RMSPropOptimizer")
tf.app.flags.DEFINE_float('decay_rate', 0.99,
                          "decay rate of RMSPropOptimizer")
tf.app.flags.DEFINE_float('momentum', 0.9,
                          "momentum of RMSPropOptimizer")


class ModelConfig(object):
    """
    testing config
    """

    def __init__(self):
        self.data_dir = FLAGS.data_dir
        self.checkpoints_dir = FLAGS.checkpoints_dir
        self.log_dir = FLAGS.log_dir
        self.batch_size = FLAGS.batch_size
        self.total_epoches = FLAGS.total_epoches
        self.hidden_size = FLAGS.hidden_size
        self.num_layers = FLAGS.num_layers
        self.input_dims = FLAGS.input_dims
        self.num_classes = FLAGS.num_classes
        self.num_steps = FLAGS.num_steps
        self.learning_rate = FLAGS.learning_rate
        self.decay_rate = FLAGS.decay_rate
        self.momentum = FLAGS.momentum

    def show(self):
        print("data_dir:", self.data_dir)
        print("checkpoints_dir:", self.checkpoints_dir)
        print("log_dir:", self.log_dir)
        print("batch_size:", self.batch_size)
        print("total_epoches:", self.total_epoches)
        print("hidden_size:", self.hidden_size)
        print("num_layers:", self.num_layers)
        print("input_dims:", self.input_dims)
        print("num_classes:", self.num_classes)
        print("num_steps:", self.num_steps)
        print("learning_rate:", self.learning_rate)
        print("decay_rate:", self.decay_rate)
        print("momentum:", self.momentum)


def main(_):
    with tf.get_default_graph().as_default() as graph:
        global_steps = tf.train.get_or_create_global_step(graph=graph)

        # TODO read data
        X = np.ones([FLAGS.batch_size, FLAGS.num_steps,
                     FLAGS.input_dims], dtype=np.float32)
        indices = np.array(
            np.ones([FLAGS.batch_size, FLAGS.num_steps], dtype=np.int32), dtype=np.int32)
        values = np.array(
            np.ones(FLAGS.batch_size, dtype=np.int32), dtype=np.int32)
        shape = np.array([FLAGS.batch_size, FLAGS.num_steps], dtype=np.int32)
        Y = tf.SparseTensorValue(indices, values, shape)

        # config setting
        config = ModelConfig()
        config.show()

        # model
        model = model_blstm.HWRModel(config)

        # summary
        # merged_op = tf.summary.merge_all()
        # train_summary_writer = tf.summary.FileWriter(
        #     FLAGS.log_dir + 'train', graph=graph)
        # valid_summary_writer = tf.summary.FileWriter(
        #     FLAGS.log_dir + 'valid', graph=graph)

        init = tf.global_variables_initializer()
        # saver
        # saver = tf.train.Saver()

        # Session
        with tf.Session() as sess:
            sess.run(init)
            model.predict(sess, X)
            # model.step(sess, X, Y, global_step=global_steps)

            # for epoch_steps in range(FLAGS.total_epoches):
            #     if (epoch_steps + 1) % 50 == 0:
            #         # Save the variables to disk.
            #         save_path = saver.save(
            #             sess, FLAGS.checkpoints_dir, global_step=epoch_steps)
            #         print("Model saved in file: %s" % save_path)


if __name__ == "__main__":
    if not os.path.exists(FLAGS.checkpoints_dir):
        os.makedirs(FLAGS.checkpoints_dir)
    tf.app.run()
