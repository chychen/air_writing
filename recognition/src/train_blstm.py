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
tf.app.flags.DEFINE_integer('batch_size', 128,
                            "mini-batch size")
tf.app.flags.DEFINE_integer('total_epoches', 100,
                            "total training epoches")
tf.app.flags.DEFINE_integer('hidden_size', 100,
                            "size of LSTM hidden memory")
tf.app.flags.DEFINE_integer('num_layers', 1,
                            "number of stacked blstm")
tf.app.flags.DEFINE_integer("input_dims", 3,
                            "input dimensions")
tf.app.flags.DEFINE_integer("num_classes", 100 + 1,
                            "num_labels + 1(blank)")
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
        self.num_steps = None
        self.learning_rate = FLAGS.learning_rate
        self.decay_rate = FLAGS.decay_rate
        self.momentum = FLAGS.momentum

    def set_num_steps(self, num_steps):
        self.num_steps = num_steps

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
        # print("num_steps:", self.num_steps)
        print("learning_rate:", self.learning_rate)
        print("decay_rate:", self.decay_rate)
        print("momentum:", self.momentum)


def train_model():
    with tf.get_default_graph().as_default() as graph:
        # global_steps = tf.train.get_or_create_global_step(graph=graph)

        # config setting
        config = ModelConfig()
        config.show()

        # load data
        # [textline_id, length, 3], 3->(x', y', time)
        input_data = np.load('data.npy')
        label_data = np.load('dense.npy')
        seq_len_list = []
        for _, v in enumerate(input_data):
            seq_len_list.append(v.shape[0])
        seq_len_list = np.array(seq_len_list)
        k = np.argmax(seq_len_list)
        max_length = input_data[k].shape[0]

        # padding each textline to maximum length -> max_length
        padded_input_data = []
        for _, v in enumerate(input_data):
            residual = max_length - v.shape[0]
            padding_array = np.zeros([residual, 3])
            padded_input_data.append(
                np.concatenate([v, padding_array], axis=0))
        padded_input_data = np.array(padded_input_data)
        # print(input_data.shape)
        # print(input_data.dtype)
        # print(padded_input_data.shape)
        # print(padded_input_data.dtype)

        num_line = label_data.shape[0]
        num_batch = int(num_line / config.batch_size)

        model = model_blstm.HWRModel(config)

        init = tf.global_variables_initializer()
        # Session
        with tf.Session() as sess:
            sess.run(init)
            for i in range(config.total_epoches):
                for j in range(num_batch):
                    batch_idx = j * config.batch_size
                    # input
                    input_batch = padded_input_data[batch_idx:batch_idx +
                                                    config.batch_size]
                    # sequence length
                    seq_len_batch = seq_len_list[batch_idx:batch_idx +
                                                 config.batch_size]
                    # label
                    dense_batch = label_data[batch_idx:batch_idx +
                                             config.batch_size]
                    indices = tf.where(tf.not_equal(dense_batch, 0))
                    Y = tf.SparseTensor(indices, tf.gather_nd(
                        dense_batch, indices), dense_batch.shape)

                    # logits = model.predict(sess, input_batch, seq_len_batch)
                    losses = model.step(sess, input_batch, seq_len_batch, Y)
                    print(losses)


def main(_):
    train_model()


if __name__ == "__main__":
    if not os.path.exists(FLAGS.checkpoints_dir):
        os.makedirs(FLAGS.checkpoints_dir)
    tf.app.run()
