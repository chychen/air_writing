from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf
import model_blstm_test


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', '../data/',
                           "data directory")
tf.app.flags.DEFINE_string('checkpoints_dir', '../checkpoints/',
                           "training checkpoints directory")
tf.app.flags.DEFINE_string('log_dir', '../train_log/',
                           "summary directory")
tf.app.flags.DEFINE_string('restore_path', None,
                           "path of saving model eg: ../checkpoints/model.ckpt-5")
tf.app.flags.DEFINE_integer('batch_size', 128,
                            "mini-batch size")
tf.app.flags.DEFINE_integer('total_epoches', 300,
                            "total training epoches")
tf.app.flags.DEFINE_integer('hidden_size', 128,
                            "size of LSTM hidden memory")
tf.app.flags.DEFINE_integer('num_layers', 2,
                            "number of stacked blstm")
tf.app.flags.DEFINE_integer("input_dims", 10,
                            "input dimensions")
tf.app.flags.DEFINE_integer("num_classes", 69,  # 68 letters + 1 blank
                            "num_labels + 1(blank)")
tf.app.flags.DEFINE_integer('save_freq', 250,
                            "frequency of saving model")
tf.app.flags.DEFINE_float('learning_rate', 0.001,
                          "learning rate of RMSPropOptimizer")
tf.app.flags.DEFINE_float('decay_rate', 0.99,
                          "decay rate of RMSPropOptimizer")
tf.app.flags.DEFINE_float('momentum', 0.9,
                          "momentum of RMSPropOptimizer")
tf.app.flags.DEFINE_float('max_length', 1940,
                          "pad to same length")
tf.app.flags.DEFINE_integer('label_pad', 63,
                            "label pad size")
tf.app.flags.DEFINE_boolean('if_valid_vr', False,
                            "label pad size")

letter_table = [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f',
                'g', 'ga', 'h', 'i', 'j', 'k', 'km', 'l', 'm', 'n', 'o', 'p', 'pt', 'q', 'r', 's', 'sc', 'sp', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '<b>']


class ModelConfig(object):
    """
    testing config
    """

    def __init__(self):
        self.data_dir = FLAGS.data_dir
        self.checkpoints_dir = FLAGS.checkpoints_dir
        self.log_dir = FLAGS.log_dir
        self.restore_path = FLAGS.restore_path
        self.batch_size = FLAGS.batch_size
        self.total_epoches = FLAGS.total_epoches
        self.hidden_size = FLAGS.hidden_size
        self.num_layers = FLAGS.num_layers
        self.input_dims = FLAGS.input_dims
        self.num_classes = FLAGS.num_classes
        self.save_freq = FLAGS.save_freq
        self.learning_rate = FLAGS.learning_rate
        self.decay_rate = FLAGS.decay_rate
        self.momentum = FLAGS.momentum
        self.max_length = FLAGS.max_length
        self.label_pad = FLAGS.label_pad
        self.if_valid_vr = FLAGS.if_valid_vr

    def show(self):
        print("data_dir:", self.data_dir)
        print("checkpoints_dir:", self.checkpoints_dir)
        print("log_dir:", self.log_dir)
        print("restore_path:", self.restore_path)
        print("batch_size:", self.batch_size)
        print("total_epoches:", self.total_epoches)
        print("hidden_size:", self.hidden_size)
        print("num_layers:", self.num_layers)
        print("input_dims:", self.input_dims)
        print("num_classes:", self.num_classes)
        print("save_freq:", self.save_freq)
        print("learning_rate:", self.learning_rate)
        print("decay_rate:", self.decay_rate)
        print("momentum:", self.momentum)
        print("max_length:", self.max_length)
        print("label_pad:", self.label_pad)
        print("if_valid_vr:", self.if_valid_vr)


def test_model():
    with tf.get_default_graph().as_default() as graph:
        # config setting
        config = ModelConfig()
        config.show()
        # load data
        input_data = np.load('data.npy')
        target_data = np.load('dense.npy').item()
        label_data = target_data['dense'].astype(np.int32)

        # label_seq_len = target_data['length'].astype(np.int32)
        seq_len_list = []
        for _, v in enumerate(input_data):
            seq_len_list.append(v.shape[0])
        seq_len_list = np.array(seq_len_list).astype(np.int32)
        k = np.argmax(seq_len_list)
        max_length = input_data[k].shape[0]

        # padding each textline to maximum length -> max_length (1940)
        padded_input_data = []
        for _, v in enumerate(input_data):
            residual = max_length - v.shape[0]
            padding_array = np.zeros([residual, FLAGS.input_dims])
            padded_input_data.append(
                np.concatenate([v, padding_array], axis=0))
        padded_input_data = np.array(padded_input_data)

        train_data, valid_data = np.split(
            padded_input_data, [padded_input_data.shape[0] * 9 // 10])
        train_seq_len, valid_seq_len = np.split(
            seq_len_list, [seq_len_list.shape[0] * 9 // 10])
        train_label, valid_label = np.split(
            label_data, [label_data.shape[0] * 9 // 10])

        # number of batches
        train_num_batch = int(train_label.shape[0] / config.batch_size)
        valid_num_batch = int(valid_label.shape[0] / config.batch_size)
        # model
        model = model_blstm_test.HWRModel(config, graph)
        # Add an op to initialize the variables.
        init = tf.global_variables_initializer()
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        # Session
        with tf.Session() as sess:
            sess.run(init)
            # restore model if exist
            if FLAGS.restore_path is not None:
                saver.restore(sess, FLAGS.restore_path)
                print("Model restored:", FLAGS.restore_path)

            v_error_rate_sum = 0.0
            for v_b in range(valid_num_batch):
                v_batch_idx = v_b * config.batch_size
                # input, sequence length, label
                v_input_batch = valid_data[v_batch_idx:v_batch_idx +
                                           config.batch_size]
                v_seq_len_batch = valid_seq_len[v_batch_idx:v_batch_idx +
                                                config.batch_size]
                v_dense_batch = valid_label[v_batch_idx:v_batch_idx +
                                            config.batch_size]
                # error_rate
                v_error_rate = model.error_rate(
                    sess, v_input_batch, v_seq_len_batch, v_dense_batch)
                v_error_rate_sum += v_error_rate
            print("v_error_rate_mean", v_error_rate_sum / valid_num_batch)


def main(_):
    test_model()


if __name__ == "__main__":
    if not os.path.exists(FLAGS.checkpoints_dir):
        os.makedirs(FLAGS.checkpoints_dir)
    tf.app.run()
