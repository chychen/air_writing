from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
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


def test_model():
    with tf.get_default_graph().as_default() as graph:
        # config setting
        config = ModelConfig()
        config.show()
        # load data
        input_data = np.load('VRdataAll.npy')
        label_data = np.load('VRlabelAll.npy')

        # label_seq_len = target_data['length'].astype(np.int32)
        seq_len_list = []
        for _, v in enumerate(input_data):
            seq_len_list.append(v.shape[0])
        seq_len_list = np.array(seq_len_list).astype(np.int32)
        # k = np.argmax(seq_len_list)
        # max_length = input_data[k].shape[0]

        # padding each textline to maximum length -> max_length (1940)
        padded_input_data = []
        for _, v in enumerate(input_data):
            residual = FLAGS.max_length - v.shape[0]
            padding_array = np.zeros([residual, FLAGS.input_dims])
            padded_input_data.append(
                np.concatenate([v, padding_array], axis=0))
        padded_input_data = np.array(padded_input_data)

        # number of batches
        num_batch = int(padded_input_data.shape[0] / config.batch_size)
        # model
        model = model_blstm.HWRModel(config, graph)
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
            # predict result
            for i, _ in enumerate(padded_input_data):
                start_time = time.time()
                predict = model.predict(
                    sess, padded_input_data[i:i+1], seq_len_list[i:i+1])
                str_decoded = ''.join(
                    [letter_table[x] for x in np.asarray(predict.values)])
                end_time = time.time()
                print('Original val: %s' % label_data[i])
                print('Decoded  val: %s' % str_decoded)
                print('Time Cost: %f' % (end_time-start_time))
                input()

            # loss_sum = 0.0
            # for b in range(num_batch):
            #     batch_idx = b * config.batch_size
            #     # input, sequence length, label
            #     input_batch = padded_input_data[batch_idx:batch_idx +
            #                                     config.batch_size]
            #     seq_len_batch = seq_len_list[batch_idx:batch_idx +
            #                                  config.batch_size]
            #     dense_batch = label_data[batch_idx:batch_idx +
            #                              config.batch_size]
            #     losses = model.compute_losses(sess, input_batch,
            #                                   seq_len_batch, dense_batch)
            #     loss_sum += losses

            # # predict result
            # predict, levenshtein = model.predict(
            #     sess, input_batch[0:1], seq_len_batch[0:1], dense_batch[0:1])
            # str_decoded = ''.join(
            #     [letter_table[x] for x in np.asarray(predict.values)])
            # val_original = ''.join(
            #     [letter_table[x] for x in dense_batch[0]])
            # print('Original val: %s' % val_original)
            # print('Decoded  val: %s' % str_decoded)
            # print("mean loss: %f, levenshtein: %f" %
            #       (loss_sum / num_batch,
            #        levenshtein))

def main(_):
    test_model()


if __name__ == "__main__":
    if not os.path.exists(FLAGS.checkpoints_dir):
        os.makedirs(FLAGS.checkpoints_dir)
    tf.app.run()
