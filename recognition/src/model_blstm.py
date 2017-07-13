from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


class HWRModel(object):
    """
    HandWriting Recognition Model
    """

    def __init__(self, config, graph):
        self.data_dir = config.data_dir
        self.checkpoints_dir = config.checkpoints_dir
        self.log_dir = config.log_dir
        self.batch_size = config.batch_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.input_dims = config.input_dims
        self.num_classes = config.num_classes
        self.learning_rate = config.learning_rate
        self.decay_rate = config.decay_rate
        self.momentum = config.momentum
        self.max_length = config.max_length
        self.label_pad = config.label_pad

        self.global_steps = tf.train.get_or_create_global_step(graph=graph)
        self.input_ph = tf.placeholder(dtype=tf.float32, shape=[
            None, None, self.input_dims], name='input_data')
        self.seq_len_ph = tf.placeholder(dtype=tf.int32, shape=[
            None], name='sequence_lenth')
        self.label_ph = tf.placeholder(dtype=tf.int32, shape=[
            self.batch_size, self.label_pad], name='label_data')
        # transform label from dense to sparse form
        # -1 -> sparse slots in dense presentation
        indices = tf.where(tf.not_equal(self.label_ph, -1))
        self.label_sparse = tf.SparseTensor(indices, tf.gather_nd(
            self.label_ph, indices), self.label_ph.shape)

        # inference
        def lstm_cell():
            return rnn.LSTMCell(self.hidden_size, use_peepholes=True, initializer=None,
                                forget_bias=1.0, state_is_tuple=True,
                                activation=tf.tanh, reuse=tf.get_variable_scope().reuse)

        with tf.variable_scope('blstm') as scope:
            # dynamic method
            fwbw, _, _ = rnn.stack_bidirectional_dynamic_rnn(
                cells_fw=[lstm_cell() for _ in range(self.num_layers)],
                cells_bw=[lstm_cell() for _ in range(self.num_layers)],
                inputs=self.input_ph,
                dtype=tf.float32,
                initial_states_fw=None,
                initial_states_bw=None,
                sequence_length=self.seq_len_ph,
                parallel_iterations=None,
                scope=scope
            )
            # merge forward and backward output by weighted combination
            fwbw_rh = tf.reshape(
                fwbw, [-1, self.max_length, 2, self.hidden_size])
            print("stack_bidirectional_dynamic_rnn:", fwbw_rh)
            weightsHidden = tf.Variable(tf.truncated_normal([2, self.hidden_size],
                                                            stddev=0.1))
            biasesHidden = tf.Variable(tf.zeros([self.hidden_size]))
            # transposed to time_major
            fwbw_rh_tp_unst = tf.unstack(fwbw_rh, axis=1)
            print(fwbw_rh_tp_unst[0])
            fb_sum = [tf.reduce_sum(tf.multiply(
                t, weightsHidden), axis=1) + biasesHidden for t in fwbw_rh_tp_unst]
            print(fb_sum[0])
            weightsClasses = tf.Variable(tf.truncated_normal([self.hidden_size,  self.num_classes],
                                                             stddev=0.1))
            biasesClasses = tf.Variable(tf.zeros([self.num_classes]))
            fb_out = [tf.matmul(t, weightsClasses) +
                      biasesClasses for t in fb_sum]
            fb_out_st = tf.stack(fb_out, axis=0)
            print(fb_out_st)

        # time_major
        self.logits_op = fb_out_st
        print(self.logits_op)
        with tf.name_scope('ctc_loss'):
            ctc_loss = tf.nn.ctc_loss(
                labels=self.label_sparse,
                inputs=self.logits_op,
                sequence_length=self.seq_len_ph,
                preprocess_collapse_repeated=False,
                ctc_merge_repeated=True,
                time_major=True
            )
            ctc_loss = tf.reduce_mean(ctc_loss)
            tf.summary.scalar('ctc_loss', ctc_loss)
        self.losses_op = ctc_loss
        print(self.losses_op)

        self.train_op = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.losses_op,
                                         global_step=self.global_steps)

        with tf.name_scope('decoder'):
            decoded, _ = tf.nn.ctc_greedy_decoder(
                inputs=self.logits_op,
                sequence_length=self.seq_len_ph,
                merge_repeated=True)
            print(decoded[0])
        self.decoded_op = decoded

        # levenshtein distance
        self.levenshtein = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                                           self.label_sparse))

        # summary
        self.merged_op = tf.summary.merge_all()
        # summary writer
        self.train_summary_writer = tf.summary.FileWriter(
            self.log_dir + 'train', graph=graph)

    def predict(self, sess, inputs, seq_len, labels=None):
        feed_dict = {self.input_ph: inputs,
                     self.seq_len_ph: seq_len}
        if labels is not None:
            feed_dict[self.label_ph] = labels
            decoded_seq, lev = sess.run(
                [self.decoded_op[0], self.levenshtein], feed_dict=feed_dict)
            return decoded_seq, lev
        else:
            decoded_seq = sess.run(
                self.decoded_op[0], feed_dict=feed_dict)
            return decoded_seq

    def step(self, sess, inputs, seq_len, labels):
        feed_dict = {self.input_ph: inputs,
                     self.seq_len_ph: seq_len,
                     self.label_ph: labels}
        gloebal_step, summary, _, losses = sess.run(
            [self.global_steps, self.merged_op, self.train_op, self.losses_op], feed_dict=feed_dict)
        # summary
        self.train_summary_writer.add_summary(
            summary, global_step=gloebal_step)
        return gloebal_step, losses

    def compute_losses(self, sess, inputs, seq_len, labels):
        feed_dict = {self.input_ph: inputs,
                     self.seq_len_ph: seq_len,
                     self.label_ph: labels}
        losses = sess.run(
            self.losses_op, feed_dict=feed_dict)
        return losses


class TestingConfig(object):
    """
    testing config
    """

    def __init__(self):
        self.data_dir = '../data/'
        self.checkpoints_dir = '../checkpoints/'
        self.log_dir = '../model_log/'
        self.batch_size = 128
        self.total_epoches = 50
        self.hidden_size = 10
        self.num_layers = 2
        self.input_dims = 10
        self.num_classes = 64
        self.learning_rate = 1e-4
        self.decay_rate = 0
        self.momentum = 0


def test_model():
    # testing runability
    with tf.get_default_graph().as_default() as graph:
        config = TestingConfig()
        X = np.ones([config.batch_size, 1940,
                     config.input_dims], dtype=np.float32)
        Y = np.random.randint(3, size=[config.batch_size, 64], dtype=np.int32)
        seq_len = [64 for _ in range(config.batch_size)]
        model = HWRModel(config, graph)
        init = tf.global_variables_initializer()
        # Session
        with tf.Session() as sess:
            sess.run(init)
            for _ in range(config.total_epoches):
                losses = model.step(sess, X, seq_len, Y)
                print(losses)


if __name__ == "__main__":
    test_model()
