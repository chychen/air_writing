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

        self.global_steps = tf.train.get_or_create_global_step(graph=graph)
        self.input_ph = tf.placeholder(dtype=tf.float32, shape=[
            None, None, self.input_dims], name='input_data')
        self.seq_len_ph = tf.placeholder(dtype=tf.int32, shape=[
            None], name='sequence_lenth')
        self.label_ph = tf.placeholder(dtype=tf.int32, shape=[
            self.batch_size, 64], name='label_data')
        indices = tf.where(tf.not_equal(self.label_ph, -1)) # -1 -> sparse value in dense presentation
        self.label_sparse = tf.SparseTensor(indices, tf.gather_nd(
            self.label_ph, indices), self.label_ph.shape)
        # self.label_seq_len_ph = tf.placeholder(dtype=tf.int32, shape=[
        #     None], name='label_sequence_lenth')

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
            # transposed to time_major
            fwbw_rh = tf.reshape(
                fwbw, [-1, 1940, 2, self.hidden_size])
            # fwbw_rh_tp = tf.transpose(fwbw_rh, perm=[1, 0, 2, 3])
            fwbw_rh_tp = fwbw_rh
            print("stack_bidirectional_dynamic_rnn:", fwbw_rh_tp)
            weightsHidden = tf.Variable(tf.truncated_normal([2, self.hidden_size],
                                                            stddev=0.1))
            biasesHidden = tf.Variable(tf.zeros([self.hidden_size]))
            fwbw_rh_tp_unst = tf.unstack(fwbw_rh_tp, axis=1)
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
        print(self.label_sparse)
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
        # with tf.name_scope('optimizer'):
        #     opt = tf.train.RMSPropOptimizer(
        #         self.learning_rate, self.decay_rate, self.momentum, 1e-10)
        #     params = tf.trainable_variables()
        #     grads_and_vars = opt.compute_gradients(self.losses_op, params)
        #     clipped_grads_and_vars = [
        #         (tf.clip_by_norm(gv[0], 1.0), gv[1]) for gv in grads_and_vars]
        # self.train_op = opt.apply_gradients(
        #     clipped_grads_and_vars, global_step=self.global_steps)

        with tf.name_scope('decoder'):
            # decoded, _ = tf.nn.ctc_beam_search_decoder(
            #     inputs=self.logits_op,
            #     sequence_length=self.seq_len_ph,
            #     beam_width=100,
            #     top_paths=1,
            #     merge_repeated=True)
            decoded, _ = tf.nn.ctc_greedy_decoder(
                inputs=self.logits_op,
                sequence_length=self.seq_len_ph,
                merge_repeated=False)
            print(decoded[0])
        self.decoded_op = decoded

        #######################################################################
        #######################################################################
        #######################################################################

        # # Defining the cell
        # # Can be:
        # #   tf.nn.rnn_cell.RNNCell
        # #   tf.nn.rnn_cell.GRUCell
        # def lstm_cell():
        #     return tf.contrib.rnn.LSTMCell(self.hidden_size, state_is_tuple=True)

        # # Stacking rnn cells
        # stack = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(self.num_layers)],
        #                                     state_is_tuple=True)

        # # The second output is the last state and we will no use that
        # outputs, _ = tf.nn.dynamic_rnn(
        #     stack, self.input_ph, self.seq_len_ph, dtype=tf.float32)

        # shape = tf.shape(self.input_ph)
        # batch_s, max_time_steps = shape[0], shape[1]

        # # Reshaping to apply the same weights over the timesteps
        # outputs = tf.reshape(outputs, [-1, self.hidden_size])

        # # Truncated normal with mean 0 and stdev=0.1
        # # Tip: Try another initialization
        # # see
        # # https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
        # W = tf.Variable(tf.truncated_normal([self.hidden_size,
        #                                      self.num_classes],
        #                                     stddev=0.1))
        # # Zero initialization
        # # Tip: Is tf.zeros_initializer the same?
        # b = tf.Variable(tf.constant(0., shape=[self.num_classes]))

        # # Doing the affine projection
        # logits = tf.matmul(outputs, W) + b

        # # Reshaping back to the original shape
        # logits = tf.reshape(logits, [batch_s, -1, self.num_classes])

        # # Time major
        # logits = tf.transpose(logits, (1, 0, 2))

        # loss = tf.nn.ctc_loss(self.label_sparse, logits, self.seq_len_ph)
        # self.losses_op = tf.reduce_mean(loss)
        # tf.summary.scalar('ctc_loss', self.losses_op)

        # # optimizer = tf.train.AdamOptimizer().minimize(cost)
        # # optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9).minimize(cost)
        # self.train_op = tf.train.MomentumOptimizer(
        # learning_rate=0.005, momentum=0.9).minimize(self.losses_op,
        # global_step=self.global_steps)

        # # Option 2: tf.contrib.ctc.ctc_beam_search_decoder
        # # (it's slower but you'll get better results)
        # self.decoded_op, _ = tf.nn.ctc_greedy_decoder(logits, self.seq_len_ph, merge_repeated=False)

        #######################################################################
        #######################################################################
        #######################################################################

        # # TODO Inaccuracy: label error rate
        # ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
        #                                       targets))

        # summary
        self.merged_op = tf.summary.merge_all()
        # summary writer
        self.train_summary_writer = tf.summary.FileWriter(
            self.log_dir + 'train', graph=graph)

    def predict(self, sess, inputs, seq_len):
        feed_dict = {self.input_ph: inputs,
                     self.seq_len_ph: seq_len}
        return sess.run(self.decoded_op[0], feed_dict=feed_dict)

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

    def getSparse(self, sess, labels):
        feed_dict = {self.label_ph: labels}
        return sess.run(self.label_sparse, feed_dict=feed_dict)


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
        self.input_dims = 3
        self.num_classes = 64
        self.learning_rate = 1e-4
        self.decay_rate = 0
        self.momentum = 0


def test_model():
    with tf.get_default_graph().as_default() as graph:
        config = TestingConfig()

        X = np.ones([config.batch_size, None,
                     config.input_dims], dtype=np.float32)

        Y = np.random.randint(3, size=[config.batch_size, 64], dtype=np.int32)

        seq_len = [64 for _ in range(config.batch_size)]

        model = HWRModel(config, graph)

        init = tf.global_variables_initializer()
        # Session
        with tf.Session() as sess:
            sess.run(init)
            for _ in range(config.total_epoches):
                # logits = model.predict(sess, X, seq_len)
                losses = model.step(sess, X, seq_len, Y)
                print(losses)


if __name__ == "__main__":
    test_model()
