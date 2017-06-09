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

    def __init__(self, config):
        self.batch_size = config.batch_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.input_dims = config.input_dims
        self.num_classes = config.num_classes
        self.num_steps = config.num_steps
        self.learning_rate = config.learning_rate
        self.decay_rate = config.decay_rate
        self.momentum = config.momentum

        # trajectory = [x, y, islifted, speed, time, direction]
        # input = [batch_size, time, trajectory]
        self.input_ph = tf.placeholder(dtype=tf.float32, shape=[
            None, self.num_steps, self.input_dims], name='input_data')
        self.label_ph = tf.sparse_placeholder(
            dtype=tf.int32, name='label_data')

        # inference
        def lstm_cell():
            return rnn.LSTMCell(self.hidden_size, use_peepholes=True, initializer=None,
                                forget_bias=1.0, state_is_tuple=True,
                                activation=tf.tanh, reuse=tf.get_variable_scope().reuse)

        with tf.variable_scope('blstm') as scope:
            # dynamic method
            outputs, _, _ = rnn.stack_bidirectional_dynamic_rnn(
                cells_fw=[lstm_cell() for _ in range(self.num_layers)],
                cells_bw=[lstm_cell() for _ in range(self.num_layers)],
                inputs=self.input_ph,
                dtype=tf.float32,
                initial_states_fw=None,
                initial_states_bw=None,
                sequence_length=[self.num_steps for _ in range(self.batch_size)],
                parallel_iterations=None,
                scope=scope
            )
            print ("stack_bidirectional_dynamic_rnn:", outputs)
        self.inference_op = outputs

        print(self.label_ph)
        print(self.inference_op)
        with tf.name_scope('ctc_loss'):
            ctc_loss = tf.nn.ctc_loss(
                labels=self.label_ph,
                inputs=self.inference_op,
                sequence_length=[self.num_steps for _ in range(self.batch_size)],
                preprocess_collapse_repeated=False,
                ctc_merge_repeated=True,
                time_major=False
            )
            ctc_loss = tf.reduce_mean(ctc_loss)
            tf.summary.scalar('ctc_loss', ctc_loss)
        self.losses_op = ctc_loss
        print(self.losses_op)

        self.train_op = tf.train.RMSPropOptimizer(
            self.learning_rate, self.decay_rate, self.momentum,
            1e-10).minimize(self.losses_op, global_step=None)

    def predict(self, sess, inputs):
        # TODO: decoder with tf.nn.ctc_beam_search_decoder
        # URL:
        # https://www.tensorflow.org/api_docs/python/tf/nn/ctc_beam_search_decoder
        feed_dict = {self.input_ph: inputs}
        return sess.run(self.inference_op, feed_dict=feed_dict)

    def step(self, sess, inputs, labels, global_step=None):
        feed_dict = {self.input_ph: inputs, self.label_ph: labels}
        _, losses = sess.run(
            [self.train_op, self.losses_op], feed_dict=feed_dict)
        return losses


class TestingConfig(object):
    """
    testing config
    """

    def __init__(self):
        self.data_dir = 'data/'
        self.checkpoints_dir = 'checkpoints/'
        self.log_dir = 'log/'
        self.batch_size = 25
        self.total_epoches = 100
        self.hidden_size = 10
        self.num_layers = 2
        self.input_dims = 15
        self.num_classes = 20
        self.num_steps = 5
        self.learning_rate = 1e-4
        self.decay_rate = 0
        self.momentum = 0


def test_model():
    with tf.get_default_graph().as_default() as graph:
        global_steps = tf.train.get_or_create_global_step(graph=graph)

        config = TestingConfig()

        X = np.ones([config.batch_size, config.num_steps,
                     config.input_dims], dtype=np.float32)
        indices = np.array([[n, 1] for n in range(config.batch_size)], dtype=np.int64)
        values = np.array([1 for _ in range(config.batch_size)], dtype=np.int32)
        shape = np.array([config.batch_size, config.num_classes], dtype=np.int64)
        Y = tf.SparseTensorValue(indices, values, shape)

        model = HWRModel(config)

        init = tf.global_variables_initializer()
        # Session
        with tf.Session() as sess:
            sess.run(init)
            for i in range(config.total_epoches):
                logits = model.predict(sess, X)
                losses = model.step(sess, X, Y)
                print(losses)

if __name__ == "__main__":
    test_model()