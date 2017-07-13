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
tf.app.flags.DEFINE_integer('save_freq', 5,
                            "ephoches of frequency of saving model")
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
tf.app.flags.DEFINE_boolean('if_lowercase_only', False,
                            "if letter table only contain lowercase")

if FLAGS.if_lowercase_only:
    letter_table = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'ga', 'h', 'i', 'j', 'k', 'km', 'l', 'm', 'n', 'o', 'p', 'pt',
                    'q', 'r', 's', 'sc', 'sp', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '<b>']
else:
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


def train_model():
    with tf.get_default_graph().as_default() as graph:
        # config setting
        config = ModelConfig()
        config.show()
        # load data
        input_data = np.load(FLAGS.data_dir + 'data.npy')
        target_data = np.load(FLAGS.data_dir + 'dense.npy').item()
        label_data = target_data['dense'].astype(np.int32)
        label_data_length = target_data['length'].astype(np.int32)

        # label_seq_len = target_data['length'].astype(np.int32)
        seq_len_list = []
        for _, v in enumerate(input_data):
            seq_len_list.append(v.shape[0])
        seq_len_list = np.array(seq_len_list).astype(np.int32)
        k = np.argmax(seq_len_list)
        max_length = input_data[k].shape[0]

        if FLAGS.if_valid_vr:
            vr_valid_data_raw = np.load('VRdataValidation.npy')[
                :FLAGS.batch_size]
            vr_valid_target = np.load('VRdenseValidation.npy').item()
            vr_valid_label_raw = vr_valid_target['dense'].astype(np.int32)[
                :FLAGS.batch_size]
            vr_seq_len_list = []
            for _, v in enumerate(vr_valid_data_raw):
                vr_seq_len_list.append(v.shape[0])
            vr_seq_len_list = np.array(vr_seq_len_list).astype(np.int32)
            # padding each textline to maximum length -> max_length (1940)
            vr_valid_data = []
            for _, v in enumerate(vr_valid_data_raw):
                residual = max_length - v.shape[0]
                padding_array = np.zeros([residual, FLAGS.input_dims])
                vr_valid_data.append(
                    np.concatenate([v, padding_array], axis=0))
            vr_valid_data = np.array(vr_valid_data)
            vr_valid_label = []
            # padding each label to same dense length -> label_pad (64)
            for _, v in enumerate(vr_valid_label_raw):
                residual = FLAGS.label_pad - v.shape[0]
                padding_array = np.zeros([residual]) - 1
                vr_valid_label.append(
                    np.concatenate([v, padding_array], axis=0))
            vr_valid_label = np.array(vr_valid_label).astype(np.int32)
            print("vr_valid_data.shape", vr_valid_data.shape)
            print("vr_valid_label.shape", vr_valid_label.shape)

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
        train_label_length, valid_label_length = np.split(
            label_data_length, [label_data_length.shape[0] * 9 // 10])

        # number of batches
        train_num_batch = int(train_label.shape[0] / config.batch_size)
        valid_num_batch = int(valid_label.shape[0] / config.batch_size)
        # model
        model = model_blstm.HWRModel(config, graph)
        # Add an op to initialize the variables.
        init = tf.global_variables_initializer()
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        train_summary_writer = tf.summary.FileWriter(
            FLAGS.log_dir + 'ephoch_train', graph=graph)
        valid_summary_writer = tf.summary.FileWriter(
            FLAGS.log_dir + 'ephoch_valid', graph=graph)
        vr_valid_summary_writer = tf.summary.FileWriter(
            FLAGS.log_dir + 'ephoch_vr_valid', graph=graph)
        # Session
        with tf.Session() as sess:
            sess.run(init)
            # restore model if exist
            if FLAGS.restore_path is not None:
                saver.restore(sess, FLAGS.restore_path)
                print("Model restored:", FLAGS.restore_path)
            # time cost evaluation
            start_time = time.time()
            end_time = 0.0
            for _ in range(config.total_epoches):
                # Shuffle the data
                shuffled_indexes = np.random.permutation(train_data.shape[0])
                train_data = train_data[shuffled_indexes]
                train_seq_len = train_seq_len[shuffled_indexes]
                train_label = train_label[shuffled_indexes]
                loss_sum = 0.0
                for b in range(train_num_batch):
                    batch_idx = b * config.batch_size
                    # input, sequence length, label
                    input_batch = train_data[batch_idx:batch_idx +
                                             config.batch_size]
                    seq_len_batch = train_seq_len[batch_idx:batch_idx +
                                                  config.batch_size]
                    dense_batch = train_label[batch_idx:batch_idx +
                                              config.batch_size]
                    # train
                    global_step, losses = model.step(sess, input_batch,
                                                     seq_len_batch, dense_batch)
                    loss_sum += losses
                global_ephoch = int(global_step // train_num_batch)
                # logging per ephoch
                # validation
                v_loss_sum = 0.0
                for v_b in range(valid_num_batch):
                    v_batch_idx = v_b * config.batch_size
                    # input, sequence length, label
                    v_input_batch = valid_data[v_batch_idx:v_batch_idx +
                                               config.batch_size]
                    v_seq_len_batch = valid_seq_len[v_batch_idx:v_batch_idx +
                                                    config.batch_size]
                    v_dense_batch = valid_label[v_batch_idx:v_batch_idx +
                                                config.batch_size]
                    v_losses = model.compute_losses(sess, v_input_batch,
                                                    v_seq_len_batch, v_dense_batch)
                    v_loss_sum += v_losses

                # predict result
                v_batch_idx = global_ephoch % valid_num_batch
                v_input_batch = valid_data[v_batch_idx:v_batch_idx +
                                           config.batch_size]
                v_seq_len_batch = valid_seq_len[v_batch_idx:v_batch_idx +
                                                config.batch_size]
                v_dense_batch = valid_label[v_batch_idx:v_batch_idx +
                                            config.batch_size]
                v_label_length_batch = valid_label_length[v_batch_idx:v_batch_idx +
                                                          config.batch_size]
                #visualize first data in validation batch
                visual_target_length = v_label_length_batch[0]
                predict, levenshtein = model.predict(
                    sess, v_input_batch, v_seq_len_batch, v_dense_batch)
                str_decoded = ''.join(
                    [letter_table[x] for x in np.asarray(predict.values[:visual_target_length])])
                val_original = ''.join(
                    [letter_table[x] for x in v_dense_batch[0]])
                end_time = time.time()
                print('Original val: %s' % val_original)
                print('Decoded  val: %s' % str_decoded)
                print("%d epoches, %d steps, mean loss: %f, valid mean loss: %f, time cost: %f(sec/batch), levenshtein: %f" %
                      (global_ephoch,
                       global_step,
                       loss_sum / train_num_batch,
                       v_loss_sum / valid_num_batch,
                       (end_time - start_time) / train_num_batch,
                       levenshtein))
                start_time = end_time
                train_summary = tf.Summary(value=[tf.Summary.Value(
                    tag="ephoch_mean_loss", simple_value=loss_sum / train_num_batch)])
                valid_summary = tf.Summary(value=[tf.Summary.Value(
                    tag="ephoch_mean_loss", simple_value=v_loss_sum / valid_num_batch)])
                train_summary_writer.add_summary(
                    train_summary, global_step=global_ephoch)
                valid_summary_writer.add_summary(
                    valid_summary, global_step=global_ephoch)
                train_summary_writer.flush()
                valid_summary_writer.flush()

                # VR validation
                if FLAGS.if_valid_vr:
                    vr_v_loss_sum = model.compute_losses(sess, vr_valid_data,
                                                         vr_seq_len_list, vr_valid_label)
                    # predict vr result
                    vr_idx = global_step % FLAGS.batch_size
                    vr_predict, vr_levenshtein = model.predict(
                        sess, vr_valid_data[vr_idx:vr_idx + 1], vr_seq_len_list[vr_idx:vr_idx + 1], vr_valid_label[vr_idx:vr_idx + 1])
                    vr_str_decoded = ''.join(
                        [letter_table[x] for x in np.asarray(vr_predict.values)])
                    vr_val_original = ''.join(
                        [letter_table[x] for x in vr_valid_label[vr_idx]])
                    print('Original vr_val: %s' % vr_val_original)
                    print('Decoded  vr_val: %s' % vr_str_decoded)

                    vr_valid_summary = tf.Summary(value=[tf.Summary.Value(
                        tag="ephoch_mean_loss", simple_value=vr_v_loss_sum)])
                    vr_valid_summary_writer.add_summary(
                        vr_valid_summary, global_step=global_ephoch)
                    vr_valid_summary_writer.flush()

                    print("VR valid mean loss: %f, vr_levenshtein: %f" %
                          (vr_v_loss_sum, vr_levenshtein))

                if (global_ephoch % FLAGS.save_freq) == 0:
                    save_path = saver.save(
                        sess, FLAGS.checkpoints_dir + "model.ckpt",
                        global_step=global_step)
                    print("Model saved in file: %s" % save_path)


def main(_):
    train_model()


if __name__ == "__main__":
    if not os.path.exists(FLAGS.checkpoints_dir):
        os.makedirs(FLAGS.checkpoints_dir)
    tf.app.run()
