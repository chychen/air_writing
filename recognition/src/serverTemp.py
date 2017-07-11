from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import socket
import time
import json
import numpy as np
import tensorflow as tf
import model_blstm
import atexit
import sys
import select
from tagProcess import transferS
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
tf.app.flags.DEFINE_integer('num_layers', 1,
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

letter_table = [' ', 'a', 'b', 'c', 'd', 'e', 'f',
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


server_address = ('140.113.210.18', 2001)
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

sock.bind(server_address)
def whenexit():
    connection.close()
    print("Exit and close connection")

sock.listen(1)


# def recvall(sock):
#     BUFF_SIZE = 1  # 4 KiB
#     data = ""
#     while True:
#         part = sock.recv(1024)
#         print("One time")
#         print("Size ", sys.getsizeof(part))
#         if sys.getsizeof(part) < BUFF_SIZE:
#             data += str(part)
#             print("go here ")
#             # either 0 or end of data
#             break
#         data += str(part)
#     print("See!")
#     print(data)
#     return data

# def recvall(sock):
#     buf = sock.recv(4096)
#     while buf:
#         yield buf
#         print("one time ",time.time())
#         buf = sock.recv(4096)

def recvall(sock):
    framents = []
    while True:
        buf = sock.recv(1024)
        sentence = buf.decode('ascii')
        # print(sentence)
        mark = sentence[-3:]
        # print("mark ",mark)
        if sentence[-3:] == 'bye':
            framents.append(sentence[:-3])
            break
            # sock.sendall("good".encode('utf-8'))
        else:
            framents.append(sentence)
    return framents

# def recvall(sock):
#     data = sock.recv(5000000000)
#     return str(data)

with tf.get_default_graph().as_default() as graph:
    # config = ModelConfig()
    # config.show()
    # model = model_blstm.HWRModel(config, graph)
    # init = tf.global_variables_initializer()
    # saver = tf.train.Saver()
    with tf.Session() as sess:
        # sess.run(init)
        # saver.restore(sess, FLAGS.restore_path)

        print("restore")
        # exit()
        while True:
            connection, clientAddr = sock.accept()
            atexit.register(whenexit)
            try:
                print('connection ', clientAddr)
                while True:
                    
                    data = ''.join(recvall(connection))
                    # print(type(data))
                    # print(data[:30])
                    json_data = json.loads(data)
                    input_data = transferS(json_data)
                    # time.sleep(5)
                    # connection.sendall("Done".encode('utf-8'))
                    # connection.close()
                    # break
                    # process data

                    #################
                    seq_len_list = []
                    for _, v in enumerate(input_data):
                        seq_len_list.append(v.shape[0])
                    seq_len_list = np.array(seq_len_list).astype(np.int32)
                    padded_input_data = []
                    for _, v in enumerate(input_data):
                        residual = FLAGS.max_length - v.shape[0]
                        padding_array = np.zeros([residual, FLAGS.input_dims])
                        padded_input_data.append(
                            np.concatenate([v, padding_array], axis=0))
                    padded_input_data = np.array(padded_input_data)
                    num_batch = int(
                        padded_input_data.shape[0] / config.batch_size)
                    for i, _ in enumerate(padded_input_data):
                        start_time = time.time()
                        predict = model.predict(
                            sess, padded_input_data[i:i + 1], seq_len_list[i:i + 1])
                        str_decoded = ''.join(
                            [letter_table[x] for x in np.asarray(predict.values)])
                        end_time = time.time()
                        # print('Original val: %s' % label_data[i])
                        print('Decoded  val: %s' % str_decoded)
                        print('Time Cost: %f' % (end_time - start_time))
                        connection.sendall(str_decoded.encode('utf-8'))
                        connection.close()
                        print("connection close")
                    # print("ok?")
                    # print(type(data))

            finally:
                print("connection close")
                connection.close()
