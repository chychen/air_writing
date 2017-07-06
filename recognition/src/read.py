import numpy as np

f = open('../data/letters.txt', 'r')

l = []

l.append(" ")
for row in f:
    print(l.append(row.strip('\n')))

f.close()
s = []
k = np.load('label.npy')
themax = 0
for row in k:
    print(row)
    gg = row.strip('\n')
    s.append(gg)
    for aaa in gg:
        if aaa in ['[', ']']:
            input()
    if len(gg) > themax:
        print(len(gg))
        themax = len(gg)
        # print(gg.strip('.'))


l += ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
for sentence in s:
    for char in sentence:
        if char not in l:
            print(sentence, "  ", char)

dense = np.zeros((len(s), 64),dtype=np.int32)
dense += -1
length = np.zeros(len(s))
print(l)
for idl, sentence in enumerate(s):
    sentence_iter = iter(sentence)
    length[idl] = len(sentence)
    for idx, char in enumerate(sentence_iter):
        if char == 'g' and sentence[idx:idx + 2] == 'ga':
            dense[idl, idx] = l.index('ga')
            
            next(sentence_iter)

        elif char == 'k' and sentence[idx:idx + 2] == 'km':
            dense[idl, idx] = l.index('km')
            next(sentence_iter)

        elif char == 'p' and sentence[idx:idx + 2] == 'pt':
            dense[idl, idx] = l.index('pt')
            next(sentence_iter)

        elif char == 's' and sentence[idx:idx + 2] == 'sc':
            dense[idl, idx] = l.index('sc')
            next(sentence_iter)

        elif char == 's' and sentence[idx:idx + 2] == 'sp':
            dense[idl, idx] = l.index('sp')
            next(sentence_iter)

        elif char in l:
            dense[idl, idx] = l.index(char)

        else:
            continue

print(dense)
alldata = {}

alldata['dense'] = dense
alldata['length'] = length
np.save("dense.npy", alldata)
# print(row.strip('\n'))

# import tensorflow as tf
# a_t = tf.constant(dense)
# idx = tf.where(tf.not_equal(a_t, 0))
# sparse = tf.SparseTensor(idx, tf.gather_nd(a_t,idx), a_t.get_shape())


"""
a = np.reshape(np.arange(24), (3, 4, 2))
with tf.Session() as sess:
    a_t = tf.constant(a)
    idx = tf.where(tf.not_equal(a_t, 0))
    # Use tf.shape(a_t, out_type=tf.int64) instead of a_t.get_shape() if tensor shape is dynamic
    sparse = tf.SparseTensor(idx, tf.gather_nd(a_t, idx), a_t.get_shape())
    dense = tf.sparse_tensor_to_dense(sparse)
    b = sess.run(dense)
np.all(a == b)
"""
