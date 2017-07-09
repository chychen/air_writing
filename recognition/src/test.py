import numpy as np

data = np.load('VRdataTag100.npy')
label = np.load('VRlabelTag100.npy')

print("data shape:", data.shape)
print("label shape:", label.shape)

for i, _ in enumerate(data):
    print("data:", data[i])
    print("data %d shape %s" % (i, str(data[i].shape)))
    print("label:", label[i])
    print("data %d shape %d" % (i, len(label[i])))
    input()
