#!/usr/bin/env python

import numpy as np
import sys
from sklearn.metrics import accuracy_score,normalized_mutual_info_score

sys.path.append('/mnt/disk1/thanasis/phd/aae/clustering/')
import utils
[x,t] = utils.load_mnist(path='/mnt/disk1/thanasis/phd/aae/datasets/mnist/',dataset="testing")

softmax = np.load(sys.argv[1])

mem = np.argmax(softmax, axis=1)

new_mem = np.zeros((softmax.shape[0]),dtype=np.int8)
conf = np.zeros((softmax.shape[0]),dtype=np.float32)
for i in xrange(softmax.shape[1]):
    pos_yi = np.argmax(softmax[:,i])
    new_lab = t[pos_yi]
    new_mem[np.where(mem==i)[0]] = new_lab
    conf[np.where(mem==i)[0]] = softmax[np.where(mem==i),i]
conf = np.zeros((softmax.shape[0]),dtype=np.float32)
for i in xrange(softmax.shape[0]):
    conf[i] = softmax[i,mem[i]]
error = []
for i in xrange(9):
    if np.where(new_mem==i)[0].size != 0:
        error.append(1-np.average(conf[np.where(new_mem==i)],weights=np.array(np.where(new_mem==i)[0])))
    else:
        error.append(1)
print error
print np.array(error).mean()
# new_mem = np.delete(new_mem, np.where(t.flatten()==2)[0])
# new_ground = np.delete(t.flatten(),np.where(t.flatten()==2)[0]).flatten()
# print new_mem.shape,new_ground.shape

# p = []
# for i in xrange(9):
#     true = np.where(t.flatten()==i)[0]
#     pred = np.where(new_mem==i)[0]
#     p.append(1-len(np.intersect1d(true,pred))/np.float32(len(true)))

# print np.array(p).mean()
print accuracy_score(t.flatten(),new_mem)
print 1-accuracy_score(t.flatten(),new_mem)
# print 1-normalized_mutual_info_score(t.flatten(),mem)
