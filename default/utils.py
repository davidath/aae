###############################################################################
# Description
###############################################################################


import cPickle
import gzip
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import ConfigParser
import sys
from datetime import datetime
import os
import struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros

MNIST_PATH = '../datasets/mnist'


def save(filename, *objects):
    # Save object and compress it at the same time (it can be used for multiple
    # objects or a single object)
    fil = gzip.open(filename, 'wb')
    for obj in objects:
        cPickle.dump(obj, fil, protocol=cPickle.HIGHEST_PROTOCOL)
    fil.close()


def load(filename):
    # Load compressed object as python generator
    fil = gzip.open(filename, 'rb')
    while True:
        try:
            yield cPickle.load(fil)
        except EOFError:
            break
    fil.close()


def load_single(filename):
    # Load compressed object as object
    fil = gzip.open(filename, 'rb')
    c = cPickle.load(fil)
    fil.close()
    return c


def plot_image(image, x, y):
    # Plot single image
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    pixels = image.reshape((x, y))
    ax.matshow(pixels, cmap=matplotlib.cm.binary)
    plt.plot()
    plt.show()


def plot_border(img):
    newimg = np.pad(img,((1,1),(1,1)),'constant',constant_values=(0))
    max_x = newimg.shape[0]-1
    max_y = newimg.shape[1]-1
    newimg[0,:] = np.ones(shape=(newimg[0,:].shape)) * np.max(newimg)
    newimg[max_x,:] = np.ones(shape=(newimg[max_x,:].shape)) * np.max(newimg)
    newimg[:,0] = np.ones(shape=(newimg[:,0].shape)) * np.max(newimg)
    newimg[:,max_y] = np.ones(shape=(newimg[:,max_y].shape)) * np.max(newimg)
    return newimg

def plot_row(images, spacing, CMAP=None):
    images = np.asarray([plot_border(image) for image in images])
    x = images.shape[1]
    y = images.shape[2]
    canvas_y = y*len(images)+spacing*(len(images)-1)
    canvas = np.zeros(shape=(x,canvas_y))
    print canvas.shape
    idx = range(0,canvas.shape[1],y+spacing)
    for pos,i in enumerate(idx):
        canvas[:,i:i+y] += images[pos]
    if CMAP:
      plt.matshow(canvas, cmap=CMAP)
    else:
      plt.matshow(canvas)
    plt.axis('off')
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.show()

def plot_grid(images, gridx, gridy, xspace, yspace, CMAP=plt.cm.binary, row_space=None, out=None,border=False):
    if border:
        images = np.asarray([plot_border(image) for image in images])
    x = images.shape[1]
    y = images.shape[2]
    canvas_x = x*gridx+xspace*(gridx-1)
    canvas_y = y*gridy+yspace*(gridy-1)
    canvas = np.zeros(shape=(canvas_x,canvas_y))
    idx_y = range(0,canvas.shape[1],y+yspace)
    idx_x = range(0,canvas.shape[0],x+xspace)
    for posi,i in enumerate(idx_x):
        for posj,j in enumerate(idx_y):
            canvas[i:i+y,j:j+y] += images[posi+posj]
    if row_space:
        zeros = np.zeros(shape=(row_space,canvas.shape[1]))
        canvas = np.append(zeros,canvas)
        canvas = canvas.reshape(row_space+canvas_x,canvas_y)
    if CMAP:
      plt.matshow(canvas, cmap=CMAP)
    else:
      plt.matshow(canvas)
    plt.axis('off')
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    if out:
        fig.savefig(out+".png")
    else:
        plt.show()

def load_mnist(dataset="training", digits=np.arange(10), path="."):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [k for k in range(size) if lbl[k] in digits]
    N = len(ind)

    images = zeros((N, rows * cols), dtype=float)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ind[i] * rows * cols: (ind[i] + 1)
                              * rows * cols]).reshape((rows * cols))
        labels[i] = lbl[ind[i]]
    for i in range(len(ind)):
        images[i] = images[i] / 255
    return images, labels


# Load AAE configuration file
def load_config(input_path):
    cp = ConfigParser.ConfigParser()
    cp.read(input_path)
    return cp

# Logging messages such as loss,loading,etc.


def log(s, label='INFO'):
    sys.stdout.write(label + ' [' + str(datetime.now()) + '] ' + str(s) + '\n')
    sys.stdout.flush()


def load_data_train(cp):
    log('Loading data........')
    out = cp.get('Experiment', 'ModelOutputPath')
    num = cp.get('Experiment', 'Enumber')
    # If 'input file' parameter not defined then assume MNIST dataset
    if cp.get('Experiment', 'DataInputPath') == '':
        # Get FULL dataset containing both training/testing
        # In this experiment MNIST used as a testing mechanism in order to
        # test the connection between layers, availabilty,etc. and not for
        # hyperparameter tweaking.
        [X1, labels1] = load_mnist(
            dataset='training', path=MNIST_PATH)
        # Shuffle the dataset for training then use the same permutation for
        # the labels.
        p = np.random.permutation(X1.shape[0])
        X = X1[p].astype(np.float32)
        labels = labels1[p]
        prefix = cp.get('Experiment', 'prefix')
        np.save(out + prefix + '_' + num + '_' + 'random_perm.npy', p)
        log('DONE........')
        log('Dataset shape: ' + str(X.shape))
        return [X, labels]
    # If 'input file' is specified then load inputfile, our script assumes that
    # the input file will always be a numpy object
    else:
        try:
            X = np.load(cp.get('Experiment', 'DataInputPath'))
        except:
            log('Input file must be a saved numpy object (*.npy)')
        # Shuffle the dataset
        p = np.random.permutation(X.shape[0])
        X = X[p].astype(np.float32)
        prefix = cp.get('Experiment', 'prefix')
        np.save(out + prefix +  '_' + num + '_' + 'random_perm.npy', p)
        log('DONE........')
        log('Dataset shape: ' + str(X.shape))
        return X

def plot_class_space(template, dataset, labels):
    z = template.get_hidden(dataset)
    plt.figure()
    color=cm.rainbow(np.linspace(0,1,10))
    for l,c in zip(range(10),color):
        ix = np.where(labels==l)[0]
        plt.scatter(z[ix,0],z[ix,1],c=c,label=l,s=8,linewidth=0)
    plt.legend()
    plt.show()

def get_batch_idx(N, batch_size):
    num_batches = (N + batch_size - 1) / batch_size

    for i in range(num_batches):
        start, end = i * batch_size, (i + 1) * batch_size
        idx = slice(start, end)
    yield idx
