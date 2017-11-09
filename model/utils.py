###############################################################################
# Description
###############################################################################


import cPickle
import gzip
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import ConfigParser
import sys
from datetime import datetime

MNIST_PATH = '.'


def save(filename, *objects):
    # Save object and compress it at the same time (it can be used for multiple
    # objects or a single object)
    fil = gzip.open(filename, 'wb')
    for obj in objects:
        cPickle.dump(obj, fil,protocol=cPickle.HIGHEST_PROTOCOL)
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

def displayz(a, x, y, startind=0, sizex=12, sizey=12, CMAP=None):
    fig = plt.figure(figsize=(sizex, sizey))
    fig.subplots_adjust(hspace=0.01, wspace=0.05)
    for i in range(x * y):
        sub = fig.add_subplot(x, y, i+1)
        # sub.imshow(a[startind+i,:,:], interpolation='nearest')
        if CMAP:
            sub.imshow(a[startind+i,:,:], cmap=CMAP, interpolation='nearest')
        else:
            sub.imshow(a[startind+i,:,:], interpolation='nearest')
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
    # for i in range(len(ind)):
    #     images[i] = images[i] / 255
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
    # If 'input file' parameter not defined then assume MNIST dataset
    if cp.get('Experiment', 'DataInputPath') == '':
        # Get FULL dataset containing both training/testing
        # In this experiment MNIST used as a testing mechanism in order to
        # test the connection between layers, availabilty,etc. and not for
        # hyperparameter tweaking.
        [X1, labels1] = load_mnist(
            dataset='training', path=MNIST_PATH)
        # Shuffle the dataset for training then use the same permutation for the labels.
        p = np.random.permutation(X.shape[0])
        X = X[p].astype(np.float32) * 0.02
        labels = labels[p]
        prefix = cp.get('Experiment', 'prefix')
        np.save(prefix + '_' + 'random_perm.npy', p)
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
        X = X[p]
        prefix = cp.get('Experiment', 'prefix')
        np.save(prefix + '_' + 'random_perm.npy', p)
        return X
    log('DONE........')
    log('Dataset shape: '+X.shape)
