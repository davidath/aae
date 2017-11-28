###############################################################################
# Contains various functions that are used throught the training,testing and
# building process (object save/load, plots, dataset loading, etc.). 
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
JUPYTER_MNIST_PATH = '../../../datasets/mnist'

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
    newimg = np.pad(img, ((1, 1), (1, 1)), 'constant', constant_values=(0))
    max_x = newimg.shape[0] - 1
    max_y = newimg.shape[1] - 1
    newimg[0, :] = np.ones(shape=(newimg[0, :].shape)) * np.max(newimg)
    newimg[max_x, :] = np.ones(shape=(newimg[max_x, :].shape)) * np.max(newimg)
    newimg[:, 0] = np.ones(shape=(newimg[:, 0].shape)) * np.max(newimg)
    newimg[:, max_y] = np.ones(shape=(newimg[:, max_y].shape)) * np.max(newimg)
    return newimg


def plot_row(images, spacing, CMAP=None):
    images = np.asarray([plot_border(image) for image in images])
    x = images.shape[1]
    y = images.shape[2]
    canvas_y = y * len(images) + spacing * (len(images) - 1)
    canvas = np.zeros(shape=(x, canvas_y))
    print canvas.shape
    idx = range(0, canvas.shape[1], y + spacing)
    for pos, i in enumerate(idx):
        canvas[:, i:i + y] += images[pos]
    if CMAP:
        plt.matshow(canvas, cmap=CMAP)
    else:
        plt.matshow(canvas)
    plt.axis('off')
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.show()


def plot_grid(images, x, y, img_x, img_y, CMAP=plt.cm.binary, out=None, border=2):
    images = (i for i in images)
    width = border * (x + 1) + x * img_x
    height = border * (y + 1) + y * img_y

    res = np.zeros([height, width])  # the resulting image

    for curr_x in range(x):
        x_offset = (curr_x + 1) * border + (curr_x * img_x)
        for curr_y in range(y):
            y_offset = (curr_y + 1) * border + (curr_y * img_y)
            # print x_offset, y_offset
            try:
                res[x_offset:x_offset + img_x, y_offset:y_offset + img_y] = images.next().reshape(img_x, img_y)
            except StopIteration:
                pass
    fig = plt.figure()
    plt.imshow(res, cmap=CMAP)
    plt.axis('off')
    if out:
        fig.savefig(out + '.png')
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
        # In this experiment MNIST is used as a testing mechanism in order to
        # test the connection between layers, availabilty,etc. and not for
        # hyperparameter tweaking.
        try:
            [X1, labels1] = load_mnist(
                dataset='training', path=MNIST_PATH)
        except:
            [X1, labels1] = load_mnist(
                dataset='training', path=JUPYTER_MNIST_PATH)
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
        np.save(out + prefix + '_' + num + '_' + 'random_perm.npy', p)
        log('DONE........')
        log('Dataset shape: ' + str(X.shape))
        return X


def load_data_test(cp):
    log('Loading data........')
    out = cp.get('Experiment', 'ModelOutputPath')
    num = cp.get('Experiment', 'Enumber')
    # If 'input file' parameter not defined then assume MNIST dataset
    if cp.get('Experiment', 'DataInputPath') == '':
        # In this experiment MNIST is used as a testing mechanism in order to
        # test the connection between layers, availabilty,etc. and not for
        # hyperparameter tweaking.
        try:
            [X, labels] = load_mnist(
                dataset='testing', path=MNIST_PATH)
        except:
            [X, labels] = load_mnist(
                dataset='testing', path=JUPYTER_MNIST_PATH)
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
        log('DONE........')
        log('Dataset shape: ' + str(X.shape))
        return X


def plot_class_space(template, dataset, labels, out=None):
    z = template.get_hidden(dataset)
    fig = plt.figure()
    color = cm.rainbow(np.linspace(0, 1, 10))
    for l, c in zip(range(10), color):
        ix = np.where(labels == l)[0]
        plt.scatter(z[ix, 0], z[ix, 1], c=c, label=l, s=8, linewidth=0)
    plt.legend()
    if out:
        fig.savefig(out + ".png")
    else:
        plt.show()
