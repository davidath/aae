###############################################################################
# Description
###############################################################################

#!/usr/bin/env python
import sys

def load_data(cp, train):
    log('Loading data........')
    # If 'input file' parameter not defined then assume MNIST dataset
    if cp.get('Experiment', 'inputfile') == '':
        # Get FULL dataset containing both training/testing
        # In this experiment MNIST used as a testing mechanism in order to
        # test the connection between layers, availabilty,etc. and not for
        # hyperparameter tweaking.
        [X1, labels1] = utils.load_mnist(
            dataset='training', path=MNIST_PATH)
        [X2, labels2] = utils.load_mnist(
            dataset='testing', path=MNIST_PATH)
        X = np.concatenate((X1, X2), axis=0)
        labels = np.concatenate((labels1, labels2), axis=0)
        # Will dataset be used for training? then shuffle the dataset then use
        # the same permutation for the labels.
        if train == 'train':
            p = np.random.permutation(X.shape[0])
            X = X[p].astype(np.float32) * 0.02
            labels = labels[p]
            prefix = cp.get('Experiment', 'prefix')
            num = cp.get('Experiment', 'num')
            np.save(prefix + '_' + num + 'random_perm.npy', p)
        return [X, labels]
    # If 'input file' is specified then load inputfile, our script assumes that
    # the input file will always be a numpy object
    else:
        try:
            X = np.load(cp.get('Experiment', 'inputfile'))
        except:
            log('Input file must be a saved numpy object (*.npy)')
        # Will dataset be used for training? then shuffle the dataset
        if train == 'train':
            p = np.random.permutation(X.shape[0])
            X = X[p]
            prefix = cp.get('Experiment', 'prefix')
            num = cp.get('Experiment', 'num')
            np.save(prefix + '_' + num + 'random_perm.npy', p)
        return X
    log('DONE........')

# Control flow
def main(path, train):
    cp = load_config(path)
    # Check if MNIST dataset was loaded
    try:
        [X, labels] = load_data(cp, train)
    except:
        X = load_data(cp, train)
    # Check training/testing flag
    if train == 'train':
        init(cp, X)
    else:
        pretrained(cp, X)
