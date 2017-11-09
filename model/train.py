#!/usr/bin/env python

###############################################################################
# Description
###############################################################################

import os
# Removing/Adding comment enables/disables theano GPU support
# os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,device=gpu,floatX=float32,nvcc.flags=-D_FORCE_INLINES'
# Removing/Adding comment forces/stops theano CPU support, usually used for model saving
os.environ['THEANO_FLAGS'] = 'device=cpu,force_device=True'

import sys
import ConfigParser
import utils
import aae

# Logging messages such as loss,loading,etc.
def log(s, label='INFO'):
    sys.stdout.write(label + ' [' + str(datetime.now()) + '] ' + str(s) + '\n')
    sys.stdout.flush()

# Control flow
def main(path):
    cp = utils.load_config(path)
    # Check if MNIST dataset was loaded
    try:
        [X, labels] = utils.load_data_train(cp)
    except:
        X = utils.load_data_train(cp)
    init(cp, X)

# Initialize neural network and train model
def train(cp, dataset):
    # Shared input variable
    input_var = theano.shared(name='input_var', value=np.asarray(dataset,
                                            dtype=theano.config.floatX),
                          borrow=True)
    # Scalar used for batch training
    index = T.lscalar()
    # Building/Stacking layers
    [layer_dict, aae] = aae.build_model(cp)
    # Get Theano functions for Objective
    recon_loss = aae.reconstruction_loss(cp, input_var, layer_dict)
    print recon_loss(0, cp.getint('hyperparameters','batchsize'), cp.getint('hyperparameters','learningrate'))


from operator import attrgetter
from argparse import ArgumentParser
if __name__ == "__main__":
    parser = ArgumentParser(
        description='Training script for adversarial autoencoder')
    # Configuration file path
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='config path file')
    opts = parser.parse_args()
    getter = attrgetter('input')
    inp = getter(opts)
    main(inp)
