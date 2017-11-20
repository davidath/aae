#!/usr/bin/env python

###############################################################################
# Description
###############################################################################

import os
# Removing/Adding comment enables/disables theano GPU support
# os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,device=cuda,floatX=float32'
# Removing/Adding comment forces/stops theano CPU support, usually used
# for model saving
os.environ['THEANO_FLAGS'] = 'device=cpu,force_device=True'
import numpy as np
import sys
import utils
import aae
import theano
import theano.tensor as T
from datetime import datetime
import time
from tqdm import *
import lasagne.layers as ll
import random

# Logging messages such as loss,loading,etc.


def log(s, label='INFO'):
    sys.stdout.write(label + ' [' + str(datetime.now()) + '] ' + str(s) + '\n')
    sys.stdout.flush()

# Timing functions


def timing(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    log("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

# Control flow


def main(path):
    cp = utils.load_config(path)
    # Check if MNIST dataset was loaded
    try:
        [X, labels] = utils.load_data_test(cp)
    except KeyboardInterrupt:
        log('Caught CTRL-C....', label='Exception')
        log('Loading data was stopped..... exiting', label='Exception')
        exit(-1)
    except:
        X = utils.load_data_train(cp)
        labels = None
    if 'jupyter' in path:
        test(cp, X, labels, fig_out=False)
    else:
        test(cp, X, labels)


# Plot batch reconstruction

def plot_batch_reconstruction(layer_dict, X_batch, out=None):
    try:
        x_size = y_size = int(np.sqrt(layer_dict['AAE_Input'].shape[1]))
        gridx = gridy = int(np.sqrt(X_batch.shape[0]))
        recon = ll.get_output(layer_dict['AAE_Output'], X_batch).eval(
        ).reshape(X_batch.shape[0], x_size, y_size)
        if out:
            utils.plot_grid(recon, gridx, gridy, 0, 0, out=out)
        else:
            utils.plot_grid(recon, gridx, gridy, 0, 0)
    except:
        log('Expected square matrices for batch and sample....')
        log('Unable to plot grid.....')


# Plot autoencoder generated digits

def plot_generated(layer_dict, random_sample, out=None):
    try:
        x_size = y_size = int(np.sqrt(layer_dict['AAE_Input'].shape[1]))
        gridx = gridy = int(np.sqrt(random_sample.shape[0]))
        gen_out = ll.get_output(layer_dict['AAE_Output'],
                                inputs={layer_dict['Z']: random_sample}).eval()
        gen_out = gen_out.reshape(random_sample.shape[0], x_size, y_size)
        if out:
            utils.plot_grid(gen_out, gridx, gridy, 0, 0, out=out)
        else:
            utils.plot_grid(gen_out, gridx, gridy, 0, 0)
    except:
        log('Expected square matrices....')
        log('Unable to plot grid.....')


# Initialize neural network and train model


def test(cp, dataset, labels=None, fig_out=True):
    # IO init
    prefix = cp.get('Experiment', 'prefix')
    out = cp.get('Experiment', 'ModelOutputPath')
    num = cp.get('Experiment', 'Enumber')
    # Load pre-trained weights
    weights = np.load(out + prefix + '_' + num + '_' + 'weights.npy')
    # Build pretrained model
    [layer_dict, adv_ae] = aae.load_pretrained(cp, weights)
    # Save as CPU model
    template = aae.make_template(layer_dict, adv_ae)
    template.save(out + prefix + '_' + num + '_' + 'model_cpu.zip')
    fig_str = out + prefix + '_' + num + '_'
    # Plot latent space
    # Warning: this works in case that the latent variable/space width is 2
    if cp.getint('Z', 'Width') == 2:
        if fig_out:
            utils.plot_class_space(template, dataset, labels, out=fig_str+'latent_space')
        else:
            utils.plot_class_space(template, dataset, labels)
    # Load batch size
    code_width = cp.getint('Z', 'Width')
    batch_size = cp.getint('Hyperparameters', 'batchsize')
    sample_dist = cp.get('Hyperparameters', 'SampleDist')
    # Number of mnist labels
    if labels is not None:
        num_labels = [i[0] for i in labels]
        num_labels = len(set(num_labels))
    else:
        num_labels = None
    # Prepare mini batch slices
    slices = xrange(0, dataset.shape[0], batch_size)
    # Random idx
    rand = random.choice(slices)
    idx = slice(rand, rand + batch_size)
    X_batch = dataset[idx]
    # Plot reconstruction
    if fig_out:
        plot_batch_reconstruction(layer_dict, X_batch, out=fig_str+'reconstruction')
    else:
        plot_batch_reconstruction(layer_dict, X_batch)
    # Sample random
    if sample_dist == 'swiss' and num_labels is not None:
        sample = aae.sample_swiss_roll(
            batch_size, code_width, num_labels)
    elif sample_dist == 'uniform':
        sample = aae.sample_uniform(batch_size, code_width)
    else:
        sample = aae.sample_normal(batch_size, code_width)
    # Plot autoencoder generated digits
    if fig_out:
        plot_generated(layer_dict, sample, out=fig_str+'generated')
    else:
        plot_generated(layer_dict, sample)

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
