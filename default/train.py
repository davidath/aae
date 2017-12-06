#!/usr/bin/env python

###############################################################################
# Training script, contains dataset loading, training and saving of the
# AAE_default model
###############################################################################

import os
# Removing/Adding comment enables/disables theano GPU support
os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,device=cuda,floatX=float32,blas.ldflags=-lopenblas'
# Removing/Adding comment forces/stops theano CPU support, usually used for model saving
# os.environ['THEANO_FLAGS'] = 'device=cpu,force_device=True'
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
        [X, labels] = utils.load_data_train(cp)
    except KeyboardInterrupt:
        log('Caught CTRL-C....', label='Exception')
        log('Loading data was stopped..... exiting', label='Exception')
        exit(-1)
    except:
        X = utils.load_data_train(cp)
        labels = None
    train(cp, X, labels)

# Plot batch reconstruction


def plot_batch_reconstruction(layer_dict, X_batch):
    try:
        x_size = y_size = int(np.sqrt(layer_dict['AAE_Input'].shape[1]))
        gridx = gridy = int(np.sqrt(X_batch.shape[0]))
        recon = ll.get_output(layer_dict['AAE_Output'], X_batch).eval(
        ).reshape(X_batch.shape[0], x_size, y_size)
        utils.plot_grid(recon, gridx, gridy, x_size, y_size)
    except:
        log('Expected square matrices for batch and sample....')
        log('Unable to plot grid.....')

# Initialize neural network and train model


def train(cp, dataset, labels=None):
    # IO init
    prefix = cp.get('Experiment', 'prefix')
    out = cp.get('Experiment', 'ModelOutputPath')
    num = cp.get('Experiment', 'Enumber')
    # Building model / Stacking layers
    [layer_dict, adv_ae] = aae.build_model(cp)
    # Pre-train inits
    code_width = cp.getint('Z', 'Width')
    batch_size = cp.getint('Hyperparameters', 'batchsize')
    ep_lr_decay1 = cp.getint('Hyperparameters', 'lrdecayepoch1')
    ep_lr_decay2 = cp.getint('Hyperparameters', 'lrdecayepoch2')
    max_epochs = cp.getint('Hyperparameters', 'maxepochs')
    lr = float(cp.get('Hyperparameters', 'AElearningrate'))
    dglr = float(cp.get('Hyperparameters', 'DGlearningrate'))
    plot_recon = cp.getboolean('Experiment', 'PlotReconstruction')
    sample_dist = cp.get('Hyperparameters', 'SampleDist')
    # Number of mnist labels, this is only used if the swiss roll dist is chosen
    if labels is not None:
        num_labels = [i[0] for i in labels]
        num_labels = len(set(num_labels))
    else:
        num_labels = None
    # Get objective functions
    recon_loss = aae.reconstruction_loss(layer_dict)
    d_z_loss = aae.d_z_discriminate(layer_dict)
    d_sample_loss = aae.d_sample_discriminate(layer_dict)
    g_z_loss = aae.g_z_discriminate(layer_dict)
    fake = np.zeros((batch_size, 1)).astype(np.float32)
    real = np.ones((batch_size, 1)).astype(np.float32)
    for epoch in xrange(max_epochs):
        # Save on CTRL-C
        try:
            # Epoch timing
            tstart = time.time()
            # Gather losses
            reconstruct = []
            cross_entropy = []
            entropy = []
            # Mini batch loop
            for row in tqdm(xrange(0, dataset.shape[0], batch_size), ascii=True):
                # Slice dataset
                idx = slice(row, row + batch_size)
                X_batch = dataset[idx]
                reconstruct.append(recon_loss(X_batch, lr))
                # Sample from swiss roll distribution as prior p(z)
                if sample_dist == 'swiss' and num_labels is not None:
                    sample = aae.sample_swiss_roll(
                        batch_size, code_width, num_labels)
                # Sample from uniform distribution as prior p(z)
                elif sample_dist == 'uniform':
                    sample = aae.sample_uniform(batch_size, code_width)
                # Sample from swiss_roll distribution as prior p(z)
                else:
                    sample = aae.sample_normal(batch_size, code_width)
                # Gather loss for each mini-batch
                cross_entropy.append(d_z_loss(X_batch, fake, dglr))
                d_sample_loss(sample, real, dglr)
                entropy.append(g_z_loss(X_batch, real, dglr))
            reconstruct = np.asarray(reconstruct)
            cross_entropy = np.asarray(cross_entropy)
            entropy = np.asarray(entropy)
            # Train loss messages and model saves
            if epoch % 1 == 0:
                log(str(epoch) + ' ' + str(np.mean(reconstruct)),
                    label='AAE-LRecon')
                log(str(epoch) + ' ' + str(np.mean(cross_entropy)),
                    label='AAE-LCross')
                log(str(epoch) + ' ' + str(np.mean(entropy)),
                    label='AAE-LEntr')
                timing(tstart, time.time())
                # Optional reconstruction plot during training
                if plot_recon:
                    plot_batch_reconstruction(layer_dict, X_batch)
            # Learning rate decay
            if (epoch == ep_lr_decay1) or (epoch == ep_lr_decay2):
                lr = lr / 10.0
                dglr = dglr / 10.0
            # Save on milestone epochs
            if (epoch % 100 == 0) and (epoch != 0):
                template = aae.make_template(layer_dict, adv_ae)
                template.save(out + prefix + '_' + num + '_' + 'model.zip')
                np.save(out + prefix + '_' + num + '_' + 'weights.npy',
                        ll.get_all_param_values(adv_ae))
        # Save on CTRl-C
        except KeyboardInterrupt:
            # log('Caught CTRL-C, Training has been stoped.......')
            # log('Saving model....')
            template = aae.make_template(layer_dict, adv_ae)
            template.save(out + prefix + '_' + num + '_' + 'model.zip')
            np.save(out + prefix + '_' + num + '_' + 'weights.npy',
                    ll.get_all_param_values(adv_ae))
            break

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
