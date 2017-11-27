#!/usr/bin/env python

###############################################################################
# Description
###############################################################################

import os
# Removing/Adding comment enables/disables theano GPU support
# os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,device=cuda,floatX=float32'
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

# Initialize neural network and train model


def train(cp, dataset, labels=None):
    # Plot during training ?
    plot_recon = cp.getboolean('Experiment', 'PlotReconstruction')
    # If flag is true, then load test dataset
    if plot_recon:
        log('Plot flag is true, loading test dataset......')
        [test, tlab] = utils.load_data_test(cp)
    # IO init
    prefix = cp.get('Experiment', 'prefix')
    out = cp.get('Experiment', 'ModelOutputPath')
    num = cp.get('Experiment', 'Enumber')
    # Building/Stacking layers
    [layer_dict, adv_ae] = aae.build_model(cp)
    # Pre-train inits
    code_width = cp.getint('Z', 'Width')
    label_width = cp.getint('Y', 'Width')
    batch_size = cp.getint('Hyperparameters', 'batchsize')
    ep_lr_decay1 = cp.getint('Hyperparameters', 'lrdecayepoch1')
    ep_lr_decay2 = cp.getint('Hyperparameters', 'lrdecayepoch2')
    max_epochs = cp.getint('Hyperparameters', 'maxepochs')
    lr = float(cp.get('Hyperparameters', 'AElearningrate'))
    dglr = float(cp.get('Hyperparameters', 'DGlearningrate'))
    sample_dist = cp.get('Hyperparameters', 'SampleDist')
    # Number of mnist labels
    if labels is not None:
        num_labels = [i[0] for i in labels]
        num_labels = len(set(num_labels))
    else:
        num_labels = None
    # Get objective functions
    recon_loss = aae.reconstruction_loss(layer_dict)
    z_dis_loss = aae.z_discriminator_loss(layer_dict)
    y_dis_loss = aae.y_discriminator_loss(layer_dict)
    z_gen_loss = aae.z_generator_loss(layer_dict)
    y_gen_loss = aae.y_generator_loss(layer_dict)
    for epoch in xrange(max_epochs):
        # Save on CTRL-C
        try:
            # Epoch timing
            tstart = time.time()
            # Gather losses
            reconstruct = []
            z_cross_entropy = []
            y_cross_entropy = []
            z_entropy = []
            y_entropy = []
            # Mini batch loop
            for row in tqdm(xrange(0, dataset.shape[0], batch_size), ascii=True):
                # Slice dataset
                idx = slice(row, row + batch_size)
                X_batch = dataset[idx]
                reconstruct.append(recon_loss(X_batch, lr))
                # Sample from normal distribution for prior p(z)
                if sample_dist == 'swiss' and num_labels is not None:
                    sample = aae.sample_swiss_roll(
                        batch_size, code_width, num_labels)
                elif sample_dist == 'uniform':
                    sample = aae.sample_uniform(batch_size, code_width)
                else:
                    sample = aae.sample_normal(batch_size, code_width)
                y_sample = aae.sample_cat(batch_size, label_width)
                z_cross_entropy.append(z_dis_loss(X_batch, sample, dglr))
                y_cross_entropy.append(y_dis_loss(X_batch, y_sample, dglr))
                z_entropy.append(z_gen_loss(X_batch, dglr))
                y_entropy.append(y_gen_loss(X_batch, dglr))
            reconstruct = np.asarray(reconstruct)
            z_cross_entropy = np.asarray(z_cross_entropy)
            y_cross_entropy = np.asarray(y_cross_entropy)
            z_entropy = np.asarray(z_entropy)
            y_entropy = np.asarray(y_entropy)
            # Train loss messages and model saves
            if epoch % 5 == 0:
                log(str(epoch) + ' ' + str(np.mean(reconstruct)),
                    label='AAE-LRecon')
                log(str(epoch) + ' ' + str(np.mean(z_cross_entropy)),
                    label='AAE-Z-Cross')
                log(str(epoch) + ' ' + str(np.mean(y_cross_entropy)),
                    label='AAE-Y-Cross')
                log(str(epoch) + ' ' + str(np.mean(z_entropy)),
                    label='AAE-Z-Entr')
                log(str(epoch) + ' ' + str(np.mean(y_entropy)),
                    label='AAE-Y-Entr')
                timing(tstart, time.time())
                # Optional reconstruction plot during training
                if plot_recon:
                    utils.plot_cluster_heads(
                        layer_dict, test, batch_size, code_width)
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
            log('Caught CTRL-C, Training has been stoped.......')
            log('Saving model....')
            plot_cluster_heads(layer_dict, batch_size, code_width)
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
