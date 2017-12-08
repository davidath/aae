#!/usr/bin/env python

###############################################################################
# Training script, contains dataset loading, training and saving of the
# AAE_Clustering model
###############################################################################

import os
# Removing/Adding comment enables/disables theano GPU support
os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,device=cuda,floatX=float32'
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
    # Number of mnist labels, this is only used if the swiss roll dist is chosen
    if labels is not None:
        num_labels = [i[0] for i in labels]
        num_labels = len(set(num_labels))
    else:
        num_labels = None
    # Get objective functions
    recon_loss = aae.reconstruction_loss(layer_dict)
    d_z_discriminate = aae.d_z_discriminate(layer_dict)
    g_z_discriminate = aae.g_z_discriminate(layer_dict)
    pz_discriminate = aae.pz_discriminate(layer_dict)
    d_y_discriminate = aae.d_y_discriminate(layer_dict)
    g_y_discriminate = aae.g_y_discriminate(layer_dict)
    py_discriminate = aae.py_discriminate(layer_dict)
    fake = np.zeros((batch_size, 1)).astype(np.float32)
    real = np.ones((batch_size, 1)).astype(np.float32)
    for epoch in xrange(max_epochs):
        # Save on CTRL-C
        try:
            # Epoch timing
            tstart = time.time()
            # Gather losses
            reconstruct = []
            z_dis_loss = []
            y_dis_loss = []
            z_gen_loss = []
            y_gen_loss = []
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
                # Gather losses
                y_sample = aae.sample_cat(batch_size, label_width)
                z_d = d_z_discriminate(X_batch, fake, dglr) + pz_discriminate(sample, real, dglr)
                y_d = d_y_discriminate(X_batch, fake, dglr) + py_discriminate(y_sample, real, dglr)
                z_g = g_z_discriminate(X_batch, real, dglr)
                y_g = g_y_discriminate(X_batch, real, dglr)
                z_d = z_d / 2.0
                y_d = y_d / 2.0
                z_dis_loss.append(z_d)
                y_dis_loss.append(y_d)
                z_gen_loss.append(z_g)
                y_gen_loss.append(y_g)
            # Print batch mean loss
            reconstruct = np.asarray(reconstruct)
            z_dis_loss = np.asarray(z_dis_loss)
            y_dis_loss = np.asarray(y_dis_loss)
            z_gen_loss = np.asarray(z_gen_loss)
            y_gen_loss = np.asarray(y_gen_loss)
            # Train loss messages and model saves
            if epoch % 1 == 0:
                log(str(epoch) + ' ' + str(np.mean(reconstruct)),
                    label='AAE-LRecon')
                log(str(epoch) + ' ' + str(np.mean(z_dis_loss)),
                    label='AAE-Z-Cross')
                log(str(epoch) + ' ' + str(np.mean(y_dis_loss)),
                    label='AAE-Y-Cross')
                log(str(epoch) + ' ' + str(np.mean(z_gen_loss)),
                    label='AAE-Z-Entr')
                log(str(epoch) + ' ' + str(np.mean(y_gen_loss)),
                    label='AAE-Y-Entr')
                timing(tstart, time.time())
                # Optional reconstruction plot during training
                if plot_recon:
                    if epoch % 10 == 0:
                        utils.plot_cluster_heads(
                              layer_dict, test, batch_size, code_width, out=str(epoch))
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
