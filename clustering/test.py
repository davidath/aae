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
from draw_net import get_pydot_graph, draw_to_file

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
    # Load batch size
    code_width = cp.getint('Z', 'Width')
    batch_size = cp.getint('Hyperparameters', 'batchsize')
    sample_dist = cp.get('Hyperparameters', 'SampleDist')
    label_width = cp.getint('Y', 'Width')
    # Prepare mini batch slices
    slices = xrange(0, dataset.shape[0], batch_size)
    # Random idx
    rand = random.choice(slices)
    idx = slice(rand, rand + batch_size)
    X_batch = dataset[idx].astype(np.float32)
    # Plot reconstruction
    if fig_out:
        utils.plot_batch_reconstruction(layer_dict, X_batch, out=fig_str+'reconstruction')
    else:
        utils.plot_batch_reconstruction(layer_dict, X_batch)
    # Sample random
    if sample_dist == 'swiss' and num_labels is not None:
        sample = aae.sample_swiss_roll(
            batch_size, code_width, num_labels)
    elif sample_dist == 'uniform':
        sample = aae.sample_uniform(batch_size, code_width)
    else:
        sample = aae.sample_normal(batch_size, code_width)
    cat_sample = aae.sample_cat(batch_size, label_width)
    # Plot autoencoder generated digits
    if fig_out:
        utils.plot_generated(layer_dict, cat_sample, sample, out=fig_str+'generated')
    else:
        utils.plot_generated(layer_dict, cat_sample, sample)
    if fig_out:
        utils.plot_cluster_heads(layer_dict, dataset[:500], batch_size, code_width, out=fig_str)
    else:
        utils.plot_cluster_heads(layer_dict, dataset[:500], batch_size, code_width)
    draw_to_file(adv_ae, fig_str+'test.pdf', verbose=True)


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
