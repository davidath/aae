#!/usr/bin/env python

###############################################################################
# Description
###############################################################################

import os
# Removing/Adding comment enables/disables theano GPU support
os.environ[
    'THEANO_FLAGS'] = 'mode=FAST_RUN,device=cuda,floatX=float32'
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
    except:
        X = utils.load_data_train(cp)
        labels = None
    train(cp, X, labels)

# Initialize neural network and train model


def train(cp, dataset, labels=None):
    # IO init
    prefix = cp.get('Experiment', 'prefix')
    out = cp.get('Experiment', 'ModelOutputPath')
    num = cp.get('Experiment', 'Enumber')
    # Shared input variable
    input_var = theano.shared(name='input_var', value=np.asarray(dataset,
                                                                 dtype=theano.config.floatX),
                              borrow=True)
    # Building/Stacking layers
    [layer_dict, adv_ae] = aae.build_model(cp)
    # Get objective functions
    recon_loss = aae.reconstruction_loss(cp, input_var, layer_dict)
    dis_loss = aae.discriminator_loss(cp, input_var, layer_dict)
    gen_loss = aae.generator_loss(cp, input_var, layer_dict)
    # Pre-train inits
    pz_std = float(cp.get('Hyperparameters', 'pz_std'))
    code_width = cp.getint('Z', 'Width')
    batch_size = cp.getint('Hyperparameters', 'batchsize')
    ep_lr_decay1 = cp.getint('Hyperparameters', 'lrdecayepoch1')
    ep_lr_decay2 = cp.getint('Hyperparameters', 'lrdecayepoch2')
    max_epochs = cp.getint('Hyperparameters', 'maxepochs')
    lr = float(cp.get('Hyperparameters', 'learningrate'))
    # Save on CTRL-C
    try:
        for epoch in xrange(max_epochs):
            # Epoch timing
            tstart = time.time()
            # Gather losses
            reconstruct = []
            cross_entropy = []
            entropy = []
            for row in tqdm(xrange(0, dataset.shape[0], batch_size),ascii=True):
                reconstruct.append(recon_loss(row, batch_size, lr))
                # Sample from normal distribution for prior p(z)
                normal_sample = aae.sample_normal(batch_size,code_width)
                cross_entropy.append(
                    dis_loss(row, batch_size, lr, normal_sample))
                entropy.append(gen_loss(row, batch_size, lr, normal_sample))
                # aae.generate_digit(layer_dict)
            reconstruct = np.asarray(reconstruct)
            cross_entropy = np.asarray(cross_entropy)
            entropy = np.asarray(entropy)
            # Train loss messages and model saves
            if epoch % 10 == 0:
                log(str(epoch) + ' ' + str(np.mean(reconstruct)),
                    label='AAE-LRecon')
                log(str(epoch) + ' ' + str(np.mean(cross_entropy)),
                    label='AAE-LCross')
                log(str(epoch) + ' ' + str(np.mean(entropy)),
                    label='AAE-LEntr')
                timing(tstart,time.time())
                if labels is not None:
                    template = aae.make_template(layer_dict, adv_ae)
                    utils.plot_class_space(template, dataset, labels)
                    recon = template.get_output(dataset[0:9,:]).reshape(9,28,28)
                #     utils.plot_grid(recon,3,3,0,0)
            if epoch % ep_lr_decay1 == 0 or epoch % ep_lr_decay2 == 0:
                lr = lr / 10
            if (epoch % 100 == 0) and (epoch != 0):
                template = aae.make_template(layer_dict, adv_ae)
                template.save(out + prefix + '_' + num + '_' + 'model.zip')
    except KeyboardInterrupt:
        # log('Caught CTRL-C, Training has been stoped.......')
        # log('Saving model....')
        template = aae.make_template(layer_dict, adv_ae)
        template.save(out + prefix + '_' + num + '_' + 'model.zip')

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
