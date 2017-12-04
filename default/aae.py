###############################################################################
# Contains everything that has to do with the AAE as a model (Layer stacking,
# loss functions, etc.) Loss functions and layer stacking were inspired by
# https://github.com/hjweide/adversarial-autoencoder that also uses Lasagne
# implementation
###############################################################################

import lasagne
import lasagne.layers as ll
import numpy as np
import theano
import theano.tensor as T
import utils
from modeltemplate import Model
from theano.tensor.shared_randomstreams import RandomStreams

# Stacking layers from config file


def build_model(cp):
    # Create activations dictionary
    relu = lasagne.nonlinearities.rectify
    linear = lasagne.nonlinearities.linear
    sigmoid = lasagne.nonlinearities.sigmoid
    act_dict = {'ReLU': relu, 'Linear': linear, 'Sigmoid': sigmoid}
    # Begin stacking layers
    ###########################################################################
    # Encoder part of AE / GAN Generator
    ###########################################################################
    # Input
    input_layer = ae_network = ll.InputLayer(
        shape=(None, cp.getint('AAE_Input', 'Width')), name='AAE_Input')
    # Stack endoder layers
    for sect in [i for i in cp.sections() if 'Encoder' in i]:
        ae_network = ll.DenseLayer(incoming=ae_network,
                                   num_units=cp.getint(sect, 'Width'), nonlinearity=act_dict[cp.get(sect, 'Activation')],
                                   name=sect)
        # Add generator flag that will be used in backward pass
        ae_network.params[ae_network.W].add('generator')
        ae_network.params[ae_network.b].add('generator')
    # Latent variable Z layer also known as q(z|x)
    ae_enc = ae_network = ll.DenseLayer(incoming=ae_network,
                                        num_units=cp.getint('Z', 'Width'), nonlinearity=act_dict[cp.get('Z', 'Activation')],
                                        name='Z')
    generator = ae_network
    # Add generator flag that will be used in backward pass
    ae_enc.params[ae_enc.W].add('generator')
    ae_enc.params[ae_enc.b].add('generator')
    # ---- End of Encoder for AE and Generator for GAN ----
    ###########################################################################
    # Decoder part of AE
    ###########################################################################
    # Stack decoder layers
    for sect in [i for i in cp.sections() if 'Decoder' in i]:
        ae_network = ll.DenseLayer(incoming=ae_network,
                                   num_units=cp.getint(sect, 'Width'), nonlinearity=act_dict[cp.get(sect, 'Activation')],
                                   name=sect)
    ae_out = ae_network = ll.DenseLayer(incoming=ae_network,
                                        num_units=cp.getint(
                                            'AAE_Output', 'Width'), nonlinearity=act_dict[cp.get('AAE_Output', 'Activation')],
                                        name='AAE_Output')
    # ---- End of Decoder for AE ----
    ###########################################################################
    # Discriminator part of GAN
    ###########################################################################
    # prior_inp = ll.InputLayer(
    #     shape=(None, cp.getint('Z', 'Width')), name='pz_inp')
    # dis_in = dis_net = ll.ConcatLayer(
    #     [ae_enc, prior_inp], axis=0, name='Dis_in')
    dis_net = generator
    # Stack discriminator layers
    for sect in [i for i in cp.sections() if 'Discriminator' in i]:
        dis_net = ll.DenseLayer(incoming=dis_net,
                                num_units=cp.getint(sect, 'Width'), nonlinearity=act_dict[cp.get(sect, 'Activation')],
                                name=sect)
        # Add generator flag that will be used in backward pass
        dis_net.params[dis_net.W].add('discriminator')
        dis_net.params[dis_net.b].add('discriminator')
    dis_out = dis_net = ll.DenseLayer(incoming=dis_net,
                                      num_units=cp.getint('Dout', 'Width'), nonlinearity=act_dict[cp.get('Dout', 'Activation')],
                                      name='Dis_out')
    dis_out.params[dis_out.W].add('discriminator')
    dis_out.params[dis_out.b].add('discriminator')
    aae = ll.get_all_layers([ae_out, dis_out])
    layer_dict = {layer.name: layer for layer in aae}
    return layer_dict, aae

# Load pretrained model a.k.a rebuild model and load pretrained weights


def load_pretrained(cp, weights):
    aae = build_model(cp)[1]
    ll.set_all_param_values(aae, weights)
    layer_dict = {layer.name: layer for layer in aae}
    return layer_dict, aae

# Create template of our model for testing,saving, etc.


def make_template(layer_dict, aae):
    return Model(layer_dict=layer_dict, aae=aae)


# Create autoencoder objective function also known as reconstruction loss

def reconstruction_loss(layer_dict):
    # Symbolic var for learning rate
    lr = T.scalar('lr')
    # Symbolic input variable
    input_var = T.fmatrix('input_var')
    # Symbolic mini batch variable
    batch = T.fmatrix('batch')
    # Get reconstructed input from AE
    reconstruction = ll.get_output(
        layer_dict['AAE_Output'], input_var, deterministic=False)
    # MSE between real input and reconstructed input
    recon_loss = T.mean(T.mean(T.sqr(input_var - reconstruction), axis=1))
    # Update trainable parameters of AE
    recon_params = ll.get_all_params(layer_dict['AAE_Output'], trainable=True)
    recon_updates = lasagne.updates.nesterov_momentum(
        recon_loss, recon_params, learning_rate=lr, momentum=0.9)
    # Reconstruction loss a.k.a Lrecon
    recon_func = theano.function(inputs=[theano.In(batch), lr],
                                 outputs=recon_loss, updates=recon_updates,
                                 givens={input_var: batch}
                                 )
    return recon_func


# Create discriminator objective function also known as the cross entropy between prior
# distribution p(z) and posterior estimate distribution q(z|x)

def d_sample_discriminate(layer_dict):
    # Symbolic var for learning rate
    dglr = T.scalar('lr')
    # Symbolic input variable
    input_var = T.fmatrix('input_var')
    # Symbolic mini batch variable
    batch = T.fmatrix('batch')
    dis_targets = T.fmatrix('dis_targets')
    # Get discriminator output
    dis_out = ll.get_output(layer_dict['Dis_out'],
                            inputs={layer_dict['Z']: input_var},
                            deterministic=False)
    # Fake/Real (0/1) labels for discriminator loss
    # dis_targets = T.vertical_stack(
    #     T.ones((batch.shape[0], 1)),
    #     T.zeros((pz_batch.shape[0], 1))
    # )
    # Cross entropy regularization term
    dis_loss = T.mean(T.nnet.binary_crossentropy(dis_out, dis_targets))
    dis_params = ll.get_all_params(layer_dict['Dis_out'], trainable=True,
                                   discriminator=True)
    dis_updates = lasagne.updates.nesterov_momentum(
        dis_loss, dis_params, learning_rate=dglr, momentum=0.1)
    # Discriminator loss aka Cross Entropy term
    dis_func = theano.function(inputs=[theano.In(batch),
                                       theano.In(dis_targets), dglr],
                               outputs=dis_loss, updates=dis_updates,
                               givens={input_var: batch}
                               )
    return dis_func




# Create generator objective function also known as the entropy of the posterior
# estimate distribution q(z|x)

def d_z_discriminate(layer_dict):
    # Symbolic var for learning rate
    dglr = T.scalar('lr')
    # Symbolic input variable
    input_var = T.fmatrix('input_var')
    gen_targets = T.fmatrix('gen_targets')
    # Symbolic mini batch variable
    batch = T.fmatrix('batch')
    # Get generator output
    gen_out = ll.get_output(layer_dict['Z'], input_var, deterministic=False)
    # Pass generator output to discriminator input
    dis_out = ll.get_output(layer_dict['Dis_out'],
                            inputs={layer_dict['Z']: gen_out},
                            deterministic=False
                            )
    # Create generator targets, confuse discriminator
    # gen_targets = T.zeros_like(batch.shape[0])
    # Entropy regularization term
    gen_loss = T.mean(T.nnet.binary_crossentropy(dis_out, gen_targets))
    gen_params = ll.get_all_params(
        layer_dict['Dis_out'], trainable=True, discriminator=True)
    gen_updates = lasagne.updates.nesterov_momentum(
        gen_loss, gen_params, learning_rate=dglr, momentum=0.1)
    # Generator loss aka Entropy term
    gen_func = theano.function(inputs=[theano.In(batch), theano.In(gen_targets), dglr],
                               outputs=gen_loss, updates=gen_updates,
                               givens={input_var: batch}
                               )
    return gen_func

def g_z_discriminate(layer_dict):
    # Symbolic var for learning rate
    dglr = T.scalar('lr')
    # Symbolic input variable
    input_var = T.fmatrix('input_var')
    gen_targets = T.fmatrix('gen_targets')
    # Symbolic mini batch variable
    batch = T.fmatrix('batch')
    # Get generator output
    gen_out = ll.get_output(layer_dict['Z'], input_var, deterministic=False)
    # Pass generator output to discriminator input
    dis_out = ll.get_output(layer_dict['Dis_out'],
                            inputs={layer_dict['Z']: gen_out},
                            deterministic=False
                            )
    # Create generator targets, confuse discriminator
    # gen_targets = T.zeros_like(batch.shape[0])
    # Entropy regularization term
    gen_loss = T.mean(T.nnet.binary_crossentropy(dis_out, gen_targets))
    gen_params = ll.get_all_params(
        layer_dict['Dis_out'], trainable=True, generator=True)
    gen_updates = lasagne.updates.nesterov_momentum(
        gen_loss, gen_params, learning_rate=dglr, momentum=0.1)
    # Generator loss aka Entropy term
    gen_func = theano.function(inputs=[theano.In(batch), theano.In(gen_targets), dglr],
                               outputs=gen_loss, updates=gen_updates,
                               givens={input_var: batch}
                               )
    return gen_func

# Sample from normal distribution


def sample_normal(batch_size, code_width, mu=0, sigma=1):
    return np.random.normal(mu, sigma, size=(batch_size, code_width)).astype(np.float32)

# Sample for uniform distribution


def sample_uniform(batch_size, code_width, low=-2, high=2):
    return np.random.uniform(low, high, size=(batch_size,
                                              code_width)).astype(np.float32)


# Sample swiss roll
# Credits to https://github.com/musyoku/adversarial-autoencoder

def sample(label, num_labels):
    uni = np.random.uniform(0.0, 1.0) / float(num_labels) + \
        float(label) / float(num_labels)
    r = np.sqrt(uni) * 3.0
    rad = np.pi * 4.0 * np.sqrt(uni)
    x = r * np.cos(rad)
    y = r * np.sin(rad)
    return np.array([x, y]).reshape((2,))


def sample_swiss_roll(batchsize, ndim, num_labels):
    z = np.zeros((batchsize, ndim), dtype=np.float32)
    for batch in range(batchsize):
        for zi in range(ndim // 2):
            z[batch, zi * 2:zi * 2 +
                2] = sample(np.random.randint(0, num_labels - 1), num_labels)
    return z


if __name__ == "__main__":
    import utils
    cp = utils.load_config('../cfg/default/normal.ini')
    from draw_net import *
    draw_to_file(build_model(cp)[1], 'test.pdf', verbose=True)
