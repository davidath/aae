###############################################################################
# Description
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
    relu = lasagne.nonlinearities.rectify
    linear = lasagne.nonlinearities.linear
    sigmoid = lasagne.nonlinearities.sigmoid
    act_dict = {'ReLU': relu, 'Linear': linear, 'Sigmoid': sigmoid}
    # Begin stacking layers
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
    # Add generator flag that will be used in backward pass
    ae_enc.params[ae_enc.W].add('generator')
    ae_enc.params[ae_enc.b].add('generator')
    # ---- End of Encoder for AE and Generator for GAN ----
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
    prior_inp = ll.InputLayer(
        shape=(None, cp.getint('Z', 'Width')), name='pz_inp')
    dis_in = dis_net = ll.ConcatLayer(
        [ae_enc, prior_inp], axis=0, name='Dis_in')
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

# Create template of our model for testing,saving, etc.


def make_template(layer_dict, aae):
    return Model(layer_dict=layer_dict, aae=aae)


# Create autoencoder objective function also known as reconstruction loss


def reconstruction_loss(cp, input_var, layer_dict):
    # Scalar used for batch training
    index = T.lscalar()
    batch_size = T.lscalar()
    # Scalar used for learning rate
    learning_rate = T.scalar(name='learning_rate')
    # Get reconstructed input from AE
    X_hat = ll.get_output(layer_dict['AAE_Output'])
    # MSE between real input and reconstructed input
    recon_loss =  T.mean(
        T.mean(T.sqr(layer_dict['AAE_Input'].input_var - X_hat), axis=1))
    # Update trainable parameters of AE
    recon_params = ll.get_all_params(layer_dict['AAE_Output'], trainable=True)
    recon_updates = lasagne.updates.nesterov_momentum(
        recon_loss, recon_params, learning_rate=learning_rate, momentum=float(cp.get('Hyperparameters', 'momentum')))
    # Reconstruction loss a.k.a Lrecon
    recon_func = theano.function(inputs=[index, batch_size, learning_rate],
                                 outputs=recon_loss, updates=recon_updates,
                                 givens={layer_dict['AAE_Input'].input_var: input_var[
                                     index:index + batch_size, :]}
                                 )
    return recon_func

# Create discriminator objective function also known as the cross entropy between prior
# distribution p(z) and posterior estimate distribution q(z|x)


def discriminator_loss(cp, input_var, layer_dict):
    # Scalar used for batch training
    index = T.lscalar()
    batch_size = T.lscalar()
    pz = T.fmatrix('pz')
    # Scalar used for learning rate
    learning_rate = T.scalar(name='learning_rate')
    # Get discriminator output
    dis_out = ll.get_output(layer_dict['Dis_out'])
    # Fake/Real (0/1) labels for discriminator loss
    dis_targets = T.vertical_stack(
        T.ones((batch_size, 1)),
        T.zeros((batch_size, 1))
    )
    # Cross entropy regularization term
    dis_loss = T.nnet.binary_crossentropy(dis_out, dis_targets).mean()
    # Train discriminator
    dis_params = ll.get_all_params(
        layer_dict['Dis_out'], trainable=True, discriminator=True)
    dis_updates = lasagne.updates.nesterov_momentum(
        dis_loss, dis_params, learning_rate=learning_rate, momentum=float(cp.get('Hyperparameters', 'momentum')))
    # Theano function
    Ldis_func = theano.function(inputs=[index, batch_size, learning_rate, pz],
                                outputs=dis_loss, updates=dis_updates,
                                givens={
                                    layer_dict['AAE_Input'].input_var: input_var[index:index + batch_size, :],
                                    layer_dict['pz_inp'].input_var: pz
    }
    )
    return Ldis_func

# Create generator objective function also known as the entropy of the posterior
# estimate distribution q(z|x)


def generator_loss(cp, input_var, layer_dict):
    # Scalar used for batch training
    index = T.lscalar()
    batch_size = T.lscalar()
    # Scalar used for learning rate
    learning_rate = T.scalar(name='learning_rate')
    pz = T.fmatrix('pz')
    # Pass generator output to discriminator input
    dis_out = ll.get_output(layer_dict['Dis_out'])
    # Create generator targets
    gen_targets = T.zeros_like(dis_out.shape[0])
    # Entropy regularization term
    gen_loss = T.nnet.binary_crossentropy(dis_out, gen_targets).mean()
    # Train generator
    gen_params = ll.get_all_params(
        layer_dict['Dis_out'], trainable=True, generator=True)
    gen_updates = lasagne.updates.nesterov_momentum(
        gen_loss, gen_params, learning_rate=learning_rate, momentum=float(cp.get('Hyperparameters', 'momentum')))
    Lgen_func = theano.function(inputs=[index, batch_size, learning_rate, pz],
                                outputs=gen_loss, updates=gen_updates,
                                givens={
                                    layer_dict['AAE_Input'].input_var: input_var[index:index + batch_size, :],
                                    layer_dict['pz_inp'].input_var: pz
    }
    )
    return Lgen_func

# Sample from normal distribution

def sample_normal(batch_size, code_width):
    rs = RandomStreams(seed=np.random.randint(1234))
    return 1.0*rs.normal(size=(batch_size, code_width),dtype=theano.config.floatX).eval()


def gpu_sample_normal(batch_size, code_width):
    rs = RandomStreams(seed=np.random.randint(1234))
    return 1.0*rs.normal(size=(batch_size, code_width),dtype=theano.config.floatX)

# Create REAL generator network

def generate_digit(layer_dict):
    input_lay = gen = layer_dict['pz_inp']
    for sect in [layer_dict[i] for i in layer_dict if 'Decoder' in i]:
        gen = ll.DenseLayer(incoming=gen,
                                   num_units=sect.num_units,
                                   W=sect.W, nonlinearity=sect.nonlinearity,
                                   name=sect)
    sect = layer_dict['AAE_Output']
    gen = ll.DenseLayer(incoming=gen,
                               num_units=sect.num_units,
                               W=sect.W, nonlinearity=sect.nonlinearity,
                               name=sect)
    input_lay.input_var = gpu_sample_normal(9,2)
    utils.plot_grid(ll.get_output(gen).eval().reshape(9,28,28),3,3,0,0)

#
# if __name__ == '__main__':
#     cp = utils.load_config('../cfg/aae_default.ini')
#     [layer_dict, aae] = build_model(cp)
# #     print('collected %d layers' % (len(layer_dict.keys())))
#     for name in layer_dict:
#         print name, ll.get_output_shape(layer_dict[name])
# #         print('%s: %r' % (name, layer_dict[name]))
# #     print ll.get_all_params(layer_dict['AAE_Output'],trainable=True)
# #     print len(ll.get_all_params(layer_dict['AAE_Output'],trainable=True))
