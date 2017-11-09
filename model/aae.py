###############################################################################
# Description
###############################################################################

import lasagne
import lasagne.layers as ll
import numpy as np
import theano.tensor as T
import utils
from modeltemplate import Model

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
                                   num_units=cp.getint(sect, 'Width'),
                                   W=lasagne.init.Normal(std=0.01), nonlinearity=act_dict[cp.get(sect, 'Activation')],
                                   name=sect)
        # Add generator flag that will be used in backward pass
        ae_network.params[ae_network.W].add('generator')
        ae_network.params[ae_network.b].add('generator')
    # Latent variable Z layer also known as q(z|x)
    ae_enc = ae_network = ll.DenseLayer(incoming=ae_network,
                                        num_units=cp.getint('Z', 'Width'),
                                        W=lasagne.init.Normal(std=0.01), nonlinearity=act_dict[cp.get('Z', 'Activation')],
                                        name='Z')
    # Add generator flag that will be used in backward pass
    ae_enc.params[ae_enc.W].add('generator')
    ae_enc.params[ae_enc.b].add('generator')
    # ---- End of Encoder for AE and Generator for GAN ----
    # Stack decoder layers
    for sect in [i for i in cp.sections() if 'Decoder' in i]:
        ae_network = ll.DenseLayer(incoming=ae_network,
                                   num_units=cp.getint(sect, 'Width'),
                                   W=lasagne.init.Normal(std=0.01), nonlinearity=act_dict[cp.get(sect, 'Activation')],
                                   name=sect)
    ae_out = ae_network = ll.DenseLayer(incoming=ae_network,
                                        num_units=cp.getint(
                                            'AAE_Output', 'Width'),
                                        W=lasagne.init.Normal(std=0.01), nonlinearity=act_dict[cp.get('AAE_Output', 'Activation')],
                                        name='AAE_Output')
    # ---- End of Decoder for AE ----
    prior_inp = ll.InputLayer(
        shape=(None, cp.getint('Z', 'Width')), name='pz_inp')
    dis_in = dis_net = ll.ConcatLayer(
        [ae_enc, prior_inp], axis=0, name='Dis_in')
    # Stack discriminator layers
    for sect in [i for i in cp.sections() if 'Discriminator' in i]:
        dis_net = ll.DenseLayer(incoming=dis_net,
                                num_units=cp.getint(sect, 'Width'),
                                W=lasagne.init.Normal(std=0.01), nonlinearity=act_dict[cp.get(sect, 'Activation')],
                                name=sect)
        # Add generator flag that will be used in backward pass
        dis_net.params[dis_net.W].add('discriminator')
        dis_net.params[dis_net.b].add('discriminator')
    dis_out = dis_net = ll.DenseLayer(incoming=dis_net,
                                      num_units=cp.getint('Dout', 'Width'),
                                      W=lasagne.init.Normal(std=0.01), nonlinearity=act_dict[cp.get('Dout', 'Activation')],
                                      name='Dis_out')
    dis_out.params[dis_out.W].add('discriminator')
    dis_out.params[dis_out.b].add('discriminator')
    aae = ll.get_all_layers([ae_out, dis_out])
    layer_dict = {layer.name: layer for layer in aae}
    return layer_dict, aae

# Create template of our model for testing,saving, etc.


def make_template(layer_dict, aae):
    return Model(layer_dict=layer_dict, aae=aae)


def reconstruction_loss(cp, input_var, layer_dict):
    learning_rate = T.scalar(name='learning_rate')
    X_recon = ll.get_output(layer_dict['AAE_Output'])
    recon_loss = lasagne.objectives.squared_error(X_recon,
                                                  layer_dict['AAE_Input'].input_var)
    recon_params = ll.get_all_params(layer_dict['AAE_Output'], trainable=True)
    recon_updates = lasagne.updates.nesterov_momentum(
        cost, params, learning_rate=learning_rate, momentum=cp.getint('hyperparameters', 'momentum'))
    recon_func = theano.function(inputs=[index, batch_size, learning_rate],
                                 outputs=recon_loss, updates=recon_updates,
                                 givens={layer_dict['AAE_Input'].input_var: input_var[
                                     index:index + batch_size, :]}
                                 )
    return recon_func

# if __name__ == '__main__':
#     cp = utils.load_config('../cfg/aae_default.ini')
#     [layer_dict,aae] = build_model(cp)
#     print('collected %d layers' % (len(layer_dict.keys())))
#     for name in layer_dict:
#         print('%s: %r' % (name, layer_dict[name]))
#     print ll.get_all_params(layer_dict['AAE_Output'],trainable=True)
#     print len(ll.get_all_params(layer_dict['AAE_Output'],trainable=True))
