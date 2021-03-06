###############################################################################
# Contains everything that has to do with the AAE as a model (Layer stacking,
# loss functions, etc.)
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
    # Check batch norm flag
    batch_norm_flag = cp.getboolean('Hyperparameters', 'BatchNorm')
    # Initialize activation functions
    relu = lasagne.nonlinearities.rectify
    linear = lasagne.nonlinearities.linear
    sigmoid = lasagne.nonlinearities.sigmoid
    softmax = lasagne.nonlinearities.softmax
    # Make activation dictionary
    act_dict = {'ReLU': relu, 'Linear': linear,
        'Sigmoid': sigmoid, 'Softmax': softmax}
    # Begin stacking layers
    ###########################################################################
    # Encoder part of AE / GAN Generator
    ###########################################################################
    # Input
    input_layer = ae_network = ll.InputLayer(
        shape=(None, cp.getint('AAE_Input', 'Width')), name='AAE_Input')
    # Add batch norm when flag is true
    if batch_norm_flag:
        ae_network = ll.BatchNormLayer(incoming=ae_network)
    # Dropout
    ae_network = ll.DropoutLayer(incoming=ae_network,
                                    p=float(cp.get('Dropout', 'rate')))
    # List of encoder sections in config file
    enc_sect = [i for i in cp.sections() if 'Encoder' in i]
    # Stack endoder layers
    for sect in enc_sect:
        ae_network = ll.DenseLayer(incoming=ae_network,
                                   num_units=cp.getint(sect, 'Width'), nonlinearity=act_dict[cp.get(sect, 'Activation')],
                                   name=sect)
        # Add generator flag that will be used in backward pass
        ae_network.params[ae_network.W].add('generator_z')
        ae_network.params[ae_network.b].add('generator_z')
        ae_network.params[ae_network.W].add('generator_y')
        ae_network.params[ae_network.b].add('generator_y')
        # Add batch norm when flag is true
        if batch_norm_flag: # and (sect != enc_sect[-1]):
            ae_network = ll.BatchNormLayer(incoming=ae_network)
    # Latent variable Y layer also known as q(y|x)
    gen_y = ll.DenseLayer(incoming=ae_network,
                                        num_units=cp.getint('Y', 'Width'), nonlinearity=act_dict[cp.get('Y', 'Activation')],
                                        name='Y')
    # Add generator flag that will be used in backward pass
    gen_y.params[gen_y.W].add('generator_y')
    gen_y.params[gen_y.b].add('generator_y')
    # Latent variable Z layer also known as q(z|x)
    gen_z = ll.DenseLayer(incoming=ae_network,
                                        num_units=cp.getint('Z', 'Width'), nonlinearity=act_dict[cp.get('Z', 'Activation')],
                                        name='Z')
    # Add generator flag that will be used in backward pass
    gen_z.params[gen_z.W].add('generator_z')
    gen_z.params[gen_z.b].add('generator_z')
    # ---- End of Encoder for AE, Generator Z for GAN1 and Generator Y for GAN2----
    ###########################################################################
    # Merging latent label+style representations
    ###########################################################################
    # Save pre-merge layers for Discriminators
    z_dis_net = gen_z
    y_dis_net = gen_y
    # Prepare Y for merging
    gen_y = ll.DenseLayer(incoming=gen_y,
                         num_units=cp.getint('Decoder1', 'Width'), nonlinearity=act_dict['Linear'],
                         b=None,
                         name='PreDecY')
    # Prepare Z for merging
    gen_z = ll.DenseLayer(incoming=gen_z,
                         num_units=cp.getint('Decoder1', 'Width'), nonlinearity=act_dict['Linear'],
                         b=None,
                         name='PreDecZ')
    if batch_norm_flag:
         gen_y = ll.BatchNormLayer(incoming=gen_y)
         gen_z = ll.BatchNormLayer(incoming=gen_z)
    ae_network = ll.ConcatLayer([gen_z, gen_y], name='MergeLatent')
    ###########################################################################
    # Decoder part of AE
    ###########################################################################
    ae_network = ll.DenseLayer(incoming=ae_network,
                             num_units=cp.getint('Decoder1', 'Width'), nonlinearity=act_dict[cp.get('Decoder1', 'Activation')],
                                        name='MergeDec')
    # List of decoder section in config file
    dec_sect = [i for i in cp.sections() if 'Decoder' in i]
    # Stack decoder layers
    for sect in dec_sect:
        ae_network = ll.DenseLayer(incoming=ae_network,
                                   num_units=cp.getint(sect, 'Width'), nonlinearity=act_dict[cp.get(sect, 'Activation')],
                                   name=sect)
    ae_out = ae_network = ll.DenseLayer(incoming=ae_network,
                                        num_units=cp.getint(
                                            'AAE_Output', 'Width'), nonlinearity=act_dict[cp.get('AAE_Output', 'Activation')],
                                        name='AAE_Output')
    # ---- End of Decoder for AE ----
    ###########################################################################
    # Z Discriminator part of GAN
    ###########################################################################
    # z_prior_inp = ll.InputLayer(
    #     shape=(None, cp.getint('Z', 'Width')), name='pz_inp')
    # z_dis_in = z_dis_net = ll.ConcatLayer(
    #     [pre_merge_z, z_prior_inp], axis=0, name='Z_Dis_in')
    z_dis_net = ll.GaussianNoiseLayer(incoming=z_dis_net,sigma=0.3)
    # Stack discriminator layers
    for sect in [i for i in cp.sections() if 'Discriminator' in i]:
        z_dis_net = ll.DenseLayer(incoming=z_dis_net,
                                num_units=cp.getint(sect, 'Width'), nonlinearity=act_dict[cp.get(sect, 'Activation')],
                                name='Z_' + sect)
        # Add generator flag that will be used in backward pass
        z_dis_net.params[z_dis_net.W].add('discriminator_z')
        z_dis_net.params[z_dis_net.b].add('discriminator_z')
    # Z discriminator output
    z_dis_out = z_dis_net = ll.DenseLayer(incoming=z_dis_net,
                                      num_units=cp.getint('Dout', 'Width'), nonlinearity=act_dict[cp.get('Dout', 'Activation')],
                                      name='Z_Dis_out')
    z_dis_out.params[z_dis_out.W].add('discriminator_z')
    z_dis_out.params[z_dis_out.b].add('discriminator_z')
    # ---- End of Z Discriminator ----
    ###########################################################################
    # Y Discriminator part of GAN
    ###########################################################################
    # y_prior_inp = ll.InputLayer(
    #     shape=(None, cp.getint('Y', 'Width')), name='py_inp')
    # y_dis_in = y_dis_net = ll.ConcatLayer(
    #     [pre_merge_y, y_prior_inp], axis=0, name='Y_Dis_in')
    y_dis_net = ll.GaussianNoiseLayer(incoming=y_dis_net,sigma=0.3)
    # Stack discriminator layers
    for sect in [i for i in cp.sections() if 'Discriminator' in i]:
        y_dis_net = ll.DenseLayer(incoming=y_dis_net,
                                num_units=cp.getint(sect, 'Width'), nonlinearity=act_dict[cp.get(sect, 'Activation')],
                                name='Y_' + sect)
        # Add generator flag that will be used in backward pass
        y_dis_net.params[y_dis_net.W].add('discriminator_y')
        y_dis_net.params[y_dis_net.b].add('discriminator_y')
    # Y discriminator output
    y_dis_out = y_dis_net = ll.DenseLayer(incoming=y_dis_net,
                                      num_units=cp.getint('Dout', 'Width'), nonlinearity=act_dict[cp.get('Dout', 'Activation')],
                                      name='Y_Dis_out')
    y_dis_out.params[y_dis_out.W].add('discriminator_y')
    y_dis_out.params[y_dis_out.b].add('discriminator_y')
    # ---- End of Y Discriminator ----
    aae = ll.get_all_layers([ae_out, z_dis_out, y_dis_out])
    layer_dict = {layer.name: layer for layer in aae}
    return layer_dict, aae

# Build pretrained model and load pretrained weights

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

def d_z_discriminate(layer_dict):
    # Symbolic var for learning rate
    dglr = T.scalar('lr')
    # Symbolic input variable
    input_var = T.fmatrix('input_var')
    # Symbolic mini batch variable
    batch = T.fmatrix('batch')
    dis_targets = T.fmatrix('dis_targets')
    # Get discriminator output
    gen_out = ll.get_output(layer_dict['Z'], input_var, deterministic=False)
    dis_out = ll.get_output(layer_dict['Z_Dis_out'],
                            inputs={layer_dict['Z']: gen_out},
                            deterministic=False)
    # Cross entropy regularization term
    dis_loss = T.mean(T.nnet.binary_crossentropy(dis_out, dis_targets))
    dis_params = ll.get_all_params(layer_dict['Z_Dis_out'], trainable=True,
                                   discriminator_z=True)
    dis_updates = lasagne.updates.nesterov_momentum(
        dis_loss, dis_params, learning_rate=dglr, momentum=0.1)
    # Discriminator loss aka Cross Entropy term
    dis_func = theano.function(inputs=[theano.In(batch),
                                       theano.In(dis_targets), dglr],
                               outputs=dis_loss, updates=dis_updates,
                               givens={input_var: batch}
                               )
    return dis_func


def pz_discriminate(layer_dict):
    # Symbolic var for learning rate
    dglr = T.scalar('lr')
    # Symbolic input variable
    input_var = T.fmatrix('input_var')
    # Symbolic mini batch variable
    batch = T.fmatrix('batch')
    dis_targets = T.fmatrix('dis_targets')
    # Get discriminator output
    dis_out = ll.get_output(layer_dict['Z_Dis_out'],
                            inputs={layer_dict['Z']: input_var},
                            deterministic=False)
    # Cross entropy regularization term
    dis_loss = T.mean(T.nnet.binary_crossentropy(dis_out, dis_targets))
    dis_params = ll.get_all_params(layer_dict['Z_Dis_out'], trainable=True,
                                   discriminator_z=True)
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

def g_z_discriminate(layer_dict):
    # Symbolic var for learning rate
    dglr = T.scalar('lr')
    # Symbolic input variable
    input_var = T.fmatrix('input_var')
    # Symbolic mini batch variable
    batch = T.fmatrix('batch')
    gen_targets = T.fmatrix('gen_targets')
    # Get generator output
    gen_out = ll.get_output(layer_dict['Z'], input_var, deterministic=False)
    # Pass generator output to discriminator input
    dis_out = ll.get_output(layer_dict['Z_Dis_out'],
                            inputs={layer_dict['Z']: gen_out},
                            deterministic=False
                            )
    # Entropy regularization term
    gen_loss = T.mean(T.nnet.binary_crossentropy(dis_out, gen_targets))
    gen_params = ll.get_all_params(
        layer_dict['Z_Dis_out'], trainable=True, generator_z=True)
    gen_updates = lasagne.updates.nesterov_momentum(
        gen_loss, gen_params, learning_rate=dglr, momentum=0.1)
    # Generator loss aka Entropy term
    gen_func = theano.function(inputs=[theano.In(batch), theano.In(gen_targets), dglr],
                               outputs=gen_loss, updates=gen_updates,
                               givens={input_var: batch}
                               )
    return gen_func



# Create discriminator objective function also known as the cross entropy between prior
# distribution p(z) and posterior estimate distribution q(z|x)

def d_y_discriminate(layer_dict):
    # Symbolic var for learning rate
    dglr = T.scalar('lr')
    # Symbolic input variable
    input_var = T.fmatrix('input_var')
    # Symbolic mini batch variable
    batch = T.fmatrix('batch')
    dis_targets = T.fmatrix('dis_targets')
    # Get discriminator output
    gen_out = ll.get_output(layer_dict['Y'], input_var, deterministic=False)
    dis_out = ll.get_output(layer_dict['Y_Dis_out'],
                            inputs={layer_dict['Y']: gen_out},
                            deterministic=False)
    # Cross entropy regularization term
    dis_loss = T.mean(T.nnet.binary_crossentropy(dis_out, dis_targets))
    dis_params = ll.get_all_params(layer_dict['Y_Dis_out'], trainable=True,
                                   discriminator_y=True)
    dis_updates = lasagne.updates.nesterov_momentum(
        dis_loss, dis_params, learning_rate=dglr, momentum=0.1)
    # Discriminator loss aka Cross Entropy term
    dis_func = theano.function(inputs=[theano.In(batch),
                                       theano.In(dis_targets), dglr],
                               outputs=dis_loss, updates=dis_updates,
                               givens={input_var: batch}
                               )
    return dis_func


def py_discriminate(layer_dict):
    # Symbolic var for learning rate
    dglr = T.scalar('lr')
    # Symbolic input variable
    input_var = T.fmatrix('input_var')
    # Symbolic mini batch variable
    batch = T.fmatrix('batch')
    dis_targets = T.fmatrix('dis_targets')
    # Get discriminator output
    dis_out = ll.get_output(layer_dict['Y_Dis_out'],
                            inputs={layer_dict['Y']: input_var},
                            deterministic=False)
    # Cross entropy regularization term
    dis_loss = T.mean(T.nnet.binary_crossentropy(dis_out, dis_targets))
    dis_params = ll.get_all_params(layer_dict['Y_Dis_out'], trainable=True,
                                   discriminator_y=True)
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

def g_y_discriminate(layer_dict):
    # Symbolic var for learning rate
    dglr = T.scalar('lr')
    # Symbolic input variable
    input_var = T.fmatrix('input_var')
    # Symbolic mini batch variable
    batch = T.fmatrix('batch')
    gen_targets = T.fmatrix('gen_targets')
    # Get generator output
    gen_out = ll.get_output(layer_dict['Y'], input_var, deterministic=False)
    # Pass generator output to discriminator input
    dis_out = ll.get_output(layer_dict['Y_Dis_out'],
                            inputs={layer_dict['Y']: gen_out},
                            deterministic=False
                            )
    # Entropy regularization term
    gen_loss = T.mean(T.nnet.binary_crossentropy(dis_out, gen_targets))
    gen_params = ll.get_all_params(
        layer_dict['Y_Dis_out'], trainable=True, generator_y=True)
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

# Sample onehot_categorical
# Credits to https://github.com/musyoku/adversarial-autoencoder


def sample_cat(batchsize, num_labels):
	y = np.zeros((batchsize, num_labels), dtype=np.float32)
	indices = np.random.randint(0, num_labels, batchsize)
	for b in range(batchsize):
		y[b, indices[b]] = 1
	return y

if __name__ == "__main__":
    import utils
    cp = utils.load_config('../cfg/clustering/normal.ini')
    from draw_net import *
    draw_to_file(build_model(cp)[1], 'test.pdf', verbose=True)
