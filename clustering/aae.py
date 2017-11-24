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
    # Input
    input_layer = ae_network = ll.InputLayer(
        shape=(None, cp.getint('AAE_Input', 'Width')), name='AAE_Input')
    # Add batch norm when flag is true
    if batch_norm_flag:
        ae_network = ll.BatchNormLayer(incoming=ae_network)
    # Dropout
    # ae_network = ll.DropoutLayer(incoming=ae_network,
    #                                 p=float(cp.get('Dropout', 'rate')))
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
        if batch_norm_flag and (sect != enc_sect[-1]):
            ae_network = ll.BatchNormLayer(incoming=ae_network)
    # Latent variable Y layer also known as q(y|x)
    gen_y = ll.DenseLayer(incoming=ae_network,
                                        num_units=cp.getint('Y', 'Width'), nonlinearity=act_dict[cp.get('Y', 'Activation')],
                                        name='Y')
    # Add generator flag that will be used in backward pass
    gen_y.params[gen_y.W].add('generator_y')
    gen_y.params[gen_y.b].add('generator_y')
    # gen_y = ll.NonlinearityLayer(incoming=gen_y, nonlinearity=act_dict['Softmax'])
    # Latent variable Z layer also known as q(z|x)
    gen_z = ll.DenseLayer(incoming=ae_network,
                                        num_units=cp.getint('Z', 'Width'), nonlinearity=act_dict[cp.get('Z', 'Activation')],
                                        name='Z')
    # Add generator flag that will be used in backward pass
    gen_z.params[gen_z.W].add('generator_z')
    gen_z.params[gen_z.b].add('generator_z')
    # ---- End of Encoder for AE, Generator Z for GAN1 and Generator Y for GAN2----
    # Save pre-merge layers for Discriminators
    pre_merge_z = gen_z
    pre_merge_y = gen_y
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
    # ae_network = ll.ElemwiseSumLayer([gen_z, gen_y], name='MergeDec')
    ae_network = ll.ConcatLayer([gen_z, gen_y], name='MergeDec')
    ae_network = ll.DenseLayer(incoming=ae_network,
                             num_units=cp.getint('Decoder1', 'Width'), nonlinearity=act_dict[cp.get('Decoder1', 'Activation')],
                                        name='M')
    # Add batch norm when flag is true
    # if batch_norm_flag:
    #     ae_network = ll.BatchNormLayer(incoming=ae_network)
    # List of decoder section in config file
    dec_sect = [i for i in cp.sections() if 'Decoder' in i]
    # Stack decoder layers
    for sect in dec_sect:
        ae_network = ll.DenseLayer(incoming=ae_network,
                                   num_units=cp.getint(sect, 'Width'), nonlinearity=act_dict[cp.get(sect, 'Activation')],
                                   name=sect)
        # if batch_norm_flag and (sect != dec_sect[-1]):
        #     ae_network = ll.BatchNormLayer(incoming=ae_network)
    ae_out = ae_network = ll.DenseLayer(incoming=ae_network,
                                        num_units=cp.getint(
                                            'AAE_Output', 'Width'), nonlinearity=act_dict[cp.get('AAE_Output', 'Activation')],
                                        name='AAE_Output')
    # ---- End of Decoder for AE ----
    z_prior_inp = ll.InputLayer(
        shape=(None, cp.getint('Z', 'Width')), name='pz_inp')
    z_dis_in = z_dis_net = ll.ConcatLayer(
        [pre_merge_z, z_prior_inp], axis=0, name='Z_Dis_in')
    z_dis_net = ll.GaussianNoiseLayer(incoming=z_dis_net,sigma=0.3)
    # Stack discriminator layers
    for sect in [i for i in cp.sections() if 'Discriminator' in i]:
        z_dis_net = ll.DenseLayer(incoming=z_dis_net,
                                num_units=cp.getint(sect, 'Width'), nonlinearity=act_dict[cp.get(sect, 'Activation')],
                                name='Z_' + sect)
        # Add generator flag that will be used in backward pass
        z_dis_net.params[z_dis_net.W].add('discriminator_z')
        z_dis_net.params[z_dis_net.b].add('discriminator_z')
        # if batch_norm_flag:
        #     z_dis_net = ll.BatchNormLayer(incoming=z_dis_net)
    # Z discriminator output
    z_dis_out = z_dis_net = ll.DenseLayer(incoming=z_dis_net,
                                      num_units=cp.getint('Dout', 'Width'), nonlinearity=act_dict[cp.get('Dout', 'Activation')],
                                      name='Z_Dis_out')
    z_dis_out.params[z_dis_out.W].add('discriminator_z')
    z_dis_out.params[z_dis_out.b].add('discriminator_z')
    # ---- End of Z Discriminator ----
    y_prior_inp = ll.InputLayer(
        shape=(None, cp.getint('Y', 'Width')), name='py_inp')
    y_dis_in = y_dis_net = ll.ConcatLayer(
        [pre_merge_y, y_prior_inp], axis=0, name='Y_Dis_in')
    y_dis_net = ll.GaussianNoiseLayer(incoming=y_dis_net,sigma=0.3)
    # Stack discriminator layers
    for sect in [i for i in cp.sections() if 'Discriminator' in i]:
        y_dis_net = ll.DenseLayer(incoming=y_dis_net,
                                num_units=cp.getint(sect, 'Width'), nonlinearity=act_dict[cp.get(sect, 'Activation')],
                                name='Y_' + sect)
        # Add generator flag that will be used in backward pass
        y_dis_net.params[y_dis_net.W].add('discriminator_y')
        y_dis_net.params[y_dis_net.b].add('discriminator_y')
        # Add batch norm when flag is true
        # if batch_norm_flag:
        #     y_dis_net = ll.BatchNormLayer(incoming=y_dis_net)
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


def load_pretrained(cp, weights):
    aae = build_model(cp)[1]
    ll.set_all_param_values(aae, weights)
    layer_dict = {layer.name: layer for layer in aae}
    return layer_dict, aae

def copy_net(aae1, aae2):
    batch_norm_params = ['gamma','beta','inv_std','mean']
    idx = [pos for pos,i in enumerate(ll.get_all_params(aae2)) if i.name not in batch_norm_params]
    no_norm_params = [ll.get_all_param_values(aae2)[i] for i in idx]
    ll.set_all_param_values(aae1, no_norm_params)
    layer_dict = {layer.name: layer for layer in aae1}
    return layer_dict, aae1

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

# forward/backward (optional) pass for Z_discriminator
def z_discriminator_loss(layer_dict):
    # Symbolic var for learning rate
    dglr = T.scalar('lr')
    # Symbolic input variable
    input_var = T.fmatrix('input_var')
    # Symbolic mini batch variable
    batch = T.fmatrix('batch')
    # Symbolic samples from p(z) prior
    pz = T.fmatrix('pz')
    # Symbolic batch for p(z)
    pz_batch = T.fmatrix('pz_batch')
    # Get discriminator output
    dis_out = ll.get_output(layer_dict['Z_Dis_out'],
                            inputs={layer_dict['pz_inp']: pz,
                                    layer_dict['AAE_Input']: input_var},
                            deterministic=False)
    # Fake/Real (0/1) labels for discriminator loss
    dis_targets = T.vertical_stack(
        T.ones((batch.shape[0], 1)),
        T.zeros((pz_batch.shape[0], 1))
    )
    # Cross entropy regularization term
    dis_loss = T.mean(T.nnet.binary_crossentropy(dis_out, dis_targets))
    dis_params = ll.get_all_params(layer_dict['Z_Dis_out'], trainable=True,
                                   discriminator_z=True)
    dis_updates = lasagne.updates.nesterov_momentum(
        dis_loss, dis_params, learning_rate=dglr, momentum=0.1)
    # Discriminator loss aka Cross Entropy term
    dis_func = theano.function(inputs=[theano.In(batch),
                                       theano.In(pz_batch), dglr],
                               outputs=dis_loss, updates=dis_updates,
                               givens={input_var: batch, pz: pz_batch}
                               )
    return dis_func

# Create discriminator objective function also known as the cross entropy between prior
# distribution p(y) and posterior estimate distribution q(y|x)

# forward/backward (optional) pass for Y_discriminator


def y_discriminator_loss(layer_dict):
    # Symbolic var for learning rate
    dglr = T.scalar('lr')
    # Symbolic input variable
    input_var = T.fmatrix('input_var')
    # Symbolic mini batch variable
    batch = T.fmatrix('batch')
    # Symbolic samples from p(z) prior
    py = T.fmatrix('py')
    # Symbolic batch for p(z)
    py_batch = T.fmatrix('py_batch')
    # Get discriminator output
    dis_out = ll.get_output(layer_dict['Y_Dis_out'],
                            inputs={layer_dict['py_inp']: py,
                                    layer_dict['AAE_Input']: input_var},
                            deterministic=False)
    # Fake/Real (0/1) labels for discriminator loss
    dis_targets = T.vertical_stack(
        T.ones((batch.shape[0], 1)),
        T.zeros((py_batch.shape[0], 1))
    )
    # Cross entropy regularization term
    dis_loss = T.mean(T.nnet.binary_crossentropy(dis_out, dis_targets))
    dis_params = ll.get_all_params(layer_dict['Y_Dis_out'], trainable=True,
                                   discriminator_y=True)
    dis_updates = lasagne.updates.nesterov_momentum(
        dis_loss, dis_params, learning_rate=dglr, momentum=0.1)
    # Discriminator loss aka Cross Entropy term
    dis_func = theano.function(inputs=[theano.In(batch),
                                       theano.In(py_batch), dglr],
                               outputs=dis_loss, updates=dis_updates,
                               givens={input_var: batch, py: py_batch}
                               )
    return dis_func


# Create generator objective function also known as the entropy of the posterior
# estimate distribution q(z|x)

def z_generator_loss(layer_dict):
    # Symbolic var for learning rate
    dglr = T.scalar('lr')
    # Symbolic input variable
    input_var = T.fmatrix('input_var')
    # Symbolic mini batch variable
    batch = T.fmatrix('batch')
    # Get generator output
    gen_out = ll.get_output(layer_dict['Z'], input_var, deterministic=False)
    # Pass generator output to discriminator input
    dis_out = ll.get_output(layer_dict['Z_Dis_out'],
                            inputs={layer_dict['Z_Dis_in']: gen_out},
                            deterministic=False
                            )
    # Create generator targets, confuse discriminator
    gen_targets = T.zeros_like(batch.shape[0])
    # Entropy regularization term
    gen_loss = T.mean(T.nnet.binary_crossentropy(dis_out, gen_targets))
    gen_params = ll.get_all_params(
        layer_dict['Z_Dis_out'], trainable=True, generator_z=True)
    gen_updates = lasagne.updates.nesterov_momentum(
        gen_loss, gen_params, learning_rate=dglr, momentum=0.1)
    # Generator loss aka Entropy term
    gen_func = theano.function(inputs=[theano.In(batch), dglr],
                               outputs=gen_loss, updates=gen_updates,
                               givens={input_var: batch}
                               )
    return gen_func


# Create generator objective function also known as the entropy of the posterior
# estimate distribution q(y|x)

def y_generator_loss(layer_dict):
    # Symbolic var for learning rate
    dglr = T.scalar('lr')
    # Symbolic input variable
    input_var = T.fmatrix('input_var')
    # Symbolic mini batch variable
    batch = T.fmatrix('batch')
    # Get generator output
    gen_out = ll.get_output(layer_dict['Y'], input_var, deterministic=False)
    # Pass generator output to discriminator input
    dis_out = ll.get_output(layer_dict['Y_Dis_out'],
                            inputs={layer_dict['Y_Dis_in']: gen_out},
                            deterministic=False
                            )
    # Create generator targets, confuse discriminator
    gen_targets = T.zeros((batch.shape[0], 1))
    # Entropy regularization term
    gen_loss = T.mean(T.nnet.binary_crossentropy(dis_out, gen_targets))
    gen_params = ll.get_all_params(
        layer_dict['Y_Dis_out'], trainable=True, generator_y=True)
    gen_updates = lasagne.updates.nesterov_momentum(
        gen_loss, gen_params, learning_rate=dglr, momentum=0.1)
    # Generator loss aka Entropy term
    gen_func = theano.function(inputs=[theano.In(batch), dglr],
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
