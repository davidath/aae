###############################################################################
# Model template is enabling the use of neural network and other models
# to be used after training.
###############################################################################

import numpy as np
import lasagne
import theano
import utils

class Model(object):

  def __init__(self, layer_dict, aae):
      self._input_layer = layer_dict['AAE_Input']
      self._z_encoder_layer = layer_dict['Z']
      self._y_encoder_layer = layer_dict['Y']
      self._decoder_layer = layer_dict['AAE_Output']
    #   self._z_discriminator_input = layer_dict['Z_Dis_in']
    #   self._y_discriminator_input = layer_dict['Y_Dis_in']
      self._z_discriminator_output = layer_dict['Z_Dis_out']
      self._y_discriminator_output = layer_dict['Y_Dis_out']
      self._network = aae
      self._layer_dict = layer_dict

  def __eq__(self, other):
      this_param = lasagne.layers.get_all_param_values(self._network)
      other_param = lasagne.layers.get_all_param_values(other._network)
      for i,val in enumerate(this_param):
          if not(np.array_equal(this_param[i],other_param[i])):
              return False
      return True

  def get_hidden(self, dataset):
      self._input_layer.input_var = theano.shared(name='input_var',
                                               value=np.asarray(dataset,
                                               dtype=theano.config.floatX),
                                               borrow=True)
      hidden = lasagne.layers.get_output(self._z_encoder_layer).eval()
      return hidden


  def get_output(self, dataset):
      self._input_layer.input_var = theano.shared(name='input_var',
                                                   value=np.asarray(dataset,
                                                   dtype=theano.config.floatX),
                                                   borrow=True)
      output = lasagne.layers.get_output(self._decoder_layer).eval()
      return output

  def save(self,filename='model_template.zip'):
      utils.save(filename,self)

  def load(self,filename='model_template.zip'):
      self = utils.load_single(filename)
