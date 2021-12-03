import os, sys
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.compat.v1 as tf1
import tensorflow.keras.backend as K

import mechanoChemML.src.pde_layers as pde_layers

"""Build NN models based on a list of layers provided from configuration file."""

def _build_one_layer(layer_dict, kl_divergence_function=None):
    """ 
    Return one keras layer based on the layer dictionary 

    Args:
        layer_dict (dict): a dictionary contains configurations of a Keras layer
        kl_divergence_function: scaled kl_divergence_function (default: None)

    Returns:
        A Keras layer

    Note:
        The following layers are supported:

        - BatchNormalization
        - Conv2D
        - Convolution2DFlipout
        - Convolution2DReparameterization
        - Dense
        - DenseFlipout
        - DenseReparameterization
        - Flatten
        - GaussianNoise
        - MaxPooling2D
        - PDERandom
        - Reshape
        - UpSampling2D
        - PDEZero

    """
    args = [] 
    tfkl = tf.keras.layers
    tfpl = tfp.layers
    tfd = tfp.distributions
    val_init = 0.1
    stddev_init = 0.2
    stddev_init = 0.1 # TF probability default value
    # remember to add the positional argument default value to the related function: add_layer_default_argument(layer_dict)
    if layer_dict['type'] == 'Convolution2DFlipout' :
        if 'padding' not in layer_dict: layer_dict['padding'] = 'valid'
        if 'activation' not in layer_dict: layer_dict['activation'] = None
        return tfpl.Convolution2DFlipout(
                filters=layer_dict['filters'], 
                kernel_size=layer_dict['kernel_size'], 
                activation=layer_dict['activation'], 
                kernel_divergence_fn=kl_divergence_function,
                bias_divergence_fn=kl_divergence_function,
                padding=layer_dict['padding'],
                kernel_posterior_fn=tfpl.default_mean_field_normal_fn(
                    loc_initializer=tf1.initializers.random_uniform(minval=-val_init, maxval=val_init), 
                    untransformed_scale_initializer=tf1.initializers.random_normal(mean=-3., stddev=stddev_init), 
                    ),
                bias_posterior_fn=tfpl.default_mean_field_normal_fn(
                    is_singular=True, # very important
                    loc_initializer=tf1.initializers.random_uniform(minval=-val_init, maxval=val_init),
                    untransformed_scale_initializer=tf1.initializers.random_normal(mean=-3., stddev=stddev_init), 
                    ),
                )
    elif layer_dict['type'] == 'Convolution2DReparameterization' :
        if 'padding' not in layer_dict: layer_dict['padding'] = 'valid'
        if 'activation' not in layer_dict: layer_dict['activation'] = None
        return tfpl.Convolution2DReparameterization(
                filters=layer_dict['filters'], 
                kernel_size=layer_dict['kernel_size'], 
                activation=layer_dict['activation'], 
                padding=layer_dict['padding'],
                kernel_divergence_fn=kl_divergence_function,
                bias_divergence_fn=kl_divergence_function,
                )
    elif layer_dict['type'] == 'Conv2D' :
        return tfkl.Conv2D(
                filters=layer_dict['filters'], 
                kernel_size=layer_dict['kernel_size'], 
                activation=layer_dict['activation'], 
                padding=layer_dict['padding'],
                )
    elif layer_dict['type'] == 'MaxPooling2D' :
        if 'padding' not in layer_dict: layer_dict['padding'] = 'valid'
        if 'strides' not in layer_dict: layer_dict['strides'] = None

        return tfkl.MaxPooling2D(
                pool_size=layer_dict['pool_size'], 
                padding=layer_dict['padding'],
                strides=layer_dict['strides'],
                )
    elif layer_dict['type'] == 'BatchNormalization' :
        return tfkl.BatchNormalization()
    elif layer_dict['type'] == 'GaussianNoise' :
        return tfkl.GaussianNoise(float(layer_dict['stddev']))
    elif layer_dict['type'] == 'GaussianDropout' :
        return tfkl.GaussianDropout(float(layer_dict['rate']))
    elif layer_dict['type'] == 'Flatten' :
        return tfkl.Flatten()
    elif layer_dict['type'] == 'DenseFlipout' :

        if 'activation' not in layer_dict: layer_dict['activation'] = None
        return tfpl.DenseFlipout(
                units=layer_dict['units'], 
                kernel_divergence_fn=kl_divergence_function,
                bias_divergence_fn=kl_divergence_function,
                activation=layer_dict['activation'], 
                kernel_posterior_fn=tfpl.default_mean_field_normal_fn(
                    loc_initializer=tf1.initializers.random_uniform(minval=-val_init, maxval=val_init),
                    untransformed_scale_initializer=tf1.initializers.random_normal(mean=-3., stddev=stddev_init), 
                    ),
                bias_posterior_fn=tfpl.default_mean_field_normal_fn(
                    is_singular=True,
                    loc_initializer=tf1.initializers.random_uniform(minval=-val_init, maxval=val_init),
                    untransformed_scale_initializer=tf1.initializers.random_normal(mean=-3., stddev=stddev_init), 
                    ),
                )
    elif layer_dict['type'] == 'DenseReparameterization' :
        if 'activation' not in layer_dict: layer_dict['activation'] = None
        return tfpl.DenseReparameterization(
                units=layer_dict['units'], 
                kernel_divergence_fn=kl_divergence_function,
                bias_divergence_fn=kl_divergence_function,
                activation=layer_dict['activation'], 
                )
    elif layer_dict['type'] == 'Dense' :
        if 'activation' not in layer_dict: layer_dict['activation'] = None
        return tfkl.Dense(
                units=layer_dict['units'], 
                activation=layer_dict['activation'], 
                )
    elif layer_dict['type'] == 'Reshape' :
        try:
            if layer_dict['input_shape'] != None:
                # print('reshape:', layer_dict['input_shape'])
                return tfkl.Reshape(target_shape=layer_dict['target_shape'], input_shape=layer_dict['input_shape'])
            else:
                return tfkl.Reshape(target_shape=layer_dict['target_shape'])
        except:
            return tfkl.Reshape(target_shape=layer_dict['target_shape'])
    elif layer_dict['type'] == 'UpSampling2D' :
        return tfkl.UpSampling2D(size=layer_dict['size'])
    elif layer_dict['type'] == 'PDERandom' :
        return pde_layers.LayerFillRandomNumber(name='input')
    elif layer_dict['type'] == 'PDEZero' :
        return pde_layers.LayerFillZeros(name='input')
    else:
        return ValueError ('The layer type = ' + layer_dict['type'] + ' is not coded yet! Please add it by yourself.')


def _is_digit(str0):
    """ 
    Check if a string is digit or not

    Args:
        str0 (str): a string

    Returns:
        bool: True if is digit, false if not.

    """
    return str0.isdigit()

def _is_tuple(str0):
    """ 
    Check if a string is tuple or not

    Args:
        str0 (str): a string

    Returns:
        bool: True if is digit, false if not.

    """
    if str0[0] == '(' and str0[-1] == ')':
        return True
    else:
        return False

def _is_list(str0):
    """ 
    Check if a string is list or not

    Args:
        str0 (str): a string

    Returns:
        bool: True if is digit, false if not.

    """
    if str0[0] == '[' and str0[-1] == ']':
        return True
    else:
        return False

def _form_parameter_value(str0):
    """ 
    Convert a string to the proper type (int, tuple, list, str)

    Args:
        str0 (str): a string

    Returns:
        Variable with the proper type (int, tuple, list, str)

    """
    if _is_digit(str0):
        return int(str0)
    elif _is_tuple(str0):
        return tuple(int(s) for s in str0.strip("()").split(","))
    elif _is_list(str0):
        return list(int(s) for s in str0.strip("[]").split(","))
    else : 
        return str0

def _form_NN_dict_from_str(str0):
    """
    Form a list with each item being a dictionary containing the layer configuration

    Args:
        str0 (str): a string
    
    Returns:
        list_of_layers_dict (dict): a list of layer dictionary

    Notes:
        - keys of the layer dictionary: 'type', 'activation', 'unit', 'padding', etc
        - the keys are different for different Keras layers. 
        - the keys are defined based on the argument name of each Keras layer

    """
    list_of_layers = [ x.strip() for x in str0.split(';') if x.strip()]
    list_of_layers_dict = []
    for s0 in list_of_layers:
        one_layer = {}
        list_of_parameters = [ x.strip() for x in s0.split('|') if x.strip()]
        # print(list_of_parameters)
        for p0 in list_of_parameters:
            _p0 = [ x.strip() for x in p0.split('=') if x.strip()]
            one_layer[_p0[0]] = _form_parameter_value(_p0[1])
        # print(one_layer)
        list_of_layers_dict.append(one_layer)
    return list_of_layers_dict

class BNN_user_weak_pde_general(tf.keras.Model):
    """ 
    User defined general weak-pde constrained BNNs. Automatically create a sequential BNN model based on the list of layers.

    Args:
        layers_str (str): a string contains all info of layers defining the NNs.
        NUM_TRAIN_EXAMPLES (int): scale factor for the kl-loss. See more explanation: https://www.tensorflow.org/probability/api_docs/python/tfp/layers/Convolution2DFlipout
        Sigma2 (float): initial value for the variance of residual. Used in the loss.
    """

    def __init__(self, layers_str, NUM_TRAIN_EXAMPLES, Sigma2=1.0e-4):
        super(BNN_user_weak_pde_general, self).__init__()
        isBNN = False
        if layers_str.find('Flipout') >= 0:
            isBNN = True

        self.list_of_layers_dict = _form_NN_dict_from_str(layers_str)

        self.NUM_TRAIN_EXAMPLES = NUM_TRAIN_EXAMPLES
        self.Sigma2 = tf.Variable(Sigma2, trainable=isBNN)

        tfd = tfp.distributions
        kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
                                  tf.cast(self.NUM_TRAIN_EXAMPLES, dtype=tf.float32))

        # 'all_layers' prefix is needed for BNN warm start indexing
        # random 1st layer
        self.all_layers = [_build_one_layer(self.list_of_layers_dict[0])]
        for l0 in self.list_of_layers_dict[1:]:
            self.all_layers.append(_build_one_layer(l0, kl_divergence_function))

    def call(self, inputs, training=False):
        """ 
        Execute each layer: See https://www.tensorflow.org/api_docs/python/tf/keras/Model

        Args:
            inputs: a keras.Input object or list of keras.Input objects.
            training (bool): One can use it to specify a different behavior in training and inference. 
        """
        x = self.all_layers[0](inputs)
        for hl in self.all_layers[1:]:
            x = hl(x)
        return tf.concat([x, inputs], 3)

class BNN_user_weak_pde_general_heter(tf.keras.Model):
    """ 
    User defined general weak-pde constrained BNNs. Automatically create a sequential BNN model based on the list of layers.

    Heterogeneous inputs with [image, scalar]

    Args:
        layers_str (str): a string contains all info of layers defining the NNs.
        NUM_TRAIN_EXAMPLES (int): scale factor for the kl-loss. See more explanation: https://www.tensorflow.org/probability/api_docs/python/tfp/layers/Convolution2DFlipout
        Sigma2 (float): initial value for the variance of residual. Used in the loss.
    """

    def __init__(self, layers_str, NUM_TRAIN_EXAMPLES, Sigma2=1.0e-4):
        super(BNN_user_weak_pde_general_heter, self).__init__()
        isBNN = False
        if layers_str.find('Flipout') >= 0:
            isBNN = True

        self.list_of_layers_dict = _form_NN_dict_from_str(layers_str)

        self.NUM_TRAIN_EXAMPLES = NUM_TRAIN_EXAMPLES
        self.Sigma2 = tf.Variable(Sigma2, trainable=isBNN)

        tfd = tfp.distributions
        kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
                                  tf.cast(self.NUM_TRAIN_EXAMPLES, dtype=tf.float32))

        def merge_two_tensor(a):
            return K.concatenate([a[0], a[1]], axis=1)

        # 'all_layers' prefix is needed for BNN warm start indexing
        # naming of part1, part2 should follow by order to avoid issue in warm start
        # random 1st layer
        self.all_layers_part1 = [_build_one_layer(self.list_of_layers_dict[0])]
        self.all_layers_part2 = [tf.keras.layers.Lambda(merge_two_tensor)]
        build_decoder = False
        for l0 in self.list_of_layers_dict[1:]:
            if l0['type'].find('Dense') >= 0 :
                build_decoder = True
            # the additional parameters will account for information of num_parameters/(num_parameters+dense unit)
            # it would not be a small fraction. And this will make sure that parameter information is well
            # blended into the whole NN structure.
            if build_decoder:
                self.all_layers_part2.append(_build_one_layer(l0, kl_divergence_function))
            else:
                self.all_layers_part1.append(_build_one_layer(l0, kl_divergence_function))
        # print('encoder:', self.all_layers_part1)
        # print('decoder:', self.all_layers_part2)
        # exit(0)
        self.pde_parameters = None

    def call(self, inputs, training=False):
        """ 
        Execute each layer: See https://www.tensorflow.org/api_docs/python/tf/keras/Model

        It is almost impossible to pass a scalar out without sacrificing the data format.
        New function call is defined to pass such data.

        Args:
            inputs: a keras.Input object or list of keras.Input objects.
            training (bool): One can use it to specify a different behavior in training and inference. 
        """
        x = self.all_layers_part1[0](inputs[0])
        for hl in self.all_layers_part1[1:]:
            x = hl(x)
        # combine parameters with dense layer
        y = self.all_layers_part2[0]([inputs[1], x])
        for hl in self.all_layers_part2[1:]:
            y = hl(y)
        self.pde_parameters = inputs[1]
        return tf.concat([y, inputs[0]], 3)

    def get_pde_parameters(self):
        """
        Pass scalar parameters from inputs to output
        """
        return self.pde_parameters

class BNN_user_general(tf.keras.Model):
    """ 
    User defined general BNNs. Automatically create a sequential BNN model based on the list of layers.

    Args:
        layers_str (str): a string contains all info of layers defining the NNs.
        NUM_TRAIN_EXAMPLES (int): scale factor for the kl-loss. See more explanation: https://www.tensorflow.org/probability/api_docs/python/tfp/layers/Convolution2DFlipout
    """

    def __init__(self, layers_str, NUM_TRAIN_EXAMPLES):
        super(BNN_user_general, self).__init__()
        self.list_of_layers_dict = _form_NN_dict_from_str(layers_str)

        self.NUM_TRAIN_EXAMPLES = NUM_TRAIN_EXAMPLES
        tfd = tfp.distributions
        kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
                                  tf.cast(self.NUM_TRAIN_EXAMPLES, dtype=tf.float32))

        # random 1st layer
        self.all_layers = [_build_one_layer(self.list_of_layers_dict[0])]
        for l0 in self.list_of_layers_dict[1:]:
            self.all_layers.append(_build_one_layer(l0, kl_divergence_function))

    def call(self, inputs, training=False):
        """ 
        Execute each layer: See https://www.tensorflow.org/api_docs/python/tf/keras/Model

        Args:
            inputs: a keras.Input object or list of keras.Input objects.
            training (bool): One can use it to specify a different behavior in training and inference. 
        """

        x = self.all_layers[0](inputs)
        
        for hl in self.all_layers[1:]:
            x = hl(x)
        return x


class NN_user_general(tf.keras.Model):
    """ 
    User defined general NNs. Automatically create a sequential NN model based on the list of layers.

    Args:
        layers_str (str): a string contains all info of layers defining the NNs.
    """


    def __init__(self, layers_str):
        super(NN_user_general, self).__init__()
        self.list_of_layers_dict = _form_NN_dict_from_str(layers_str)

        # random 1st layer
        self.all_layers = [_build_one_layer(self.list_of_layers_dict[0])]
        for l0 in self.list_of_layers_dict[1:]:
            self.all_layers.append(_build_one_layer(l0))

    def call(self, inputs, training=False):
        """ 
        Execute each layer: See https://www.tensorflow.org/api_docs/python/tf/keras/Model

        Args:
            inputs: a keras.Input object or list of keras.Input objects.
            training (bool): One can use it to specify a different behavior in training and inference. 
        """
        x = self.all_layers[0](inputs)
        
        for hl in self.all_layers[1:]:
            x = hl(x)

        return x

def merge_two_tensor(a):
    return K.concatenate([a[0], a[1]], axis=1)

class BNN_user_weak_pde_general_dynamic(tf.keras.Model):
    """ 
    User defined general weak-pde constrained BNNs. Automatically create a sequential BNN model based on the list of layers.

    Args:
        layers_str (str): a string contains all info of layers defining the NNs.
        NUM_TRAIN_EXAMPLES (int): scale factor for the kl-loss. See more explanation: https://www.tensorflow.org/probability/api_docs/python/tfp/layers/Convolution2DFlipout
        Sigma2 (float): initial value for the variance of residual. Used in the loss.
    """

    def __init__(self, layers_str, NUM_TRAIN_EXAMPLES, Sigma2=1.0e-4):
        super(BNN_user_weak_pde_general_dynamic, self).__init__()
        isBNN = False
        if layers_str.find('Flipout') >= 0:
            isBNN = True

        self.list_of_layers_dict = _form_NN_dict_from_str(layers_str)

        self.NUM_TRAIN_EXAMPLES = NUM_TRAIN_EXAMPLES
        self.Sigma2 = tf.Variable(Sigma2, trainable=isBNN)

        tfd = tfp.distributions
        kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(q, p) /  # pylint: disable=g-long-lambda
                                  tf.cast(self.NUM_TRAIN_EXAMPLES, dtype=tf.float32))

        # random 1st layer
        self.all_layers = [_build_one_layer(self.list_of_layers_dict[0])]
        ind0 = 1
        for l0 in self.list_of_layers_dict[1:]:
            self.all_layers.append(_build_one_layer(l0, kl_divergence_function))
            # print(l0)
            if l0['type'] == 'Flatten':
                self.flatten_index = ind0
            ind0 += 1

        # self.merge_layer = tf.keras.layers.Lambda(merge_two_tensor)

    def call(self, inputs, training=False):
        """ 
        Execute each layer: See https://www.tensorflow.org/api_docs/python/tf/keras/Model

        Args:
            inputs: a keras.Input object or list of keras.Input objects.
            training (bool): One can use it to specify a different behavior in training and inference. 
        """
        inp0 = inputs[0] # [Dirichlet, Neumann, Initial]
        inp1 = inputs[1] # [NN] batch_x_time

        x = self.all_layers[0](inp0)
        for hl in self.all_layers[1:self.flatten_index+1]:
            # print(hl)
            x = hl(x)
        # print('-----------------')
        # x = self.merge_layer([inp1, x])  #,
        for hl in self.all_layers[self.flatten_index+1:]:
            # print(hl)
            x = hl(x) # size of x is determined by the NN structure from config.ini file.

        # current_time is in the size of [batch, :, :, 1]
        current_time = tf.expand_dims(inp1,axis=1)
        current_time = tf.expand_dims(current_time,axis=1)
        current_time = tf.multiply(tf.ones_like(inp0[:,:,:,0:1]), current_time)

        # channels = outputs + 3*dof + 1
        return tf.concat([x, inp0, current_time], 3)


if __name__ == '__main__':

    """ example for setting up an encoder-decoder structure with deterministic layers """
    example_NN = """ 
    type=PDERandom;
    type=Conv2D | filters=8 | kernel_size=5 | activation=relu | padding=same;
    type=MaxPooling2D | pool_size=(2,2) | padding=same;
    type=Conv2D | filters=16 | kernel_size=5 | activation=relu | padding=same;
    type=MaxPooling2D | pool_size=(2,2) | padding=same;
    type=Flatten;
    type=Dense | units=64 | activation=relu;
    type=Dense | units=32 | activation=relu;
    type=Reshape | target_shape=[4,4,2];
    type=Conv2D | filters=8 | kernel_size=5 | activation=relu | padding=same;
    type=UpSampling2D | size=(2,2);
    type=Conv2D | filters=8 | kernel_size=5 | activation=relu | padding=same;
    type=Conv2D | filters=1 | kernel_size=5 | activation=relu | padding=same;
    """

    model = NN_user_general(example_NN)
    input_shape=(None, 16, 16, 1)
    model.build(input_shape) 
    model.summary()

    """ example for setting up an encoder-decoder structure with probabilistic layers """
    example_BNN = """ 
    type=PDERandom;
    type=Convolution2DFlipout | filters=8 | kernel_size=5 | activation=relu | padding=same;
    type=MaxPooling2D | pool_size=(2,2) | padding=same;
    type=Convolution2DFlipout | filters=16 | kernel_size=5 | activation=relu | padding=same;
    type=MaxPooling2D | pool_size=(2,2) | padding=same;
    type=Flatten;
    type=DenseFlipout | units=64 | activation=relu;
    type=DenseFlipout | units=32 | activation=relu;
    type=Reshape | target_shape=[4,4,2];
    type=Convolution2DFlipout | filters=8 | kernel_size=5 | activation=relu | padding=same;
    type=UpSampling2D | size=(2,2);
    type=Convolution2DFlipout | filters=8 | kernel_size=5 | activation=relu | padding=same;
    type=Convolution2DFlipout | filters=1 | kernel_size=5 | activation=relu | padding=same;
     """
    model = BNN_user_general(example_BNN, NUM_TRAIN_EXAMPLES=16)
    input_shape=(None, 16, 16, 1)
    model.build(input_shape) 
    model.summary()
