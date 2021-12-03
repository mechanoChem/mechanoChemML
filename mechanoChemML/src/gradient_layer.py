import keras
import keras.backend as K
from keras.layers import Lambda, Concatenate

class Gradient(keras.layers.Layer):
  """
  Gradient is a custom keras layer.
  It is called with a list consisting of a function and the inputs,
  e.g. Gradient()([f(x),x])

  Note that if x is an array of length n
  and f(x) returns an array of length m,
  this layer will return a 1D array of length m*n,
  instead of a 2D array
  i.e. the result is always flattened

  The df_i/dx_j component will be at the n*i+j location
  of the returned array.
  """

  def __init__(self, **kwargs):
    super(Gradient, self).__init__(**kwargs)

  def call(self, ins):
    """ The user interface function, used to call this layer

    :param ins: The inputs are a list with two components, consisting of a keras model or layer and the keras input layer, e.g. [output,x]

    """

    if isinstance(ins[0], list):
      y = ins[0][0]
    else:
      y = ins[0] #y = f(x)
    x = ins[1] #x

    if y.shape[-1] == 1:
      return K.gradients(y,x)
    else:
      gradVec = []
      for i in range(y.shape[-1]):
        gradVec += [K.gradients(y[:,i],x)[0]]
      return K.concatenate(gradVec)

  def compute_output_shape(self, input_shape):
    return (None,input_shape[0][1]*input_shape[1][1])

