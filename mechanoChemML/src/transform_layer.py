import keras
from keras.layers import Lambda
import keras.backend as K
import marshal, base64, types
import numpy as np

#def _Transform(transforms):
#  """ Function to define a Keras transformation layer
#
#  :param transforms: A function that takes the input array x and returns a list of transformation outputs
#  :type transforms: func 
#
#  :returns: A Keras Transformation layer
#
#  """
#  
#  def func(x):
#    y = Lambda(transforms)(x)
#    return Lambda(lambda x: K.stack(x,axis=-1))(y)
#  return func


class Transform(keras.layers.Layer):
  """ Class to define a Keras transformation layer

  :param transforms: A function that takes the input array x and returns a list of transformation outputs
  :type transforms: func 

  """

  def __init__(self, transforms, **kwargs):
    super(Transform, self).__init__(**kwargs)
    self.transforms = transforms

  def call(self, inputs):
    return K.stack(self.transforms(inputs),axis=-1)

  def compute_output_shape(self, input_shape):
    return (None,len(self.transforms(np.ones((1,input_shape[1])))))

  def get_config(self):
    config = super(Transform, self).get_config()
    config.update({"transforms": base64.b64encode(marshal.dumps(self.transforms.__code__)).decode('utf-8')})
    return config

  @classmethod
  def from_config(cls, config):
    code = marshal.loads(base64.b64decode(config["transforms"]))
    config["transforms"] = types.FunctionType(code, globals(), "transforms")
    return cls(**config)
