import numpy as np
import json
import keras

import sys, os

from mechanoChemML.src.idnn import IDNN
from mechanoChemML.src.gradient_layer import Gradient
from mechanoChemML.src.transform_layer import Transform

def transforms(x):

    h0 = x[:,0]
    h1 = 16.*x[:,1]*x[:,2]*x[:,3]
    h2 = 4.*(x[:,1]*x[:,1] + x[:,2]*x[:,2] + x[:,3]*x[:,3])
    h3 = 64.*(x[:,2]*x[:,2]*x[:,3]*x[:,3] +
              x[:,1]*x[:,1]*x[:,3]*x[:,3] +
              x[:,1]*x[:,1]*x[:,2]*x[:,2])
    
    return [h0,h1,h2,h3]


rnd = 12
idnn = keras.models.load_model(f'idnn_{rnd}.h5',
                               custom_objects={'Gradient': Gradient, 
                                               'Transform': Transform(transforms)})
i = 0
weights = []
biases = []
for layer in idnn.layers[1:]:
    w = layer.get_weights()
    if len(w)==2:
        weights.append(w[0])
        biases.append(w[1])
    elif len(w)==1:
        weights.append(w[0])

last = max(len(weights) - 1,len(biases) - 1)

for i,weight in enumerate(weights):
    if i == last:
        weight *= 0.01
    np.savetxt('weights_'+str(i)+'.txt',weight,header=str(weight.shape[0])+' '+str(weight.shape[1]))

for i,bias in enumerate(biases):
    if i == last:
        bias *= 0.01
    np.savetxt('bias_'+str(i)+'.txt',bias,header=str(bias.shape[0]))
