import numpy as np
import json
import keras

import sys, os

from mechanoChemML.src.idnn import IDNN
from mechanoChemML.src.gradient_layer import Gradient
from mechanoChemML.src.transform_layer import Transform

def transforms(x):

    h0 = x[:,0]
    h1 = 2./3.*(x[:,1]**2 + x[:,2]**2 + x[:,3]**2 +
                x[:,4]**2 + x[:,5]**2 + x[:,6]**2)
    h2 = 8./3.*(x[:,1]**4 + x[:,2]**4 + x[:,3]**4 +
                x[:,4]**4 + x[:,5]**4 + x[:,6]**4)
    h3 = 4./3.*((x[:,1]**2 + x[:,2]**2)*
                (x[:,3]**2 + x[:,4]**2 + x[:,5]**2 + x[:,6]**2) +
                (x[:,3]**2 + x[:,6]**2)*(x[:,4]**2 + x[:,5]**2))
    h4 = 16./3.*(x[:,1]**2*x[:,2]**2 + x[:,3]**2*x[:,6]**2 + x[:,4]**2*x[:,5]**2)
    h5 = 32./3.*(x[:,1]**6 + x[:,2]**6 + x[:,3]**6 +
                 x[:,4]**6 + x[:,5]**6 + x[:,6]**6)
    h6 = 8./3.*((x[:,1]**4 + x[:,2]**4)*
                (x[:,3]**2 + x[:,4]**2 + x[:,5]**2 + x[:,6]**2) +
                (x[:,3]**4 + x[:,6]**4)*(x[:,4]**2 + x[:,5]**2) + 
                (x[:,1]**2 + x[:,2]**2)*
                (x[:,3]**4 + x[:,4]**4 + x[:,5]**4 + x[:,6]**4) +
                (x[:,3]**2 + x[:,6]**2)*(x[:,4]**4 + x[:,5]**4))
    h7 = 16./3.*(x[:,1]**2*x[:,2]**2*(x[:,3]**2 + x[:,4]**2 + x[:,5]**2 + x[:,6]**2) + 
                 x[:,3]**2*x[:,6]**2*(x[:,1]**2 + x[:,2]**2 + x[:,4]**2 + x[:,5]**2) + 
                 x[:,4]**2*x[:,5]**2*(x[:,1]**2 + x[:,2]**2 + x[:,3]**2 + x[:,6]**2))
    h8 = 32./3.*(x[:,1]**4*x[:,2]**2 + x[:,3]**4*x[:,6]**2 + x[:,4]**4*x[:,5]**2 +
                 x[:,1]**2*x[:,2]**4 + x[:,3]**2*x[:,6]**4 + x[:,4]**2*x[:,5]**4)
    h9 = 8.*(x[:,1]**2 + x[:,2]**2)*(x[:,3]**2 + x[:,6]**2)*(x[:,4]**2 + x[:,5]**2)
    h10 = 64./5.*((x[:,1]**2 - x[:,2]**2)*(x[:,3]*x[:,5] + x[:,4]*x[:,6])*(x[:,3]*x[:,4] - x[:,5]*x[:,6]) +
                  x[:,1]*x[:,2]*(x[:,3]**2 - x[:,6]**2)*(x[:,4]**2 - x[:,5]**2))
    h11 = 64.*np.sqrt(5)*x[:,1]*x[:,2]*x[:,3]*x[:,4]*x[:,5]*x[:,6]

    return [h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11]


rnd = 11
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
