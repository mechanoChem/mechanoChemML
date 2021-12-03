import keras
import sys, os

import numpy as np
from mechanoChemML.src.idnn import IDNN
from mechanoChemML.src.gradient_layer import Gradient
from mechanoChemML.src.transform_layer import Transform
import json

def transforms(x):

    h0 = x[:,0]
    h1 = 16.*x[:,1]*x[:,2]*x[:,3]
    h2 = 4.*(x[:,1]*x[:,1] + x[:,2]*x[:,2] + x[:,3]*x[:,3])
    h3 = 64.*(x[:,2]*x[:,2]*x[:,3]*x[:,3] +
              x[:,1]*x[:,1]*x[:,3]*x[:,3] +
              x[:,1]*x[:,1]*x[:,2]*x[:,2])
    
    return [h0,h1,h2,h3]

#print('load model...')
#idnn = keras.models.load_model(os.path.dirname(__file__)+'/idnn_test.h5',custom_objects={'Gradient': Gradient})

print('recreate model...')
hidden_layers = [70, 70]
idnn = IDNN(4,
            hidden_layers,
            transforms=transforms,
            final_bias=True)

for i in range(len(hidden_layers)+1):
    w = np.loadtxt(os.path.dirname(__file__)+'/test_weights/weights_{}.txt'.format(i),ndmin=2)
    b = np.loadtxt(os.path.dirname(__file__)+'/test_weights/bias_{}.txt'.format(i),ndmin=1)
    idnn.layers[i+2].set_weights([w,b])

print('read input...')
# Read in the casm Monte Carlo input file
input_file = sys.argv[1]
with open(input_file) as fin:
    inputs = json.load(fin)

phi = []
kappa  = []
for comp in inputs["driver"]["conditions_list"]:
    T = comp["temperature"]
    phi.append([comp["phi"][0][0],
                comp["phi"][1][0],
                comp["phi"][2][0],
                comp["phi"][3][0]])
    kappa.append([comp["kappa"][0][0],
                  comp["kappa"][1][0],
                  comp["kappa"][2][0],
                  comp["kappa"][3][0]])

phi = np.array(phi)
eta = np.array(kappa) # Since it's just for testing the workflow, we'll take eta as kappa

print('predicting...')
pred = idnn.predict(eta)
mu = pred[1]
kappa = eta + 0.5*mu/phi # Back out what kappa would have been

#keras.backend.clear_session()

print('write output...')
# Write out a limited CASM-like results file
results = {"T": len(eta)*[T]}
for i in range(4):
    results["kappa_{}".format(i)] = kappa[:,i].tolist()
    results["phi_{}".format(i)] = phi[:,i].tolist()
    results["<op_val({})>".format(i)] = eta[:,i].tolist()

with open('results.json','w') as fout:
    json.dump(results,fout,sort_keys=True, indent=4)
