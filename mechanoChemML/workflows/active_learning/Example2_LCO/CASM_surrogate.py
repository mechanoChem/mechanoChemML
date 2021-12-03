import keras
import sys, os

import numpy as np
from mechanoChemML.src.idnn import IDNN
from mechanoChemML.src.gradient_layer import Gradient
from mechanoChemML.src.transform_layer import Transform
import json

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


#print('load model...')
#idnn = keras.models.load_model(os.path.dirname(__file__)+'/idnn_test.h5',
#                               custom_objects={'Gradient': Gradient, 
#                                               'Transform2': Transform(transforms)})

print('recreate model...')
hidden_layers = [174, 174, 174]
idnn = IDNN(7,
            hidden_layers,
            activation='tanh',
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
for comp in inputs["driver"]["custom_conditions"]:
    T = comp["temperature"]
    phi.append([comp["bias_phi"]["0"],
                comp["bias_phi"]["1"],
                comp["bias_phi"]["2"],
                comp["bias_phi"]["3"],
                comp["bias_phi"]["4"],
                comp["bias_phi"]["5"],
                comp["bias_phi"]["6"]])
    kappa.append([comp["bias_kappa"]["0"],
                  comp["bias_kappa"]["1"],
                  comp["bias_kappa"]["2"],
                  comp["bias_kappa"]["3"],
                  comp["bias_kappa"]["4"],
                  comp["bias_kappa"]["5"],
                  comp["bias_kappa"]["6"]])

phi = np.array(phi)
eta = np.array(kappa) # Since it's just for testing the workflow, we'll take eta as kappa

print('predicting...')
pred = idnn.predict(eta)
keras.backend.clear_session()

mu = pred[1]
kappa = eta + 0.5*mu/phi # Back out what kappa would have been

print('write output...')
# Write out a limited CASM-like results file
results = {"T": len(eta)*[T]}
for i in range(7):
    results[f"Bias_kappa({i})"] = kappa[:,i].tolist()
    results[f"Bias_phi({i})"] = phi[:,i].tolist()
    results[f"<order_param({i})>"] = eta[:,i].tolist()

with open('results.json','w') as fout:
    json.dump(results,fout,sort_keys=True, indent=4)
