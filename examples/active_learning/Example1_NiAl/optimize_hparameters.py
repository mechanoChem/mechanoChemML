#!/usr/bin/env python

import sys, os

import numpy as np
from mechanoChemML.src.idnn import IDNN
from mechanoChemML.workflows.active_learning.Example1_NiAl.CASM_wrapper import loadCASMOutput
import sys
import keras

set_i = int(sys.argv[1])
read = int(sys.argv[2])
rnd = int(sys.argv[3])

# Randomly choose hyperparameters
if (read == 0 or rnd <= 2):
    learning_rate = 5.*np.power(10.,-1.-2.*np.random.rand(1)[0],dtype=np.float32)
    n_layers = 2
    hidden_units = n_layers*[np.random.randint(20,200)]
else:
    readHP = open('data/sortedHyperParameters_'+str(rnd-1)+'.txt','r')
    HP = readHP.read().splitlines()
    exec ('hparam = ['+HP[read]+']')
    learning_rate = hparam[0]
    hidden_units = hparam[1]

def IDNN_transforms():

    def transforms(x):
        h0 = x[:,0]
        h1 = 16.*x[:,1]*x[:,2]*x[:,3]
        h2 = 4.*(x[:,1]*x[:,1] + x[:,2]*x[:,2] + x[:,3]*x[:,3])
        h3 = 64.*(x[:,2]*x[:,2]*x[:,3]*x[:,3] +
                  x[:,1]*x[:,1]*x[:,3]*x[:,3] +
                  x[:,1]*x[:,1]*x[:,2]*x[:,2])
        
        return [h0,h1,h2,h3]

    return transforms

# Callbacks for training either model
csv_logger = keras.callbacks.CSVLogger('training/training_{}_{}.txt'.format(rnd,set_i),append=True)
reduceOnPlateau = keras.callbacks.ReduceLROnPlateau(factor=0.5,
                                                    patience=100,
                                                    min_lr=1.e-4)
earlyStopping = keras.callbacks.EarlyStopping(patience=150)

# Define model(s)
idnn = IDNN(4,
            hidden_units,
            transforms=IDNN_transforms(),
            dropout=0.06,
            unique_inputs=True,
            final_bias=True)
idnn.compile(loss=['mse','mse',None],
             loss_weights=[0.01,1,None],
             optimizer=keras.optimizers.Adagrad(lr=np.float32(learning_rate)))

# read in casm data
eta_train, mu_train = loadCASMOutput(rnd)
        
# shuffle the training set (otherwise, the most recent results
# will be put in the validation set by Keras)
inds = np.arange(eta_train.shape[0])
np.random.shuffle(inds)
eta_train = eta_train[inds]
mu_train = mu_train[inds]
        
# create energy dataset (zero energy at origin)
eta_train0 = np.zeros(eta_train.shape)
g_train0 = np.zeros((eta_train.shape[0],1))
        
# train
history = idnn.fit([eta_train0,eta_train,eta_train],
                   [100.*g_train0,100.*mu_train],
                   validation_split=0.25,
                   epochs=250,
                   batch_size=100,
                   callbacks=[csv_logger,
                              reduceOnPlateau,
                              earlyStopping])
idnn.save('idnn_{}_{}.h5'.format(rnd,set_i))

valid_loss = history.history['val_loss'][-1]

# Write out hyperparameters and l2norm
if not np.isnan(valid_loss):
    fout = open('hparameters_'+str(set_i)+'.txt','w')
    fout.write('hparameters += [['+str(learning_rate)+','+str(hidden_units)+',"'+str(rnd)+'_'+str(set_i)+'",'+str(valid_loss)+']]')
    fout.close()
