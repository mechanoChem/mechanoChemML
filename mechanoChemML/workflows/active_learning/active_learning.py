#!/usr/bin/env python

import sys, os

import numpy as np
import shutil
from shutil import copyfile
from mechanoChemML.src.idnn import IDNN, find_wells
from mechanoChemML.src.gradient_layer import Gradient
from mechanoChemML.src.transform_layer import Transform
from mechanoChemML.workflows.active_learning.Example1_NiAl.hp_search import hyperparameterSearch

from importlib import import_module
from mechanoChemML.workflows.active_learning.Example1_NiAl.CASM_wrapper import submitCASM, compileCASMOutput, loadCASMOutput
import tensorflow as tf
from sobol_seq import i4_sobol

import keras
from keras.callbacks import CSVLogger, ReduceLROnPlateau, EarlyStopping
from configparser import ConfigParser

############ Active learning class #######

class Active_learning(object):
    """
    Class to define the active learning workflow
    used to create a deep neural network
    representation of the free energy of a system.
    """

    ########################################
    
    def __init__(self,config_path,test=False):

        self.test = test

        if not os.path.exists('training'):
            os.mkdir('training')
        if not os.path.exists('data'):
            os.mkdir('data')
        if not os.path.exists('outputFiles'):
            os.mkdir('outputFiles')

        self.read_config(config_path)
        self.seed = 1 # initial Sobol sequence seed

        # initialize IDNN
        self.hidden_units = [20,20]
        self.lr = 0.2
        self.idnn = IDNN(self.dim,
                         self.hidden_units,
                         transforms=self.IDNN_transforms(),
                         dropout=self.Dropout,
                         unique_inputs=True,
                         final_bias=True)
        self.idnn.compile(loss=['mse','mse',None],
                          loss_weights=[0.01,1,None],
                          optimizer=keras.optimizers.Adagrad(lr=self.lr))

    ########################################

    def read_config(self,config_path):

        config = ConfigParser()
        config.read(config_path)

        self.casm_project_dir = config['DNS']['CASM_project_dir']
        self.job_manager = config['DNS']['JOB_MANAGER']
        self.N_jobs = int(config['HPC']['CPUNum']) #number of processors to use
        self.N_global_pts = int(config['WORKFLOW']['N_global_pts']) #global sampling points each iteration
        self.N_rnds = int(config['WORKFLOW']['Iterations'])
        self.Epochs = int(config['NN']['Epochs'])
        self.Batch_size = int(config['NN']['Batch_size'])
        self.N_hp_sets = int(config['HYPERPARAMETERS']['N_sets'])
        self.Dropout = float(config['HYPERPARAMETERS']['Dropout'])
        self.dim = 4
        
    ########################################

    def create_test_set(self,N_points,dim,bounds=[0.,1.],seed=1):

        Q = 0.25*np.array([[1, 1, 1, 1],
                           [1, 1, -1, -1],
                           [1, -1, -1, 1],
                           [1, -1, 1, -1]])
        
        # Create test set
        x_test = np.zeros((N_points,dim))
        eta = np.zeros((N_points,dim))
        i = 0
        while (i < N_points):
            x_test[i],seed = i4_sobol(dim,seed)
            x_test[i] = (bounds[1] - bounds[0])*x_test[i] + bounds[0] # shift/scale according to bounds
            eta[i] = np.dot(x_test[i],Q.T).astype(np.float32)
            if eta[i,0] <= 0.25:
                i += 1

        return x_test, eta, seed

    ########################################
    
    def ideal(self,x_test):

        T = 600.
        kB = 8.61733e-5
        Q = 0.25*np.array([[1, 1, 1, 1],
                           [1, 1, -1, -1],
                           [1, -1, -1, 1],
                           [1, -1, 1, -1]])
        invQ = np.linalg.inv(Q)

        g_test = 0.25*kB*T*np.sum((x_test*np.log(x_test) + (1.-x_test)*np.log(1.-x_test)),axis=1)
        mu_test = 0.25*kB*T*np.log(x_test/(1.-x_test)).dot(invQ)

        return mu_test, g_test

    ########################################

    def global_sampling(self,rnd):
        
        # sample with sobol
        if rnd==0:
            x_bounds = [1.e-5,1-1.e-5]
        elif rnd<11:
            x_bounds = [-0.05,1.05]
        else:
            #x_bounds = [-0.02,1.02]
            x_bounds = [0.,1.]
        x_test,eta,self.seed = self.create_test_set(self.N_global_pts,
                                                    self.dim,
                                                    bounds=x_bounds,
                                                    seed=self.seed)

        # approximate mu
        if rnd==0:
            mu_test,_ = self.ideal(x_test)
        else:
            mu_test = 0.01*self.idnn.predict([eta,eta,eta])[1]
            
        # submit casm
        submitCASM(self.N_jobs,mu_test,eta,rnd,casm_project_dir=self.casm_project_dir,test=self.test,job_manager=self.job_manager)
        compileCASMOutput(rnd)

    ########################################
    
    def local_sampling(self,rnd):
        
        # local error
        eta_test, mu_test = loadCASMOutput(rnd-1,singleRnd=True)
        mu_pred = 0.01*self.idnn.predict([eta_test,eta_test,eta_test])[1]
        error = np.sum((mu_pred - mu_test)**2,axis=1)
        etaE =  eta_test[np.argsort(error)[::-1]]

        # find wells
        _,eta_test,self.seed = self.create_test_set(30*self.N_global_pts,
                                                    self.dim,
                                                    seed=self.seed)
        etaW = find_wells(self.idnn,eta_test)

        # randomly perturbed samples
        eta_a = np.repeat(etaE[:200],4,axis=0)
        eta_b = np.repeat(etaE[200:400],2,axis=0)
        eta_c = np.repeat(etaW[:400],4,axis=0)
        if self.test:
            eta_a = np.repeat(etaE[:2],3,axis=0)
            eta_b = np.repeat(etaE[2:4],2,axis=0)
            eta_c = np.repeat(etaW[:4],3,axis=0)
        eta_local = np.vstack((eta_a,eta_b,eta_c))
        eta_local += 0.25*(1.5/(2.*self.N_global_pts**(1./self.dim)))*(np.random.rand(*eta_local.shape)-0.5) #perturb points randomly
        mu_local = 0.01*self.idnn.predict([eta_local,eta_local,eta_local])[1]
        
        # submit casm
        submitCASM(self.N_jobs,mu_local,eta_local,rnd,casm_project_dir=self.casm_project_dir,test=self.test,job_manager=self.job_manager)
        compileCASMOutput(rnd)

    ########################################
        
    def hyperparameter_search(self,rnd):
        # submit
        self.hidden_units, self.lr = hyperparameterSearch(rnd,self.N_hp_sets,job_manager=self.job_manager)

    ########################################
        
    def IDNN_transforms(self):

        def transforms(x):
            h0 = x[:,0]
            h1 = 16.*x[:,1]*x[:,2]*x[:,3]
            h2 = 4.*(x[:,1]*x[:,1] + x[:,2]*x[:,2] + x[:,3]*x[:,3])
            h3 = 64.*(x[:,2]*x[:,2]*x[:,3]*x[:,3] +
                      x[:,1]*x[:,1]*x[:,3]*x[:,3] +
                      x[:,1]*x[:,1]*x[:,2]*x[:,2])
            
            return [h0,h1,h2,h3]

        return transforms

    ########################################
    
    def surrogate_training(self,rnd):
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
        lr_decay = 0.9**rnd
        self.idnn.compile(loss=['mse','mse',None],
                          loss_weights=[0.01,1,None],
                          optimizer=keras.optimizers.Adagrad(lr=self.lr*lr_decay))
        csv_logger = CSVLogger('training/training_{}.txt'.format(rnd),append=True)
        reduceOnPlateau = ReduceLROnPlateau(factor=0.5,patience=100,min_lr=1.e-4)
        earlyStopping = EarlyStopping(patience=150)
        self.idnn.fit([eta_train0,eta_train,eta_train],
                      [100.*g_train0,100.*mu_train],
                      validation_split=0.25,
                      epochs=self.Epochs,
                      batch_size=self.Batch_size,
                      callbacks=[csv_logger,
                                 reduceOnPlateau,
                                 earlyStopping])
        self.idnn.save('idnn_{}.h5'.format(rnd))

    ########################################
        
    def main_workflow(self):
        """
        Main function outlining the workflow.

        - Global sampling

        - Surrogate training (including hyperparameter search)

        - Local sampling
        """

        for rnd in range(self.N_rnds):
            self.global_sampling(2*rnd)

            if rnd==1:
                self.hyperparameter_search(rnd)
                custom_objects = {'Gradient': Gradient, 
                                  'Transform': Transform(self.IDNN_transforms())}
                self.idnn = keras.models.load_model('idnn_1.h5',
                                                    custom_objects=custom_objects)
                
            self.surrogate_training(rnd)
            self.local_sampling(2*rnd+1)
