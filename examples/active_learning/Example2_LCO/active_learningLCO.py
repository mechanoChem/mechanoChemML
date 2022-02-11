#!/usr/bin/env python

import sys, os

import numpy as np
import shutil
from shutil import copyfile
from mechanoChemML.src.idnn import IDNN
from mechanoChemML.src.gradient_layer import Gradient
from mechanoChemML.src.transform_layer import Transform
from mechanoChemML.workflows.active_learning.Example2_LCO.hp_search import hyperparameterSearch2

from importlib import import_module
from mechanoChemML.workflows.active_learning.Example2_LCO.CASM_wrapper import submitCASM, compileCASMOutput, loadCASMOutput
import tensorflow as tf
from sobol_seq import i4_sobol
from mechanoChemML.workflows.active_learning.Example2_LCO.hitandrun import billiardwalk

import keras
from keras.callbacks import CSVLogger, ReduceLROnPlateau, EarlyStopping
from configparser import ConfigParser
import keras.backend as K
from keras.layers import Lambda

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
        self.lr = 0.002
        self.idnn = IDNN(self.dim,
                         self.hidden_units,
                         activation='tanh',
                         transforms=self.IDNN_transforms(),
                         dropout=self.Dropout,
                         unique_inputs=True,
                         final_bias=True)
        self.idnn.compile(loss=['mse','mse',None],
                          loss_weights=[0.01,1,None],
                          optimizer=keras.optimizers.RMSprop(lr=self.lr))

    ########################################

    def read_config(self,config_path):

        config = ConfigParser()
        config.read(config_path)

        self.casm_project_dir = config['DNS']['CASM_project_dir']
        self.N_jobs = int(config['HPC']['CPUNum']) #number of processors to use
        self.N_global_pts = int(config['WORKFLOW']['N_global_pts']) #global sampling points each iteration
        self.N_rnds = int(config['WORKFLOW']['Iterations'])
        self.Epochs = int(config['NN']['Epochs'])
        self.Batch_size = int(config['NN']['Batch_size'])
        self.N_hp_sets = int(config['HYPERPARAMETERS']['N_sets'])
        self.Dropout = float(config['HYPERPARAMETERS']['Dropout'])
        self.LR_range = [float(p) for p in config['HYPERPARAMETERS']['LearningRate'].split(',')]
        self.Layers_range = [int(p) for p in config['HYPERPARAMETERS']['Layers'].split(',')]
        self.Neurons_range = [int(p) for p in config['HYPERPARAMETERS']['Neurons'].split(',')]
        self.dim = 7
        self.T = 260. #300.
        self.phi = np.array([10.,0.1,0.1,0.1,0.1,0.1,0.1])
        self.Q = np.loadtxt(f'{os.path.dirname(__file__)}/Q_LCO.txt')
        self.invQ = np.linalg.inv(self.Q)[:,:self.dim]
        self.Q = self.Q[:self.dim]
        self.n_planes = np.vstack((self.invQ,-self.invQ))
        self.c_planes = np.hstack((np.ones(self.invQ.shape[0]),np.zeros(self.invQ.shape[0])))
        self.x0 = np.zeros(self.dim)
        self.x0[0] = 0.5
        
    ########################################

    def create_test_set(self,N_points,N_boundary=0):

        # Create test set
        tau = 1
        
        eta, eta_b = billiardwalk(self.x0,self.n_planes,self.c_planes,N_points,tau)
        self.x0 = eta[-1] # Take last point to be next initial point
        np.savetxt('x0.txt',self.x0)
        eta = np.vstack((eta,eta_b[np.random.permutation(np.arange(len(eta_b)))[:N_boundary]]))

        return eta

    ########################################

    def create_local_test_set(self,N_points,N_boundary=0):

        # Create test set
        tau = 1

        x0 = np.zeros(self.dim)
        x0[0] = 0.5

        local_n_planes = np.zeros((2,self.dim))
        local_n_planes[0,0] = -1.
        local_n_planes[1,0] = 1.
        local_n_planes = np.vstack((self.n_planes,local_n_planes))
        local_c_planes = np.hstack((self.c_planes,np.array([-0.45,0.55])))
        
        eta, eta_b = billiardwalk(x0,local_n_planes,local_c_planes,N_points,tau)
        eta = np.vstack((eta,eta_b[np.random.permutation(np.arange(len(eta_b)))[:N_boundary]]))

        return eta

    ########################################
    
    def global_sampling(self,rnd):
        
        # sample quasi-uniformly
        if rnd<6:
            N_b = int(self.N_global_pts/4)
        else:
            N_b = 0
        print('Create sample set...')
        eta = self.create_test_set(self.N_global_pts,
                                   N_boundary=N_b)

        # define bias parameters
        if rnd==0:
            kappa = eta
        else:
            mu_test = 0.01*self.idnn.predict([eta,eta,eta])[1]
            kappa = eta + 0.5*mu_test/self.phi

        # Sample wells (and end members), around eta0=0.5, etai+/- 0.5
        print('Sampling wells and end members...')
        etaW = np.zeros((2*self.dim,self.dim))
        # wells
        etaW[:,0] = 0.5
        for i in range(1,self.dim):
            etaW[2*i,i] = 0.425
            etaW[2*i+1,i] = -0.425
        # end members
        etaW[0,0] = 0.075
        etaW[1,0] = 0.925

        # define bias parameters
        if rnd==0:
            kappaW = etaW
        else:
            muW = 0.01*self.idnn.predict([etaW,etaW,etaW])[1]
            kappaW = etaW + 0.5*muW/self.phi

        N_w = 25
        if self.test:
            N_w = 2
        kappaW = np.repeat(kappaW,N_w,axis=0)
        kappaW  += 0.15*(np.random.rand(*kappaW.shape)-0.5)

        # Sample between wells
        # Get vertices
        etaB = np.zeros((2*(self.dim-1),self.dim))
        # wells
        etaB[:,0] = 0.5
        for i in range(1,self.dim):
            etaB[2*i-2,i] = 0.5
            etaB[2*i-1,i] = -0.5
        if rnd==0:
            kappaB = etaB
        else:
            muB = 0.01*self.idnn.predict([etaB,etaB,etaB])[1]
            kappaB = etaB + 0.5*muB/self.phi

        N_w2 = 20 # Number of random points per vertex
        if self.test:
            N_w2 = 2
        kappaW2 = np.zeros((2*(self.dim-1)*N_w2,self.dim))
        kappaW2[:,0] = kappaB[0,0]
        kappaW2 += 0.05*(np.random.rand(*kappaW2.shape)-0.5) # Small random perterbation
        for i in range(1,self.dim):
            for j in range(2*N_w2):
                kappaW2[2*(i-1)*N_w2 + j,i] = np.random.rand()*(kappaB[2*i-2,i] - kappaB[2*i-1,i]) + kappaB[2*i-1,i] # Random between positive and negative well

        kappa = np.vstack((kappa,kappaW,kappaW2))
            
        # submit casm
        print('Submit jobs to CASM...')
        submitCASM(self.N_jobs,self.phi,kappa,self.T,rnd,casm_project_dir=self.casm_project_dir,test=self.test)
        print('Compile output...')
        compileCASMOutput(rnd)

    ########################################
    
    def local_sampling(self,rnd):
        
        # local error
        print('Loading data...')
        kappa_test, eta_test, mu_test = loadCASMOutput(rnd-1,singleRnd=True)
        #kappa_test, eta_test, mu_test = loadCASMOutput(rnd-1)
        print('Predicting...')
        mu_pred = 0.01*self.idnn.predict([eta_test,eta_test,eta_test])[1]
        print('Finding high pointwise error...')
        error = np.sum((mu_pred - mu_test)**2,axis=1)
        kappaE =  kappa_test[np.argsort(error)[::-1]]

        # randomly perturbed samples
        if self.test:
            kappa_a = np.repeat(kappaE[:3],3,axis=0)
            kappa_b = np.repeat(kappaE[3:6],2,axis=0)
        else:
            kappa_a = np.repeat(kappaE[:200],3,axis=0)
            kappa_b = np.repeat(kappaE[200:400],2,axis=0)
        kappa_local = np.vstack((kappa_a,kappa_b))
        kappa_local += 0.02*2.*(np.random.rand(*kappa_local.shape)-0.5) #perturb points randomly
        
        # submit casm
        print('Submitting to CASM...')
        submitCASM(self.N_jobs,self.phi,kappa_local,self.T,rnd,casm_project_dir=self.casm_project_dir,test=self.test)
        print('Compiling CASM output data...')
        compileCASMOutput(rnd)

    ########################################

    # Define function creating IDNN with random hyperparameters
    def train_rand_idnn(self,rnd,set_i):
        learning_rate = np.power(10,(np.log10(self.LR_range[1]) - np.log10(self.LR_range[0]))*np.random.rand(1)[0] + np.log10(0.0001),dtype=np.float32)
        n_layers = np.random.randint(self.Layers_range[0],self.Layers_range[1]+1)
        hidden_units = n_layers*[np.random.randint(self.Neurons_range[0],self.Neurons_range[1]+1)]

        idnn = IDNN(self.dim,
                    hidden_units,
                    activation='tanh',
                    transforms=self.IDNN_transforms(),
                    dropout=self.Dropout,
                    unique_inputs=True,
                    final_bias=True)
        idnn.compile(loss=['mse','mse',None],
                     loss_weights=[0.01,1,None],
                     optimizer=keras.optimizers.RMSprop(lr=learning_rate))

        valid_loss,_ = self.surrogate_training()(rnd,idnn,f'_{set_i}')

        return hidden_units, learning_rate, valid_loss

    ########################################
        
    def hyperparameter_search(self,rnd):

        # submit
        commands = [f"sys.path.append('{os.path.dirname(__file__)}')",
                    'from active_learningLCO import Active_learning']
        training_func = 'Active_learning("../LCO_free_energy.ini").train_rand_idnn'
        if self.test:
            training_func = 'Active_learning("../LCO_test.ini").train_rand_idnn'
        self.hidden_units, self.lr = hyperparameterSearch2(rnd,self.N_hp_sets,commands,training_func)

    ########################################
        
    def IDNN_transforms(self):

        def transforms(x):
            #import numpy as np
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

        return transforms

    ########################################
    
    def surrogate_training(self):

        def training(rnd,idnn,set_i=''):
            # read in casm data
            print('Loading data...')
            _, eta_train, mu_train = loadCASMOutput(2*rnd)
            
            # weight the most recent high error points as high as all the other points
            n_points = len(eta_train)
            sample_weight = np.ones(n_points)
            if rnd > 0:
                sample_weight[-1000:] = max(1,(n_points-1000)/(2*1000))
                
            # shuffle the training set (otherwise, the most recent results
            # will be put in the validation set by Keras)
            inds = np.arange(eta_train.shape[0])
            np.random.shuffle(inds)
            eta_train = eta_train[inds]
            mu_train = mu_train[inds]
            sample_weight = sample_weight[inds]

            # create energy dataset (zero energy at origin)
            eta_train0 = np.zeros(eta_train.shape)
            g_train0 = np.zeros((eta_train.shape[0],1))
        
            # train
            lr_decay = 0.95**rnd
            idnn.compile(loss=['mse','mse',None],
                         loss_weights=[0.01,1,None],
                         optimizer=keras.optimizers.RMSprop(lr=self.lr*lr_decay))
            csv_logger = CSVLogger('training/training_{}{}.txt'.format(rnd,set_i),append=True)
            reduceOnPlateau = ReduceLROnPlateau(factor=0.5,patience=100,min_lr=1.e-6)
            print('Training...')
            history = idnn.fit([eta_train0,eta_train,eta_train],
                               [100.*g_train0,100.*mu_train],
                               validation_split=0.25,
                               epochs=self.Epochs,
                               batch_size=self.Batch_size,
                               sample_weight=[sample_weight,sample_weight],
                               callbacks=[csv_logger,
                                          reduceOnPlateau])
            print('Saving IDNN...')
            idnn.save('idnn_{}{}.h5'.format(rnd,set_i))

            valid_loss = history.history['val_loss'][-1]
            
            return valid_loss, idnn

        return training

    ########################################
        
    def main_workflow(self):
        """
        Main function outlining the workflow.

        - Global sampling

        - Surrogate training (including hyperparameter search)

        - Local sampling
        """

        for rnd in range(self.N_rnds):
            print('Begin global sampling, round ',rnd,'...')
            self.global_sampling(2*rnd)

            if rnd==1:
                print('Perform hyperparameter search...')
                self.hyperparameter_search(rnd)
                print('Load best model...')
                self.idnn = keras.models.load_model('idnn_1.h5',
                                                    custom_objects={'Gradient': Gradient, 
                                                                    'Transform': Transform(self.IDNN_transforms())})

            print('Train surrogate model, round ',rnd,'...')
            _, self.idnn = self.surrogate_training()(rnd,self.idnn)

            # Get rid of the memory leak
            keras.backend.clear_session()
            self.idnn = keras.models.load_model(f'idnn_{rnd}.h5',
                                                custom_objects={'Gradient': Gradient, 
                                                                'Transform': Transform(self.IDNN_transforms())})
            
            print('Begin local sampling, round ',rnd,'...')
            self.local_sampling(2*rnd+1)
