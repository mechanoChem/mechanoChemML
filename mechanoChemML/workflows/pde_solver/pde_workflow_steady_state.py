from __future__ import absolute_import, division, print_function, unicode_literals

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
#os.environ["CUDA_VISIBLE_DEVICES"]="3"



# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import datetime
import glob
import time
import pickle
import logging
from natsort import natsorted, ns
import socket
from configparser import ConfigParser, ExtendedInterpolation
import argparse

# # TensorFlow and tf.keras
import tensorflow as tf
tf.keras.backend.set_floatx('float32')
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.training import py_checkpoint_reader
import tensorflow_probability as tfp
# tf.logging.set_verbosity(tf.logging.ERROR)

from mechanoChemML.src.nn_models import BNN_user_weak_pde_general
import mechanoChemML.src.pde_layers as pde_layers
from mechanoChemML.workflows.pde_solver.pde_utility import plot_PDE_solutions, plot_fields, split_data, expand_dataset, exe_cmd, BatchData, plot_one_field_hist, plot_one_field_stat, plot_one_field,plot_PDE_solutions_new

print('host:', socket.gethostname())
if socket.gethostname() == 'Destiny':
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
      tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
      # Invalid device or cannot modify virtual devices once initialized.
      pass
elif socket.gethostname().find('gpu-cn') >= 0:
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
      tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
      # Invalid device or cannot modify virtual devices once initialized.
      pass
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    print('all devices: ', physical_devices)
    print("dynamic memory growth host", physical_devices)
else:
    physical_devices = tf.config.list_physical_devices('GPU')
    print('all devices: ', physical_devices)
    print("no dynamic memory growth")

class PDEWorkflowSteadyState:
    """
    General Weak PDE constrained workflow.

    Workflow for any specific physical system should inherit from this general workflow.

    """

    def __init__(self):
        self.restart_dir_to_load = ''
        self.now_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.today_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M")

        self._parse_sys_args()
        if self.args.restartfrom:
            print("Note: old config.ini content is loaded. The specified config.ini will be totally neglected.")
            self._load_saved_states()
            # retract old now string from .pickle file
            self.now_str_old = self.args.restartfrom.split('-')[-1].split('.')[0]
        else:
            # 
            self._read_config_file()

        self._debugger()

        # prepare the folder
        cmd = "mkdir -p results restart "
        exe_cmd(cmd)

        self.features = None #: training inputs with Dirichlet and Neumann BCs: features
        self.labels = None #: training labels (not used during training, for comparison purpose only.): labels 

        if self.config['NN']['NNArchitecture'].find('Flipout') >= 0:
            self.isBNN = True
        else:
            self.isBNN = False

        if self.args.restartfrom:
            # only these two parameters in the old configuration file will be altered during restart.
            self.epochs = self.args.continuerun
            self.InitialEpoch = 0
        else:
            self.epochs = int(self.config['NN']['Epochs'])
            self.InitialEpoch = int(self.config['NN']['InitialEpoch'])

        if self.isBNN:
            self.filename_base = self.today_str + '-' + socket.gethostname() + '-BNN-' + self.args.configfile[:-4]
            self.monte_carlo_num = int(self.config['NN']['MonteCarloNum'])
            self.Sigma1 = float(self.config['NN']['Sigma1'])
            self.Sigma2 = float(self.config['NN']['Sigma2'])
            self.tot_img = 8
        else:
            self.filename_base = self.today_str + '-' + socket.gethostname() + '-NN-' + self.args.configfile[:-4]
            self.monte_carlo_num = 1
            self.Sigma1 = 0.0
            self.Sigma2 = 0.0
            self.tot_img = 6

        self.expand_times = int(self.config['NN']['DataAugTimes'])
        self.batch_size = int(self.config['NN']['BatchSize'])
        self.data_path = self.config['NN']['DataPath']
        self.NNOptimizer = self.config['NN']['Optimizer']
        self.LR0 = float(self.config['NN']['LearningRate'])
        self.NeumannFirst = int(self.config['NN']['NeumannFirst'])

        try:
            self.FixLoc = int(self.config['NN']['FixLoc'])
        except:
            self.FixLoc = 0

        self.model = None

        try:
            self.data_folder = [x.strip() for x in self.config['NN']['DataFolder'].split()]
        except:
            self.data_folder = ['DNS/']
            print('Default DNS/ is used for DataFolder in NN in config.ini')

        if self.args.restartfrom:
            # make sure that new filename starts the same as old filename. 
            self.filename_base += (
            '-x' + str(self.expand_times) 
            + '-B' + str(self.batch_size) 
            + '-E' + self.config['NN']['Epochs'] 
            + '-I' + self.config['NN']['InitialEpoch'] 
            + '-mc' + str(self.monte_carlo_num) 
            + '-1S' + "{:.1e}".format(self.Sigma1) 
            + '-2S' + "{:.1e}".format(self.Sigma2) 
            + '-' + self.NNOptimizer 
            + '-' + "{:.1e}".format(self.LR0) 
            + '-' + self.data_path.replace('/', '')
            + '-'  + self.now_str_old 
            + '-e' + str(self.args.continuerun)
            )
        else:
            self.filename_base += (
            '-x' + str(self.expand_times) 
            + '-B' + str(self.batch_size) 
            + '-E' + str(self.epochs)            
            + '-I' + str(self.InitialEpoch)            
            + '-mc' + str(self.monte_carlo_num) 
            + '-1S' + "{:.1e}".format(self.Sigma1) 
            + '-2S' + "{:.1e}".format(self.Sigma2) 
            + '-' + self.NNOptimizer 
            + '-' + "{:.1e}".format(self.LR0) 
            + '-' + self.data_path.replace('/', '')
            + '-' + self.now_str 
            )

        self.restart_dir = 'restart/' +  self.filename_base
        self.filename = 'results/' +  self.filename_base

    def _load_saved_states(self):
        """ load saved information from the pickle file during restart """
        saved_config = pickle.load(open(self.args.restartfrom, "rb"))
        self.config = saved_config ['configdata']
        self.restart_dir_to_load = saved_config['savedckpdir']
        print('Content of old config.ini to be used: ', self.config)

    def _read_config_file(self):
        """ read configurations from the config.ini file """
        config = ConfigParser(interpolation=ExtendedInterpolation())
        config.read(self.args.configfile)
        self.config = config

    def _parse_sys_args(self):
        """ parse system args """
        parser = argparse.ArgumentParser(description='Run BNN', prog="'" + (sys.argv[0]) + "'")
        parser.add_argument('configfile', type=str, help='simulation configuration file')
        parser.add_argument('-rf', '--restartfrom', type=str, default='', help='restart from saved .pickle files (with previous predictions)')
        parser.add_argument('-ra', '--restartat', type=int, default=0, help='restart at which epoch')
        parser.add_argument('-init', '--initfrom', type=str, default='', help='initialize from saved .pickle files (overwrite the info given in config.ini)')
        parser.add_argument('-c', '--continuerun', type=int, default=-1,  help='continue run how many epoches from the restart file)')
        try:
            args = parser.parse_args()
            self.args = args
        except:
            parser.print_help()
            exit(0)

    def _debugger(self):
        """ setup the debugger """
        logger = logging.getLogger('root')
        FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
        logging.basicConfig(format=FORMAT)
        logger.setLevel(logging.DEBUG)
        self.logger = logger

    def _output_bc_stats(self):

        self.bc_data_seq = BatchData(data=(self.features, self.labels), batch_size=20)
        bc_data_stats = tf.zeros_like(self.features[0:1,:,:,:]).numpy()
        bc_val_min = None
        bc_val_max = None
        for step, (batch_x, batch_y) in enumerate(self.bc_data_seq):
            bc_counts = tf.where(batch_x > 0, tf.ones_like(batch_x, dtype=tf.float32), tf.zeros_like(batch_y, dtype=tf.float32))
            bc_counts = tf.reduce_sum(bc_counts, [0], keepdims=True).numpy()
            bc_data_stats = bc_data_stats + bc_counts
            bc_values = batch_x
            bc_values = np.ma.masked_where(bc_values <= 0, bc_values)
            bc_min = np.amin(bc_values, axis=(1,2))
            bc_max = np.amax(bc_values, axis=(1,2))
            if bc_val_min is None:
                bc_val_min = bc_min
                bc_val_max = bc_max
            else:
                bc_val_min = np.concatenate((bc_val_min, bc_min), axis=0)
                bc_val_max = np.concatenate((bc_val_max, bc_max), axis=0)
            print(step, np.shape(bc_data_stats), 'bc_min=', np.shape(bc_val_min), 'bc_max=', np.shape(bc_val_max))

        bc_data_info = {
                'count':bc_data_stats,
                'min': bc_val_min,
                'max': bc_val_max,
                }
        pickle_out = open('bc_data_' + str(np.shape(bc_val_min)[0]) + '.pickle', "wb")
        pickle.dump(bc_data_info, pickle_out)
        pickle_out.close()

        exit(0)

    def _output_bc_stats_good_bad(self):

        self.bc_data_seq = BatchData(data=(self.features, self.labels), batch_size=1)
        bc_data_stats = tf.zeros_like(self.features[0:1,:,:,:]).numpy()
        bc_val_min = None
        bc_val_max = None
        for step, (batch_x, batch_y) in enumerate(self.bc_data_seq):

# top 20 Neumann:  [248, 95, 187, 273, 19, 242, 101, 89, 166, 93, 110, 6, 310, 82, 302, 224, 260, 190, 207, 209]
# bad 20 Neumann:  [138, 38, 61, 55, 186, 119, 47, 26, 281, 90, 249, 42, 54, 279, 56, 276, 319, 30, 259, 257]
# top 20 no Neumann:  [105, 244, 191, 261, 35, 8, 99, 115, 136, 255, 18, 22, 243, 171, 32, 12, 0, 270, 23, 193]
# bad 20 no Neumann:  [152, 295, 307, 206, 308, 60, 298, 204, 218, 256, 219, 141, 240, 226, 174, 250, 289, 262, 285, 316]

            # if step in [248, 95, 187, 273, 19, 242, 101, 89, 166, 93, 110, 6, 310, 82, 302, 224, 260, 190, 207, 209]:
            # if step in [138, 38, 61, 55, 186, 119, 47, 26, 281, 90, 249, 42, 54, 279, 56, 276, 319, 30, 259, 257]:
            # if step in [105, 244, 191, 261, 35, 8, 99, 115, 136, 255, 18, 22, 243, 171, 32, 12, 0, 270, 23, 193]:
            if step in [152, 295, 307, 206, 308, 60, 298, 204, 218, 256, 219, 141, 240, 226, 174, 250, 289, 262, 285, 316]:
                bc_counts = tf.where(batch_x > 0, tf.ones_like(batch_x, dtype=tf.float32), tf.zeros_like(batch_y, dtype=tf.float32))
                bc_counts = tf.reduce_sum(bc_counts, [0], keepdims=True).numpy()
                bc_data_stats = bc_data_stats + bc_counts
                bc_values = batch_x
                bc_values = np.ma.masked_where(bc_values <= 0, bc_values)
                bc_min = np.amin(bc_values, axis=(1,2))
                bc_max = np.amax(bc_values, axis=(1,2))
                if bc_val_min is None:
                    bc_val_min = bc_min
                    bc_val_max = bc_max
                else:
                    bc_val_min = np.concatenate((bc_val_min, bc_min), axis=0)
                    bc_val_max = np.concatenate((bc_val_max, bc_max), axis=0)
                print(step, np.shape(bc_data_stats), 'bc_min=', np.shape(bc_val_min), 'bc_max=', np.shape(bc_val_max))

        bc_data_info = {
                'count':bc_data_stats,
                'min': bc_val_min,
                'max': bc_val_max,
                }
        pickle_out = open('bc_data_' + str(np.shape(bc_val_min)[0]) + '.pickle', "wb")
        pickle.dump(bc_data_info, pickle_out)
        pickle_out.close()

        exit(0)



    def _load_data(self, only_neumann_data=False, test_folder=''):
        """ 
        load data 

        Args:
            only_neumann_data (bool): only load the BVP setup with Neumann BCs. Use this flag when train the NN with Neumann BCs first.
            test_folder (str): the default location of data is in 'DNS'. If test_folder is specified, the data in this folder will be loaded for testing purpose.
        """

        # waiting for gpu resources without killing the program
        # while (True):
            # try:
                # tf.math.ceil(0.1)
                # cmd = "echo 'gpu resource is allowed' > " + self.filename+'-gpu.log'
                # exe_cmd(cmd)
                # break
            # except:
                # cmd = "echo 'gpu resource is not available, waiting...' > " + self.filename+'-gpu.log'
                # exe_cmd(cmd)
                # time.sleep(1)
                # pass

        self.features = None
        self.labels = None

        if test_folder == '':
            only_testing = False
        else:
            only_testing = True

        if only_testing:
            data_folder_list = [test_folder]
            # data_folder = self.data_path + '/' + test_folder
        else:
            data_folder_list = self.data_folder
            # data_folder = self.data_path + '/' + self.data_folder + '/'

        for one_folder in data_folder_list:
            data_folder = self.data_path + '/' + one_folder + '/'

            file_list = glob.glob(data_folder + '/np-features*.npy')
            file_list = natsorted(file_list, alg=ns.IGNORECASE)
            # print (file_list)

            count = 0
            for f1 in file_list:
                print('file: ', count, f1)
                count += 1
                one_feature = np.load(f1)
                label_path = f1.replace('features', 'labels')
                one_label = np.load(label_path)
                print('file:', f1, 'label:', np.shape(one_label), 'feature:', np.shape(one_feature))
                if (self.features is None):
                    self.features = np.copy(one_feature)
                    self.labels = np.copy(one_label)
                else:
                    self.features = np.concatenate((self.features, one_feature), axis=0)
                    self.labels = np.concatenate((self.labels, one_label), axis=0)

            if (not only_testing) and only_neumann_data:
                raise ValueError("only neumann data option is disabled")
                # selected_index = []
                # for i0 in range(0, np.shape(self.features)[0]):
                    # # print('i0=', i0, np.any(np.greater(self.features[i0,:,:,2:4], 0)))
                    # # the following should still be compatible with the extra Neumann channel
                    # if np.any(np.greater(self.features[i0,:,:,self.dof:2*self.dof], 0)): # with Neumann BCs 
                        # selected_index.append(i0)
                # selected_index = np.array(selected_index)
                # # print(np.shape(selected_index))
                # # print(np.shape(self.features[selected_index]))
                # self.features = self.features[selected_index]
                # self.labels = self.labels[selected_index]
        print('len of self.features: ', np.shape(self.features))
        self.dh = 1.0 / (np.shape(self.features)[2] - 1.0)

        # the_feature = pde_layers.LayerFillZeros()(self.features)
        # the_feature = pde_layers.LayerFillRandomNumber()(self.features)
        # for i in range(0, 1000, 50):
            # plot_fields(
                    # list_of_field = [
                        # the_feature[i:i+1, :, :, 0:1], 
                        # the_feature[i:i+1, :, :, 1:2], 
                        # the_feature[i:i+1, :, :, 2:3], 
                        # ],
                    # list_of_field_name = [
                        # 'Dirichlet', 
                        # 'Neumann x', 
                        # 'Neumann y', 
                        # ], 
                    # dof = 1, 
                    # dof_name = ['c'],
                    # filename = 'results/' + self.problem_name + '-' + str(i) + '-Sol.png')

        # R_red, y_pred, y_true_dummy, _, _, _ = self._compute_residual(the_feature, self.labels)
        # print(np.shape(R_red), np.shape(y_pred))
        # for i in range(0, 1000, 50):
            # print(i, R_red[i,20:30,20:30,0])
            # # print(i, self.labels[i,20:30,20:30,0])

        # BC perturbation
        # the_feature = pde_layers.LayerFillRandomToBCs(stddev=0.05)(the_feature)
        # the_feature = the_feature.numpy()

        # self._output_bc_stats()
        # self._output_bc_stats_good_bad()
        self.features = self.features.astype(np.single)    
        self.labels = self.labels.astype(np.single)    
        print(self.features.dtype)

        if only_testing:
            self.test_dataset = self.features
            self.test_label = self.labels
            # for scaling test
            # the_feature, the_label = expand_dataset(self.features, self.labels, times=12)
            # self.test_seq = BatchData(data=(the_feature, the_label), batch_size=4096)

            self.test_seq = BatchData(data=(self.test_dataset, self.test_label), batch_size=1)
            self.train_seq = self.test_seq
        else:
            the_feature, the_label = expand_dataset(self.features, self.labels, times=self.expand_times)
            self.train_dataset, self.train_label, self.val_dataset, self.val_label, self.test_dataset, self.test_label = split_data(the_feature, the_label, self.batch_size, split_ratio=['0.8', '0.1', '0.1'])
            # self.train_dataset, self.train_label, self.val_dataset, self.val_label, self.test_dataset, self.test_label = split_data(the_feature, the_label, split_ratio=['0.1', '0.1', '0.8'])
            self.train_seq = BatchData(data=(self.train_dataset, self.train_label), batch_size=self.batch_size)
            self.val_seq   = BatchData(data=(self.val_dataset,   self.val_label),   batch_size=self.batch_size)
            self.test_seq  = BatchData(data=(self.test_dataset,  self.test_label),  batch_size=self.batch_size)
            print('len of features: ', np.shape(the_feature), 
                  'len of training data: ', np.shape(self.train_dataset), 
                  'len of test data: ', np.shape(self.test_dataset), 
                  'batch size: ', self.batch_size, 
                  'total train batches: ', len(self.train_seq),
                  'total val batches: ', len(self.val_seq),
                  'total test batches: ', len(self.test_seq),
                  )

    def _bulk_residual(self):
        """
        Dummy _bulk_residual function. The actual residual should be implemented in each physical problem. 

        todo:
            - Please implement the residual in each related physical systems (PDEs).

        error:
            - Not implemented.
        """
        raise ValueError('Residual is not implemented! Please implement it in the specific problem!')


    def _compute_residual(self, features, y_pred, only_y_pred=False):
        """
        Compute different residuals, and apply the Dirichlet BCs to the NN predicted solutions.

        args:
            features (tensor): size of [None, :, :, 2*dof]
            y_pred (tensor): size of [None, :, :, dof]

        return:
            - different residuals and the y_pred with applied Dirichlet BCs.
        """

        # mask contains the region not on the Dirichlet boundary
        bc_mask_dirichlet = pde_layers.ComputeBoundaryMaskNodalData(features, dof=self.dof, opt=1)
        reverse_bc_mask_dirichlet = tf.where( bc_mask_dirichlet == 0, tf.fill(tf.shape(bc_mask_dirichlet), 1.0), tf.fill(tf.shape(bc_mask_dirichlet), 0.0))

        # apply the Dirichlet BCs to y_pred
        input_dirichlet = features[:,:,:,0:self.dof]
        dirichlet_bc = tf.multiply(input_dirichlet, reverse_bc_mask_dirichlet)
        # print(y_pred.dtype, bc_mask_dirichlet.dtype, input_dirichlet.dtype)
        y_pred = tf.multiply(y_pred, bc_mask_dirichlet)
        y_pred = y_pred + dirichlet_bc

        if only_y_pred:
            return y_pred

        bc_mask_neumann = pde_layers.ComputeBoundaryMaskNodalData(features, dof=self.dof, opt=2)
        reverse_bc_mask_neumann = tf.where( bc_mask_neumann == 0, tf.fill(tf.shape(bc_mask_neumann), 1.0), tf.fill(tf.shape(bc_mask_neumann), 0.0))

        if self.UseTwoNeumannChannel :
            neumann_residual = pde_layers.ComputeNeumannBoundaryResidualNodalDataNew(features, dh=self.dh, dof=self.dof)
        else:
            neumann_residual = pde_layers.ComputeNeumannBoundaryResidualNodalData(features, dh=self.dh, dof=self.dof)

        y_true_dummy = pde_layers.LayerFillRandomNumber()(input_dirichlet)
        elem_bulk_residual=self._bulk_residual(y_pred)
        elem_residual_mask = pde_layers.GetElementResidualMask(y_true_dummy)
        R = pde_layers.GetNodalInfoFromElementInfo(elem_bulk_residual, elem_residual_mask, dof=self.dof)
        R_fix = tf.where(dirichlet_bc==0.5, R, 0.0)

        # get only neumann part
        R_neumann = tf.multiply(R, reverse_bc_mask_neumann) 
        # neumann BCs
        dR_neumann = R_neumann - neumann_residual 

        # remove boundary part
        R_no_dirichlet = tf.multiply(R, bc_mask_dirichlet) 
        R_no_dirichlet_no_neumann = tf.multiply(R_no_dirichlet, bc_mask_neumann) 
        R_body = R_no_dirichlet_no_neumann

        # actual residual without the essential BCs
        R_red = R_no_dirichlet - neumann_residual
        return R_red, y_pred, y_true_dummy, R_body, dR_neumann, R_fix

    def _loss_probabilistic(self):
        """
        General probabilistic loss functions. The _compute_residual() has to be specified in each problem.
        """
        def loss(y_true, y_pred):

            if self.UseTwoNeumannChannel :
                inputs = y_pred[:,:,:,self.dof:4*self.dof] # new Neumann Channel
            else:
                inputs = y_pred[:,:,:,self.dof:3*self.dof] # old Neumann Channel
            y_pred = y_pred[:,:,:,0:self.dof]
            dist = tfp.distributions.Normal(loc=tf.zeros_like(y_pred), scale=self.Sigma1)
            y_noise = tf.squeeze(dist.sample(1), [0]) # only sample 1, thus, lead dimension can be squeezed. 
            y_pred = y_pred + y_noise
            R_red, y_pred, y_true_dummy, _, _, _ = self._compute_residual(inputs, y_pred)
            dist = tfp.distributions.Normal(loc=tf.zeros_like(y_true_dummy), scale=self.model.Sigma2)
            return self.BetaMSELoss * tf.reduce_mean(tf.square(tf.where(y_true_dummy > -0.9, tf.random.normal(tf.shape(y_true_dummy), 0.5, 0.05, tf.float32, seed=1), tf.zeros_like(y_true_dummy)) - tf.where(y_true_dummy > -0.9, y_pred, tf.zeros_like(y_pred)))) + self.BetaPDELoss * tf.keras.backend.sum(tf.reduce_mean(-dist.log_prob(R_red), 0))
        return loss

    def _loss_deterministic(self):
        """
        General deterministic loss functions. The _compute_residual() has to be specified in each problem.
        """
        def loss(y_true, y_pred):
            if self.UseTwoNeumannChannel :
                inputs = y_pred[:,:,:,self.dof:4*self.dof] # new Neumann Channel
            else:
                inputs = y_pred[:,:,:,self.dof:3*self.dof] # old Neumann Channel
            y_pred = y_pred[:,:,:,0:self.dof]
            R_red, y_pred, y_true_dummy, _, _, _ = self._compute_residual(inputs, y_pred)
            return self.BetaMSELoss * tf.reduce_mean(tf.square(tf.where(y_true_dummy > -0.9, tf.random.normal(tf.shape(y_true_dummy), 0.5, 0.05, tf.float32, seed=1), tf.zeros_like(y_true_dummy)) - tf.where(y_true_dummy > -0.9, y_pred, tf.zeros_like(y_pred)))) + self.BetaPDELoss * tf.reduce_mean(tf.reduce_sum(tf.square(R_red), axis=[1,2,3]))
        return loss

    def _build_loss(self):
        """ 
        Build the loss for weak-PDE constrained NN
        """
        self.BetaMSELoss = tf.Variable(1.0, trainable=False, dtype=tf.float32)
        self.BetaPDELoss = tf.Variable(1.0, trainable=False, dtype=tf.float32)
        if self.isBNN:
            loss = self._loss_probabilistic()
        else: 
            loss = self._loss_deterministic(),
        return loss
    
    def _build_optimizer(self):
        """ 
        Build the optimizer for weak-PDE constrained NN
        """
        # not using the decay learning rate function.
        # global_step = tf.Variable(0, name='global_step', trainable=False)
        # LearningRate = tf.compat.v1.train.exponential_decay(
            # learning_rate=self.LR0,
            # global_step=global_step,
            # decay_steps=100,
            # decay_rate=0.8,
            # staircase=True)

        if self.NNOptimizer.lower() == 'RMSprop'.lower() :
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.LR0)
        elif self.NNOptimizer.lower() == 'Adadelta'.lower() :
            optimizer = tf.keras.optimizers.Adadelta(learning_rate=self.LR0)
        elif self.NNOptimizer.lower() == 'Adam'.lower() :
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.LR0)
        elif self.NNOptimizer.lower() == 'Nadam'.lower() :
            optimizer = tf.keras.optimizers.Nadam(learning_rate=self.LR0)
        elif self.NNOptimizer.lower() == 'SGD'.lower() : # not very well
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.LR0, momentum=0.9) 
        else:
            raise ValueError('Unknown optimizer option:', self.NNOptimizer)
        return optimizer


    def _check_weights(self):
        """
        Print the weights of layers. For debug purpose, not using anywhere.
        """
        for one_layer in self.model.layers:
            print ('layer = ',one_layer)
            print ('weights =', one_layer.get_weights())
            print ('weights shape = ', np.shape(one_layer.get_weights()))

    def _load_saved_cnn_model(self):
        """
        Use saved optimized CNN parameters to initialize the mean of BNN parameters.
        """
        try:
            self.BestCNNWeight = self.config['NN']['SaveCNNModel']
        except:
            self.BestCNNWeight = ''

        if self.args.initfrom:
            self.BestCNNWeight = self.args.initfrom
        if not self.BestCNNWeight:
            return 0 

        print('----- Loading saved NN optimized model parameter to initialize BNN----------')
        if self.BestCNNWeight.find('.pickle') >= 0:
            # It is encouraged to use pickle file to auto find the check point info and load best CNN weight. 
            saved_config = pickle.load(open(self.BestCNNWeight, "rb"))
            print('best cnn weight (pickle): ', saved_config['savedckpdir'])
            best_cnn_weights = tf.train.latest_checkpoint(saved_config['savedckpdir'])
            print('best_cnn_weights: ', best_cnn_weights)
        else:
            # Manually provide the best CNN weight checkpoint info. 
            # It is discouraged to use this approach. 
            print('best cnn weight (others): ', self.BestCNNWeight)
            print('auto restart dir to load: ', self.restart_dir)
            print('check: ', '/'.join(self.restart_dir.split('/')[:-1]) + '/' + self.BestCNNWeight)
            best_cnn_weights = tf.train.latest_checkpoint('/'.join(self.restart_dir.split('/')[:-1]) + '/' + self.BestCNNWeight)
            print('best_cnn_weights: ', best_cnn_weights)

        reader = py_checkpoint_reader.NewCheckpointReader(best_cnn_weights)
        var_to_shape_map = reader.get_variable_to_shape_map()
        var_to_dtype_map = reader.get_variable_to_dtype_map()
        #print(var_to_shape_map)
        #print(var_to_dtype_map)

        saved_kernel = []
        saved_bias = []
        for key, value in natsorted(var_to_shape_map.items()):
            # print(key, value)
            if key.find('all_layers') >= 0 and key.find('OPTIMIZER') < 0 :
                # print(key, value)
                val0 = None
                if key.find('kernel') >= 0:
                    val0 = reader.get_tensor(key)
                    saved_kernel.append(val0)
                if key.find('bias') >= 0:
                    val0 = reader.get_tensor(key)
                    saved_bias.append(val0)

        # print('all trainable variable:',self.model.trainable_variables )
        # print('total(all trainable variable):',len(self.model.trainable_variables))
        kernel_ind = 0 
        bias_ind = 0
        # to_untrainable = []
        v_ind = 0
        for v0 in self.model.trainable_variables :
            if v0.name.find('kernel_posterior_loc') >= 0 :
                v0_value = v0.value().numpy()
                # print('v0= ', type(v0), v0.name, np.shape(v0_value))
                v0.assign(saved_kernel[kernel_ind])
                kernel_ind += 1
                # to_untrainable.append(v_ind)
                #v0.assign(v0_value)
            if v0.name.find('bias_posterior_loc') >= 0 :
                v0_value = v0.value().numpy()
                # print('v0= ', type(v0), v0.name, np.shape(v0_value))
                v0.assign(saved_bias[bias_ind])
                bias_ind += 1
                #v0.assign(v0_value)
            if v0.name.find('untransformed_scale') >= 0 :
                # v0.assign(v0.value().numpy()*0.001)
                v0.assign(v0.value().numpy()*2.0)
                # v0.assign(v0.value().numpy()) # still bad
                # v0.assign(v0.value().numpy()*0.5) # very bad
                # v0.assign(v0.value().numpy()*0.25) # very bad, the distribution is wide, and loss are huge
                # print('v0= ', type(v0), v0.name, np.shape(v0_value))
            v_ind += 1
        if len(saved_kernel) != kernel_ind:
            print('saved kernel: ', saved_kernel, 'kernel_ind: ', kernel_ind)
            raise ValueError("WARNING: loaded cnn saved kernel numbers != bnn kernel numbers, might load wrong model")
        if len(saved_bias) != bias_ind:
            print('saved bias: ', saved_bias, 'bias_ind: ', bias_ind)
            raise ValueError("WARNING: loaded cnn saved bias numbers != bnn bias numbers, might load wrong model")

        # #gets a reference to the list containing the trainable variables
        # print('trainable variable: ', len(self.model.trainable_variables))
        # # -----following not working-----
        # trainable_collection = tf.compat.v1.get_collection_ref(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
        # print(trainable_collection)
        # variables_to_remove = list()
        # for vari in trainable_collection:
            # #uses the attribute 'name' of the variable
            # if vari.name=="batch_normalization/gamma:0" or vari.name=="batch_normalization/beta:0":
                # variables_to_remove.append(vari)
        # for rem in variables_to_remove:
            # trainable_collection.remove(rem)
        # -----following not working-----
        
        # #It is very difficult to make mean to untrainable, and we should not do so.
        # print('type of trainable_variables: ', type(self.model.trainable_variables))
        # to_untrainable = [x for x in to_untrainable[::-1]]
        # print('to_untrainable: ', to_untrainable)
        # for v0 in to_untrainable:
            # del self.model.trainable_variables[v0]

            # # print('new v0: ', v0)
        # # print('all trainable weights:',self.model.trainable_weights )
        # exit(0)
        # print('get layers: ', self.model.get_layer(index=1))
        # print('layers kernel: ', self.model.get_layer(index=1).trainable_weights)
        # # print('layers bias: ', self.model.get_layer(index=1).bias)
        # print('layer trainable_variables: ', self.model.get_layer(index=1).trainable_variables)
        # print('all trainable variable:',self.model.trainable_variables )
        # print('all trainable weights:',self.model.trainable_weights )
        # print("Successfully load weight: ", latest)
        # print('layers: ', self.model.layers)
        # print('get layers: ', self.model.get_layer(index=0))
        # self.model.summary()
        return 1

    def _build_model(self):
        """ 
        Build the weak-PDE constrained NN model.
        """

        # if tf.config.list_physical_devices('GPU'):
          # # physical_devices = tf.config.list_physical_devices('GPU')
          # # print(physical_devices)
          # # strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
          # self.mirrored_strategy = tf.distribute.MirroredStrategy()
        # else:  # use default strategy
          # self.mirrored_strategy = tf.distribute.get_strategy() 

        # with self.mirrored_strategy.scope():
            # print(tf.Variable(1.))
            # self.model = BNN_user_weak_pde_general(
                    # layers_str=self.config['NN']['NNArchitecture'],
                    # NUM_TRAIN_EXAMPLES=len(self.train_seq), # total batch numbers
                    # Sigma2=self.Sigma2)
            # print(self.model)
            # self.optimizer = self._build_optimizer()

        self.model = BNN_user_weak_pde_general(
                layers_str=self.config['NN']['NNArchitecture'],
                NUM_TRAIN_EXAMPLES=len(self.train_seq), # total batch numbers
                Sigma2=self.Sigma2)
        self.optimizer = self._build_optimizer()

        self.model.compile(
            loss = self._build_loss(),
            optimizer = self.optimizer,
            experimental_run_tf_function = False,    # allow the kl-call in the layer structure
        )

    # all the data need to be converted to dataset, requires significant change of the code structure. 
    # @tf.function
    # def distributed_train_step(dist_inputs):
      # per_replica_losses = self.mirrored_strategy.run(train_step, args=(dist_inputs,))
      # return self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,axis=None)


    def _train(self):
        """
        Use batch-optimization to train the model.
        """

        if self.epochs > 0:
            cmd = "mkdir -p " + self.restart_dir
            exe_cmd(cmd)

        self.model_train_loss = []
        self.model_val_loss = []
        model_loss = 1000

        checkpoint_path = self.restart_dir + "/ckpt"    # it's difficult to include time info, as we need to restart simulation

        restart_at = 0
        if self.args.restartfrom:
            print('checkpoint_dir to load restart: ', self.restart_dir_to_load)
            latest = tf.train.latest_checkpoint(self.restart_dir_to_load)

            # load specific ckpt.
            if self.args.restartat > 0:
                latest = self.restart_dir_to_load + '/ckpt' + str(self.args.restartat).zfill(4)
            print("latest checkpoint: ", latest)
            
            # check if ckpt exist
            if (latest != None):
                self.filename += '-ra-' + latest.split('/')[-1]
                self.model.load_weights(latest)
                print("Successfully load weight: ", latest)
                # return 1
            else:
                print("No saved weights, start to train the model from the beginning!")
                pass

        self.losses = {'loss': [], 'val_loss': [], 'mse_loss':[], 'res_body':[], 'res_neu':[]}
        self.var_sigma2 = []

        # print model information
        input_shape=(None, np.shape(self.features)[1], np.shape(self.features)[2], np.shape(self.features)[3])
        self.model.build(input_shape) # `input_shape` is the shape of the input data
        self.model.summary()

        # load optimized CNN parameters to BNN if specified.
        print('self.args.restartfrom:', self.args.restartfrom)
        if self.args.restartfrom == '' :
            if (self._load_saved_cnn_model()):
                self.InitialEpoch = 0

        if self.FixLoc == 1:
            customized_trainer = True
        else:
            customized_trainer = False
        print("use customized_trainer:", customized_trainer)
        if customized_trainer:
            tvars = self.model.trainable_variables
            none_loc_vars = [var for var in tvars if '_loc' not in var.name]
            none_scale_vars = [var for var in tvars if '_scale' not in var.name]
            print(len(tvars), len(none_loc_vars), len(none_scale_vars))

        def my_loss(y_true, y_pred):

            if self.UseTwoNeumannChannel :
                inputs = y_pred[:,:,:,self.dof:4*self.dof] # new Neumann Channel
            else:
                inputs = y_pred[:,:,:,self.dof:3*self.dof] # old Neumann Channel
            y_pred = y_pred[:,:,:,0:self.dof]
            dist = tfp.distributions.Normal(loc=tf.zeros_like(y_pred), scale=self.Sigma1)
            y_noise = tf.squeeze(dist.sample(1), [0]) # only sample 1, thus, lead dimension can be squeezed. 
            y_pred = y_pred + y_noise
            R_red, y_pred, y_true_dummy, _, _, _ = self._compute_residual(inputs, y_pred)
            dist = tfp.distributions.Normal(loc=tf.zeros_like(y_true_dummy), scale=self.model.Sigma2)
            return self.BetaMSELoss * tf.reduce_mean(tf.square(tf.where(y_true_dummy > -0.9, tf.random.normal(tf.shape(y_true_dummy), 0.5, 0.05, tf.float32, seed=1), tf.zeros_like(y_true_dummy)) - tf.where(y_true_dummy > -0.9, y_pred, tf.zeros_like(y_pred)))) + self.BetaPDELoss * tf.keras.backend.sum(tf.reduce_mean(-dist.log_prob(R_red), 0))

        time_elapsed_list = []
        for epoch in range(self.epochs):
            start_time = time.time()
            # The first half of training only train data with Neumann BCs.
            # The second half train all the data.
            if self.NeumannFirst:
                if epoch == 0:
                    self._load_data(only_neumann_data=True)
                elif epoch == int(self.epochs * 0.5):
                    self._load_data(only_neumann_data=False)

            # Coefficients to enable the epoch initialization.
            if epoch < self.InitialEpoch:
                self.BetaMSELoss.assign(float(1.0))
                self.BetaPDELoss.assign(float(0.0))
            else:
                self.BetaMSELoss.assign(float(0.0))
                self.BetaPDELoss.assign(float(1.0))

            # if (epoch+1)%50 == 0:
            # print('epoch:', epoch)
            epoch_loss = []
            for step, (batch_x, batch_y) in enumerate(self.train_seq):
                if customized_trainer:
                    with tf.GradientTape() as t:
                        current_loss = my_loss(batch_y, self.model(batch_x))
                    # grads = t.gradient(current_loss, tvars)
                    # self.optimizer.apply_gradients(zip(grads,tvars))
                    grads = t.gradient(current_loss, none_loc_vars)
                    self.optimizer.apply_gradients(zip(grads,none_loc_vars))
                    # print(current_loss)
                    # print("...training...", current_loss)
                    # exit(0)
                    epoch_loss.append(current_loss)
                else:
                    batch_loss = self.model.train_on_batch(batch_x, batch_y)
                    epoch_loss.append(batch_loss)
                    # print("...training...", batch_loss)


            epoch_val_loss = []
            epoch_mse_loss = []
            epoch_res_body = []
            epoch_res_neu = []
            if self.val_seq is not None:
                for step, (batch_x, batch_y) in enumerate(self.val_seq):
                    val_loss = self.model.test_on_batch(batch_x, batch_y)
                    epoch_val_loss.append(val_loss)

                    #-------------------------- DEBUG Loss--------------------------
                    y_pred = self.model.predict_on_batch(batch_x) 
                    y_true = batch_y

                    if self.UseTwoNeumannChannel :
                        inputs = y_pred[:,:,:,self.dof:4*self.dof] # New Neumann Channel
                    else:
                        inputs = y_pred[:,:,:,self.dof:3*self.dof] # Old Neumann Channel
                    y_pred = y_pred[:,:,:,0:self.dof]
                    R_red, y_pred, y_true_dummy, R_body, dR_neumann, _ = self._compute_residual(inputs, y_pred)

                    mse_loss = 0.0
                    res_body = 0.0
                    res_neu = 0.0

                    mse_loss = self.BetaMSELoss * (tf.reduce_mean(tf.square(tf.where(y_true_dummy > -0.9, 0.5 * tf.ones_like(y_true_dummy, dtype=tf.float32), tf.zeros_like(y_true_dummy, dtype=tf.float32)) - tf.where(y_true_dummy > -0.9, y_pred, tf.zeros_like(y_pred, dtype=tf.float32)))) )
                    res_body = self.BetaPDELoss * tf.reduce_mean(tf.reduce_sum(tf.square(R_body), axis=[1,2,3]))
                    res_neu = self.BetaPDELoss * tf.reduce_mean(tf.reduce_sum(tf.square(dR_neumann), axis=[1,2,3]))

                    epoch_mse_loss.append(mse_loss)
                    epoch_res_body.append(res_body)
                    epoch_res_neu.append(res_neu)
                    #------------------------------------------------- end of loss debug-----------------

            time_elapsed = time.time() - start_time
            time_elapsed_list.append(time_elapsed)
            print("time_elapsed = %s " % time_elapsed)

            self.losses['loss'].append(tf.reduce_mean(epoch_loss))
            self.losses['val_loss'].append(tf.reduce_mean(epoch_val_loss))

            #----------------------- DEBUG -----------------------
            epoch_mse_loss = tf.reduce_mean(epoch_mse_loss)
            epoch_res_body = tf.reduce_mean(epoch_res_body)
            epoch_res_neu = tf.reduce_mean(epoch_res_neu)
            self.losses['mse_loss'].append(epoch_mse_loss.numpy())
            self.losses['res_body'].append(epoch_res_body.numpy())
            self.losses['res_neu'].append(epoch_res_neu.numpy())
            #----------------------- DEBUG -----------------------

            # if epoch % 50 == 0:
            if epoch % 1 == 0:
                # print('Epoch: {}, loss: {:.4e}, val_loss: {:.4e}'.format(epoch, self.losses['loss'][epoch], self.losses['val_loss'][epoch]))
                print('Epoch: {}, loss: {:.4e}, val_loss: {:.4e}'.format(epoch, self.losses['loss'][epoch], self.losses['val_loss'][epoch]),
                        'mse: {:.4e}'.format(epoch_mse_loss.numpy()), 
                        'res_body: {:.4e}'.format(epoch_res_body.numpy()), 
                        'res_neu: {:.4e}'.format(epoch_res_neu.numpy()), 
                        'var(Sigma2): {:.4e}'.format(tf.math.pow(self.model.Sigma2.numpy(),2)),
                        'std(Sigma2): {:.4e}'.format(self.model.Sigma2.numpy()),
                        )
            self.var_sigma2.append(tf.math.pow(self.model.Sigma2.numpy(),2))
            # exit(0)

            # save check points every 10 epoches
            if epoch % 100 == 0:
                self.model.save_weights(checkpoint_path + str(epoch).zfill(4), save_format='tf')


            # save as pickle every 100 epoches
            if epoch % 100 == 0 or epoch == 1:
                self.simulation_results = {
                        'configdata':self.config,
                        'restartedfrom': self.restart_dir_to_load,
                        'savedckpdir':self.restart_dir,
                        'losses':self.losses,
                        'var_sigma2': self.var_sigma2,
                        }
                pickle_out = open(self.filename + '.pickle', "wb")
                pickle.dump(self.simulation_results, pickle_out)
                pickle_out.close()
                print('save to: ', self.filename + '.pickle')
        print("BatchSize: {}, Averaged time per epoch: {:.8f} s".format(self.batch_size, np.mean(np.array(time_elapsed_list[1:]))))

        # save the last epoch
        if self.epochs > 0:
            self.model.save_weights(checkpoint_path + str(epoch).zfill(4), save_format='tf')

            plt.semilogy(self.losses['loss'], 'b')
            plt.semilogy(self.losses['val_loss'], 'r')
            plt.legend(['loss', 'val_loss'])
            plt.xlabel('epoch')
            plt.savefig(self.filename+'-loss.png')
            print('save to:', self.filename+'-loss.png')

            if self.isBNN:
                plt.clf()
                plt.semilogy(self.var_sigma2, 'b')
                plt.legend(['var(sigma2)'])
                plt.xlabel('epoch')
                plt.savefig(self.filename+'-sigma2.png')
                print('save to:', self.filename+'-sigma2.png')

    def debug_problem(self, use_label=False):
        """ for debugging purpose """
        self._load_data()

        # show the plots how many times:
        features = self.features
        labels = self.labels

        # features = tf.convert_to_tensor(features, dtype=tf.float32)
        # labels = tf.convert_to_tensor(labels, dtype=tf.float32)
        # print( pde_layers.LayerFillRandomToBCs(stddev=0.005)(features) )

        try: 
            prediction = self.model.predict(self.features)
            # features = y_pred[:,:,:,self.dof:3*self.dof]
            y_pred = prediction[:,:,:,0:self.dof] 
        except:
            prediction = pde_layers.LayerFillRandomNumber()(features)
            y_pred = prediction[:,:,:,0:self.dof] 

        bc_mask_dirichlet = pde_layers.ComputeBoundaryMaskNodalData(features, dof=self.dof, opt=1)
        bc_mask_neumann = pde_layers.ComputeBoundaryMaskNodalData(features, dof=self.dof, opt=2)
        if self.UseTwoNeumannChannel :
            neumann_residual = pde_layers.ComputeNeumannBoundaryResidualNodalDataNew(features, dh=self.dh, dof=self.dof)
        else:
            neumann_residual = pde_layers.ComputeNeumannBoundaryResidualNodalData(features, dh=self.dh, dof=self.dof)

        if use_label:
            y_pred = labels

        print('features',tf.shape(features))
        print('labels',tf.shape(labels))
        print('bc_mask_dirichlet:', tf.shape(bc_mask_dirichlet))

        print('bc_mask_neumann:', tf.shape(bc_mask_neumann))

        reverse_bc_mask_dirichlet = tf.where( bc_mask_dirichlet == 0, tf.fill(tf.shape(bc_mask_dirichlet), 1.0), tf.fill(tf.shape(bc_mask_dirichlet), 0.0))
        print('reverse_bc_mask_dirichlet :', tf.shape(reverse_bc_mask_dirichlet ))

        reverse_bc_mask_neumann = tf.where( bc_mask_neumann == 0, tf.fill(tf.shape(bc_mask_neumann), 1.0), tf.fill(tf.shape(bc_mask_neumann), 0.0))
        print('reverse_bc_mask_neumann :', tf.shape(reverse_bc_mask_neumann ))

        mask_labels = tf.multiply(labels, reverse_bc_mask_dirichlet) 
        print('mask_labels :', tf.shape(mask_labels ))
        print('neumann_residual :', tf.shape(neumann_residual ), tf.reduce_sum(tf.square( neumann_residual )))
        
        print('y_pred',tf.shape(y_pred))

        input_dirichlet = labels
        dirichlet_bc = tf.multiply(input_dirichlet, reverse_bc_mask_dirichlet)
        y_pred = tf.multiply(y_pred, bc_mask_dirichlet)
        y_pred = y_pred + dirichlet_bc
        print('dirichlet_bc', tf.shape(dirichlet_bc))

        elem_bulk_residual=self._bulk_residual(y_pred)

        print('elem_bulk_residual',tf.shape(elem_bulk_residual))
        elem_residual_mask = pde_layers.GetElementResidualMask(labels)
        print('elem_residual_mask',tf.shape(elem_residual_mask))
        R = pde_layers.GetNodalInfoFromElementInfo(elem_bulk_residual, elem_residual_mask, dof=self.dof)
        R_fix = tf.where(dirichlet_bc==0.5, R, 0.0)
        F_mean = tf.reduce_sum(R_fix, axis=[1,2])
        print('F_mean', F_mean)
        print('R',tf.shape(R))
        R_norm = tf.norm( R, axis=[1,2])
        R_reduce_mean = tf.reduce_mean(tf.square( R ))
        R_reduce_sum = tf.reduce_sum(tf.square( R ))
        R_reduce_mean_norm = tf.reduce_mean(R_norm)
        # c = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        # R_norm = tf.norm( c, axis=1, keepdims=True)
        # print(R_norm)
        # print(R_reduce_mean)
        # print(R_reduce_mean_norm)
        # print(R_reduce_sum)
        # R_new = R - neumann_residual

        R_neumann = tf.multiply(R, reverse_bc_mask_neumann)
        print('R_neumann :', tf.shape(R_neumann ))
        delta_R_neumann = R_neumann - neumann_residual
        print('delta_R_neumann :', tf.shape(delta_R_neumann ))

        R = R - neumann_residual

        dist = tfp.distributions.Normal(loc=tf.zeros_like(R), scale=0.00001)
        print( tf.keras.backend.sum(-dist.log_prob(R)) )

        R_bc_mask_dirichlet = tf.multiply(R, bc_mask_dirichlet) 
        print('R_bc_mask_dirichlet:', tf.shape(R_bc_mask_dirichlet))
        R_bc_mask_dirichlet_neumann = tf.multiply(R_bc_mask_dirichlet, bc_mask_neumann) 
        print('R_bc_mask_dirichlet_neumann:', tf.shape(R_bc_mask_dirichlet_neumann))

        print(tf.reduce_mean(tf.square( R_bc_mask_dirichlet )))
        print(tf.reduce_mean(tf.reduce_sum(tf.square(R_bc_mask_dirichlet_neumann), axis=[1,2,3]))) 
        print(tf.reduce_mean(tf.reduce_sum(tf.square(delta_R_neumann), axis=[1,2,3])))

        plot_fields(
                list_of_field = [
                    features[0:1, :, :, 0:self.dof], 
                    features[0:1, :, :, self.dof:2*self.dof], 
                    labels[0:1, :, :, 0:self.dof],  
                    y_pred[0:1, :, :, 0:self.dof]],
                list_of_field_name = [
                    'Dirichlet', 
                    'Neumann', 
                    'Label', 
                    'Pred. Sol.'], 
                dof = self.dof, 
                dof_name = self.dof_name,
                filename = 'results/' + self.problem_name + '-Sol.png')

        plot_fields(
                list_of_field = [
                    bc_mask_dirichlet[0:1, :, :, 0:self.dof], 
                    reverse_bc_mask_dirichlet[0:1, :, :, 0:self.dof], 
                    mask_labels[0:1, :, :, 0:self.dof], 
                    bc_mask_neumann[0:1, :, :, 0:self.dof],  
                    reverse_bc_mask_neumann[0:1, :, :, 0:self.dof]],
                list_of_field_name = [
                    'BC_Mask_Dirichlet', 
                    'Rev. BC_Mask_Dir.', 
                    'Rev. BC_Mask_Dir. * Labels', 
                    'BC_Mask_Neumann', 
                    'Rev. BC_Mask_Neu'], 
                dof = self.dof, 
                dof_name = self.dof_name,
                filename = 'results/' + self.problem_name + '-BCs.png')

        plot_fields(
                list_of_field = [
                    R[0:1, :, :, 0:self.dof], 
                    tf.tile(elem_residual_mask[0:1, :, :, 0:1], [1,1,1,self.dof]), 
                    neumann_residual[0:1, :, :, 0:self.dof], 
                    delta_R_neumann[0:1, :, :, 0:self.dof],  
                    R_bc_mask_dirichlet_neumann[0:1, :, :, 0:self.dof]],
                list_of_field_name = [
                    'Nodal R', 
                    'Elem Residual Mask', 
                    'Neumann R', 
                    'dR Neumann', 
                    'R * BC_M_Dir. * BC_M_Neu.'], 
                dof = self.dof, 
                dof_name = self.dof_name,
                filename = 'results/' + self.problem_name + '-R.png')


    def test(self, test_folder='', plot_png=True, output_reaction_force=False):
        """
        Make prediction with the surrogate model

        Args:
            test_folder (str): folder name under relative to the DataPath in the config.ini file.

        Note:
            - If test_folder is not specified, this subroutine will make prediction based on test_seq that is split from the training dataset. 
            - If test_folder is specified, this subroutine will load all the data from the folder to test_seq to make prediction.
            - For deterministic model, the MonteCarloNum from config.ini is over written to 1.

        """
        if test_folder == '':
            only_testing = False
            # return
        else:
            only_testing = True
            test_folder.replace('/', '')
            self._load_data(test_folder=test_folder + '/')

        if self.args.restartfrom:
            if self.model is None:
                self._build_model()
                self._train()

        print(' ... Running monte carlo inference')
        # Compute log prob of heldout set by averaging draws from the model:
        # p(heldout | train) = int_model p(heldout|model) p(model|train)
        #                   ~= 1/n * sum_{i=1}^n p(heldout | model_i)
        # where model_i is a draw from the posterior p(model|train).
        predictions = []
        reaction_force = []
        print(len(self.test_seq))
        # if not test_folder:
            # self.monte_carlo_num = 200
        for _ in range(self.monte_carlo_num):
            y_pred = self.model.predict(self.test_seq, verbose=1) 

            # # for scaling test
            # start_time = time.time()
            # y_pred = self.model.predict(self.test_seq, verbose=1) 
            # print("--- %s m seconds ---" % ((time.time() - start_time) * 1000/4096))
            # exit(0)

            if self.UseTwoNeumannChannel :
                inputs = y_pred[:,:,:,self.dof:4*self.dof] # New Neumann Channel
            else:
                inputs = y_pred[:,:,:,self.dof:3*self.dof] # Old Neumann Channel
            y_pred = y_pred[:,:,:,0:self.dof]
            if output_reaction_force:
                _, y_pred, _, _, _, R_fix = self._compute_residual(inputs, y_pred)
                reaction_force.append(R_fix)
            else:
                y_pred= self._compute_residual(inputs, y_pred, only_y_pred=True)
            predictions.append(y_pred)

        probs = tf.stack(predictions, axis=0)
        # print(" ... probs ...", np.shape(probs))
        mean_probs = tf.reduce_mean(probs, axis=0)
        # print(" ... mean_probs ...", np.shape(mean_probs))
        std_probs = tf.math.reduce_std(probs, axis=0)
        # print(" ... std_probs ...", np.shape(std_probs))
        expand_mean_probs = tf.tile(tf.expand_dims(mean_probs, axis=0), [self.monte_carlo_num, 1, 1, 1, 1] )
        # print(" ... expand mean_probs ...", np.shape(expand_mean_probs))
        var_probs = tf.reduce_mean( tf.math.pow(probs - expand_mean_probs, 2), axis=0)
        # print(" ... var_probs ...", np.shape(var_probs))

        if output_reaction_force:
            # output reaction forces at the location where x=0, and y=0.
            # the loadings are the DNS actually value without scaling
            # ux, uy, tx, ty, fx_mean, fy_mean, fx_std, fy_std 
            R_f = tf.stack(reaction_force, axis=0)
            mean_R_f = tf.reduce_mean(R_f, axis=0) 
            std_R_f = tf.math.reduce_std(R_f, axis=0) 
            # print(np.shape(mean_R_f), np.shape(std_R_f))
            F_mean = tf.reduce_sum(mean_R_f, axis=[1,2])
            F_std = tf.reduce_sum(std_R_f, axis=[1,2])
            print('reaction force(mean): ', F_mean)
            print('reaction force(std): ', F_std)
            Loadings = 2.0 * tf.reduce_max(inputs, axis=[1,2])-1
            _filename = self.filename + '-' + test_folder + 'F' + '.npy'
            all_force_info = tf.concat([Loadings, F_mean, F_std], axis=1)
            np.save(_filename, all_force_info)

            # print('F_mean ', F_mean)
            # print('F_std ', F_std)
            # print('BCs max (scaled)', tf.reduce_max(inputs, axis=[1,2]))
            # print('BCs max (original)', 2.0 * tf.reduce_max(inputs, axis=[1,2])-1)
        
        # heldout_log_prob = tf.reduce_mean(tf.math.log(mean_probs))
        # print(' ... Held-out nats: {:.3f}'.format(heldout_log_prob))

        if only_testing:
            if plot_png:
                for i in range(0, tf.shape(self.test_label)[0]):
                    if self.UseTwoNeumannChannel :
                        plot_PDE_solutions_new(
                                img_input = self.test_dataset[i:i+1, :, :, 0:3*self.dof], 
                                img_label = self.test_label[i:i+1, :, :, 0:self.dof], 
                                img_pre_mean = mean_probs[i:i+1, :, :, 0:self.dof], 
                                img_pre_var = var_probs[i:i+1, :, :, 0:self.dof], 
                                img_pre_std = std_probs[i:i+1, :, :, 0:self.dof], 
                                dof=self.dof, 
                                dof_name=self.dof_name, 
                                tot_img=self.tot_img, 
                                filename = self.filename + '-' + test_folder.replace('/','_') + '-' + str(i) + '.png'
                                )
                    else:
                        plot_PDE_solutions(
                                img_input = self.test_dataset[i:i+1, :, :, 0:2*self.dof], 
                                img_label = self.test_label[i:i+1, :, :, 0:self.dof], 
                                img_pre_mean = mean_probs[i:i+1, :, :, 0:self.dof], 
                                img_pre_var = var_probs[i:i+1, :, :, 0:self.dof], 
                                img_pre_std = std_probs[i:i+1, :, :, 0:self.dof], 
                                dof=self.dof, 
                                dof_name=self.dof_name, 
                                tot_img=self.tot_img, 
                                filename = self.filename + '-' + test_folder.replace('/','_') + '-' + str(i) + '.png'
                                )
                    print('save to: ', self.filename + '-' + test_folder.replace('/','_') + '-' + str(i) + '.png')

            return self.test_dataset[:, :, :, 0:3*self.dof], self.test_label[:, :, :, 0:self.dof], mean_probs[:, :, :, 0:self.dof], var_probs[:, :, :, 0:self.dof], std_probs[:, :, :, 0:self.dof]
        else:
            # just plot one data point for visualization.

            if self.UseTwoNeumannChannel :
                plot_PDE_solutions_new(
                        img_input = self.test_dataset[0:1, :, :, 0:3*self.dof], 
                        img_label = self.test_label[0:1, :, :, 0:self.dof], 
                        img_pre_mean = mean_probs[0:1, :, :, 0:self.dof], 
                        img_pre_var = var_probs[0:1, :, :, 0:self.dof], 
                        img_pre_std = std_probs[0:1, :, :, 0:self.dof], 
                        dof=self.dof, 
                        dof_name=self.dof_name, 
                        tot_img=self.tot_img, 
                        filename = self.filename + '.png'
                        )
            else:
                plot_PDE_solutions(
                        img_input = self.test_dataset[0:1, :, :, 0:2*self.dof], 
                        img_label = self.test_label[0:1, :, :, 0:self.dof], 
                        img_pre_mean = mean_probs[0:1, :, :, 0:self.dof], 
                        img_pre_var = var_probs[0:1, :, :, 0:self.dof], 
                        img_pre_std = std_probs[0:1, :, :, 0:self.dof], 
                        dof=self.dof, 
                        dof_name=self.dof_name, 
                        tot_img=self.tot_img, 
                        filename = self.filename + '.png'
                        )
            print('save to: ', self.filename + '.png')


            # only simulations with new restarted files are saved 
            if self.epochs >= 100:
                # additional simulation results are saved to pickle for future post processing purpose.
                self.simulation_results['features'] = self.test_dataset[0:10, :, :, 0:2*self.dof]
                self.simulation_results['labels'] = self.test_label[0:10, :, :, 0:self.dof]
                self.simulation_results['mean'] = mean_probs[0:10, :, :, 0:self.dof]
                self.simulation_results['var'] = var_probs[0:10, :, :, 0:self.dof]
                self.simulation_results['std'] = std_probs[0:10, :, :, 0:self.dof]

                pickle_out = open(self.filename + '.pickle', "wb")
                pickle.dump(self.simulation_results, pickle_out)
                pickle_out.close()
                print('save to: ', self.filename + '.pickle')


    def test_residual_gaussian(self, noise_std=1.e-3, sample_num=10000):
        """
        Test the residual noise distribution based on a Gaussian perturbation to inputs.

        Args:
            noise_std (float): default (1.0e-3)
            sample_num (int): default (10000)

        Note:
        It is preferred to use the DNS label data to make the test as the actually residual (mean) from such data is very small. 
        By default, it will load data from DataPath/DNS. Only the first data point will be used.
        """

        self._load_data()

        x_dim = tf.shape(self.labels).numpy()[1]
        y_dim = tf.shape(self.labels).numpy()[2]

        filename = 'results/' + self.problem_name + '-num-' + str(sample_num) + '-std-' + "{:.1e}".format(noise_std)

        dummy_batch = 500 # if too big, we get gpu memory issue
        dpi = 150

        if tf.shape(self.labels)[0] != 1:
            self.labels = self.labels[0:1,:,:,:]
            self.features = self.features[0:1,:,:,:]
            # raise ValueError("the residual Gaussian test only work for self.labels with first dim = 1, tf.shape(self.labels)=", tf.shape(self.labels))

        self.features = tf.convert_to_tensor(self.features, dtype=tf.float32)
        self.labels = tf.convert_to_tensor(self.labels, dtype=tf.float32)
        dist = tfp.distributions.Normal(loc=tf.zeros_like(self.labels), scale=noise_std)

        features = tf.tile(self.features, [sample_num,1,1,1])
        labels   = tf.tile(self.labels,   [sample_num,1,1,1])
        y_pred = labels
        # print(tf.shape(y_pred))
        y_noise = tf.squeeze(dist.sample(sample_num), [1]) # after sampling, the 2nd dim is 1.
        y_pred = y_pred + y_noise
        y_pred = tf.concat([self.labels, y_pred[1:,:,:,:]], axis=0) # replace 0 with data point without perturbation
        # print(tf.shape(y_pred))

        dummy_seq = np.array_split(np.arange(sample_num), int(sample_num/dummy_batch)+1)
        # print(dummy_seq)
        R_all = []
        R_list = []
        for s0 in dummy_seq:
            start = s0[0]
            end = s0[-1] + 1 # because start:end will not count the end, +1 will make sure end index will be counted
            R, _, _, _, _, _ = self._compute_residual(features[start:end,:,:,:], y_pred[start:end,:,:,:])
            # print(tf.shape(R))
            R_list.append(R)
        R = np.vstack(R_list)
        # print(tf.shape(R))

        for i0 in range(0, self.dof):
            plot_one_field(data=R[:,:,:,i0], x_dim=2, y_dim=2, dpi=dpi, name=filename + '-' + self.dof_name[i0] + '-R.png')
            plot_one_field_stat(data=R[:,:,:,i0], dpi=dpi, name=filename + '-' + self.dof_name[i0] + '-R-stat.png')
            plot_one_field_hist(data=R[:,:,:,i0], x_dim=x_dim, y_dim=y_dim, dpi=dpi, name=filename + '-' + self.dof_name[i0] + '-R-hist.png')
            plot_one_field_hist(data=y_pred[:,:,:,i0], x_dim=x_dim, y_dim=y_dim, dpi=dpi, name=filename + '-' + self.dof_name[i0] + '-solution-hist.png')


    def run(self):
        """ 
        Run the model by performing:

        - load data
        - build model
        - train
        - test
        """
        self._load_data()
        self._build_model()
        self._train()
        self.test()
            
if (__name__ == '__main__'):
    model = PDEWorkflowSteadyState()
    model.run()
