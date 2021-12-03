# for legacy python compatibility
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os, sys
import datetime

import mechanoChemML.workflows.multi_resolution_learning.mrnn_utility as mrnn_utility
import mechanoChemML.workflows.multi_resolution_learning.mrnn_models as mrnn_models
import mechanoChemML.src.hparameters_dnn_grid as HParametersGridDNN
import mechanoChemML.src.kfold_train as KFoldTrain

args = mrnn_utility.sys_args()

args.configfile = 'kbnn-load-dnn-1-frame-hyperparameter-search-comet.ini'
args.platform = 'gpu'
args.inspect = 0
args.debug = False
args.verbose = 1
args.show = 0

mrnn_utility.notebook_args(args)
config = mrnn_utility.read_config_file(args.configfile, args.debug)
dataset, labels, derivative, train_stats = mrnn_utility.load_all_data(config, args)

str_form = config['FORMAT']['PrintStringForm']
epochs = int(config['MODEL']['Epochs'])
batch_size = int(config['MODEL']['BatchSize'])
verbose = int(config['MODEL']['Verbose'])
n_splits = int(config['MODEL']['KFoldTrain'])

parameter = HParametersGridDNN.HyperParametersDNN(
    config,
    input_shape=len(dataset.keys()),
    output_shape=1,
    uniform_sample_number=25,
    neighbor_sample_number=1,
    iteration_time=3,
    sample_ratio=0.3,
    best_model_number=20,
    max_total_parameter=1280,
    repeat_train=n_splits,
    debug=args.debug)

the_kfolds = KFoldTrain.MLKFold(n_splits, dataset)

model_summary_list = []
while True:
    para_id, para_str, train_flag = parameter.get_next_model()
    if (train_flag and the_kfolds.any_left_fold()):
        model_name_id = str(para_id) + '-' + para_str
        print(str_form.format('Model: '), model_name_id)

        checkpoint_dir = config['RESTART']['CheckPointDir'] + config['MODEL']['ParameterID']
        model_path = checkpoint_dir + '/' + 'model.h5'

        train_dataset, train_labels, val_dataset, val_labels, test_dataset, test_labels, test_derivative = the_kfolds.get_next_fold(dataset, labels, derivative)

        model = mrnn_models.build_model(config, train_dataset, train_labels, train_stats=train_stats)
        # https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/Model

        metrics = mrnn_utility.getlist_str(config['MODEL']['Metrics'])
        optimizer = mrnn_models.build_optimizer(config)
        loss = mrnn_models.my_mse_loss_with_grad(BetaP=0.0)
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        label_scale = float(config['TEST']['LabelScale'])

        callbacks = mrnn_models.build_callbacks(config)

        train_dataset = train_dataset.to_numpy()
        train_labels = train_labels.to_numpy()
        val_dataset = val_dataset.to_numpy()
        val_labels = val_labels.to_numpy()
        test_dataset = test_dataset.to_numpy()
        test_labels = test_labels.to_numpy()

        # make sure that the derivative data is scaled correctly

        # The NN/DNS scaled derivative data should be: * label_scale * train_stats['std'] (has already multiplied by label_scale )

        # Since the feature is scaled, and label psi is scaled, the S_NN will be scaled to: label_scale * train_stats['std']
        # the model will scale S_NN back to no-scaled status.
        # here we scale F, and P to no-scaled status
        modified_label_scale = np.array(
            [1.0, 1.0 / label_scale, 1.0 / label_scale, 1.0 / label_scale, 1.0 / label_scale, 1.0 / label_scale, 1.0 / label_scale, 1.0 / label_scale, 1.0 / label_scale])
        train_labels = train_labels * modified_label_scale
        val_labels = val_labels * modified_label_scale
        test_labels = test_labels * modified_label_scale

        history = model.fit(
            train_dataset,
            train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(val_dataset, val_labels),    # or validation_split= 0.1,
            verbose=verbose,
            callbacks=callbacks)

        model.summary()
        parameter.update_model_info(para_id, history.history)
        the_kfolds.update_kfold_status()
