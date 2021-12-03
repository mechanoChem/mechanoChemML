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
import mechanoChemML.src.kfold_train as KFoldTrain

args = mrnn_utility.sys_args()

args.configfile = 'dnn-free-energy-1dns-final.ini'

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

the_kfolds = KFoldTrain.MLKFold(n_splits, dataset)
train_dataset, train_labels, val_dataset, val_labels, test_dataset, test_labels, test_derivative = the_kfolds.get_next_fold(dataset, labels, derivative, final_data=True)

model_summary_list = []

config['RESTART']['CheckPointDir'] = './saved_weight'
config['MODEL']['ParameterID'] = ''
checkpoint_dir = config['RESTART']['CheckPointDir'] + config['MODEL']['ParameterID']
model_path = checkpoint_dir + '/' + 'model.h5'

model = mrnn_models.build_model(config, train_dataset, train_labels)

if (config['RESTART']['RestartWeight'].lower() == 'y'):
    print('checkpoint_dir for restart: ', checkpoint_dir)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    print("latest checkpoint: ", latest)
    if (latest != None):
        model.load_weights(latest)
        print("Successfully load weight: ", latest)
    else:
        print("No saved weights, start to train the model from the beginning!")
        pass

metrics = mrnn_utility.getlist_str(config['MODEL']['Metrics'])
optimizer = mrnn_models.build_optimizer(config)
loss = mrnn_models.build_loss(config)
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

callbacks = mrnn_models.build_callbacks(config)
history = model.fit(
    train_dataset,
    train_labels,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(val_dataset, val_labels),
    verbose=verbose,
    callbacks=callbacks)

model.summary()

label_scale = float(config['TEST']['LabelScale'])
all_data = {'test_label': [], 'test_nn': [], 'val_label': [], 'val_nn': [], 'train_label': [], 'train_nn': []}

test_nn = model.predict(test_dataset, verbose=0, batch_size=batch_size)
val_nn = model.predict(val_dataset, verbose=0, batch_size=batch_size)
train_nn = model.predict(train_dataset, verbose=0, batch_size=batch_size)

for i in np.squeeze(test_nn):
    all_data['test_nn'].append(i / label_scale)
for i in np.squeeze(val_nn):
    all_data['val_nn'].append(i / label_scale)
for i in np.squeeze(train_nn):
    all_data['train_nn'].append(i / label_scale)

for i in test_labels['Psi_me']:
    all_data['test_label'].append(i / label_scale)
for i in val_labels['Psi_me']:
    all_data['val_label'].append(i / label_scale)
for i in train_labels['Psi_me']:
    all_data['train_label'].append(i / label_scale)
# print('all_data: ', all_data)

import pickle
import time
now = time.strftime("%Y%m%d%H%M%S")
pickle_out = open('all_data_' + now + '.pickle', "wb")
pickle.dump(all_data, pickle_out)
pickle_out.close()

pickle_out = open('history_' + now + '.pickle', "wb")
pickle.dump(history.history, pickle_out)
pickle_out.close()
print('save to: ', 'all_data_' + now + '.pickle', 'history_' + now + '.pickle')
