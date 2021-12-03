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

args.configfile = 'kbnn-load-dnn-1-frame.ini'

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

model = mrnn_models.build_model(config, train_dataset, train_labels, train_stats=train_stats)

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
# print(type(train_dataset))
history = model.fit(
    train_dataset,
    train_labels,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(val_dataset, val_labels),    # or validation_split= 0.1,
    verbose=verbose,
    callbacks=callbacks)

model.summary()
# print("history: " , history.history['loss'], history.history['val_loss'], history.history)

all_data = {'test_label': [], 'test_nn': [], 'val_label': [], 'val_nn': [], 'train_label': [], 'train_nn': []}

test_nn = model.predict(test_dataset, verbose=0, batch_size=batch_size)
val_nn = model.predict(val_dataset, verbose=0, batch_size=batch_size)
train_nn = model.predict(train_dataset, verbose=0, batch_size=batch_size)

for i in np.squeeze(test_nn):
    # print('test_nn:', i)
    all_data['test_nn'].append(i[0] / label_scale)
for i in np.squeeze(val_nn):
    all_data['val_nn'].append(i[0] / label_scale)
for i in np.squeeze(train_nn):
    all_data['train_nn'].append(i[0] / label_scale)

for i in test_labels:
    all_data['test_label'].append(i[0] / label_scale)
    # print('test_label: ', i)
for i in val_labels:
    all_data['val_label'].append(i[0] / label_scale)
for i in train_labels:
    all_data['train_label'].append(i[0] / label_scale)
# print('all_data: ', all_data)
print('test_nn shape: ', np.shape(np.squeeze(test_nn)))
print('test_labels shape: ', np.shape(test_labels))

import pickle
import time
now = time.strftime("%Y%m%d%H%M%S")
pickle_out = open('all_data_' + now + '.pickle', "wb")
pickle.dump(all_data, pickle_out)
pickle_out.close()

pickle_out = open('history_' + now + '.pickle', "wb")
pickle.dump(history.history, pickle_out)
pickle_out.close()

# all_data['P_DNS']= test_labels[:,1:4]/label_scale/train_stats['std'].to_numpy()[0:3]
# all_data['P_NN'] = test_nn[:,1:4]/label_scale/train_stats['std'].to_numpy()[0:3]
all_data['P_DNS'] = test_labels[:, 1:5]
all_data['P_NN'] = test_nn[:, 1:5]

pickle_out = open('all_P_' + now + '.pickle', "wb")
pickle.dump(all_data, pickle_out)
pickle_out.close()

print('save to: ', 'all_data_' + now + '.pickle', 'history_' + now + '.pickle', 'all_P_' + now + '.pickle')
print('the prediction of P and delta Psi_me is not the best model fit with lowest loss!')
