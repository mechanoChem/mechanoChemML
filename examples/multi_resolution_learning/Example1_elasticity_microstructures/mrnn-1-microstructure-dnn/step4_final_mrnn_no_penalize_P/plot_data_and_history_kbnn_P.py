#!/usr/bin/env python3
import sys
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np

if __name__ == '__main__':
    if (len(sys.argv) == 1):
        print('please provide all_data_20200601081735.pickle or history_20200601081735.pickle file')
        exit(0)

    timemark = sys.argv[1].split('_')[-1].split('.pickle')[0]
    print('timemark:', timemark)

    history_file = 'history_' + timemark + '.pickle'
    all_data_file = 'all_data_' + timemark + '.pickle'
    print('loading data:', history_file, all_data_file)

    all_data = pickle.load(open(all_data_file, "rb"))
    history = pickle.load(open(history_file, "rb"))

    epoches = range(0, len(history['loss']))

    #----------------------plot 1---------------------------------------
    plt.clf()
    plt.semilogy(epoches, history['loss'], 'b', lw=1.0, label='Training')
    plt.semilogy(epoches, history['val_loss'], 'k', lw=1.0, label='Validation')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    # plt.axis('equal')
    plt.savefig('kbnn-dnn-1-frame-learning.pdf', bbox_inches='tight', format='pdf')
    plt.show()

    #----------------------plot 2---------------------------------------
    plt.clf()
    plt.plot(all_data['test_label'], all_data['test_nn'], 'k.')
    xmin = min(min(all_data['test_label']), min(all_data['test_nn']))
    xmax = max(max(all_data['test_label']), max(all_data['test_nn']))
    plt.plot([xmin, xmax], [xmin, xmax], 'k-', lw=1.0)

    plt.axes().set_aspect('equal', 'box')
    plt.xlim([xmin, xmax])
    plt.ylim([xmin, xmax])
    plt.xlabel('$\Delta\Psi_{\mathrm{mech,DNS}}$')
    plt.ylabel('$\Delta\Psi_{\mathrm{mech,KBNN}}$')
    plt.savefig('kbnn-dnn-1-frame-test.pdf', bbox_inches='tight', format='pdf')
    plt.show()

    #----------------------plot 3---------------------------------------
    all_P_file = 'all_P_' + timemark + '.pickle'
    print('loading P:', all_P_file)
    all_P = pickle.load(open(all_P_file, "rb"))

    def plot_P_one_field(plt, P_true, P_pred, ind0):
        plt.clf()
        plt.plot(P_true[:, ind0], P_pred[:, ind0], 'k.')
        amax = max(np.amax(P_pred[:, ind0]), np.amax(P_true[:, ind0]))
        amin = min(np.amin(P_pred[:, ind0]), np.amin(P_true[:, ind0]))
        if (ind0 == 0):
            pre_fix = 'P11'
        elif (ind0 == 1):
            pre_fix = 'P12'
        elif (ind0 == 2):
            pre_fix = 'P21'
        elif (ind0 == 3):
            pre_fix = 'P22'
        else:
            raise ValueError('Unknown value for ind0')
        # plt.gca().set_title(pre_fix)
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.axis('equal')
        plt.axis('square')
        _ = plt.plot([amin, amax], [amin, amax], 'k-', lw=1.0)
        plt.xlim([amin, amax])
        plt.ylim([amin, amax])
        plt.savefig('kbnn-dnn-1-frame-' + pre_fix + '.pdf', bbox_inches='tight', format='pdf')
        plt.show()

    plot_P_one_field(plt, all_P['P_DNS'], all_P['P_NN'], 0)
    plot_P_one_field(plt, all_P['P_DNS'], all_P['P_NN'], 1)
    plot_P_one_field(plt, all_P['P_DNS'], all_P['P_NN'], 2)
    plot_P_one_field(plt, all_P['P_DNS'], all_P['P_NN'], 3)
