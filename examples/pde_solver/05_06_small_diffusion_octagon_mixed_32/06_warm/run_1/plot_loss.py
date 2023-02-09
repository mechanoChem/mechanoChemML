# from mechanoChemML.workflows.physics_constrained_learning.src.PDESystemDiffusionSteadyState import WeakPDESteadyStateDiffusion as thisPDESystem
# from mechanoChemML.utility.PDEUtil import plot_PDE_solutions, plot_fields, plot_tex, plot_one_loss, plot_sigma2, get_cm
import sys,os
import numpy as np
import pickle
import matplotlib.pyplot as plt
# import tensorflow as tf
import matplotlib.image as mpimg
from PIL import Image
import pandas as pd
import matplotlib as mpl
import glob

def plot_tex(tex=False):
    mpl.style.reload_library()
    plt.style.use('zxx')
    print('find zxx: ', os.path.isfile('zxx.mplstyle'))
    if (os.path.isfile('zxx.mplstyle')):
        plt.style.use('zxx.mplstyle')
    if (tex) :
        plt.style.use('tex')
    print(plt.style.available)
    print(mpl.get_configdir())

def plot_one_loss(losses, png_filename):
    # for key, item in losses.items():
        # print(key)
    plt.clf()
    plot_tex(True)
    print('min loss: ', np.min(losses['loss']))

    if np.min(losses['loss']) < 0:
        plt.plot(losses['loss'], 'b')
        plt.plot(losses['val_loss'], 'k')
        # plt.yscale('symlog')
    else:
        plt.semilogy(losses['loss'], 'b')
        plt.semilogy(losses['val_loss'], 'k')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    # plt.xlim([0,5000])
    plt.tight_layout()
    plt.savefig(png_filename)
    # plt.show()
    print('save to:', png_filename)
    # exit(0)

def plot_sigma2(sigma2, png_filename, sigma1=''):
    plt.clf()
    plot_tex(True)

    plt.semilogy(sigma2, 'k')

    plt.xlabel('epoch')
    plt.ylabel(r'$\Sigma_2$')
    # plt.xlim([0,5000])
    # plt.legend([sigma1])
    plt.title(sigma1)
    plt.tight_layout()
    plt.savefig(png_filename)
    # plt.show()
    print('save to:', png_filename)
    # exit(0)

def flatten_list(regular_list):
    flat_list = [item for sublist in regular_list for item in sublist]
    return flat_list
def plot_train_loss(f1):
    """ Plot loss in the paper format """
    pickle_file = f1
    filename = f1.replace('.pickle', '')

    sigma2 = []
    loss = []
    val_loss = []

    
    pickle_data = pickle.load(open(pickle_file, "rb"))
    for key, val in pickle_data.items():
        print(key, type(val))
    sigma2.append(pickle_data['var_sigma2']) 
    loss.append(pickle_data['losses']['loss'])
    val_loss.append(pickle_data['losses']['val_loss'])
    while pickle_data['restartedfrom'] != '':
        all_potential_pickles = glob.glob(pickle_data['restartedfrom'].replace('restart', 'results')+'*.pickle')
        all_potential_pickles = [x for x in all_potential_pickles if x.find('_pred') < 0]
        # print(all_potential_pickles)
        if len(all_potential_pickles) > 1:
            print("all_potential_pickles should not be more than 1 (exiting):", all_potential_pickles)
            exit(0)
        f1 = all_potential_pickles[0]

        pickle_data = pickle.load(open(f1, "rb"))
        sigma2.insert(0, pickle_data['var_sigma2']) 
        loss.insert(0, pickle_data['losses']['loss'])
        val_loss.insert(0, pickle_data['losses']['val_loss'])
    print(len(sigma2), len(loss), len(val_loss))
    sigma2 = flatten_list(sigma2)
    loss = flatten_list(loss)
    val_loss = flatten_list(val_loss)
    losses = {'loss':loss, 'val_loss':val_loss}

    plot_sigma2(sigma2, filename+'_sigma2.png', sigma1='')
    plot_one_loss(losses, filename+'_loss.png')
    plt.show()

for f1 in sys.argv[1:]:
    # if f1.find('pred') < 0:
    plot_train_loss(f1)
