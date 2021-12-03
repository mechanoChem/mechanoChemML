import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
import numpy as np
import datetime
import pickle

import os
import numpy as np
import tensorflow as tf

import mechanoChemML.src.pde_layers as pde_layers

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

def get_cm():
    from matplotlib.colors import LinearSegmentedColormap
    colors = [(0, 0, 1), (0,1,1), (0, 1, 0), (1,1,0), (1, 0, 0)] 
    cmap_name = 'hot'
    # Create the colormap
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)
    cm.set_bad(color='white')
    return cm

def plot_one_loss(pickle_file, png_filename, show_line=True):
    saved_config = pickle.load(open(pickle_file, "rb"))
    # for key, item in saved_config.items():
        # print(key)
    losses = saved_config['losses']
    # for key, item in losses.items():
        # print(key)
    plt.clf()
    plot_tex(True)

    if np.min(losses['loss']) < 0:
        plt.plot(losses['loss'], 'b')
        plt.plot(losses['val_loss'], 'k')
        plt.yscale('symlog')
        if show_line:
            plt.axvline(100, color ='k', linestyle ="--") 
            plt.axvline(500, color ='k', linestyle ="--") 
            plt.axvline(1000, color ='k', linestyle ="--") 
            plt.axvline(2000, color ='k', linestyle ="--") 
            plt.axvline(4000, color ='k', linestyle ="--") 
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

def plot_sigma2(pickle_file, png_filename, show_line=True, sigma1=''):
    saved_config = pickle.load(open(pickle_file, "rb"))
    sigma2 = saved_config['var_sigma2']
    plt.clf()
    plot_tex(True)

    plt.semilogy(sigma2, 'k')

    if show_line: 
        plt.axvline(100, color ='k', linestyle ="--") 
        plt.axvline(500, color ='k', linestyle ="--") 
        plt.axvline(1000, color ='k', linestyle ="--") 
        plt.axvline(2000, color ='k', linestyle ="--") 
        plt.axvline(4000, color ='k', linestyle ="--") 
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

def plot_PDE_solutions_new(img_input, img_label, img_pre_mean, img_pre_var, img_pre_std, dof=1, dof_name=['c'], tot_img=6, filename='', fig_size=3.3):
    """
    plot the results of PDEs

    Args:
        img_input (numpy array): size of [1, :, :, dof*3]
        img_label (numpy array): size of [1, :, :, dof]
        img_pre_mean (numpy array): size of [1, :, :, dof]
        img_pre_var (numpy array): size of [1, :, :, dof]
        img_pre_std (numpy array): size of [1, :, :, dof] 
        dof (int): default (=1)
        dof_name (list): list of string (default ['c'])
        tot_img (int): without plotting std (tot_img=6, default), with std (tot_img=7)
        filename (str): default ('')
    """
    dof_name = ['','']
    hot=get_cm()
    bc_mask_dirichlet = pde_layers.ComputeBoundaryMaskNodalData(img_input, dof=dof, opt=1)
    the_bc_mask_dirichlet = tf.squeeze(bc_mask_dirichlet, [0])

    the_img_input = tf.squeeze(img_input, [0])
    the_img_label = tf.squeeze(img_label, [0])
    the_img_pre_mean = tf.squeeze(img_pre_mean, [0])
    the_img_pre_var = tf.squeeze(img_pre_var, [0])
    the_img_pre_std = tf.squeeze(img_pre_std, [0])
    # print('Dirichlet BC shape:', tf.shape(bc_mask_dirichlet))
    # print('Dirichlet BC:', bc_mask_dirichlet)

    # magic number from: https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    fraction=0.046
    pad=0.04

    # make the margin as NaN ( margin = -1)
    # the problem domain internal part of the input is not filled with random numbers (thus it is = -2)
    tmp_img_label = np.concatenate((the_img_label, the_img_label), axis=2)
    # the_img_input = np.ma.masked_where(tmp_img_label < -0.9, the_img_input)
    the_img_input = np.ma.masked_where(the_img_input <= 0.0, the_img_input)

    the_img_pre_mean = np.ma.masked_where(the_img_label < -0.9, the_img_pre_mean)
    the_img_pre_var = np.ma.masked_where(the_img_label < -0.9, the_img_pre_var)
    the_img_pre_std = np.ma.masked_where(the_img_label < -0.9, the_img_pre_std)

    the_img_label = np.ma.masked_where(the_img_label < -0.9, the_img_label)

    the_img_mark = 1.0e-10 * np.ones(np.shape(the_img_label))

    # remove the Dirichlet BCs region for mean, var, std by setting the value to NaN
    # the_img_pre_mean = np.ma.masked_where(the_bc_mask_dirichlet == 0.0, the_img_pre_mean)
    the_img_pre_var = np.ma.masked_where(the_bc_mask_dirichlet == 0.0, the_img_pre_var)
    the_img_pre_std = np.ma.masked_where(the_bc_mask_dirichlet == 0.0, the_img_pre_std)
    # the_img_label = np.ma.masked_where(the_bc_mask_dirichlet == 0.0, the_img_label)

    figsize_list_x = [x*fig_size*1.1 for x in range(1, 20)]
    figsize_list_y = [x*fig_size for x in range(1, 20)]

    # for debugging purpose only to show intermediate results for NN predictions
    tot_img = tot_img +1 

    fig = plt.figure(figsize=(figsize_list_x[tot_img-1], figsize_list_y[dof-1] ))

    for i0 in range(0, dof):
        # display Dirichlet BCs
        ax = plt.subplot(dof, tot_img, 1 + tot_img * i0)
        c_img = plt.imshow(the_img_input[:, :, i0], cmap=hot)  # tensor
        plt.colorbar(c_img, fraction=fraction, pad=pad)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title('Dirichlet BC ' + dof_name[i0])

        # display Neumann BCs
        ax = plt.subplot(dof, tot_img, 2 + tot_img * i0)
        c_img = plt.imshow(the_img_input[:, :, dof+i0], cmap=hot)  # tensor
        plt.colorbar(c_img, fraction=fraction, pad=pad)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title('Neumann BC (x) ' + dof_name[i0])

        # display Neumann BCs
        ax = plt.subplot(dof, tot_img, 3 + tot_img * i0)
        c_img = plt.imshow(the_img_input[:, :, dof+i0+1], cmap=hot)  # tensor
        plt.colorbar(c_img, fraction=fraction, pad=pad)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title('Neumann BC (y) ' + dof_name[i0])

        # display label
        ax = plt.subplot(dof, tot_img, 4 + tot_img * i0)
        c_img = plt.imshow(the_img_label[:, :, i0], cmap=hot)  # tensor
        plt.colorbar(c_img, fraction=fraction, pad=pad)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        label_min = np.amin(the_img_label[:, :, i0])
        label_max = np.amax(the_img_label[:, :, i0])
        plt.title('DNS ' + dof_name[i0])

        # display reconstruction: mean
        ax = plt.subplot(dof, tot_img, 5 + tot_img * i0)
        # use the same range for better visual comparison
        print('Pred. Mean. is using label_min and label_max as colorbar range. Thus, the plot might not look so right.')
        c_img = plt.imshow(the_img_pre_mean[:, :, i0], cmap=hot, vmin=label_min, vmax=label_max)  # tensor
        # c_img = plt.imshow(the_img_pre_mean[:, :, i0], cmap=hot)  # tensor
        plt.colorbar(c_img, fraction=fraction, pad=pad)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if tot_img >= 8:
            plt.title('Pred. Mean ' + dof_name[i0])
        else:
            plt.title('Pred. ' + dof_name[i0])

        # display error
        ax = plt.subplot(dof, tot_img, 6 + tot_img * i0)
        # the denominator - 0.5 is needed. Otherwise, the relative error is too small and not correct, as the scaled zero is 0.5.
        # c_img = plt.imshow((the_img_label[:, :, i0] - the_img_pre_mean[:, :, i0]) / ((the_img_label[:, :, i0] + the_img_mark[:, :, i0]) - 0.5), cmap=hot)  # tensor
        c_img = plt.imshow(the_img_pre_mean[:, :, i0] - the_img_label[:, :, i0], cmap=hot)  # tensor
        plt.colorbar(c_img, fraction=fraction, pad=pad)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # plt.title('Rel. Error ' + dof_name[i0])
        plt.title('Pointwise Error ' + dof_name[i0])

        if tot_img >= 7:
            # # display reconstruction: var
            # ax = plt.subplot(dof, tot_img, 7 + tot_img * i0)
            # c_img = plt.imshow(the_img_pre_var[:, :, i0], cmap=hot)  # tensor
            # plt.colorbar(c_img, fraction=fraction, pad=pad)
            # ax.get_xaxis().set_visible(False)
            # ax.get_yaxis().set_visible(False)
            # plt.title('Pred. Var. ' + dof_name[i0])


            # display reconstruction: mean
            ax = plt.subplot(dof, tot_img, 7 + tot_img * i0)
            # use the same range for better visual comparison
            print('Pred. Mean. is using label_min and label_max as colorbar range. Thus, the plot might not look so right.')
            c_img = plt.imshow(the_img_pre_mean[:, :, i0], cmap=hot)  # tensor
            # c_img = plt.imshow(the_img_pre_mean[:, :, i0], cmap=hot)  # tensor
            plt.colorbar(c_img, fraction=fraction, pad=pad)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.title('Pred. (actual range) ' + dof_name[i0])

            if tot_img == 8 :
                # display reconstruction: std
                ax = plt.subplot(dof, tot_img, 8 + tot_img * i0)
                c_img = plt.imshow(2.0 * the_img_pre_std[:, :, i0], cmap=hot)  # tensor
                plt.colorbar(c_img, fraction=fraction, pad=pad)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                plt.title('Pred. (2xStd.) ' + dof_name[i0])

    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    else:
        now_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        plt.savefig("prediction" + now_str + ".png")
    # plt.show()
    # exit(0)

def plot_PDE_solutions(img_input, img_label, img_pre_mean, img_pre_var, img_pre_std, dof=1, dof_name=['c'], tot_img=6, filename='', fig_size=2.2):
    """
    plot the results of PDEs

    Args:
        img_input (numpy array): size of [1, :, :, dof*2]
        img_label (numpy array): size of [1, :, :, dof]
        img_pre_mean (numpy array): size of [1, :, :, dof]
        img_pre_var (numpy array): size of [1, :, :, dof]
        img_pre_std (numpy array): size of [1, :, :, dof] 
        dof (int): default (=1)
        dof_name (list): list of string (default ['c'])
        tot_img (int): without plotting std (tot_img=6, default), with std (tot_img=7)
        filename (str): default ('')
    """
    hot=get_cm()
    bc_mask_dirichlet = pde_layers.ComputeBoundaryMaskNodalData(img_input, dof=dof, opt=1)
    the_bc_mask_dirichlet = tf.squeeze(bc_mask_dirichlet, [0])

    the_img_input = tf.squeeze(img_input, [0])
    the_img_label = tf.squeeze(img_label, [0])
    the_img_pre_mean = tf.squeeze(img_pre_mean, [0])
    the_img_pre_var = tf.squeeze(img_pre_var, [0])
    the_img_pre_std = tf.squeeze(img_pre_std, [0])
    # print('Dirichlet BC shape:', tf.shape(bc_mask_dirichlet))
    # print('Dirichlet BC:', bc_mask_dirichlet)

    # magic number from: https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    fraction=0.046
    pad=0.04

    # make the margin as NaN ( margin = -1)
    # the problem domain internal part of the input is not filled with random numbers (thus it is = -2)
    tmp_img_label = np.concatenate((the_img_label, the_img_label), axis=2)
    # the_img_input = np.ma.masked_where(tmp_img_label < -0.9, the_img_input)
    the_img_input = np.ma.masked_where(the_img_input <= 0.0, the_img_input)

    the_img_pre_mean = np.ma.masked_where(the_img_label < -0.9, the_img_pre_mean)
    the_img_pre_var = np.ma.masked_where(the_img_label < -0.9, the_img_pre_var)
    the_img_pre_std = np.ma.masked_where(the_img_label < -0.9, the_img_pre_std)

    the_img_label = np.ma.masked_where(the_img_label < -0.9, the_img_label)

    the_img_mark = 1.0e-10 * np.ones(np.shape(the_img_label))

    # remove the Dirichlet BCs region for mean, var, std by setting the value to NaN
    the_img_pre_mean = np.ma.masked_where(the_bc_mask_dirichlet == 0.0, the_img_pre_mean)
    the_img_pre_var = np.ma.masked_where(the_bc_mask_dirichlet == 0.0, the_img_pre_var)
    the_img_pre_std = np.ma.masked_where(the_bc_mask_dirichlet == 0.0, the_img_pre_std)
    the_img_label = np.ma.masked_where(the_bc_mask_dirichlet == 0.0, the_img_label)

    figsize_list_x = [x*fig_size*1.1 for x in range(1, 20)]
    figsize_list_y = [x*fig_size for x in range(1, 20)]

    fig = plt.figure(figsize=(figsize_list_x[tot_img-1], figsize_list_y[dof-1] ))

    for i0 in range(0, dof):
        # display Dirichlet BCs
        ax = plt.subplot(dof, tot_img, 1 + tot_img * i0)
        c_img = plt.imshow(the_img_input[:, :, i0], cmap=hot)  # tensor
        plt.colorbar(c_img, fraction=fraction, pad=pad)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title('Dirichlet BC ' + dof_name[i0])

        # display Neumann BCs
        ax = plt.subplot(dof, tot_img, 2 + tot_img * i0)
        c_img = plt.imshow(the_img_input[:, :, dof+i0], cmap=hot)  # tensor
        plt.colorbar(c_img, fraction=fraction, pad=pad)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title('Neumann BC ' + dof_name[i0])

        # display label
        ax = plt.subplot(dof, tot_img, 3 + tot_img * i0)
        c_img = plt.imshow(the_img_label[:, :, i0], cmap=hot)  # tensor
        plt.colorbar(c_img, fraction=fraction, pad=pad)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        label_min = np.amin(the_img_label[:, :, i0])
        label_max = np.amax(the_img_label[:, :, i0])
        plt.title('DNS ' + dof_name[i0])

        # display reconstruction: mean
        ax = plt.subplot(dof, tot_img, 4 + tot_img * i0)
        # use the same range for better visual comparison
        print('Pred. Mean. is using label_min and label_max as colorbar range. Thus, the plot might not look so right.')
        # c_img = plt.imshow(the_img_pre_mean[:, :, i0], cmap=hot, vmin=label_min, vmax=label_max)  # tensor
        c_img = plt.imshow(the_img_pre_mean[:, :, i0], cmap=hot)  # tensor
        plt.colorbar(c_img, fraction=fraction, pad=pad)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title('Pred. Mean ' + dof_name[i0])

        # display error
        ax = plt.subplot(dof, tot_img, 5 + tot_img * i0)
        # the denominator - 0.5 is needed. Otherwise, the relative error is too small and not correct, as the scaled zero is 0.5.
        # c_img = plt.imshow((the_img_label[:, :, i0] - the_img_pre_mean[:, :, i0]) / ((the_img_label[:, :, i0] + the_img_mark[:, :, i0]) - 0.5), cmap=hot)  # tensor
        c_img = plt.imshow(the_img_pre_mean[:, :, i0] - the_img_label[:, :, i0], cmap=hot)  # tensor
        plt.colorbar(c_img, fraction=fraction, pad=pad)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # plt.title('Rel. Error ' + dof_name[i0])
        plt.title('Pointwise Error ' + dof_name[i0])

        if tot_img >= 6:
            # display reconstruction: var
            ax = plt.subplot(dof, tot_img, 6 + tot_img * i0)
            c_img = plt.imshow(the_img_pre_var[:, :, i0], cmap=hot)  # tensor
            plt.colorbar(c_img, fraction=fraction, pad=pad)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.title('Pred. Var. ' + dof_name[i0])

            if tot_img == 7 :
                # display reconstruction: std
                ax = plt.subplot(dof, tot_img, 7 + tot_img * i0)
                c_img = plt.imshow(2.0 * the_img_pre_std[:, :, i0], cmap=hot)  # tensor
                plt.colorbar(c_img, fraction=fraction, pad=pad)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                plt.title('Pred. (2xStd.) ' + dof_name[i0])

    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    else:
        now_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        plt.savefig("prediction" + now_str + ".png")
    # plt.show()


def plot_fields(list_of_field, list_of_field_name, dof, dof_name,  filename='', print_data=False, vmin=None, vmax=None, Tex=False, fig_size=2.2, mask=False):
    """
    plot the fields

    Args:
        list_of_field (list): list of numpy array [1, :, :, dof]
        list_of_field_name (list): list of strings
        dof (int): dof per node
        dof_name (list): list of string 
        filename (str): default ('')
    """
    if Tex:
        plot_tex(Tex)

    hot=get_cm()

    # magic number from: https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    fraction=0.046
    pad=0.04

    tot_img = len(list_of_field_name)
    # print('tot_img:', tot_img)

    figsize_list_x = [x*fig_size*1.1 for x in range(1, 20)]
    figsize_list_y = [x*fig_size for x in range(1, 20)]

    fig = plt.figure(figsize=(figsize_list_x[tot_img-1], figsize_list_y[dof-1] ))
    for i0 in range(0, dof):
        for j0 in range(0, tot_img):
            _field_name = list_of_field_name[j0] + ' ' + dof_name[i0]
            ax = plt.subplot(dof, tot_img, j0 + 1 + tot_img * i0)
            one_field = list_of_field[j0][0, :, :, i0]
            if mask:
                one_field = np.ma.masked_where(one_field < -0.9, one_field)
            c_img = plt.imshow(one_field, cmap=hot, vmin=vmin, vmax=vmax)  # tensor
            if print_data:
                print(_field_name, list_of_field[j0][0, :, :, i0])
            plt.colorbar(c_img, fraction=fraction, pad=pad)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.title(_field_name)

    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    else:
        now_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        plt.savefig("prediction" + now_str + ".png")
    # plt.show()


def split_data(datax, datay, batch_size, split_ratio=['0.8', '0.1', '0.1']):
    """ split data according to a specific ratio """

    split_ratio = [float(x) for x in split_ratio]
    if (len(split_ratio) != 3 or abs(sum(split_ratio) - 1.0) > 1.0e-5):
        raise ValueError(
            'split ratio should be a list containing three float values with sum() == 1.0!!! Your current split_ratio = ',
            split_ratio, ' with sum = ', sum(split_ratio))
    tr_ratio = float(split_ratio[0])
    cv_ratio = float(split_ratio[1])
    tt_ratio = float(split_ratio[2])

    number_examples = datax.shape[0]
    idx = np.arange(0, number_examples)
    np.random.shuffle(idx)
    datax = [datax[i] for i in idx]    # get list of `num` random samples
    datay = [datay[i] for i in idx]    # get list of `num` random samples

    start = 0
    end_tr = int(tr_ratio * number_examples / batch_size) * batch_size 
    end_cv = int((tr_ratio + cv_ratio) * number_examples)
    end_tt = number_examples
    tr_datax = np.array(datax[start:end_tr])
    tr_datay = np.array(datay[start:end_tr])
    cv_datax = np.array(datax[end_tr:end_cv])
    cv_datay = np.array(datay[end_tr:end_cv])
    tt_datax = np.array(datax[end_cv:end_tt])
    tt_datay = np.array(datay[end_cv:end_tt])

    return tr_datax, tr_datay, cv_datax, cv_datay, tt_datax, tt_datay

def split_data_heter(datax, datay, dataz, batch_size, split_ratio=['0.8', '0.1', '0.1']):
    """ split data according to a specific ratio """

    split_ratio = [float(x) for x in split_ratio]
    if (len(split_ratio) != 3 or abs(sum(split_ratio) - 1.0) > 1.0e-5):
        raise ValueError(
            'split ratio should be a list containing three float values with sum() == 1.0!!! Your current split_ratio = ',
            split_ratio, ' with sum = ', sum(split_ratio))
    tr_ratio = float(split_ratio[0])
    cv_ratio = float(split_ratio[1])
    tt_ratio = float(split_ratio[2])

    number_examples = datax.shape[0]
    idx = np.arange(0, number_examples)
    np.random.shuffle(idx)
    datax = [datax[i] for i in idx]    # get list of `num` random samples
    datay = [datay[i] for i in idx]    # get list of `num` random samples
    dataz = [dataz[i] for i in idx]    # get list of `num` random samples

    start = 0
    end_tr = int(tr_ratio * number_examples / batch_size) * batch_size 
    end_cv = int((tr_ratio + cv_ratio) * number_examples)
    end_tt = number_examples
    tr_datax = np.array(datax[start:end_tr])
    tr_datay = np.array(datay[start:end_tr])
    tr_dataz = np.array(dataz[start:end_tr])
    cv_datax = np.array(datax[end_tr:end_cv])
    cv_datay = np.array(datay[end_tr:end_cv])
    cv_dataz = np.array(dataz[end_tr:end_cv])
    tt_datax = np.array(datax[end_cv:end_tt])
    tt_datay = np.array(datay[end_cv:end_tt])
    tt_dataz = np.array(dataz[end_cv:end_tt])

    return tr_datax, tr_datay, tr_dataz, cv_datax, cv_datay, cv_dataz, tt_datax, tt_datay, tt_dataz


def expand_dataset(features, labels, times):
    """ expand the features and labels to 2^(n+1) with n=times """
    for i in range(0, times):
        features = np.concatenate((features, features), axis=0)
        labels = np.concatenate((labels, labels), axis=0)
        # print('2^(' + '{}'.format(i + 1) + '): feature shape = ', np.shape(features), ' label shape = ', np.shape(labels))
    return features, labels

def ExpandDatasetHeter(features, mats, labels, times):
    """ expand the features and labels to 2^(n+1) with n=times """
    for i in range(0, times):
        features = np.concatenate((features, features), axis=0)
        mats = np.concatenate((mats, mats), axis=0)
        labels = np.concatenate((labels, labels), axis=0)
        # print('2^(' + '{}'.format(i + 1) + '): feature shape = ', np.shape(features), ' label shape = ', np.shape(labels))
    return features, mats, labels


class BatchDataHeter(tf.keras.utils.Sequence):
    """Produces a sequence of the data with labels."""
    """Borrowed from: class MNISTSequence(tf.keras.utils.Sequence) """

    def __init__(self, data, batch_size=128):
        """Initializes the sequence.

    Args:
      data: Tuple of numpy `array` instances, the first representing images and
            the second labels.
      batch_size: Integer, number of elements in each training batch.
    """
        self.features, self.mats, self.labels = data
        # self.features, self.labels = BatchData.__preprocessing(images, labels)
        self.batch_size = batch_size

    # @staticmethod
    # def __preprocessing(images, labels):
    # """Preprocesses image and labels data.

    # Args:
    # images: Numpy `array` representing the image data.
    # labels: Numpy `array` representing the labels data (range 0-9).

    # Returns:
    # images: Numpy `array` representing the image data, normalized
    # and expanded for convolutional network input.
    # labels: Numpy `array` representing the labels data (range 0-9),
    # as one-hot (categorical) values.
    # """
    # # images = 2 * (images / 255.) - 1. # normalization
    # images = images[..., tf.newaxis]

    # labels = tf.keras.utils.to_categorical(labels)
    # return images, labels

    def __len__(self):
        return int(tf.math.ceil(len(self.features) / self.batch_size))  # contains batches less than the size of batch_size
        # return int(len(self.features) / self.batch_size) # all batches are equal-sized.

    def __getitem__(self, idx):
        batch_x = self.features[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_m = self.mats[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        return [batch_x, batch_m], batch_y


class BatchData(tf.keras.utils.Sequence):
    """Produces a sequence of the data with labels."""
    """Borrowed from: class MNISTSequence(tf.keras.utils.Sequence) """

    def __init__(self, data, batch_size=128):
        """Initializes the sequence.

    Args:
      data: Tuple of numpy `array` instances, the first representing images and
            the second labels.
      batch_size: Integer, number of elements in each training batch.
    """
        self.features, self.labels = data
        # self.features, self.labels = BatchData.__preprocessing(images, labels)
        self.batch_size = batch_size

    # @staticmethod
    # def __preprocessing(images, labels):
    # """Preprocesses image and labels data.

    # Args:
    # images: Numpy `array` representing the image data.
    # labels: Numpy `array` representing the labels data (range 0-9).

    # Returns:
    # images: Numpy `array` representing the image data, normalized
    # and expanded for convolutional network input.
    # labels: Numpy `array` representing the labels data (range 0-9),
    # as one-hot (categorical) values.
    # """
    # # images = 2 * (images / 255.) - 1. # normalization
    # images = images[..., tf.newaxis]

    # labels = tf.keras.utils.to_categorical(labels)
    # return images, labels

    def __len__(self):
        return int(tf.math.ceil(len(self.features) / self.batch_size))  # contains batches less than the size of batch_size
        # return int(len(self.features) / self.batch_size) # all batches are equal-sized.

    def __getitem__(self, idx):
        batch_x = self.features[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y


class BatchDataTime(tf.keras.utils.Sequence):
    """Produces a sequence of the data with labels."""
    """Borrowed from: class MNISTSequence(tf.keras.utils.Sequence) """

    def __init__(self, data, batch_size=128):
        """Initializes the sequence.

    Args:
      data: Tuple of numpy `array` instances, the first representing images and
            the second labels.
      batch_size: Integer, number of elements in each training batch.
    """
        self.features, self.time, self.labels,  = data
        self.batch_size = batch_size

    def __len__(self):
        # return int(tf.math.ceil(len(self.features) / self.batch_size))  # contains batches less than the size of batch_size
        return int(len(self.features) / self.batch_size) # all batches are equal-sized.

    def __getitem__(self, idx):
        batch_x    = self.features[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x_time = self.time[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y    = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_x_time, batch_y


def exe_cmd(cmd, output=False):
    import subprocess, os
    if output:
        output_info = os.popen(cmd).read()
        return output_info
    else:
        os.system(cmd)



def plot_one_field_stat(data, dpi=150, name='stat.png'):
    """
    Plot the statistics of a data

    Args:
        data (numpy array): data[:, :, :]
        dpi (int): dpi of png (=150)
        name (str): name of png output (='stat.png') 
    """
    sample_num = tf.shape(data).numpy()[0]
    # print('data', tf.shape(data))
    mean_data = tf.reduce_mean(data, axis=0)
    # print('mean-data', tf.shape(mean_data))
    std_data = tf.math.reduce_std(data, axis=0)
    # print('std-data', tf.shape(std_data))
    expand_mean_data = tf.tile(tf.expand_dims(mean_data, axis=0), [sample_num, 1, 1] )
    # print('exp mean data', tf.shape(expand_mean_data))
    var_data = tf.reduce_mean( tf.math.pow(data - expand_mean_data, 2), axis=0)
    # print('var data', tf.shape(var_data))

    # data = [:,:,:]
    hot=get_cm()
    fig, axes = plt.subplots(nrows=2, ncols=2)
    ax = axes[0][0]
    c_img=ax.imshow(data[0,:,:], cmap=hot)
    fig.colorbar(c_img, ax=ax)
    ax.set_title('sample')
    ax = axes[0][1]
    c_img=ax.imshow(mean_data, cmap=hot)
    fig.colorbar(c_img, ax=ax)
    ax.set_title('mean')
    ax = axes[1][0]
    c_img=ax.imshow(std_data, cmap=hot)
    fig.colorbar(c_img, ax=ax)
    ax.set_title('std')
    ax = axes[1][1]
    c_img=ax.imshow(var_data, cmap=hot)
    fig.colorbar(c_img, ax=ax)
    ax.set_title('var')

    plt.savefig(name, dpi=dpi)
    plt.clf()

def plot_one_field(data, x_dim, y_dim, dpi=150, name='solution.png'):
    """
    Plot the histogram of a data

    Args:
        data (numpy array): data[:, :, :]
        x_dim (int): subplots in the x_dim to plot
        y_dim (int): subplots in the y_dim to plot
        dpi (int): dpi of png (=150)
        name (str): name of png output (='solution.png') 
    """

    hot=get_cm()
    fig, axes = plt.subplots(nrows=x_dim, ncols=y_dim)
    for i in range(0, x_dim):
        for j in range(0, y_dim):
            ax = axes[i][j]
            c_img=ax.imshow(data[i+y_dim*j,:,:], cmap=hot)
            fig.colorbar(c_img, ax=ax)
            ax.set_xticks([])
            ax.set_yticks([])
    plt.savefig(name, dpi=dpi)
    plt.clf()

def plot_one_field_hist(data, x_dim, y_dim, dpi=150, name='hist.png'):
    """
    Plot the histogram of a data

    Args:
        data (numpy array): data[:, :, :]
        x_dim (int): subplots in the x_dim to plot
        y_dim (int): subplots in the y_dim to plot
        dpi (int): dpi of png (=150)
        name (str): name of png output (='hist.png') 
    """
    # data = [:,:,:]
    fig, axes = plt.subplots(nrows=x_dim, ncols=y_dim)
    for i in range(0, x_dim):
        for j in range(0, y_dim):
            ax = axes[i][j]
            ax.hist(data[:,i,j], bins='auto')
            ax.set_xticks([])
            ax.set_yticks([])
    plt.savefig(name, dpi=dpi)
    plt.clf()


if __name__ == '__main__':
    tot_img = 7


    # dof = 1
    # mat1 = np.random.rand(2,16,16,dof)
    # mat2 = np.random.rand(2,16,16,dof*2)
    # plot_PDE_solutions(
            # img_input = mat2[0:1, :, :, 0:2*dof], 
            # img_label = mat1[0:1, :, :, 0:1*dof], 
            # img_pre_mean = mat1[0:1, :, :, 0:1*dof], 
            # img_pre_var = mat1[0:1, :, :, 0:1*dof], 
            # img_pre_std = mat1[0:1, :, :, 0:1*dof], 
            # dof=dof, 
            # dof_name=['c'], 
            # tot_img = tot_img,
            # filename='test1.png')

    # dof = 2
    # mat1 = np.random.rand(2,16,16,dof)
    # mat2 = np.random.rand(2,16,16,dof*2)
    # plot_PDE_solutions(
            # img_input = mat2[0:1, :, :, 0:2*dof], 
            # img_label = mat1[0:1, :, :, 0:1*dof], 
            # img_pre_mean = mat1[0:1, :, :, 0:1*dof], 
            # img_pre_var = mat1[0:1, :, :, 0:1*dof], 
            # img_pre_std = mat1[0:1, :, :, 0:1*dof], 
            # dof=dof, 
            # dof_name=['ux', 'uy'], 
            # tot_img = tot_img,
            # filename='test2.png')

    # dof = 3
    # mat1 = np.random.rand(2,16,16,dof)
    # mat2 = np.random.rand(2,16,16,dof*2)
    # plot_PDE_solutions(
            # img_input = mat2[0:1, :, :, 0:2*dof], 
            # img_label = mat1[0:1, :, :, 0:1*dof], 
            # img_pre_mean = mat1[0:1, :, :, 0:1*dof], 
            # img_pre_var = mat1[0:1, :, :, 0:1*dof], 
            # img_pre_std = mat1[0:1, :, :, 0:1*dof], 
            # dof=dof, 
            # dof_name=['ux', 'uy', 'c'], 
            # tot_img = tot_img,
            # filename='test3.png')

    tot_img = 6
    for dof in range(1, 3):
        mat1 = np.random.rand(2,16,16,dof)
        mat2 = np.random.rand(2,16,16,dof*2)
        plot_PDE_solutions(
                img_input = mat2[0:1, :, :, 0:2*dof], 
                img_label = mat1[0:1, :, :, 0:1*dof], 
                img_pre_mean = mat1[0:1, :, :, 0:1*dof], 
                img_pre_var = mat1[0:1, :, :, 0:1*dof], 
                img_pre_std = mat1[0:1, :, :, 0:1*dof], 
                dof=dof, 
                dof_name=['a']*dof, 
                tot_img = tot_img,
                filename='test-'+str(tot_img) + '-' + str(dof) + '.png')


    # for dof in range(1, 3):
        # for tot_img in range(1, 8):
            # mat1 = np.random.rand(2,16,16,dof)
            # names = []
            # fields = []
            # dof_name = ['test'] * dof
            # for i in range(0, tot_img):
                # fields.append(mat1)
                # names.append(str(i))
            # plot_fields(fields, names, dof, dof_name, filename='test-'+str(tot_img) + '-' + str(dof) + '-2.45.png')
