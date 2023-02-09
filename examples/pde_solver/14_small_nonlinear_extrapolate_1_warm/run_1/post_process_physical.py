from mechanoChemML.workflows.pde_solver.pde_system_elasticity_nonlinear import WeakPDENonLinearElasticity as thisPDESystem
from mechanoChemML.workflows.pde_solver.pde_utility import plot_PDE_solutions, plot_fields, plot_tex, plot_one_loss, plot_sigma2, get_cm
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.image as mpimg
from PIL import Image
import pandas as pd

def plot_field_results(list_of_field, list_of_field_name, dof, dof_name,  filename='', print_data=False, vmin=None, vmax=None, Tex=False):
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

    tot_img = len(list_of_field_name) - 1
    # print('tot_img:', tot_img)

    # figsize_list_x = [x*2.45 for x in range(1, 20)]
    # figsize_list_y = [x*2.2 for x in range(1, 20)]
    fig_size = 4
    figsize_list_x = [x*fig_size*1.1 for x in range(1, 20)]
    figsize_list_y = [x*fig_size for x in range(1, 20)]

    dirichlet_data = list_of_field[0]
    dirichlet_reverse_mask = tf.where( dirichlet_data < 0, tf.fill(tf.shape(dirichlet_data), 1.0), tf.fill(tf.shape(dirichlet_data), 0.0))

    nn = np.shape(dirichlet_data)[1]
    dh = 1.0/(nn - 1)
    coor = np.array(list(range(0, nn)))*dh
    loc = int(nn/2) - 1
    dirichelt_sum = np.sum(dirichlet_data[0,:,:,0], axis=0)
    x_direction = np.sum(dirichelt_sum > 0) > 0
    label_text = 'coordinate'

    for i0 in range(0, dof):
        plt.clf()
        # fig = plt.figure(figsize=(figsize_list_x[tot_img-1], figsize_list_y[dof-1] ))
        fig = plt.figure(figsize=(figsize_list_x[tot_img-1], figsize_list_y[0] ))
        for j0 in range(1, tot_img+1):
            # masked_field = np.ma.masked_where(dirichlet_reverse_mask[0,:,:,i0] == 0.0, list_of_field[j0][0, :, :, i0]) # remove the Dirichlet BCs location
            masked_field = np.ma.masked_where(dirichlet_data[0,:,:,i0] == -1, list_of_field[j0][0, :, :, i0]) # remove the margin

            if list_of_field_name[j0].lower() == 'dns':
                label_field = masked_field
                label_min = np.amin(2.0*(masked_field-0.5))
                label_max = np.amax(2.0*(masked_field-0.5))
                print('min', label_min, 'max', label_max)
            _field_name = list_of_field_name[j0] + ' ' + dof_name[i0]
            ax = plt.subplot(1, tot_img, j0 + 1 + tot_img * 0 - 1)
            
            if _field_name.find('Sol') >= 0 or  _field_name.find('Mean') >= 0 or  _field_name.find('DNS') >= 0:
                c_img = plt.imshow(2.0*(masked_field-0.5), cmap=hot, vmin=label_min, vmax=label_max)  # tensor
            elif _field_name.find('Std') >= 0:
                c_img = plt.imshow(2.0*masked_field, cmap=hot, vmin=vmin, vmax=vmax)  # tensor
            else :
                print(_field_name, 'do not know what to do, exit...')
                xxx

            if list_of_field_name[j0].lower().find('sol') >=0:
                det_field = masked_field

            if list_of_field_name[j0].lower().find('mean') >=0:
                mean_field = masked_field
            if list_of_field_name[j0].lower().find('std') >=0:
                std_field = masked_field

            if _field_name.find('DNS') >= 0 or  _field_name.find('Mean') >= 0 or _field_name.find('Std') >= 0:
                if i0==0:
                    plt.axhline(loc, color ='k', linestyle ="--") 
                elif i0==1:
                    plt.axvline(loc, color ='k', linestyle ="--") 

            plt.colorbar(c_img, fraction=fraction, pad=pad)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.title(_field_name)



        plt.tight_layout()
        if filename:
            plt.savefig(filename.replace('.png', '-dof-' + str(i0) + '.png'))
        plt.clf()


        # plot quantitative data
        fig = plt.figure(figsize=(fig_size+1, fig_size))

        if i0==0:
            plt.plot(coor, 2.0*(label_field[loc, :]-0.5), 'k')
            plt.plot(coor, 2.0*(det_field[loc, :]-0.5), 'b')
            plt.plot(coor, 2.0*(mean_field[loc, :]-0.5), 'r')
            plt.fill_between(coor, 2.0*(mean_field[loc, :]-0.5) - 2*2*std_field[loc, :], 2.0*(mean_field[loc, :]-0.5) + 2*2*std_field[loc, :], alpha=0.7, facecolor='gray')
        elif i0==1:
            coor = np.flip(coor)
            plt.plot(coor, 2.0*(label_field[:, loc]-0.5), 'k')
            plt.plot(coor, 2.0*(det_field[loc, :]-0.5), 'b')
            plt.plot(coor, 2.0*(mean_field[:, loc]-0.5), 'r')
            plt.fill_between(coor, 2.0*(mean_field[:, loc]-0.5) - 2*2*std_field[:, loc], 2.0*(mean_field[:, loc]-0.5) + 2*2*std_field[:,loc], alpha=0.7, facecolor='gray')
        plt.xlabel(label_text)
        plt.ylabel('value')
        plt.legend(['DNS', 'Sol. (det)', 'Mean $\pm$ 2 Std'])
        plt.xlim([0,1.0])
        # plt.title('Mean $\pm$ 2 Std')
        plt.tight_layout()
        if filename:
            plt.savefig(filename.replace('.png', '-uq-' + str(i0) + '.png'))

# def plot_each_bvp_results_with_quantitative_line(ra=0):
    # """ """
    # # cnn
    # saved_data = "results/2022-03-12T05-01-exp-9-60-NN-cnn-x10-B256-E10000-I100-mc1-1S0.0e+00-2S0.0e+00-Nadam-2.5e-04-..data-20220312050122.pickle"
    # sys.argv.extend(['cnn.ini', '-rf', saved_data])

    # inputs = []
    # labels = []
    # mean = []
    # var = []
    # std = []
    # problem = thisPDESystem()
    # problem.run()
    # _inputs, _labels, _mean, _var, _std = problem.test(test_folder='DNS', plot_png=False)
    # inputs.append(_inputs)
    # labels.append(_labels)
    # mean.append(_mean)
    # var.append(_var)
    # std.append(_std)

    # # # bnn
    # # saved_data = "results/2022-05-16T21-05-gpu-cn001-BNN-bnn-20k-x10-B256-E20000-I100-mc100-1S1.0e-05-2S1.0e-03-Nadam-2.0e-06-..data-20220516210556.pickle"
    # # saved_data = "results/2022-05-16T21-05-gpu-cn001-BNN-bnn-x10-B256-E100-I100-mc100-1S1.0e-05-2S1.0e-03-Nadam-2.0e-06-..data-20220516210556.pickle"
    # saved_data = "results/2022-06-13T08-35-exp-3-58-BNN-bnn-x10-B256-E20000-I100-mc100-1S1.0e-05-2S1.0e-03-Nadam-2.0e-06-..data-20220516210556-e20000-ra-ckpt19300.pickle"
    # # ra = 100
    # # # restart_name = '-r' + str(ra)
    # # sys.argv.extend(['-ra', str(ra)])
    # restart_name = ''

    # sys.argv[3] = saved_data
    # problem = thisPDESystem()
    # problem.run()
    # _inputs, _labels, _mean, _var, _std = problem.test(test_folder='DNS', plot_png=False)
    # inputs.append(_inputs)
    # labels.append(_labels)
    # mean.append(_mean)
    # var.append(_var)
    # std.append(_std)

    # for i0 in range(0, np.shape(_inputs)[0]):
        # plot_fields(
                # list_of_field = [
                    # inputs[0][i0:i0+1,:,:,0:2],
                    # inputs[0][i0:i0+1,:,:,2:4],
                    # ],
                # list_of_field_name = [
                    # 'Dirichelet BCs', 
                    # 'Neumann BCs',
                    # ],
                # dof = 2, 
                # dof_name = ['(X)', '(Y)'],
                # filename = 'nonlinear-input-dns-'+str(i0) + restart_name +'.png',
                # Tex = True,
                # vmin = -2,
                # vmax = 0.5,
                # fig_size = 4.0,
                # )

    # for i0 in range(0, np.shape(_inputs)[0]):
        # plot_field_results(
                # list_of_field = [
                    # inputs[0][i0:i0+1,:,:,0:2],
                    # labels[0][i0:i0+1,:,:,0:2],
                    # mean[0][i0:i0+1,:,:,0:2],
                    # mean[1][i0:i0+1,:,:,0:2],
                    # std[1][i0:i0+1,:,:,0:2],
                    # # mean[1][i0:i0+1,:,:,0:1]-labels[0][i0:i0+1,:,:,0:1],
                    # ],
                # list_of_field_name = [
                    # 'input',
                    # 'DNS', 
                    # 'Sol. (det)', 
                    # 'Mean (BNN)',
                    # 'Std. (BNN)',
                    # # 'Error',
                    # ],
                # dof = 2, 
                # dof_name = ['(X)', '(Y)'],
                # filename = 'nonlinear-results-dns-'+str(i0) + restart_name +'.png',
                # Tex = True,
                # )

def output_inter_extra_with_reaction_force(sim_id=0):
    # saved_data = "results/2022-03-12T05-01-exp-9-60-NN-cnn-x10-B256-E10000-I100-mc1-1S0.0e+00-2S0.0e+00-Nadam-2.5e-04-..data-20220312050122.pickle"
    # saved_data = "results/2022-05-16T21-05-gpu-cn001-BNN-bnn-x10-B256-E100-I100-mc100-1S1.0e-05-2S1.0e-03-Nadam-2.0e-06-..data-20220516210556.pickle"

    if sim_id == 0:
        # saved_data = "results/2022-05-16T21-05-gpu-cn001-BNN-bnn-20k-x10-B256-E20000-I100-mc100-1S1.0e-05-2S1.0e-03-Nadam-2.0e-06-..data-20220516210556.pickle"
        saved_data = "results/2022-06-13T08-35-exp-3-58-BNN-bnn-x10-B256-E20000-I100-mc100-1S1.0e-05-2S1.0e-03-Nadam-2.0e-06-..data-20220516210556-e20000-ra-ckpt19300.pickle"
        # saved_data = "results/2022-06-20T09-42-exp-3-59-BNN-bnn-20k-x10-B256-E20000-I100-mc100-1S1.0e-05-2S1.0e-03-Nadam-2.0e-06-..data-20220620094200.pickle"

        sys.argv.extend(['bnn.ini', '-rf', saved_data]) #, '-ra', '4000'
    if sim_id == 1:
        saved_data = "results/2022-03-12T05-01-exp-9-60-NN-cnn-x10-B256-E10000-I100-mc1-1S0.0e+00-2S0.0e+00-Nadam-2.5e-04-..data-20220312050122.pickle"
        sys.argv.extend(['cnn.ini', '-rf', saved_data]) #, '-ra', '4000'

    # else:
        # if sim_id == 1:
            # saved_data = "results/2021-01-02T16-14-BNN-nonlinear-inter-2-bnn-x9-B64-E100-I50-mc50-1S1.0e-04-2S1.0e-04-Nadam-1.0e-08-data16x16-nonlinear-DNS-1-E30-Nu0.3-Inter-Extra-2-20210102161433.pickle"
        # elif sim_id == 2:
            # saved_data = "results/2021-01-02T16-33-BNN-nonlinear-inter-3-bnn-x9-B64-E100-I50-mc50-1S1.0e-04-2S1.0e-04-Nadam-1.0e-08-data16x16-nonlinear-DNS-1-E30-Nu0.3-Inter-Extra-3-20210102163312.pickle"
        # elif sim_id == 3:
            # saved_data = "results/2021-01-02T17-10-BNN-nonlinear-inter-4-bnn-x9-B64-E100-I50-mc50-1S1.0e-04-2S1.0e-04-Nadam-1.0e-08-data16x16-nonlinear-DNS-1-E30-Nu0.3-Inter-Extra-4-20210102171001.pickle"
        # sys.argv[3] = saved_data

    problem = thisPDESystem()
    problem.run()
    _inputs, _labels, _mean, _var, _std = problem.test(test_folder='DNS', plot_png=False, output_reaction_force=True)
    _inputs, _labels, _mean, _var, _std = problem.test(test_folder='Inter', plot_png=False, output_reaction_force=True)
    _inputs, _labels, _mean, _var, _std = problem.test(test_folder='Extra', plot_png=False, output_reaction_force=True)

def plot_forces(sim_id=0):
    plot_tex(True)
    dns = 'postprocess/force.csv'

    F_dns = pd.read_csv(dns)  
    if sim_id == 0:
        # train = "results/2022-05-16T23-01-Destiny-BNN-cnn-x10-B256-E100-I100-mc100-1S1.0e-05-2S1.0e-03-Nadam-2.0e-06-..data-20220516210556-e-1-ra-ckpt0099-DNSF.npy"
        # extra = "results/2022-05-16T23-01-Destiny-BNN-cnn-x10-B256-E100-I100-mc100-1S1.0e-05-2S1.0e-03-Nadam-2.0e-06-..data-20220516210556-e-1-ra-ckpt0099-ExtraF.npy"
        # inter = "results/2022-05-16T23-01-Destiny-BNN-cnn-x10-B256-E100-I100-mc100-1S1.0e-05-2S1.0e-03-Nadam-2.0e-06-..data-20220516210556-e-1-ra-ckpt0099-InterF.npy"
        # train = "results/2022-05-17T07-52-Destiny-BNN-bnn-x10-B256-E20000-I100-mc100-1S1.0e-05-2S1.0e-03-Nadam-2.0e-06-..data-20220516210556-e-1-ra-ckpt1000-DNSF.npy"
        # extra = "results/2022-05-17T07-52-Destiny-BNN-bnn-x10-B256-E20000-I100-mc100-1S1.0e-05-2S1.0e-03-Nadam-2.0e-06-..data-20220516210556-e-1-ra-ckpt1000-ExtraF.npy"
        # inter = "results/2022-05-17T07-52-Destiny-BNN-bnn-x10-B256-E20000-I100-mc100-1S1.0e-05-2S1.0e-03-Nadam-2.0e-06-..data-20220516210556-e-1-ra-ckpt1000-InterF.npy"
        # train = "results/2022-05-17T08-22-Destiny-BNN-bnn-x10-B256-E20000-I100-mc100-1S1.0e-05-2S1.0e-03-Nadam-2.0e-06-..data-20220516210556-e-1-ra-ckpt4000-DNSF.npy"
        # extra = "results/2022-05-17T08-22-Destiny-BNN-bnn-x10-B256-E20000-I100-mc100-1S1.0e-05-2S1.0e-03-Nadam-2.0e-06-..data-20220516210556-e-1-ra-ckpt4000-ExtraF.npy"
        # inter = "results/2022-05-17T08-22-Destiny-BNN-bnn-x10-B256-E20000-I100-mc100-1S1.0e-05-2S1.0e-03-Nadam-2.0e-06-..data-20220516210556-e-1-ra-ckpt4000-InterF.npy"
        # train = "results/2022-05-18T21-05-Destiny-BNN-bnn-x10-B256-E20000-I100-mc100-1S1.0e-05-2S1.0e-03-Nadam-2.0e-06-..data-20220516210556-e-1-ra-ckpt19300-DNSF.npy"
        # extra = "results/2022-05-18T21-05-Destiny-BNN-bnn-x10-B256-E20000-I100-mc100-1S1.0e-05-2S1.0e-03-Nadam-2.0e-06-..data-20220516210556-e-1-ra-ckpt19300-ExtraF.npy"
        # inter = "results/2022-05-18T21-05-Destiny-BNN-bnn-x10-B256-E20000-I100-mc100-1S1.0e-05-2S1.0e-03-Nadam-2.0e-06-..data-20220516210556-e-1-ra-ckpt19300-InterF.npy"
        train = "results/2022-07-08T10-32-Destiny-BNN-bnn-x10-B256-E20000-I100-mc100-1S1.0e-05-2S1.0e-03-Nadam-2.0e-06-..data-ckpt19300-e-1-ra-ckpt19999-DNSF.npy"
        extra = "results/2022-07-08T10-32-Destiny-BNN-bnn-x10-B256-E20000-I100-mc100-1S1.0e-05-2S1.0e-03-Nadam-2.0e-06-..data-ckpt19300-e-1-ra-ckpt19999-ExtraF.npy"
        inter = "results/2022-07-08T10-32-Destiny-BNN-bnn-x10-B256-E20000-I100-mc100-1S1.0e-05-2S1.0e-03-Nadam-2.0e-06-..data-ckpt19300-e-1-ra-ckpt19999-InterF.npy"
        # train = "results/2022-07-08T10-35-Destiny-BNN-bnn-x10-B256-E20000-I100-mc100-1S1.0e-05-2S1.0e-03-Nadam-2.0e-06-..data-20220620094200-e-1-ra-ckpt19999-DNSF.npy"
        # extra = "results/2022-07-08T10-35-Destiny-BNN-bnn-x10-B256-E20000-I100-mc100-1S1.0e-05-2S1.0e-03-Nadam-2.0e-06-..data-20220620094200-e-1-ra-ckpt19999-ExtraF.npy"
        # inter = "results/2022-07-08T10-35-Destiny-BNN-bnn-x10-B256-E20000-I100-mc100-1S1.0e-05-2S1.0e-03-Nadam-2.0e-06-..data-20220620094200-e-1-ra-ckpt19999-InterF.npy"




    elif sim_id == 1:
        train = "results/2021-01-02T21-39-BNN-nonlinear-30-bvp-final-1-cnn-x9-B64-E100-I50-mc50-1S1.0e-04-2S1.0e-04-Nadam-1.0e-08-data16x16-nonlinear-DNS-1-E30-Nu0.3-Inter-Extra-2-20210102161433-e-1-ra-ckpt0099-DNSF.npy"
        extra = "results/2021-01-02T21-39-BNN-nonlinear-30-bvp-final-1-cnn-x9-B64-E100-I50-mc50-1S1.0e-04-2S1.0e-04-Nadam-1.0e-08-data16x16-nonlinear-DNS-1-E30-Nu0.3-Inter-Extra-2-20210102161433-e-1-ra-ckpt0099-ExtraF.npy"
        inter = "results/2021-01-02T21-39-BNN-nonlinear-30-bvp-final-1-cnn-x9-B64-E100-I50-mc50-1S1.0e-04-2S1.0e-04-Nadam-1.0e-08-data16x16-nonlinear-DNS-1-E30-Nu0.3-Inter-Extra-2-20210102161433-e-1-ra-ckpt0099-InterF.npy"
    elif sim_id == 2:
        train = "results/2021-01-02T21-40-BNN-nonlinear-30-bvp-final-1-cnn-x9-B64-E100-I50-mc50-1S1.0e-04-2S1.0e-04-Nadam-1.0e-08-data16x16-nonlinear-DNS-1-E30-Nu0.3-Inter-Extra-3-20210102163312-e-1-ra-ckpt0099-DNSF.npy"
        extra = "results/2021-01-02T21-40-BNN-nonlinear-30-bvp-final-1-cnn-x9-B64-E100-I50-mc50-1S1.0e-04-2S1.0e-04-Nadam-1.0e-08-data16x16-nonlinear-DNS-1-E30-Nu0.3-Inter-Extra-3-20210102163312-e-1-ra-ckpt0099-ExtraF.npy"
        inter = "results/2021-01-02T21-40-BNN-nonlinear-30-bvp-final-1-cnn-x9-B64-E100-I50-mc50-1S1.0e-04-2S1.0e-04-Nadam-1.0e-08-data16x16-nonlinear-DNS-1-E30-Nu0.3-Inter-Extra-3-20210102163312-e-1-ra-ckpt0099-InterF.npy"
    elif sim_id == 3:
        train =  "results/2021-01-02T21-40-BNN-nonlinear-30-bvp-final-1-cnn-x9-B64-E100-I50-mc50-1S1.0e-04-2S1.0e-04-Nadam-1.0e-08-data16x16-nonlinear-DNS-1-E30-Nu0.3-Inter-Extra-4-20210102171001-e-1-ra-ckpt0099-DNSF.npy"
        extra = "results/2021-01-02T21-40-BNN-nonlinear-30-bvp-final-1-cnn-x9-B64-E100-I50-mc50-1S1.0e-04-2S1.0e-04-Nadam-1.0e-08-data16x16-nonlinear-DNS-1-E30-Nu0.3-Inter-Extra-4-20210102171001-e-1-ra-ckpt0099-ExtraF.npy"
        inter = "results/2021-01-02T21-40-BNN-nonlinear-30-bvp-final-1-cnn-x9-B64-E100-I50-mc50-1S1.0e-04-2S1.0e-04-Nadam-1.0e-08-data16x16-nonlinear-DNS-1-E30-Nu0.3-Inter-Extra-4-20210102171001-e-1-ra-ckpt0099-InterF.npy"

    F_train = np.load(train)
    F_inter = np.load(inter)
    F_extra = np.load(extra)
    F_all = np.vstack([F_train, F_inter, F_extra])
    F_all = F_all[F_all[:,0].argsort()]
    # print('f_all', F_all)

    # # 0,  1   2   3   4        5        6       7
    # # ux, uy, tx, ty, fx_mean, fy_mean, fx_std, fy_std 
    # print('f_dns', F_dns)
    # print('f_train', F_train)

    plt.clf()
    n_std = 1
    fig = plt.figure(figsize=(5, 4))
    plt.plot(F_dns['dUx'],F_dns['Fx'], '-k')
    plt.plot(F_train[:,0], F_train[:,4], 'r.')
    plt.plot(F_inter[:,0], F_inter[:,4], 'g.')
    plt.plot(F_extra[:,0], F_extra[:,4], 'b.')
    plt.fill_between(F_all[:,0], F_all[:,4]-n_std*F_all[:,6], F_all[:,4]+n_std*F_all[:,6], alpha=0.7, facecolor='gray')

    plt.xlabel(r'$\bar{u}_x$')
    plt.ylabel('$F_x$')
    plt.legend(['DNS', 'Train', 'Inter.', 'Extra.'])
    plt.xlim([0,1.0])
    plt.title('Mean $\pm$ '+str(n_std)+' Std')
    plt.tight_layout()
    plt.savefig('nonlinear-Fx-std-'  + str(n_std) + '-inter-extra-' + str(sim_id) + '.png')

    plt.clf()
    fig = plt.figure(figsize=(5, 4))
    plt.plot(F_dns['dUx'],F_dns['Fy'], '-k')
    plt.plot(F_train[:,0], F_train[:,5], 'r.')
    plt.plot(F_inter[:,0], F_inter[:,5], 'g.')
    plt.plot(F_extra[:,0], F_extra[:,5], 'b.')
    plt.fill_between(F_all[:,0], F_all[:,5]-n_std*F_all[:,7], F_all[:,5]+n_std*F_all[:,7], alpha=0.7, facecolor='gray')

    plt.xlabel(r'$\bar{u}_x$')
    plt.ylabel('$F_y$')
    plt.legend(['DNS', 'Train', 'Inter.', 'Extra.'])
    plt.xlim([0,1.0])
    plt.title('Mean $\pm$ '+str(n_std)+' Std')
    plt.tight_layout()
    plt.savefig('nonlinear-Fy-std-' + str(n_std) + '-inter-extra-' + str(sim_id) + '.png')

def plot_last_extra_bvp_results_with_quantitative_line(sim_id=0):
    """ """
    # cnn
    saved_data = "results/2022-03-12T05-01-exp-9-60-NN-cnn-x10-B256-E10000-I100-mc1-1S0.0e+00-2S0.0e+00-Nadam-2.5e-04-..data-20220312050122.pickle"
    sys.argv.extend(['bnn.ini', '-rf', saved_data])

    if sim_id == 0:
        test_folder = 'Extra'
    elif sim_id == 1:
        test_folder = 'Inter'

    inputs = []
    labels = []
    mean = []
    var = []
    std = []
    problem = thisPDESystem()
    problem.run()
    _inputs, _labels, _mean, _var, _std = problem.test(test_folder=test_folder, plot_png=False)
    inputs.append(_inputs)
    labels.append(_labels)
    mean.append(_mean)
    var.append(_var)
    std.append(_std)

    # # bnn
    # saved_data = "results/2022-05-16T21-05-gpu-cn001-BNN-bnn-20k-x10-B256-E20000-I100-mc100-1S1.0e-05-2S1.0e-03-Nadam-2.0e-06-..data-20220516210556.pickle"
    # saved_data = "results/2022-05-16T21-05-gpu-cn001-BNN-bnn-x10-B256-E100-I100-mc100-1S1.0e-05-2S1.0e-03-Nadam-2.0e-06-..data-20220516210556.pickle"
    saved_data = "results/2022-06-13T08-35-exp-3-58-BNN-bnn-x10-B256-E20000-I100-mc100-1S1.0e-05-2S1.0e-03-Nadam-2.0e-06-..data-20220516210556-e20000-ra-ckpt19300.pickle"

    # ra = 100
    # # restart_name = '-r' + str(ra)
    # sys.argv.extend(['-ra', str(ra)])
    restart_name = ''

    sys.argv[3] = saved_data
    problem = thisPDESystem()
    problem.run()
    _inputs, _labels, _mean, _var, _std = problem.test(test_folder=test_folder, plot_png=False)
    inputs.append(_inputs)
    labels.append(_labels)
    mean.append(_mean)
    var.append(_var)
    std.append(_std)

    all_fields = {'inputs':inputs, 'labels':labels, 'mean':mean, 'var':var, 'std':std}
    if sim_id == 0:
        pickle_out = open('all_fields_Extra_large.pickle', "wb")
    if sim_id == 1:
        pickle_out = open('all_fields_Inter_large.pickle', "wb")
    pickle.dump(all_fields, pickle_out)
    pickle_out.close()


    # all_fields = {'inputs':inputs, 'labels':labels, 'mean':mean, 'var':var, 'std':std}
    # pickle_out = open('all_fields_' + test_folder + '.pickle', "wb")
    # pickle.dump(all_fields, pickle_out)
    # pickle_out.close()

    for i0 in range(0, np.shape(_inputs)[0]):
        plot_fields(
                list_of_field = [
                    inputs[0][i0:i0+1,:,:,0:2],
                    inputs[0][i0:i0+1,:,:,2:4],
                    ],
                list_of_field_name = [
                    'Dirichelet BCs', 
                    'Neumann BCs',
                    ],
                dof = 2, 
                dof_name = ['(X)', '(Y)'],
                filename = 'nonlinear-input-' + test_folder.lower() +'-dns-'+str(i0) + restart_name +'.png',
                Tex = True,
                vmin = -2,
                vmax = 0.5,
                fig_size = 4.0,
                )

    for i0 in range(0, np.shape(_inputs)[0]):
        plot_field_results(
                list_of_field = [
                    inputs[0][i0:i0+1,:,:,0:2],
                    labels[0][i0:i0+1,:,:,0:2],
                    mean[0][i0:i0+1,:,:,0:2],
                    mean[1][i0:i0+1,:,:,0:2],
                    std[1][i0:i0+1,:,:,0:2],
                    # mean[1][i0:i0+1,:,:,0:1]-labels[0][i0:i0+1,:,:,0:1],
                    ],
                list_of_field_name = [
                    'input',
                    'DNS', 
                    'Sol. (det)', 
                    'Mean (BNN)',
                    'Std. (BNN)',
                    # 'Error',
                    ],
                dof = 2, 
                dof_name = ['(X)', '(Y)'],
                filename = 'nonlinear-results-' + test_folder.lower() + '-dns-'+str(i0) + restart_name +'.png',
                Tex = True,
                )

def compute_l2_error_inter_extra():
    """ """

    # cnn
    # saved_data =  "results/2020-12-15T20-46-NN-nonlinear-inter-1-cnn-x10-B256-E10000-I100-mc1-1S0.0e+00-2S0.0e+00-Nadam-2.5e-04-data16x16-nonlinear-DNS-1-E30-Nu0.3-Inter-Extra-1-20201215204611.pickle"
    saved_data = "results/2022-03-12T05-01-exp-9-60-NN-cnn-x10-B256-E10000-I100-mc1-1S0.0e+00-2S0.0e+00-Nadam-2.5e-04-..data-20220312050122.pickle"

    # # for scaling test only # need to manually uncomment the scaling part code
    # saved_data = "results/2022-06-13T08-35-exp-3-58-BNN-bnn-x10-B256-E20000-I100-mc100-1S1.0e-05-2S1.0e-03-Nadam-2.0e-06-..data-20220516210556-e20000-ra-ckpt19300.pickle"
    # sys.argv.extend(['nonlinear-inter-1-bnn.ini', '-rf', saved_data])
    sys.argv.extend(['bnn.ini', '-rf', saved_data]) # bnn.ini will be replaced with the original cnn.ini
    test_folder = 'All'

    inputs = []
    labels = []
    mean = []
    var = []
    std = []
    problem = thisPDESystem()
    problem.run()
    _inputs, _labels, _mean, _var, _std = problem.test(test_folder=test_folder, plot_png=False)
    print('labels:',tf.shape(_labels))
    print('mean:', tf.shape(_mean))
    dy_DNS_NN = 2.0*(_labels-0.5) - 2.0*(_mean-0.5)
    l2_error = tf.sqrt(tf.reduce_sum (tf.multiply(dy_DNS_NN, dy_DNS_NN), axis=[1,2,3])/(16.0*16.0))
    print(tf.shape(l2_error), l2_error)
    l2_error = tf.reduce_sum(l2_error)/30.0
    print('all 30 BVP l2 error:', l2_error.numpy())

    # # bnn
    # saved_data = "results/2021-01-02T15-52-BNN-nonlinear-inter-1-bnn-x9-B64-E100-I50-mc50-1S1.0e-04-2S1.0e-04-Nadam-1.0e-08-data16x16-nonlinear-DNS-1-E30-Nu0.3-Inter-Extra-1-20210102155228.pickle"
    saved_data = "results/2022-06-13T08-35-exp-3-58-BNN-bnn-x10-B256-E20000-I100-mc100-1S1.0e-05-2S1.0e-03-Nadam-2.0e-06-..data-20220516210556-e20000-ra-ckpt19300.pickle"


    sys.argv[3] = saved_data
    problem = thisPDESystem()
    problem.run()
    _inputs, _labels, _mean, _var, _std = problem.test(test_folder=test_folder, plot_png=False)
    print('labels:',tf.shape(_labels))
    print('mean:', tf.shape(_mean))
    dy_DNS_NN = 2.0*(_labels-0.5) - 2.0*(_mean-0.5)
    l2_error = tf.sqrt(tf.reduce_sum (tf.multiply(dy_DNS_NN, dy_DNS_NN), axis=[1,2,3])/(16.0*16.0))
    print(tf.shape(l2_error), l2_error)
    l2_error = tf.reduce_sum(l2_error)/30.0
    print('all 30 BVP l2 error:', l2_error.numpy())


if __name__ == '__main__':
    # ['post_process.py', 'test_right_0_small.ini', '-rf', 'results/2020-12-30T17-18-NN-test_right_100_small-x11-B128-E2000-I100-mc1-1S0.0e+00-2S0.0e+00-Nadam-2.5e-04-data16x16_right_flux-20201230171851.pickle', '-ra', '10']

    # already generate data
    # output_inter_extra_with_reaction_force(sim_id=0)
    # plot_forces(sim_id=0)
    # plot_last_extra_bvp_results_with_quantitative_line(sim_id = 0) # extra=0
    # plot_last_extra_bvp_results_with_quantitative_line(sim_id = 1) # inter=1

    # output_inter_extra_with_reaction_force(sim_id=1)
    # output_inter_extra_with_reaction_force(sim_id=2)
    # output_inter_extra_with_reaction_force(sim_id=3)
    # post_process 
    # plot_last_extra_bvp_results_with_quantitative_line(sim_id = 1)
    compute_l2_error_inter_extra() # also for scaling test
