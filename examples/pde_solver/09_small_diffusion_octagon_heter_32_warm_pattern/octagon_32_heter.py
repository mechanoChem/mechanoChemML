from mechanoChemML.workflows.pde_solver.pde_system_diffusion_steady_state_heter import WeakPDESteadyStateDiffusion as thisPDESystem
from mechanoChemML.workflows.pde_solver.pde_utility import plot_PDE_solutions, plot_fields, plot_tex, plot_one_loss, plot_sigma2, get_cm
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.image as mpimg
from PIL import Image
import pandas as pd

def plot_train_loss():
    """ Plot loss in the paper format """
    pickle_file = 'results/2020-12-17T08-20-NN-diffusion-20-bvp-final-1-cnn-x10-B256-E20000-I100-mc1-1S0.0e+00-2S0.0e+00-Nadam-2.5e-04-data16x16-diffusion-DNS-20-D1.0-20201217082006.pickle'
    plot_one_loss(pickle_file, 'diffusion-20-bvp-cnn-loss.png')
    pickle_file = 'results/2020-12-30T15-50-BNN-diffusion-20-bvp-final-1-bnn-x9-B64-E5000-I50-mc100-1S1.0e-04-2S1.0e-04-Nadam-1.0e-08-data16x16-diffusion-DNS-20-D1.0-20201230155035.pickle'
    plot_one_loss(pickle_file, 'diffusion-20-bvp-bnn-loss.png')
    plot_sigma2(pickle_file, 'diffusion-20-bvp-bnn-sigma2.png')


def plot_one_example_zero_init_neumann(steps, filename):
    steps_name = [str(x) + ' epochs' for x in steps]
    steps = [str(x) for x in steps]
    inputs = []
    labels = []
    mean = []
    var = []
    std = []
    for _s in steps[1:]:
        sys.argv[5] = _s
        problem = thisPDESystem()
        problem.run()
        _inputs, _labels, _mean, _var, _std = problem.test(test_folder='DNS')
        # inputs = _inputs
        labels = _labels
        mean.append(_mean)
        # var.append(_var)
        # std.append(_std)
        # print(_mean, _var, _std)
        # print(_mean, _var, _std)
        # print(labels)
    mean.insert(0, labels)

    plot_fields(
            list_of_field = mean,
            list_of_field_name = steps_name, 
            dof = 1, 
            dof_name = [''],
            filename = filename,
            vmin = 0.0,
            vmax = 0.70,
            Tex = True,
            )

def plot_zero_initialization_illustration():
    """ show the progressive results for training diffusion with Neumann BC case in the paper """
    # without zero
    sys.argv.extend(['test_right_100_small.ini', '-rf', 'results/2020-12-30T18-59-NN-test_right_0_small-x11-B128-E2000-I0-mc1-1S0.0e+00-2S0.0e+00-Nadam-2.5e-04-data16x16_right_flux-20201230185902.pickle', '-ra', '0'])

    steps = ['DNS', 10, 100, 500, 1000, 1500, 1900]
    plot_one_example_zero_init_neumann(steps, filename='diffusion-neumann-no-zero-init.png')

    # # sys.argv.extend(['test_right_100_small.ini', '-rf', 'results/2020-12-30T17-18-NN-test_right_100_small-x11-B128-E2000-I100-mc1-1S0.0e+00-2S0.0e+00-Nadam-2.5e-04-data16x16_right_flux-20201230171851.pickle', '-ra', '10'])
    sys.argv[3] = 'results/2020-12-30T17-18-NN-test_right_100_small-x11-B128-E2000-I100-mc1-1S0.0e+00-2S0.0e+00-Nadam-2.5e-04-data16x16_right_flux-20201230171851.pickle'
    steps = ['DNS', 50, 100, 200, 300, 400, 500]
    plot_one_example_zero_init_neumann(steps, filename='diffusion-neumann-with-zero-init.png')

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
        fig = plt.figure(figsize=(figsize_list_x[tot_img-1], figsize_list_y[dof-1] ))
        for j0 in range(1, tot_img+1):
            # masked_field = np.ma.masked_where(dirichlet_reverse_mask[0,:,:,i0] == 0.0, list_of_field[j0][0, :, :, i0]) # remove the Dirichlet BCs location
            masked_field = np.ma.masked_where(dirichlet_data[0,:,:,i0] == -1, list_of_field[j0][0, :, :, i0]) # remove the margin

            if list_of_field_name[j0].lower() == 'dns':
                label_field = masked_field
                label_min = np.amin(masked_field)
                label_max = np.amax(masked_field)
                print('min', label_min, 'max', label_max)
            _field_name = list_of_field_name[j0] + ' ' + dof_name[i0]
            ax = plt.subplot(dof, tot_img, j0 + 1 + tot_img * i0 - 1)
            
            if _field_name.find('Sol') >= 0 or  _field_name.find('Mean') >= 0:
                c_img = plt.imshow(masked_field, cmap=hot, vmin=label_min, vmax=label_max)  # tensor
            else:
                c_img = plt.imshow(masked_field, cmap=hot, vmin=vmin, vmax=vmax)  # tensor

            if list_of_field_name[j0].lower().find('mean') >=0:
                mean_field = masked_field
            if list_of_field_name[j0].lower().find('std') >=0:
                std_field = masked_field

            if _field_name.find('DNS') >= 0 or  _field_name.find('Mean') >= 0 or _field_name.find('Std') >= 0:
                if x_direction:
                    plt.axhline(loc, color ='k', linestyle ="--") 
                else:
                    plt.axvline(loc, color ='k', linestyle ="--") 

            plt.colorbar(c_img, fraction=fraction, pad=pad)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.title(_field_name)



        plt.tight_layout()
        if filename:
            plt.savefig(filename.replace('.png', '-dof-' + str(dof) + '.png'))
        plt.clf()


        # plot quantitative data
        fig = plt.figure(figsize=(fig_size+1, fig_size))

        if x_direction:
            plt.plot(coor, label_field[loc, :], 'r')
            plt.plot(coor, mean_field[loc, :], 'k')
            plt.fill_between(coor, mean_field[loc, :] - 2*std_field[loc, :], mean_field[loc, :] + 2*std_field[loc, :], alpha=0.7, facecolor='gray')
        else:
            coor = np.flip(coor)
            plt.plot(coor, label_field[:, loc], 'r')
            plt.plot(coor, mean_field[:, loc], 'k')
            plt.fill_between(coor, mean_field[:, loc] - 2*std_field[:, loc], mean_field[:, loc] + 2*std_field[:,loc], alpha=0.7, facecolor='gray')
        plt.xlabel(label_text)
        plt.ylabel('value')
        plt.legend(['DNS', 'Mean'])
        plt.xlim([0,1.0])
        plt.title('Mean $\pm$ 2 Std')
        plt.tight_layout()
        if filename:
            plt.savefig(filename.replace('.png', '-uq-' + str(dof) + '.png'))

def plot_each_bvp_results_with_quantitative_line(ra=0):
    """ """
    # cnn
    saved_data = "results/2020-12-17T08-20-NN-diffusion-20-bvp-final-1-cnn-x10-B256-E20000-I100-mc1-1S0.0e+00-2S0.0e+00-Nadam-2.5e-04-data16x16-diffusion-DNS-20-D1.0-20201217082006.pickle"
    sys.argv.extend(['diffusion-20-bvp-final-1-cnn.ini', '-rf', saved_data])

    inputs = []
    labels = []
    mean = []
    var = []
    std = []
    problem = thisPDESystem()
    problem.run()
    _inputs, _labels, _mean, _var, _std = problem.test(test_folder='DNS', plot_png=False)
    inputs.append(_inputs)
    labels.append(_labels)
    mean.append(_mean)
    var.append(_var)
    std.append(_std)

    if ra != 0:
        restart_name = '-r' + str(ra)
        sys.argv.extend(['-ra', str(ra)])
    else:
        restart_name = ''

    # # bnn
    # lr = 1e-8
    saved_data = "results/2020-12-30T22-48-BNN-diffusion-20-bvp-final-2-bnn-x9-B64-E5000-I50-mc100-1S1.0e-04-2S1.0e-04-Nadam-1.0e-08-data16x16-diffusion-DNS-20-D1.0-20201230224824.pickle"
    # lr = 1e-6
    saved_data = "results/2021-05-10T10-09-BNN-diffusion-20-bvp-final-1-bnn-x9-B64-E10000-I50-mc100-1S1.0e-04-2S1.0e-04-Nadam-1.0e-06-data16x16-diffusion-DNS-20-D1.0-20210510100934.pickle"
    # lr = 1e-5, sigma1=0.01
    saved_data = "results/2021-05-11T23-18-BNN-diffusion-20-bvp-final-1-bnn-x9-B64-E5000-I50-mc100-1S1.0e-02-2S1.0e-04-Nadam-1.0e-05-data16x16-diffusion-DNS-20-D1.0-20210511231832.pickle"
    # lr = 1e-5, sigma1=0.05
    saved_data = "results/2021-05-11T23-46-BNN-diffusion-20-bvp-final-1-bnn-x9-B64-E500-I50-mc100-1S5.0e-02-2S1.0e-04-Nadam-1.0e-05-data16x16-diffusion-DNS-20-D1.0-20210511234611.pickle"

    # lr = 1e-5, sigma1=0.1
    saved_data = "results/2021-05-12T07-42-BNN-diffusion-20-bvp-final-1-bnn-x9-B64-E1000-I50-mc100-1S1.0e-01-2S1.0e-04-Nadam-1.0e-05-data16x16-diffusion-DNS-20-D1.0-20210512074236.pickle"

    # lr = 1e-5, sigma1=0.001
    saved_data = "results/2021-05-12T09-02-BNN-diffusion-20-bvp-final-1-bnn-x9-B64-E1000-I50-mc100-1S1.0e-03-2S1.0e-04-Nadam-1.0e-05-data16x16-diffusion-DNS-20-D1.0-20210512090212.pickle"
    sys.argv[3] = saved_data
    problem = thisPDESystem()
    problem.run()
    _inputs, _labels, _mean, _var, _std = problem.test(test_folder='DNS', plot_png=False)
    inputs.append(_inputs)
    labels.append(_labels)
    mean.append(_mean)
    var.append(_var)
    std.append(_std)

    # for i0 in range(0, np.shape(_inputs)[0]):
        # plot_fields(
                # list_of_field = [
                    # inputs[0][i0:i0+1,:,:,0:1],
                    # inputs[0][i0:i0+1,:,:,1:2],
                    # ],
                # list_of_field_name = [
                    # 'Dirichelet BCs', 
                    # 'Neumann BCs',
                    # ],
                # dof = 1, 
                # dof_name = [''],
                # filename = 'diffusion-input-dns-'+str(i0) + restart_name +'.png',
                # Tex = True,
                # vmin = -2,
                # vmax = 0.5,
                # fig_size = 4.0,
                # )

    for i0 in range(0, np.shape(_inputs)[0]):
        plot_field_results(
                list_of_field = [
                    inputs[0][i0:i0+1,:,:,0:1],
                    labels[0][i0:i0+1,:,:,0:1],
                    mean[0][i0:i0+1,:,:,0:1],
                    mean[1][i0:i0+1,:,:,0:1],
                    std[1][i0:i0+1,:,:,0:1],
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
                dof = 1, 
                dof_name = [''],
                filename = 'diffusion-results-dns-'+str(i0) + restart_name +'.png',
                Tex = True,
                )

def plot_field_results_oct_pixel_data(list_of_field, list_of_field_name, dof, dof_name,  filename='', print_data=False, vmin=None, vmax=None, Tex=False):
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

    dns_field = np.ma.masked_where(dirichlet_data[0,:,:,0] == -1, list_of_field[1][0, :, :, 0]) # remove the margin
    label_min = 0.5
    label_max = np.max(dns_field)
    print('---label min ---', label_min, '---max---', label_max)

    for i0 in range(0, dof):
        plt.clf()
        fig = plt.figure(figsize=(figsize_list_x[tot_img-1], figsize_list_y[dof-1] ))
        for j0 in range(1, tot_img+1):
            # masked_field = np.ma.masked_where(dirichlet_reverse_mask[0,:,:,i0] == 0.0, list_of_field[j0][0, :, :, i0]) # remove the Dirichlet BCs location
            masked_field = np.ma.masked_where(dirichlet_data[0,:,:,i0] == -1, list_of_field[j0][0, :, :, i0]) # remove the margin

            if list_of_field_name[j0].lower() == 'dns':
                label_field = masked_field
                print('min', label_min, 'max', label_max)
            _field_name = list_of_field_name[j0] + ' ' + dof_name[i0]
            ax = plt.subplot(dof, tot_img, j0 + 1 + tot_img * i0 - 1)
            
            if _field_name.find('Sol') >= 0 or  _field_name.find('Mean') >= 0:
                c_img = plt.imshow(masked_field, cmap=hot, vmin=label_min, vmax=label_max)  # tensor
            elif _field_name.find('Std') >= 0 :
                c_img = plt.imshow(masked_field, cmap=hot)  # tensor
            elif _field_name.find('DNS') >= 0 :
                c_img = plt.imshow(masked_field, cmap=hot, vmin=label_min, vmax=label_max)  # tensor
                # img = load_octagon_dns()
                # imgplot = plt.imshow(img)
                # loc_x = np.shape(img)[0]/2.0
                # loc_y = np.shape(img)[1]/2.0
                plt.axhline(loc, color ='k', linestyle ="--") 
                plt.axvline(loc, color ='k', linestyle ="--") 

            if list_of_field_name[j0].lower().find('mean') >=0:
                mean_field = masked_field
            if list_of_field_name[j0].lower().find('std') >=0:
                std_field = masked_field
            if list_of_field_name[j0].lower().find('dns') >=0:
                dns_field = masked_field

            if  _field_name.find('Mean') >= 0 or _field_name.find('Std') >= 0:
                plt.axhline(loc, color ='k', linestyle ="--") 
                plt.axvline(loc, color ='k', linestyle ="--") 

            if _field_name.find('Mean') >= 0 or _field_name.find('Std') >= 0:
                plt.colorbar(c_img, fraction=fraction, pad=pad)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.title(_field_name)


        plt.tight_layout()
        if filename:
            plt.savefig(filename.replace('.png', '-dof-' + str(dof) + '.png'))

        plt.clf()
        # note: as pixel 32, 64, are used, it's difficult to get exact x=0.5 or y=0.5
        # for 32/2-1 = 15, and 64/2-1=31. 
        # in x-direction, one pixel less
        # in y-direction, one pixel more.
        # for h-0.515, the horizontal plot with y=0.515
        # for v-0.485, the vertical plot with x=0.485
        # 1/63=0.01587, thus 0.5 +- 0.015 is chosen from paraview for visualization.
        # x_label = pd.read_csv('postprocess/h-0.515.csv')  
        # y_label = pd.read_csv('postprocess/v-0.485.csv')  


        # plot quantitative data
        fig = plt.figure(figsize=(fig_size+1, fig_size))
        # plt.plot(coor, label_field[loc, :], 'r')
        # plt.plot(x_label['Points:0'],x_label['Result'], 'r')
        plt.plot(coor, dns_field[loc, :], 'r')
    
        plt.plot(coor, mean_field[loc, :], 'k')
        plt.fill_between(coor, mean_field[loc, :] - 2*std_field[loc, :], mean_field[loc, :] + 2*std_field[loc, :], alpha=0.7, facecolor='gray')
        plt.xlabel(label_text)
        plt.ylabel('value')
        plt.legend(['DNS', 'Mean'])
        plt.xlim([0,1.0])
        plt.title('Mean $\pm$ 2 Std')
        plt.tight_layout()
        if filename:
            plt.savefig(filename.replace('.png', '-uq-x-' + str(dof) + '.png'))

        plt.clf()
        # plot quantitative data
        fig = plt.figure(figsize=(fig_size+1, fig_size))

        coor = np.flip(coor)

        # plt.plot(coor, label_field[:, loc], 'r')
        # plt.plot(y_label['Points:1'],y_label['Result'], 'r')
        plt.plot(coor, dns_field[:, loc], 'r')
        plt.plot(coor, mean_field[:, loc], 'k')
        plt.fill_between(coor, mean_field[:, loc] - 2*std_field[:, loc], mean_field[:, loc] + 2*std_field[:,loc], alpha=0.7, facecolor='gray')
        plt.xlabel(label_text)
        plt.ylabel('value')
        plt.legend(['DNS', 'Mean'])
        plt.xlim([0,1.0])
        plt.title('Mean $\pm$ 2 Std')
        plt.tight_layout()
        if filename:
            plt.savefig(filename.replace('.png', '-uq-y-' + str(dof) + '.png'))

def plot_field_results_oct(list_of_field, list_of_field_name, dof, dof_name,  filename='', print_data=False, vmin=None, vmax=None, Tex=False):
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

    label_min = vmin
    label_max = vmax

    for i0 in range(0, dof):
        plt.clf()
        fig = plt.figure(figsize=(figsize_list_x[tot_img-1], figsize_list_y[dof-1] ))
        for j0 in range(1, tot_img+1):
            # masked_field = np.ma.masked_where(dirichlet_reverse_mask[0,:,:,i0] == 0.0, list_of_field[j0][0, :, :, i0]) # remove the Dirichlet BCs location
            masked_field = np.ma.masked_where(dirichlet_data[0,:,:,i0] == -1, list_of_field[j0][0, :, :, i0]) # remove the margin

            if list_of_field_name[j0].lower() == 'dns':
                label_field = masked_field
                print('min', label_min, 'max', label_max)
            _field_name = list_of_field_name[j0] + ' ' + dof_name[i0]
            ax = plt.subplot(dof, tot_img, j0 + 1 + tot_img * i0 - 1)
            
            if _field_name.find('Sol') >= 0 or  _field_name.find('Mean') >= 0:
                c_img = plt.imshow(masked_field, cmap=hot, vmin=label_min, vmax=label_max)  # tensor
            elif _field_name.find('Std') >= 0 :
                c_img = plt.imshow(masked_field, cmap=hot)  # tensor
            elif _field_name.find('DNS') >= 0 :
                img = load_octagon_dns()
                imgplot = plt.imshow(img)
                loc_x = np.shape(img)[0]/2.0
                loc_y = np.shape(img)[1]/2.0
                plt.axhline(loc_x, color ='k', linestyle ="--") 
                plt.axvline(loc_y, color ='k', linestyle ="--") 

            if list_of_field_name[j0].lower().find('mean') >=0:
                mean_field = masked_field
            if list_of_field_name[j0].lower().find('std') >=0:
                std_field = masked_field

            if  _field_name.find('Mean') >= 0 or _field_name.find('Std') >= 0:
                plt.axhline(loc, color ='k', linestyle ="--") 
                plt.axvline(loc, color ='k', linestyle ="--") 

            if _field_name.find('Mean') >= 0 or _field_name.find('Std') >= 0:
                plt.colorbar(c_img, fraction=fraction, pad=pad)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.title(_field_name)


        plt.tight_layout()
        if filename:
            plt.savefig(filename.replace('.png', '-dof-' + str(dof) + '.png'))

        plt.clf()
        # note: as pixel 32, 64, are used, it's difficult to get exact x=0.5 or y=0.5
        # for 32/2-1 = 15, and 64/2-1=31. 
        # in x-direction, one pixel less
        # in y-direction, one pixel more.
        # for h-0.515, the horizontal plot with y=0.515
        # for v-0.485, the vertical plot with x=0.485
        # 1/63=0.01587, thus 0.5 +- 0.015 is chosen from paraview for visualization.
        x_label = pd.read_csv('postprocess/h-0.515.csv')  
        y_label = pd.read_csv('postprocess/v-0.485.csv')  


        # plot quantitative data
        fig = plt.figure(figsize=(fig_size+1, fig_size))
        # plt.plot(coor, label_field[loc, :], 'r')
        plt.plot(x_label['Points:0'],x_label['Result'], 'r')
    
        plt.plot(coor, mean_field[loc, :], 'k')
        plt.fill_between(coor, mean_field[loc, :] - 2*std_field[loc, :], mean_field[loc, :] + 2*std_field[loc, :], alpha=0.7, facecolor='gray')
        plt.xlabel(label_text)
        plt.ylabel('value')
        plt.legend(['DNS', 'Mean'])
        plt.xlim([0,1.0])
        plt.title('Mean $\pm$ 2 Std')
        plt.tight_layout()
        if filename:
            plt.savefig(filename.replace('.png', '-uq-x-' + str(dof) + '.png'))

        plt.clf()
        # plot quantitative data
        fig = plt.figure(figsize=(fig_size+1, fig_size))

        coor = np.flip(coor)

        # plt.plot(coor, label_field[:, loc], 'r')
        plt.plot(y_label['Points:1'],y_label['Result'], 'r')
        plt.plot(coor, mean_field[:, loc], 'k')
        plt.fill_between(coor, mean_field[:, loc] - 2*std_field[:, loc], mean_field[:, loc] + 2*std_field[:,loc], alpha=0.7, facecolor='gray')
        plt.xlabel(label_text)
        plt.ylabel('value')
        plt.legend(['DNS', 'Mean'])
        plt.xlim([0,1.0])
        plt.title('Mean $\pm$ 2 Std')
        plt.tight_layout()
        if filename:
            plt.savefig(filename.replace('.png', '-uq-y-' + str(dof) + '.png'))

def plot_bvp_results_with_quantitative_line(pixel=32):
    """ """
    if pixel == 32:
        # cnn
        saved_data = "results/2022-03-12T05-01-exp-2-60-NN-cnn-x13-B512-E5000-I50-mc1-1S0.0e+00-2S0.0e+00-Nadam-2.5e-04-data-20220312050122.pickle"
    elif pixel == 64:
        saved_data = None
    restart_name = '-' + str(pixel)

    # sys.argv.extend(['cnn.ini', '-rf', saved_data, '-ra', '19999'])
    sys.argv.extend(['cnn.ini', '-rf', saved_data])

    inputs = []
    labels = []
    mean = []
    var = []
    std = []
    problem = thisPDESystem()
    problem.run()
    _inputs, _labels, _mean, _var, _std = problem.test(test_folder='DNS', plot_png=False)
    inputs.append(_inputs)
    labels.append(_labels)
    mean.append(_mean)
    var.append(_var)
    std.append(_std)

    # # bnn
    if pixel == 32:
        # cnn
        saved_data = "results/2022-03-19T21-09-exp-5-59-BNN-bnn-5-x10-B256-E20000-I100-mc100-1S2.0e-04-2S1.0e-03-Nadam-2.0e-06-data-20220319210943.pickle"
    elif pixel == 64:
        saved_data = None
    sys.argv[3] = saved_data
    problem = thisPDESystem()
    # problem.run()
    _inputs, _labels, _mean, _var, _std = problem.test(test_folder='DNS', plot_png=False)
    inputs.append(_inputs)
    labels.append(_labels)
    mean.append(_mean)
    var.append(_var)
    std.append(_std)

    for i0 in range(0, np.shape(_inputs)[0]):
        plot_fields(
                list_of_field = [
                    inputs[0][i0:i0+1,:,:,0:1],
                    inputs[0][i0:i0+1,:,:,1:2],
                    ],
                list_of_field_name = [
                    'Dirichelet BCs', 
                    'Neumann BCs',
                    ],
                dof = 1, 
                dof_name = [''],
                filename = 'diffusion-input-oct-'+str(i0) + restart_name +'.png',
                Tex = True,
                vmin = -2,
                vmax = 0.5,
                fig_size = 4.0,
                )

    for i0 in range(0, np.shape(_inputs)[0]):
        # plot_field_results_oct(
        plot_field_results_oct_pixel_data(
                list_of_field = [
                    inputs[0][i0:i0+1,:,:,0:1],
                    labels[0][i0:i0+1,:,:,0:1],
                    mean[0][i0:i0+1,:,:,0:1],
                    mean[1][i0:i0+1,:,:,0:1],
                    std[1][i0:i0+1,:,:,0:1],
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
                dof = 1, 
                dof_name = [''],
                filename = 'diffusion-results-oct-'+str(i0) + restart_name +'.png',
                Tex = True,
                vmin = 0.5, 
                vmax = 0.85,
                )


def plot_mean_std(mean, std):
    """ testing purpose only """
    dirichlet_input = mean
    dirichelt_sum = np.sum(dirichlet_input[0,:,:,0], axis=0)
    x_direction = np.sum(dirichelt_sum > 0) > 0
    if x_direction:
        label_text = 'X coordinate'
    else:
        label_text = 'Y coordinate'
    nn = np.shape(mean)[1]
    dh = 1.0/(nn - 1)
    coor = np.array(list(range(0, 16)))*dh
    loc = int(nn/2)
    plt.plot(coor, mean[0, :, loc, 0], 'r')
    plt.plot(coor, mean[0, :, loc, 0], 'k')
    plt.fill_between(coor, mean[0, :,loc,0] - std[0, :,loc,0], mean[0, :,loc,0] + std[0, :,loc,0], alpha=0.4, facecolor='gray')
    plt.xlabel(label_text)
    plt.ylabel('Val.')
    plt.legend(['DNS', 'Mean'])
    plt.title('Mean $\pm$ Std')
    plt.show()

def plot_paraview_csv():
    """ testing purpose only """ 
    # generate paraview data
    # 1. plot over line
    # 2. change the line resolution from 1000 to 50
    # 3. output to csv

    x_label = pd.read_csv('h.csv')  
    plt.plot(x_label['Points:0'],x_label['Result'])
    y_label = pd.read_csv('v.csv')  
    plt.plot(y_label['Points:1'],y_label['Result'])
    # print(x_label)
    plt.show()


def crop_img_white_boundary(img):
    #!/usr/bin/python3
    # Created by Silencer @ Stackoverflow 
    # 2018.01.23 14:41:42 CST
    # 2018.01.23 18:17:42 CST
    import cv2
    import numpy as np

    
    ## (1) Convert to gray, and threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    ## (2) Morph-op to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)
    
    ## (3) Find the max-area contour
    cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnt = sorted(cnts, key=cv2.contourArea)[-1]
    
    ## (4) Crop and save it
    x,y,w,h = cv2.boundingRect(cnt)
    dst = img[y:y+h, x:x+w]
    return dst
    # cv2.imwrite("001.png", dst)

def load_octagon_dns():
    import PIL.Image
    img_file = 'postprocess/octagon-dns-1.png'
    img = Image.open(img_file)
    if np.shape(img)[2] == 4:
        rgba_image = PIL.Image.open(img_file)
        rgb_image = rgba_image.convert('RGB')
        img = np.array(rgb_image) # im2arr.shape: height x width x channel
    img = crop_img_white_boundary(img)
    return img

def plot_img():
    img = load_octagon_dns()
    imgplot = plt.imshow(img)
    loc_x = np.shape(img)[0]/2.0
    loc_y = np.shape(img)[1]/2.0
    plt.axhline(loc_x, color ='k', linestyle ="--") 
    plt.axvline(loc_y, color ='k', linestyle ="--") 
    plt.show()

def plot_new_bnn_loss_simga2(pickle_file='', lr='', sigma1=''):
    if pickle_file == '':
        pickle_file = 'results/2020-12-30T15-50-BNN-diffusion-20-bvp-final-1-bnn-x9-B64-E5000-I50-mc100-1S1.0e-04-2S1.0e-04-Nadam-1.0e-08-data16x16-diffusion-DNS-20-D1.0-20201230155035.pickle'

    plot_one_loss(pickle_file, 'diffusion-20-bvp-bnn-loss-' + lr + '.png', show_line=False)
    plot_sigma2(pickle_file, 'diffusion-20-bvp-bnn-sigma2-' + lr + '.png', show_line=False, sigma1=sigma1)

if __name__ == '__main__':
    # ['post_process.py', 'test_right_0_small.ini', '-rf', 'results/2020-12-30T17-18-NN-test_right_100_small-x11-B128-E2000-I100-mc1-1S0.0e+00-2S0.0e+00-Nadam-2.5e-04-data16x16_right_flux-20201230171851.pickle', '-ra', '10']

    # mat1 = np.random.rand(2,16,16,1)
    # mat2 = np.random.rand(2,16,16,1)
    # plot_mean_std(mean=mat1, std=mat2)
    # plot_paraview_csv()
    # plot_img()

    #############################################################
    # run the following three command one by one
    #------------------ 20 BVPs ----------------
    # plot_zero_initialization_illustration()
    # plot_train_loss()
    # plot_each_bvp_results_with_quantitative_line()
    # plot_each_bvp_results_with_quantitative_line(100)
    # plot_each_bvp_results_with_quantitative_line(450)
    # plot_each_bvp_results_with_quantitative_line(500)
    # plot_each_bvp_results_with_quantitative_line(600)
    # plot_each_bvp_results_with_quantitative_line(1000)
    # plot_each_bvp_results_with_quantitative_line(2000)
    # plot_each_bvp_results_with_quantitative_line(4000)
    #------------------ 1 BVP on octagon ----------------
    plot_bvp_results_with_quantitative_line(pixel=32)
    # plot_bvp_results_with_quantitative_line(pixel=64)
    # plot_new_bnn_loss_simga2('results/2020-12-30T15-50-BNN-diffusion-20-bvp-final-1-bnn-x9-B64-E5000-I50-mc100-1S1.0e-04-2S1.0e-04-Nadam-1.0e-08-data16x16-diffusion-DNS-20-D1.0-20201230155035.pickle', '1e-8')
    # plot_new_bnn_loss_simga2('results/2021-05-10T10-09-BNN-diffusion-20-bvp-final-1-bnn-x9-B64-E10000-I50-mc100-1S1.0e-04-2S1.0e-04-Nadam-1.0e-06-data16x16-diffusion-DNS-20-D1.0-20210510100934.pickle', '1e-6')
    # plot_new_bnn_loss_simga2('results/2021-05-10T17-46-BNN-diffusion-20-bvp-final-1-bnn-x9-B64-E10000-I50-mc100-1S1.0e-04-2S1.0e-04-Nadam-1.0e-05-data16x16-diffusion-DNS-20-D1.0-20210510174621.pickle', '1e-5')
    # plot_new_bnn_loss_simga2('results/2021-05-11T23-18-BNN-diffusion-20-bvp-final-1-bnn-x9-B64-E5000-I50-mc100-1S1.0e-02-2S1.0e-04-Nadam-1.0e-05-data16x16-diffusion-DNS-20-D1.0-20210511231832.pickle', '1e-5', sigma1='$\Sigma_1=1.0e-4$')
    # plot_new_bnn_loss_simga2('results/2021-05-11T23-46-BNN-diffusion-20-bvp-final-1-bnn-x9-B64-E500-I50-mc100-1S5.0e-02-2S1.0e-04-Nadam-1.0e-05-data16x16-diffusion-DNS-20-D1.0-20210511234611.pickle', '1e-5', sigma1='$\Sigma_1=2.5e-3$')
    # plot_new_bnn_loss_simga2('results/2021-05-12T07-42-BNN-diffusion-20-bvp-final-1-bnn-x9-B64-E1000-I50-mc100-1S1.0e-01-2S1.0e-04-Nadam-1.0e-05-data16x16-diffusion-DNS-20-D1.0-20210512074236.pickle', '1e-5', sigma1='$\Sigma_1=1.0e-2$')
    # plot_new_bnn_loss_simga2('results/2021-05-12T09-02-BNN-diffusion-20-bvp-final-1-bnn-x9-B64-E1000-I50-mc100-1S1.0e-03-2S1.0e-04-Nadam-1.0e-05-data16x16-diffusion-DNS-20-D1.0-20210512090212.pickle', '1e-5', sigma1='$\Sigma_1=1.0e-6$')



