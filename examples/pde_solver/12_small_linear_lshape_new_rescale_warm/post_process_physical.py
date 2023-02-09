# from mechanoChemML.workflows.pde_solver.pde_system_elasticity_linear import WeakPDELinearElasticity as thisPDESystem
from mechanoChemML.workflows.pde_solver.pde_utility import plot_PDE_solutions, plot_fields, plot_tex, plot_one_loss, plot_sigma2, get_cm
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.image as mpimg
from PIL import Image
import pandas as pd


import numpy as np
import tensorflow as tf
from mechanoChemML.workflows.pde_solver.pde_workflow_steady_state import PDEWorkflowSteadyState
from mechanoChemML.workflows.pde_solver.pde_system_elasticity_linear import LayerLinearElasticityBulkResidual
from mechanoChemML.workflows.pde_solver.pde_system_elasticity_linear import WeakPDELinearElasticity as thisPDESystem

scale_factor = 0.1 # 


class thisPDESystem(PDEWorkflowSteadyState):
    def __init__(self):
        super().__init__()
        self.dof = 2
        self.dof_name = ['Ux', 'Uy']
        self.problem_name = 'linear-elasticity'
        self.E0 = 25
        self.nu0 = 0.3
        self.UseTwoNeumannChannel = False
        self.normalization_factor = scale_factor

    def _bulk_residual(self, y_pred):
        """
        bulk residual for linear elasticity
        """
        print('normal_factor: ', self.normalization_factor)
        elem_bulk_residual=LayerLinearElasticityBulkResidual(dh=self.dh, E0=self.E0, nu0=self.nu0, normalization_factor=self.normalization_factor)(y_pred)
        return elem_bulk_residual
    
    def _build_optimizer(self):
        """ 
        Build the optimizer for weak-PDE constrained NN
        """
        # not using the decay learning rate function.
        # global_step = tf.Variable(0, name='global_step', trainable=False)
        # LearningRate = tf.compat.v1.train.exponential_decay(
            # learning_rate=self.LR0,
            # global_step=global_step,
            # decay_steps=1000,
            # decay_rate=0.8,
            # staircase=True)
        # print("new learning rate")
        initial_learning_rate = self.LR0
        LearningRate = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=500,
            decay_rate=0.97,
            staircase=True)

        if self.NNOptimizer.lower() == 'Adam'.lower() :
            optimizer = tf.keras.optimizers.Adam(learning_rate=LearningRate)
        elif self.NNOptimizer.lower() == 'Nadam'.lower() :
            optimizer = tf.keras.optimizers.Nadam(learning_rate=self.LR0)
        elif self.NNOptimizer.lower() == 'SGD'.lower() : # not very well
            optimizer = tf.keras.optimizers.SGD(learning_rate=LearningRate) 
        else:
            raise ValueError('Unknown optimizer option:', self.NNOptimizer)
        return optimizer


def output_inter_extra_with_reaction_force(sim_id=0):

    if sim_id == 0:
        # saved_data = "results/2021-01-05T09-36-BNN-l-shape-32-bnn-new-x9-B64-E100-I50-mc50-1S1.0e-04-2S1.0e-04-Nadam-1.0e-08-datal-shape-32x32-correct-20210105093652.pickle"
        saved_data = "results/2022-06-30T21-30-exp-8-57-NN-cnn-x11-B1024-E50000-I100-mc1-1S0.0e+00-2S0.0e+00-Nadam-2.5e-04-data-20220630213057.pickle"
        sys.argv.extend(['cnn.ini', '-rf', saved_data])
    elif sim_id == 1:
        # saved_data = "results/2021-01-05T00-33-NN-l-shape-32-adam-2-x10-B256-E50000-I100-mc1-1S0.0e+00-2S0.0e+00-Adam-2.5e-04-datal-shape-32x32-correct-20210105003316.pickle"
        # saved_data = "results/2022-03-15T08-07-exp-12-57-NN-cnn-50k-x11-B1024-E50000-I100-mc1-1S0.0e+00-2S0.0e+00-Adam-2.5e-04-..data-20220315080711.pickle"
        saved_data = "results/2022-07-06T21-41-exp-13-57-BNN-bnn-x11-B256-E50000-I100-mc100-1S1.0e-05-2S1.0e-03-Nadam-2.0e-06-data-20220706214140.pickle"
        sys.argv.extend(['bnn.ini', '-rf', saved_data])

    problem = thisPDESystem()
    # problem.run()
    _inputs, _labels, _mean, _var, _std = problem.test(test_folder='DNS', plot_png=False, output_reaction_force=True)
    _inputs, _labels, _mean, _var, _std = problem.test(test_folder='Inter', plot_png=False, output_reaction_force=True)

def plot_forces(sim_id=0):
    plot_tex(True)
    dns = 'postprocess/force.csv'

    F_dns = pd.read_csv(dns)  


    if sim_id == 0:
        # train = "results/2021-01-05T14-00-BNN-l-shape-32-bnn-new-x9-B64-E100-I50-mc50-1S1.0e-04-2S1.0e-04-Nadam-1.0e-08-datal-shape-32x32-correct-20210105093652-e-1-ra-ckpt0099-DNSF.npy"
        # inter = "results/2021-01-05T14-00-BNN-l-shape-32-bnn-new-x9-B64-E100-I50-mc50-1S1.0e-04-2S1.0e-04-Nadam-1.0e-08-datal-shape-32x32-correct-20210105093652-e-1-ra-ckpt0099-InterF.npy"

        train = "results/2022-07-08T09-57-Destiny-BNN-bnn-x11-B256-E50000-I100-mc100-1S1.0e-05-2S1.0e-03-Nadam-2.0e-06-data-20220706214140-e-1-ra-ckpt49999-DNSF.npy"
        # inter = "results/2022-07-08T09-57-Destiny-BNN-bnn-x11-B256-E50000-I100-mc100-1S1.0e-05-2S1.0e-03-Nadam-2.0e-06-data-20220706214140-e-1-ra-ckpt49999-InterF.npy"
    elif sim_id == 1:
        # train = "results/2021-01-05T15-24-NN-l-shape-32-bnn-new-x10-B256-E50000-I100-mc1-1S0.0e+00-2S0.0e+00-Adam-2.5e-04-datal-shape-32x32-correct-20210105003316-e-1-ra-ckpt11000-DNSF.npy"
        # inter = "results/2021-01-05T15-24-NN-l-shape-32-bnn-new-x10-B256-E50000-I100-mc1-1S0.0e+00-2S0.0e+00-Adam-2.5e-04-datal-shape-32x32-correct-20210105003316-e-1-ra-ckpt11000-InterF.npy"
        # train = "results/2022-05-17T10-02-Destiny-NN-bnn-x11-B1024-E50000-I100-mc1-1S0.0e+00-2S0.0e+00-Adam-2.5e-04-..data-20220315080711-e-1-ra-ckpt49999-DNSF.npy"
        # inter = "results/2022-05-17T10-02-Destiny-NN-bnn-x11-B1024-E50000-I100-mc1-1S0.0e+00-2S0.0e+00-Adam-2.5e-04-..data-20220315080711-e-1-ra-ckpt49999-InterF.npy"
        train = "results/2022-07-08T09-56-Destiny-NN-cnn-x11-B1024-E50000-I100-mc1-1S0.0e+00-2S0.0e+00-Nadam-2.5e-04-data-20220630213057-e-1-ra-ckpt49999-DNSF.npy"
        # inter = "results/2022-07-08T09-56-Destiny-NN-cnn-x11-B1024-E50000-I100-mc1-1S0.0e+00-2S0.0e+00-Nadam-2.5e-04-data-20220630213057-e-1-ra-ckpt49999-InterF.npy"


    F_train = np.load(train)
    # F_inter = np.load(inter)
    # F_all = np.vstack([F_train, F_inter])
    F_all = F_train #np.vstack([F_train, F_inter])
    F_all = F_all[F_all[:,1].argsort()]
    # print('f_all', F_all)
    print(F_train)

    # # 0,  1   2   3   4        5        6       7
    # # ux, uy, tx, ty, fx_mean, fy_mean, fx_std, fy_std 
    # print('f_dns', F_dns)
    # print('f_train', F_train)

    if sim_id == 0: # BNN
        plt.clf()
        n_std = 1
        fig = plt.figure(figsize=(5, 4))
        plt.plot(F_dns['dUy'],F_dns['Fx'], '-k')
        plt.plot(F_train[:,1], F_train[:,4], 'r.')
        # plt.plot(F_inter[:,1], F_inter[:,4], 'g.')
        plt.fill_between(F_all[:,1], F_all[:,4]-n_std*F_all[:,6], F_all[:,4]+n_std*F_all[:,6], alpha=0.7, facecolor='gray')

        plt.xlabel(r'$\bar{u}_y$')
        plt.ylabel('$F_x$')
        plt.legend(['DNS', 'NN'])
        # plt.xlim([0,1.0])
        plt.title('Mean $\pm$ '+str(n_std)+' Std')
        plt.tight_layout()
        plt.savefig('linear-Fx-std-'  + str(n_std) + '-inter-' + str(sim_id) + '.png')

        plt.clf()
        fig = plt.figure(figsize=(5, 4))
        plt.plot(F_dns['dUy'],F_dns['Fy'], '-k')
        plt.plot(F_train[:,1], F_train[:,5], 'r.')
        # plt.plot(F_inter[:,1], F_inter[:,5], 'g.')
        plt.fill_between(F_all[:,1], F_all[:,5]-n_std*F_all[:,7], F_all[:,5]+n_std*F_all[:,7], alpha=0.7, facecolor='gray')

        plt.xlabel(r'$\bar{u}_y$')
        plt.ylabel('$F_y$')
        plt.legend(['DNS', 'NN'])
        # plt.xlim([0,1.0])
        plt.title('Mean $\pm$ '+str(n_std)+' Std')
        plt.tight_layout()
        plt.savefig('linear-Fy-std-' + str(n_std) + '-inter-' + str(sim_id) + '.png')
    elif sim_id == 1: # CNN
        plt.clf()
        fig = plt.figure(figsize=(5, 4))
        plt.plot(F_dns['dUy'],F_dns['Fx'], '-k')
        plt.plot(F_train[:,1], F_train[:,4], 'r.')
        # plt.plot(F_inter[:,1], F_inter[:,4], 'g.')

        plt.xlabel(r'$\bar{u}_y$')
        plt.ylabel('$F_x$')
        plt.legend(['DNS', 'NN'])
        # plt.xlim([0,1.0])
        plt.title('Deterministic')
        plt.tight_layout()
        plt.savefig('linear-Fx-inter-' + str(sim_id) + '.png')

        plt.clf()
        fig = plt.figure(figsize=(5, 4))
        plt.plot(F_dns['dUy'],F_dns['Fy'], '-k')
        plt.plot(F_train[:,1], F_train[:,5], 'r.')
        # plt.plot(F_inter[:,1], F_inter[:,5], 'g.')

        plt.xlabel(r'$\bar{u}_y$')
        plt.ylabel('$F_y$')
        plt.legend(['DNS', 'NN'])
        # plt.xlim([0,1.0])
        plt.title('Deterministic')
        plt.tight_layout()
        plt.savefig('linear-Fy-inter-' + str(sim_id) + '.png')


def plot_lshape_bvp_results_with_quantitative_line():
    """ """
    # cnn
    # saved_data = "results/2020-12-31T13-52-NN-l-shape-32-x10-B256-E5000-I100-mc1-1S0.0e+00-2S0.0e+00-Nadam-2.5e-04-datal-shape-32x32-20201231135224.pickle" # old
    # saved_data =   "results/2021-01-03T00-28-NN-l-shape-32-x10-B256-E50000-I100-mc1-1S0.0e+00-2S0.0e+00-Nadam-2.5e-04-datal-shape-32x32-20210103002821.pickle" # new
    # saved_data = "results/2021-01-05T00-33-NN-l-shape-32-adam-2-x10-B256-E50000-I100-mc1-1S0.0e+00-2S0.0e+00-Adam-2.5e-04-datal-shape-32x32-correct-20210105003316.pickle"
    # saved_data = "results/2022-03-15T08-07-exp-12-57-NN-cnn-50k-x11-B1024-E50000-I100-mc1-1S0.0e+00-2S0.0e+00-Adam-2.5e-04-..data-20220315080711.pickle"
    saved_data = "results/2022-06-30T21-30-exp-8-57-NN-cnn-x11-B1024-E50000-I100-mc1-1S0.0e+00-2S0.0e+00-Nadam-2.5e-04-data-20220630213057.pickle"
    sys.argv.extend(['cnn.ini', '-rf', saved_data])

    inputs = []
    labels = []
    mean = []
    var = []
    std = []
    problem = thisPDESystem()
    # problem.run()
    _inputs, _labels, _mean, _var, _std = problem.test(test_folder='Inter', plot_png=False)
    # _inputs, _labels, _mean, _var, _std = problem.test(test_folder='DNS', plot_png=False)
    inputs.append(_inputs)
    labels.append(_labels)
    mean.append(_mean)
    var.append(_var)
    std.append(_std)

    # get the loading location where vertical displacement is 0.55 (scaled) -> or 0.1 (no scaled)
    Loadings = tf.reduce_max(_inputs, axis=[1,2,3])
    for i0 in range(0, np.shape(Loadings)[0]):
        # print(i0)
        if Loadings[i0] == 0.545:
            nn = i0
    # print(Loadings, nn)
    # exit(0)

    # # bnn
    # saved_data = "results/2021-01-03T06-53-BNN-l-shape-32-bnn-x9-B64-E100-I50-mc50-1S1.0e-04-2S1.0e-04-Nadam-1.0e-08-datal-shape-32x32-20210103065355.pickle"
    # saved_data = "results/2021-01-05T09-36-BNN-l-shape-32-bnn-new-x9-B64-E100-I50-mc50-1S1.0e-04-2S1.0e-04-Nadam-1.0e-08-datal-shape-32x32-correct-20210105093652.pickle"
    # saved_data = "results/2022-03-15T08-07-exp-12-57-NN-cnn-50k-x11-B1024-E50000-I100-mc1-1S0.0e+00-2S0.0e+00-Adam-2.5e-04-..data-20220315080711.pickle"
    saved_data = "results/2022-07-06T21-41-exp-13-57-BNN-bnn-x11-B256-E50000-I100-mc100-1S1.0e-05-2S1.0e-03-Nadam-2.0e-06-data-20220706214140.pickle"
    # ra = 100
    # # restart_name = '-r' + str(ra)
    # sys.argv.extend(['-ra', str(ra)])
    restart_name = ''

    sys.argv[3] = saved_data
    problem = thisPDESystem()
    # problem.run()
    _inputs, _labels, _mean, _var, _std = problem.test(test_folder='Inter', plot_png=False)
    # _inputs, _labels, _mean, _var, _std = problem.test(test_folder='DNS', plot_png=False)
    inputs.append(_inputs)
    labels.append(_labels)
    mean.append(_mean)
    var.append(_var)
    std.append(_std)

    all_fields = {'inputs':inputs, 'labels':labels, 'mean':mean, 'var':var, 'std':std}
    pickle_out = open('all_fields_Inter_lshape.pickle', "wb")
    pickle.dump(all_fields, pickle_out)
    pickle_out.close()

    for i0 in range(0, np.shape(_inputs)[0]):
        # if i0 != nn:
            # continue
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
                filename = 'linear-input-inter-dns-'+str(i0) + restart_name +'.png',
                Tex = True,
                vmin = -2,
                vmax = 0.5,
                fig_size = 4.0,
                )

    for i0 in range(0, np.shape(_inputs)[0]):
        # if i0 != nn:
            # continue
        plot_field_results_lshape(
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
                filename = 'linear-results-inter-dns-'+str(i0) + restart_name +'.png',
                Tex = True,
                )

def plot_field_results_lshape(list_of_field, list_of_field_name, dof, dof_name,  filename='', print_data=False, vmin=None, vmax=None, Tex=False):
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
    print('coor:', coor)

    label_min = vmin
    label_max = vmax

    for i0 in range(0, dof):
        plt.clf()
        fig = plt.figure(figsize=(figsize_list_x[tot_img-1], figsize_list_y[0] ))
        for j0 in range(1, tot_img+1):
            # masked_field = np.ma.masked_where(dirichlet_reverse_mask[0,:,:,i0] == 0.0, list_of_field[j0][0, :, :, i0]) # remove the Dirichlet BCs location
            masked_field = np.ma.masked_where(dirichlet_data[0,:,:,i0] == -1, list_of_field[j0][0, :, :, i0]) # remove the margin
            if i0 == 1: # uy
                loc = int(nn/4)
            elif i0 == 0: #ux
                loc = int(nn/4)*3

            print('loc:', loc, coor[loc])

            if list_of_field_name[j0].lower() == 'dns':
                label_field = masked_field
                label_min = np.amin(masked_field)
                label_max = np.amax(masked_field)
                print('min', label_min, 'max', label_max)
            _field_name = list_of_field_name[j0] + ' ' + dof_name[i0]
            ax = plt.subplot(1, tot_img, j0 + 1 + tot_img * 0 - 1)
            
            if _field_name.find('Sol') >= 0 or  _field_name.find('Mean') >= 0:
                c_img = plt.imshow(scale_factor * (masked_field - 0.5), cmap=hot, vmin=label_min, vmax=label_max)  # tensor
            elif _field_name.find('Std') >= 0 :
                c_img = plt.imshow(scale_factor * masked_field, cmap=hot)  # tensor
            elif _field_name.find('DNS') >= 0 :
                c_img = plt.imshow(scale_factor * (masked_field - 0.5), cmap=hot, vmin=label_min, vmax=label_max)  # tensor
                # if i0 == 0: # x
                    # # label_min = 0.5
                    # # label_max = 0.535
                    # img = load_img('l-shape-x-0.5-0.535.png')
                    # imgplot = plt.imshow(img)
                    # loc_x = np.shape(img)[0]/4.0*3.0
                    # plt.axvline(loc_x, color ='k', linestyle ="--") 
                # elif i0 == 1: # y
                    # label_min = 0.49
                    # label_max = 0.55
                    # img = load_img('l-shape-y-0.49-0.55.png')
                    # imgplot = plt.imshow(img)
                    # loc_y = np.shape(img)[1]/4.0
                    # plt.axhline(loc_y, color ='k', linestyle ="--") 

            if list_of_field_name[j0].lower().find('sol') >=0:
                det_field = masked_field
            if list_of_field_name[j0].lower().find('mean') >=0:
                mean_field = masked_field
            if list_of_field_name[j0].lower().find('std') >=0:
                std_field = masked_field

            if  _field_name.find('Mean') >= 0 or _field_name.find('Std') >= 0 or _field_name.find('DNS') >= 0:
                if i0 == 1:
                    plt.axhline(loc, color ='k', linestyle ="--") 
                if i0 == 0:
                    plt.axvline(loc, color ='k', linestyle ="--") 

            if _field_name.find('Mean') >= 0 or _field_name.find('Std') >= 0:
                plt.colorbar(c_img, fraction=fraction, pad=pad)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.title(_field_name)


        plt.tight_layout()
        if filename:
            plt.savefig(filename.replace('.png', '-dof-' + str(i0) + '.png'))

        plt.clf()
        # # note: as pixel 32, 64, are used, it's difficult to get exact x=0.5 or y=0.5
        # # for 32/2-1 = 15, and 64/2-1=31. 
        # # in x-direction, one pixel less
        # # in y-direction, one pixel more.
        # # for h-0.515, the horizontal plot with y=0.515
        # # for v-0.485, the vertical plot with x=0.485
        # # 1/63=0.01587, thus 0.5 +- 0.015 is chosen from paraview for visualization.
        # x_label = pd.read_csv('ux-x-0.75.csv')  
        # y_label = pd.read_csv('uy-y-0.75.csv')  
        if i0 == 0:
            plt.clf()
            # plot quantitative data
            fig = plt.figure(figsize=(fig_size+1, fig_size))

            coor_flip = np.flip(coor)

            plt.plot(coor_flip, scale_factor*(label_field[:, loc]-0.5), 'k')
            # plt.plot(x_label['Points:1'],x_label['ResultX'], 'r')
            plt.plot(coor_flip, scale_factor * (det_field[:, loc]-0.5), 'b')
            plt.plot(coor_flip, scale_factor * (mean_field[:, loc]-0.5), 'r')
            plt.fill_between(coor_flip, scale_factor * (mean_field[:, loc]-0.5) - 2*scale_factor*std_field[:, loc], scale_factor * (mean_field[:, loc]-0.5) + 2*scale_factor*std_field[:, loc], alpha=0.7, facecolor='gray')
            plt.xlabel(label_text)
            plt.ylabel('value')
            plt.legend(['DNS', 'Sol. (det)', 'Mean $\pm$ 2 Std'])
            plt.xlim([0,1.0])
            # plt.title('Mean $\pm$ 2 Std')
            plt.tight_layout()
            if filename:
                plt.savefig(filename.replace('.png', '-uq-x.png'))

        elif i0 == 1:
            plt.clf()
            # plot quantitative data
            fig = plt.figure(figsize=(fig_size+1, fig_size))
            plt.plot(coor, scale_factor*(label_field[loc, :]-0.5), 'k')
            # plt.plot(y_label['Points:0'],y_label['ResultY'], 'r')
    
            plt.plot(coor, scale_factor * (det_field[loc,:]-0.5), 'b')
            plt.plot(coor, scale_factor * (mean_field[loc,:]-0.5), 'r')
            plt.fill_between(coor, scale_factor * (mean_field[loc, :]-0.5) - 2*scale_factor*std_field[loc, :], scale_factor*(mean_field[loc, :]-0.5) + 2*scale_factor*std_field[loc, :], alpha=0.7, facecolor='gray')
            plt.xlabel(label_text)
            plt.ylabel('value')
            plt.legend(['DNS', 'Sol. (det)', 'Mean $\pm$ 2 Std'])
            plt.xlim([0,1.0])
            # plt.title('Mean $\pm$ 2 Std')
            plt.tight_layout()
            if filename:
                plt.savefig(filename.replace('.png', '-uq-y.png'))





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

def load_img(img_file):
    import PIL.Image
    img = Image.open(img_file)
    if np.shape(img)[2] == 4:
        rgba_image = PIL.Image.open(img_file)
        rgb_image = rgba_image.convert('RGB')
        img = np.array(rgb_image) # im2arr.shape: height x width x channel
    img = crop_img_white_boundary(img)
    return img

def plot_deformed_shape():
    print(""" this script is not good, the reference square block is not preserved. Thus, inkscape is used to manually generate this figure.""")
    # img = load_img('linear-bvp-i.png')
    # imgplot = plt.imshow(img)
    # plt.show()

    plot_tex(True)

    hot=get_cm()

    # magic number from: https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    fraction=0.046
    pad=0.04

    tot_img = 3
    fig_size = 4
    figsize_list_x = [x*fig_size*1.1 for x in range(1, 20)]
    figsize_list_y = [x*fig_size for x in range(1, 20)]

    dof = 1
    for i0 in range(0, dof):
        plt.clf()
        fig = plt.figure(figsize=(figsize_list_x[tot_img-1], figsize_list_y[dof-1] ))
        for j0 in range(0, tot_img):
            ax = plt.subplot(dof, tot_img, j0 + 1 + tot_img * i0)
            if j0 == 0:
                img = load_img('linear-bvp-i.png')
                title = 'BVP (i)'
            if j0 == 1:
                img = load_img('linear-bvp-ii.png')
                title = 'BVP (ii)'
            if j0 == 2:
                img = load_img('linear-bvp-iii.png')
                title = 'BVP (iii)'
            imgplot = plt.imshow(img)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.title(title)
            plt.box(on=None)

        plt.tight_layout()
        plt.savefig('linear-deformed-bvp-i-ii-iii.png')

if __name__ == '__main__':
    # ['post_process.py', 'test_right_0_small.ini', '-rf', 'results/2020-12-30T17-18-NN-test_right_100_small-x11-B128-E2000-I100-mc1-1S0.0e+00-2S0.0e+00-Nadam-2.5e-04-data16x16_right_flux-20201230171851.pickle', '-ra', '10']

    # mat1 = np.random.rand(2,16,16,1)
    # mat2 = np.random.rand(2,16,16,1)
    # plot_mean_std(mean=mat1, std=mat2)
    # plot_paraview_csv()
    # plot_img()

    #############################################################
    # run the following three command one by one
    #------------------ 30 BVPs ----------------
    # plot_each_bvp_results_with_quantitative_line()
    #--------------------L-shape------------
    # # # plot_deformed_shape() # not good.
    # plot_lshape_bvp_results_with_quantitative_line()
    # output_inter_extra_with_reaction_force(sim_id=0)
    output_inter_extra_with_reaction_force(sim_id=1)
    # plot_forces(sim_id=0)
    # plot_forces(sim_id=1)
