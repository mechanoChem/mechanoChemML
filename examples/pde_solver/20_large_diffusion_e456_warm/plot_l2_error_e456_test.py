import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import pandas as pd

import numpy as np
import pickle
import sys

"""
l2 error is computed in the main_test_min_max.py 
this script is only for plotting purpose.
"""

def plot_tex(tex=False):
    import os
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.style.reload_library()
    plt.style.use('zxx')
    print('find zxx: ', os.path.isfile('zxx.mplstyle'))
    if (os.path.isfile('zxx.mplstyle')):
        plt.style.use('zxx.mplstyle')
    if (tex) :
        plt.style.use('tex')
    print(plt.style.available)
    print(mpl.get_configdir())

f1 = 'all_summary.csv'
summary = pd.read_csv(f1)
summary_pentagon = summary[summary['e5-all-mean'].notna()]
summary_e456 = summary[summary['s4-all-mean'].notna()].reset_index()
# print(summary_pentagon, len(summary_pentagon))
# print(summary_e456, len(summary_e456))

plot_tex(True)

# for i0 in range(len(summary_pentagon)):
    # loc = []
    # tick = []
    # l2_error_a_mean = []
    # x_a =[]
    # l2_error_a_std = []
    # l2_error_n_mean = []
    # x_n = []
    # l2_error_n_std = []
    # l2_error_d_mean = []
    # l2_error_d_std = []
    # x_d = []

    # # training information
    # x_a.append(1-0.2)
    # x_n.append(1+0.0)
    # x_d.append(1+0.2)
    # l2_error_a_mean.append(summary_pentagon['e5-all-mean'   ][i0])
    # l2_error_a_std.append (summary_pentagon['e5-all-std'    ][i0])
    # l2_error_n_mean.append(summary_pentagon['e5-w-Neu-mean' ][i0])
    # l2_error_n_std.append (summary_pentagon['e5-w-Neu-std'  ][i0])
    # l2_error_d_mean.append(summary_pentagon['e5-wo-Neu-mean'][i0])
    # l2_error_d_std.append (summary_pentagon['e5-wo-Neu-std' ][i0])
    # loc.append(1)
    # tick.append('')

    # # testing regular
    # x_a.append(2-0.2)
    # x_n.append(2+0.0)
    # x_d.append(2+0.2)
    # l2_error_a_mean.append(summary_pentagon['e5-regular-all-mean'   ][i0])
    # l2_error_a_std.append (summary_pentagon['e5-regular-all-std'    ][i0])
    # l2_error_n_mean.append(summary_pentagon['e5-regular-w-Neu-mean' ][i0])
    # l2_error_n_std.append (summary_pentagon['e5-regular-w-Neu-std'  ][i0])
    # l2_error_d_mean.append(summary_pentagon['e5-regular-wo-Neu-mean'][i0])
    # l2_error_d_std.append (summary_pentagon['e5-regular-wo-Neu-std' ][i0])
    # loc.append(2)
    # tick.append('regular')
    
    # # testing extreme
    # x_a.append(3-0.2)
    # x_n.append(3+0.0)
    # x_d.append(3+0.2)
    # l2_error_a_mean.append(summary_pentagon['e5-extreme-all-mean'   ][i0])
    # l2_error_a_std.append (summary_pentagon['e5-extreme-all-std'    ][i0])
    # l2_error_n_mean.append(summary_pentagon['e5-extreme-w-Neu-mean' ][i0])
    # l2_error_n_std.append (summary_pentagon['e5-extreme-w-Neu-std'  ][i0])
    # l2_error_d_mean.append(summary_pentagon['e5-extreme-wo-Neu-mean'][i0])
    # l2_error_d_std.append (summary_pentagon['e5-extreme-wo-Neu-std' ][i0])
    # loc.append(3)
    # tick.append('extreme')

    # # print(x_a, x_n, x_d, l2_error_a_mean)
    # plt.rcParams["figure.figsize"] = (4,4)
    # fig = plt.figure()
    # plt.errorbar(x_a, l2_error_a_mean, yerr=l2_error_a_std, label='all BCs', ls='none', marker='s', color='k')
    # plt.errorbar(x_n, l2_error_n_mean, yerr=l2_error_n_std, label='w Neu. BCs', ls='none', marker='v', color='k')
    # plt.errorbar(x_d, l2_error_d_mean, yerr=l2_error_d_std, label='w/o Neu. BCs', ls='none', marker='o', color='k')
    # # plt.title('testing results')
    # plt.legend(loc='upper center')
    # plt.xlabel('')
    # plt.ylabel('$L_2$ error')
    # plt.ylim([0,0.2])
    # # plt.xticks([2.2, 3.2], ['regular', 'extreme'])
    # plt.xticks(loc, tick)
    # plt.axvline(x=1.5, ls='--',color='k')
    # plt.text(0.75, -0.035, "train", fontsize=20)
    # plt.text(2.3, -0.035, "test", fontsize=20)
    # plt.savefig('l2_error_pentagon_'+str(i0)+'.png', bbox_inches='tight', dpi=600)
    # # plt.show()
    # # exit(0)


edges = [4,5,6,7,8,9,10,11]
plot_tex(True)
e456=-1
e4569=-1
for i0 in range(len(summary_e456)):
    loc = []
    tick = []

    l2_error_a_mean = []
    x_a =[]
    l2_error_a_std = []
    l2_error_n_mean = []
    x_n = []
    l2_error_n_std = []
    l2_error_d_mean = []
    l2_error_d_std = []
    x_d = []

    # # training information: e4
    # x_a.append(0-0.2)
    # x_n.append(0+0.0)
    # x_d.append(0+0.2)
    # l2_error_a_mean.append(summary_e456['e4-32k-all-mean'   ][i0])
    # l2_error_a_std.append (summary_e456['e4-32k-all-std'    ][i0])
    # l2_error_n_mean.append(summary_e456['e4-32k-w-Neu-mean' ][i0])
    # l2_error_n_std.append (summary_e456['e4-32k-w-Neu-std'  ][i0])
    # l2_error_d_mean.append(summary_e456['e4-32k-wo-Neu-mean'][i0])
    # l2_error_d_std.append (summary_e456['e4-32k-wo-Neu-std' ][i0])
    # loc.append(0)
    # tick.append('4')

    # # training information: e5
    # x_a.append(1-0.2)
    # x_n.append(1+0.0)
    # x_d.append(1+0.2)
    # l2_error_a_mean.append(summary_e456['e5-64k-all-mean'   ][i0])
    # l2_error_a_std.append (summary_e456['e5-64k-all-std'    ][i0])
    # l2_error_n_mean.append(summary_e456['e5-64k-w-Neu-mean' ][i0])
    # l2_error_n_std.append (summary_e456['e5-64k-w-Neu-std'  ][i0])
    # l2_error_d_mean.append(summary_e456['e5-64k-wo-Neu-mean'][i0])
    # l2_error_d_std.append (summary_e456['e5-64k-wo-Neu-std' ][i0])
    # loc.append(1)
    # tick.append('5')
    
    # # training information: e6
    # x_a.append(2-0.2)
    # x_n.append(2+0.0)
    # x_d.append(2+0.2)
    # l2_error_a_mean.append(summary_e456['e6-96k-all-mean'   ][i0])
    # l2_error_a_std.append (summary_e456['e6-96k-all-std'    ][i0])
    # l2_error_n_mean.append(summary_e456['e6-96k-w-Neu-mean' ][i0])
    # l2_error_n_std.append (summary_e456['e6-96k-w-Neu-std'  ][i0])
    # l2_error_d_mean.append(summary_e456['e6-96k-wo-Neu-mean'][i0])
    # l2_error_d_std.append (summary_e456['e6-96k-wo-Neu-std' ][i0])
    # loc.append(2)
    # tick.append('6')

    # # training information: e9
    # x_a.append(3-0.2)
    # x_n.append(3+0.0)
    # x_d.append(3+0.2)
    # l2_error_a_mean.append(summary_e456['e9-96k-all-mean'   ][i0])
    # l2_error_a_std.append (summary_e456['e9-96k-all-std'    ][i0])
    # l2_error_n_mean.append(summary_e456['e9-96k-w-Neu-mean' ][i0])
    # l2_error_n_std.append (summary_e456['e9-96k-w-Neu-std'  ][i0])
    # l2_error_d_mean.append(summary_e456['e9-96k-wo-Neu-mean'][i0])
    # l2_error_d_std.append (summary_e456['e9-96k-wo-Neu-std' ][i0])
    # loc.append(3)
    # if not np.isnan(summary_e456['e9-96k-all-mean'   ][i0]):
        # tick.append('9')
    # else:
        # tick.append('')

    for e0 in edges:
        x_a.append(e0-0.2)
        x_n.append(e0+0.0)
        x_d.append(e0+0.2)
        l2_error_a_mean.append(summary_e456['s'+str(e0)+'-all-mean'   ][i0])
        l2_error_a_std.append (summary_e456['s'+str(e0)+'-all-std'    ][i0])
        l2_error_n_mean.append(summary_e456['s'+str(e0)+'-w-Neu-mean' ][i0])
        l2_error_n_std.append (summary_e456['s'+str(e0)+'-w-Neu-std'  ][i0])
        l2_error_d_mean.append(summary_e456['s'+str(e0)+'-wo-Neu-mean'][i0])
        l2_error_d_std.append (summary_e456['s'+str(e0)+'-wo-Neu-std' ][i0])
        loc.append(e0)
        tick.append(str(e0))

    print(x_a, x_n, x_d, l2_error_a_mean)

    plt.rcParams["figure.figsize"] = (12,4)
    fig = plt.figure()
    plt.errorbar(x_a, l2_error_a_mean, yerr=l2_error_a_std, label='all BCs', ls='none', marker='s', color='k')
    plt.errorbar(x_n, l2_error_n_mean, yerr=l2_error_n_std, label='w Neu. BCs', ls='none', marker='v', color='k')
    plt.errorbar(x_d, l2_error_d_mean, yerr=l2_error_d_std, label='w/o Neu. BCs', ls='none', marker='o', color='k')
    # plt.title('testing results')
    plt.legend(loc='upper left')
    plt.ylabel('$L_2$ error')
    plt.xlabel('')
    plt.ylim([0,0.5])
    plt.xlim([-0.8,11.8])
    plt.xticks(loc, tick)
    plt.text(1.1, -0.075, "train", fontsize=20)
    plt.text(7.25, -0.075, "test", fontsize=20)
    plt.axvline(x=3.5, ls='--',color='k')
    e456 += 1
    # plt.savefig('l2_error_e456_'+str(e456)+'.png', bbox_inches='tight')
    plt.savefig('l2_error_e456_'+str(e456)+'_cnn_rescale_physical.png', bbox_inches='tight')
