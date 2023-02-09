#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt 
import sys

def example_plot(ax, img0):
    c_img = ax.imshow(img0)
    return c_img

f1 = sys.argv[1]
ind0 = -1
try:
    ind0 = int(sys.argv[2])
except:
    pass
all_data = np.load(f1)
print('data shape:', np.shape(all_data))

all_data = np.ma.masked_where(all_data == -1, all_data) # remove the margin
all_data = np.ma.masked_where(all_data == -2, all_data) # remove the margin
for i0 in range(0, np.shape(all_data)[0]):
    if ind0 >= 0:
        i0 = ind0
    fig, axs = plt.subplots(1, np.shape(all_data)[3])
    j0 = 0
    if np.shape(all_data)[3] > 1:
        for ax in axs.flat:
            c_img = example_plot(ax, all_data[i0,:,:, j0])
            # plt.colorbar(c_img)
            ax.set_title("channel " + str(j0))
            j0 += 1
    else:
        c_img = example_plot(axs, all_data[i0,:,:, j0])
        axs.set_title("channel " + str(j0))
    plt.savefig(f1.replace('/','')+'-'+str(i0)+'.png')
    plt.show()
    exit(0)
