#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt 
import sys

labels = np.load(sys.argv[1])
# features = np.load('np-features.npy')

print('data shape:', np.shape(labels))

labels = np.ma.masked_where(labels == -1, labels) # remove the margin
for i0 in range(0, np.shape(labels)[0]):
    plt.clf()
    # plt.figure()
    # plt.title('label 1')
    # plt.imshow(labels[i0,:,:,0])
    plt.figure()
    plt.title('label 2')
    c_img = plt.imshow(labels[i0,:,:,1])
    plt.colorbar(c_img)
    plt.savefig('l-shape-'+str(i0)+'.png')
    # plt.show()
    # exit(0)
