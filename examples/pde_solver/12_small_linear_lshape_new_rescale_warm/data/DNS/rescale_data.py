#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt 
import sys
import os

def plot_one_img(img):
    plt.figure()
    c_img = plt.imshow(img[0, :,:,1])
    plt.colorbar(c_img)
    plt.show()

if __name__ == '__main__':
    # if len(sys.argv) < 4:
        # print(sys.argv[0], ' [DNS label.npy]  [dummy label.npy] [features.npy]')
        # print("only works for Dirichlet BCs")
        # exit(0)

    # DNS_datafile = sys.argv[1]
    # Dummy_datafile = sys.argv[2]
    # Feature_datafile = sys.argv[3]
        
    DNS_datafile = 'np-labels-0.npy'
    Feature_datafile = 'np-features-0.npy'

    # loop over dummy datafile
    DNS_data = np.load(DNS_datafile)
    Feature_data = np.load(Feature_datafile)
    print('DNS data:',np.shape(DNS_data))
    print('Feature data:',np.shape(Feature_data))

    # x_new = 0.5*x_0 +0.5     [-1, 1]
    # x_0 = 2 * (x_new - 0.5)

    # x_new_new = 1.0*x_0 +0.5 [-0.5, 0.5]
    # x_0 = 1.0 * (x_new - 0.5)

    # x_new_new = 2.0*x_0 +0.5 [-0.25, 0.25]
    # x_0 = 0.5 * (x_new - 0.5)

    # x_new_new = 5*x_0 +0.5  [-0.1, 0.1]
    # x_0 = 0.2 * (x_new - 0.5)

    # x_new_new = 10*x_0 +0.5  [-0.05, 0.05]
    # x_0 = 0.1 * (x_new - 0.5)

    coef = 10 #5.0 #5.0 #1.0 #0.5


    new_dns_data = []
    for i0 in range(0, np.shape(DNS_data)[0]):
        print('processing ... ', i0, ' total: ', np.shape(DNS_data)[0])
        one_dns_data = DNS_data[i0:i0+1, :,:,:]
        one_margin = np.ma.masked_where(one_dns_data >= 0.0, one_dns_data) # keep the margin
        one_margin = np.ma.filled(one_margin, 0.0)
        x_new = np.ma.masked_where(one_dns_data < 0.0, one_dns_data) # remove the margin
        # print('dns_data:', one_dns_data)
        # print('margin:', one_margin)
        # print('solution:', x_new)
        x_0 = 2.0 * (x_new - 0.5)
        # print('solution_0:', x_0)
        x_new_new = coef * x_0 + 0.5
        x_new_new = np.ma.filled(x_new_new, 0.0)
        # print('solution_new_new:', x_new_new)
        one_new_dns_data = x_new_new + one_margin  # have to fill the masked region with zeros, otherwise, it won't work.
        print('one_new_dns_data:', one_new_dns_data)
        new_dns_data.append(one_new_dns_data)
    new_labels = np.concatenate(new_dns_data, axis=0)
    print('new_labels', np.shape(new_labels))
    np.save('np-labels-rescaled.npy', new_labels)
    # print(new_dns_data)

    new_feature_data = []
    for i0 in range(0, np.shape(Feature_data)[0]):
        print('processing ... ', i0, ' total: ', np.shape(Feature_data)[0])
        one_feature_data = Feature_data[i0:i0+1, :,:,:]
        # print('feature_data:', one_feature_data)
        one_margin = np.ma.masked_where(one_feature_data > 1.0e-5, one_feature_data) # keep the margin
        # print('margin:', one_margin)
        one_margin = np.ma.filled(one_margin, 0.0)
        # print('margin:', one_margin)
        x_new = np.ma.masked_where(one_feature_data < 1.0e-5, one_feature_data) # remove the margin
        # print('solution:', x_new)
        x_0 = 2.0 * (x_new - 0.5)
        # print('solution_0:', x_0)
        x_new_new = coef * x_0 + 0.5
        x_new_new = np.ma.filled(x_new_new, 0.0)
        # print('solution_new_new:', x_new_new)
        one_new_feature_data = x_new_new + one_margin  # have to fill the masked region with zeros, otherwise, it won't work.
        print('one_new_feature_data:', one_new_feature_data)
        new_feature_data.append(one_new_feature_data)
    # print(new_feature_data)
    new_features = np.concatenate(new_feature_data, axis=0)
    print('new_features', np.shape(new_features))
    np.save('np-features-rescaled.npy', new_features)

