#!/usr/bin/env python3
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt

from myfile import *

# f1 = 'larger-sigma2-init.log' 
# content = read_file(f1)
# content = [x.strip() for x in content if x.find('Epoch') >= 0]

# loss = []
# sigma2 = []
# for c1 in content:
    # c1_list = c1.split()
    # loss.append(float(c1_list[3][:-1]))

# plt.clf()
# # plt.semilogy(loss, 'b')
# plt.plot(loss)
# plt.show()

if __name__ == '__main__' :
    if len(sys.argv) == 1:
        print(sys.argv[0], " +.pickle files")
        exit(0)
    else:
        inputs = sys.argv[1:]
    for i0 in inputs:
        filename = i0.replace('.pickle', '-loss-others.png')
        saved_config = pickle.load(open(i0, "rb"))
        losses = saved_config ['losses']
        # print(losses['loss'],filename)
        plt.clf()

        plt.semilogy(losses['loss'], 'b')
        plt.semilogy(losses['val_loss'], 'r')
        plt.semilogy(losses['mse_loss'], 'g')
        # plt.semilogy(losses['pde_loss'], 'm')
        # plt.semilogy(losses['neu_loss'], 'y')
        # plt.semilogy(losses['elbo_loss'], 'c')
        plt.xlabel('epoch')
        # plt.legend(['loss', 'val_loss', 'mse', 'pde', 'nue', 'abs(elbo)'])

        # ratio = []
        # for i0 in range(0, len(losses['neu_loss'])):
            # ratio.append(losses['neu_loss'][i0]/losses['elbo_loss'][i0])
        # plt.plot(ratio)

        plt.savefig(filename)
        plt.show()
        print('save to:', filename)
        # exit(0)
