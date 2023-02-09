#!/usr/bin/env python3

import numpy as np
import os
from myfile import read_file
import sys

feature_list = sys.argv[1:]
print('feature_list:', feature_list)

for f0 in feature_list:
    feature_name = f0
    feature_data = np.load(feature_name)
    shapes = list(np.shape(feature_data))
    if len(shapes) > 1:
        print(feature_name,np.shape(feature_data), )
        for i0 in range(0, shapes[0]):
            new_feature_name = feature_name[0:-4] + '-' + str(i0) + '.npy'
            np.save(new_feature_name, feature_data[i0:i0+1,:,:,:])
            print(new_feature_name)

    # cmd = 'del ' + feature_name 
    # print(cmd)
    # os.system(cmd)
# cmd = 'del npy.list' 
# os.system(cmd)

