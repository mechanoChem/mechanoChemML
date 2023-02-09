#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt 
import sys

f1 = sys.argv[1]
all_mat_data = np.load(f1)
print("all-mats.npy:", all_mat_data)
