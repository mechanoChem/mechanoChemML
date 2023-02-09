#!/usr/bin/env python3

# https://stackoverflow.com/questions/3761391/boostpython-python-list-to-stdvector
# https://wiki.python.org/moin/boost.python/extract
# https://www.boost.org/doc/libs/1_39_0/libs/python/test/vector_indexing_suite.cpp

import numpy as np
import sys
import ctypes
ctypes.CDLL("libmpi.so", mode=ctypes.RTLD_GLOBAL)
sys.path.append('/home/xiaoxuan/Desktop/P201910-ML-PDE/4_mechanochem-ml/dns_wrapper/mechanoChemFEM/linear-elasticity/build/')
sys.path.append('/home/xiaoxuan/Desktop/P201910-ML-PDE/4_mechanochem-ml/dns_wrapper/mechanoChemFEM/linear-elasticity/')

#print('location 1')
from PYmechanoChem import PYmechanoChem
#print('location 2')

import itertools
from parameter_study import ParametersDomain
from parameter_study import ParametersLoading


if (__name__ == '__main__'):
  args = sys.argv[:]
  print(args)
  
  problem = PYmechanoChem(args) # read args
  # re_load parameters

  problem.setup_mechanoChemFEM() # parameters.prm for 10 times steps.
  print("------------------------------------")
  #problem.simulate() # 11th
  #problem.simulate() # 12th
  # ./main parameters.prm
  # parameter2.prm
