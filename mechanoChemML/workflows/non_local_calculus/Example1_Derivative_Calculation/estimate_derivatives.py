#!/usr/bin/env python
import os
os.environ['MPLCONFIGDIR'] = os.getcwd() + '/configs/'
#
# Import python modules
import sys,glob,copy,itertools,os
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt


# Import user modules

# Global Variables
DELIMETER='__'
MAX_PROCESSES = 1
PARALLEL = 0

from mechanoChemML.src.graph_main import main

if __name__ == '__main__':



##############################################################
# Choose directories and data files

	cwd = os.path.abspath(os.path.expanduser('.'))
	directories_load = ['data']
	directories_dump = ['result']
	file = 'func_val.csv'
##############################################################
# Choose model parameters

	model_p = 3
	model_order = 2

##############################################################
# Set settings
	


	settings = {
		#Set Paths
		'cwd': cwd,
		'directories_load':directories_load,
		'directories_dump':directories_dump,
		'data_filename':file,


			
		#Model settings
		'model_order':[model_order],
		'model_p':[model_p],			

		#Graph structure settings
		'algebraic_operations': [[
			{'func':lambda df: df['x_1'] + df['x_2'] + df['x_3'], 'labels':'u_3'} , 
			]],
		
		#Graph structure settings
		'differential_operations':[[
			*[{'function':'u_%i'%i,
			'variable': ['x_%i'%j],
			'weight':['stencil'], 
			'adjacency':['nearest'], #(symmetric finite difference like scheme),
			'manifold':[['x_1', 'x_2', 'x_3']], 
			'accuracy': [2],						
			'dimension':[j-1],	#Index of dimension taking partial derivative about 			
			'order':1,
			'operation':['partial'],
			} for i in range(1,4) for j in range(1,4)]
			]]
		}
##############################################################
# Call main function
	
	main(settings = settings)
		