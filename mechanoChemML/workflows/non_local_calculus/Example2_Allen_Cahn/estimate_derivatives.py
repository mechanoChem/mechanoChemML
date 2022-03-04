#!/usr/bin/env python
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


	cwd = '.'
	samples = 'dns/data/Sample%i'			
	file = 'data_c.csv'
	folder = 'ProcessDump'
	samples_ID = [i for i in range(0,100)]
	samples_ID = [0,]
	cwd = os.path.abspath(os.path.expanduser(cwd))
	directories_load = [samples%(i) for i in samples_ID]
	directories_dump = [os.path.join(samples%(i),folder) for i in samples_ID]

##############################################################
# Choose model parameters

	model_p = 1
	model_order = 2
	
##############################################################
#Temporary variable for calculation of derivative 
	_derivative_list = ['Phi_1P','Phi_2P','Phi_3P','Phi_4P','Phi_5P',
				'Phi_1M','Phi_2M','Phi_3M','Phi_4M','Phi_5M',
				'GradE_P','LapC_P',
				'LanE_P','dLan_P',	
				'GradE_M','LapC_M',
				'LanE_M','dLan_M',]
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
		#'algebraic_operations': [[
		#	{'func':lambda df: df['partial__1__TE__Phi_1P__stencil'] + df['partial__1__Phi_1P__Time__stencil'], 'labels':'DiffVal'} , 
		#	]],
		
		'algebraic_operations': [[
			*[{'func':lambda df: df['Phi_%iP'%i] + df['Phi_%iM'%i], 'labels':'Phi_%i'%i}
			for i in range(6)],
			]], 	

		#Graph structure settings
		'differential_operations':[[

			{'function':'Phi_1P',
			'variable': ['Time'],
			'weight':['stencil'], 
			'adjacency':['backward_nearest'], #(symmetric finite difference like scheme),
			'manifold':[['Time']], #(or could specify 't' if the vertices are strictly defined by time and adjacency is defined by that, and not per say euclidean distance in 'u' space)
			'accuracy': [1],						
			'dimension':[0],				
			'order':1,
			'operation':['partial'],
			},
			*[{'function':'TE',
			'variable': [x],
			'weight':['stencil'], 
			'adjacency':['nearest'], #(symmetric finite difference like scheme),
			'manifold':[[x]], #(or could specify 't' if the vertices are strictly defined by time and adjacency is defined by that, and not per say euclidean distance in 'u' space)
			'accuracy': [2],		
			'dimension':[0],
			'order':1,
			'operation':['partial'],
			} for x in _derivative_list]					
			]]}
##############################################################
# Call main function
	
	main(settings = settings)
		