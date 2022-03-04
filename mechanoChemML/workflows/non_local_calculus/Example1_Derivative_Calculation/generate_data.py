###################################
####### Required libraries: #######
#######      scikit-learn   #######
#######         numpy       #######
###################################

import sys, os
import numpy as np
import pandas as pd 

np.set_printoptions(precision=5)


if __name__ == "__main__":    
	
	
	# Choose directories and data files
	out_dir = './data/'
	out_file = 'func_val.csv'
	
	NumPoints = 1000
	#Randomly sample [x_1, x_2, x_3]\in [0,1]^3 
	x_1 = np.random.uniform(0,1,[NumPoints,1])
	x_2 = np.random.uniform(0,1,[NumPoints,1])
	x_3 = np.random.uniform(0,1,[NumPoints,1])
	
	#Define some functions
	u_1 = x_1**2 + x_2**2 + x_3**2 
	u_2 = (x_1**2 + x_2**2)*np.sin(10*x_1)
	
	df = pd.DataFrame(np.hstack((x_1, x_2, x_3, u_1, u_2)), columns = ['x_1','x_2','x_3','u_1','u_2'])
	
	
	if not os.path.exists(out_dir):
	  os.makedirs(out_dir)
	  print('New directory created: %s'%out_dir)
	df.to_csv(out_dir + out_file)