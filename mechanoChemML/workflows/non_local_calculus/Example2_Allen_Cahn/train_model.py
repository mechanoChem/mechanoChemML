import sys, os
import pandas as pd 
#sys.path.append('../../../src/')
from mechanoChemML.src.systemID import *
np.set_printoptions(precision=5)

#Run as: python train_model config_allen_cahn.ini
if __name__ == "__main__":    
	
	print('======= NonLocal Calculus Example 2: Allen Cahn dynamics =======')
	
	# Choose directories and data files
	samples = './dns/data/Sample%i'			
	file = 'data.csv'
	folder = 'ProcessDump'
	samples_ID = [i for i in range(100)]
	directories_dump = [os.path.join(samples%(i),folder) for i in samples_ID]
	
	#Read data
	dataset = {'set%i'%i: pd.read_csv(directories_dump[i]+ '/'+ file) for i in range(len(samples_ID))}
	
	#Concatenate different datasets
	training_df = pd.concat([dataset['set%i'%i] for i in range(len(samples_ID))])
	
	#Residue data for dc/dt = \sum_i^N (coef_i *basis_i)
	###Extract important basis terms
	keys = [
		'B1',
		'B2',
		'B3',
		]
	for key in keys:
		lhs_basis = ['partial__1__Phi_1P__Time__stencil']
		_rhs_basis = [
				'Phi_1P','Phi_2P','Phi_3P','Phi_4P','Phi_5P',
				'Phi_1M','Phi_2M','Phi_3M','Phi_4M','Phi_5M',
				'GradE_P','LapC_P',
				'LanE_P','dLan_P',	
				'GradE_M','LapC_M',
				'LanE_M','dLan_M',					
				]
		rhs_basis = { 'B1': [*['partial__1__TE_P__'+_term+'__stencil' for _term in _rhs_basis],], 
			'B2': [*['partial__1__TE__'+_term+'__stencil' for _term in _rhs_basis], 
			*['partial__1__TE_P__'+_term+'__stencil' for _term in _rhs_basis]], 
			'B3': [*['partial__1__TE__'+_term+'__stencil' for _term in _rhs_basis], 
			*['partial__1__TE_P__'+_term+'__stencil' for _term in _rhs_basis],
			*[_term for _term in _rhs_basis]]
			}[key]
		training_data_lhs = training_df.filter(lhs_basis).to_numpy()
		training_data_rhs = training_df.filter(rhs_basis).to_numpy()
		#ttraining_data_lhs = training_data_rhs*gamma => arget_index = 0 in config file and, 
		data_mat = np.hstack((training_data_lhs,training_data_rhs)) 
		print('Num data points: %i'%data_mat.shape[0])
		print('Num basis: %i'%data_mat.shape[1])
		#Stepwise regression
		problem = systemID()
		problem.identifying(data_mat)

		#Output results
		print('System identification results:')
		prefactor=-problem.results['prefactor']

		active_basis = [i for i in range(len(rhs_basis)) if abs(prefactor[i])>1e-12]
		result_str = lhs_basis[0] + '=' + '+'.join([str(prefactor[i])+rhs_basis[i] for i in active_basis])
		print('Prefactors:',prefactor)
		print(result_str)
		gamma_matrix = problem.results['model'].gamma_matrix
		for i in range(gamma_matrix.shape[1]):
			print(' Gamma matrix at iterations %i: %s and loss:%f'%(i,gamma_matrix[:,i],problem.results['model'].loss[i]) )
		#Saving as csv
		gamma_matrix = gamma_matrix.transpose()
		_temp = np.abs(gamma_matrix) > 1e-14
		_temp = np.flip(np.argsort(_temp.sum(axis=0)))
		result_df = pd.DataFrame(gamma_matrix[:,_temp], columns =  [rhs_basis[i] for i in _temp])
		
		result_df.insert(0, 'loss', problem.results['model'].loss[:])
		result_df.insert(0, 'iter', np.arange(1,gamma_matrix.shape[0]+1))
		
		result_df.to_csv (r'./result/model_'+key+ '.csv', index = False, header=True)

