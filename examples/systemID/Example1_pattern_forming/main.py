###################################
####### Required libraries: #######
#######        FEniCS       #######
#######      scikit-learn   #######
#######         numpy       #######
###################################
#Plaese use "python main.py config_Example1.ini" to run the problem with the configuration file.#

import sys
import numpy as np
from mechanoChemML.workflows.systemID.systemID import systemID
#from systemID import systemID
np.set_printoptions(precision=3)

print('start')
if __name__ == "__main__":    
    print('=====COMMENTS: SystemID Example 1: Pattern formation-Schnakenberg_model =======')
    ####### Require FEniCS: #######
    #generate data 
    #from forward_model import *
    #Schnakenberg_model()
    
    #construct operators
    # choose data to use
    # data_list=np.arange(50,60)
    # from generate_basis import *
    # generate_basis(data_list=data_list)
    print('\n=====COMMENTS: In this example we constructed 13 operators in Eq. 19 and Eq.20 =======')

    ###################################

    used_time_step=[51]
    sigma=0
    data=np.zeros(0)
    for step in used_time_step:
        if np.size(data)<1:
            data=np.loadtxt('../datasets/Turing_system_basis/basis_sigma_'+str(sigma)+'_step_'+str(step)+'.dat')
        else:
            data=np.append(data,np.loadtxt('../datasets/Turing_system_basis/basis_sigma_'+str(sigma)+'_step_'+str(step)+'.dat'),0)   
    #read data to be identified
    
    # #
    # # #################
    print('\n=====COMMENTS: SystemID by stepwise regression by specified_target =======')
    print('\n=====COMMENTS:In default setting:\n the target is dC_1/dt, and the correct solution is dC_1/dt=0.05nabla^2C_1+0.1-1C_1+1C_1^2C_2=======')
    print('\n=====COMMENTS: \nabla^2C_1, constant(1), C_1 and C_1^2C_2 are the 1th, third, forth and sixth operators in the lists=======')
    print('\n=====COMMENTS:"next the System identification results will summarize all prefactors for all operators.')
    problem = systemID()
    problem.identifying(data)
    print('System identification results:')
    prefactor=problem.results['prefactor']
    print('Final result:',prefactor)
    print(' loss at each iterations',problem.results['model'].loss )
    print('\n=====COMMENTS:We correctly identify all operators and eliminate all non-active operators (with prefactor "0")=======')

    # #
    # #
    # #################
    print('\n=====COMMENTS: SystemID by stepwise regression by confirmation_of_consistency =======')
    problem.confirmation_of_consistency(data)
    print('results of confirmation_of_consistency :\n',problem.results['prefactor'])
    cos_similiarity=np.triu(problem.results['cos_similiarity'])
    #
    index=np.where(np.abs(np.abs(cos_similiarity)-1+np.identity(cos_similiarity.shape[0]))<1.0e-5)
    print('consistent pairs :\n',list(zip(index[0], index[1])))
