###################################
####### Required libraries: #######
#######        FEniCS       #######
#######      scikit-learn   #######
#######         numpy       #######
###################################

import sys
import numpy as np
from mechanoChemML.workflows.systemID.systemID import systemID
np.set_printoptions(precision=3)


if __name__ == "__main__":    
    print('======= SystemID Example 2: Constitutive modeling of soft materials =======')
    print('\n=====COMMENTS: This is the "simple version" of Exmaple in our paper: =======')
    print(' Z. Wang, J.B. Estrada, E.M. Arruda, K. Garikipati, Inference of deformation mechanisms and constitutive response of soft material surrogates of biological tissue by full-field characterization and data-driven variational system identification, Journal of the Mechanics and Physics of Solids, Volume 153, 2021. =======')

    #generate data 
    #from forward_model import *
    #threeField_neo_Hookean()
    
    #construct operators
    # from generate_basis import *
    # data_list=['extension','extension_2','bending','torsion']
    # generate_basis(data_list=data_list)
    data_list=['bending']
    data=np.zeros(0)
    for shape in data_list:
      data=np.loadtxt('../datasets/soft_materials/'+shape+'.dat')
      # #
      # # #################
      # # print('\n======= SystemID by stepwise regression by specified_target =======')
      problem = systemID()
      problem.identifying(data)
      print('System identification results:')
      prefactor=-problem.results['prefactor']
      print('Final result:',prefactor)
      print(' loss at each iterations',problem.results['model'].loss )
