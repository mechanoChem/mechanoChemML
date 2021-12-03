###################################
####### Required libraries: #######
#######        FEniCS       #######
#######      scikit-learn   #######
#######         numpy       #######
###################################

import sys
from mechanoChemML.src.systemID import systemID
np.set_printoptions(precision=3)


if __name__ == "__main__":    
    print('======= SystemID Example 2: Constitutive modeling of soft materials =======')
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
      data=np.loadtxt('basis/'+shape+'.dat')
      # #
      # # #################
      # # print('\n======= SystemID by stepwise regression by specified_target =======')
      problem = systemID()
      problem.identifying(data)
      print('System identification results:')
      prefactor=-problem.results['prefactor']
      print('Final result:',prefactor)
      print(' loss at each iterations',problem.results['model'].loss )
