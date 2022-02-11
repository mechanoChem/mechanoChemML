import sys
from mechanoChemML.src.systemID import systemID
np.set_printoptions(precision=3)

if __name__ == "__main__":    
    print('======= SystemID demo: polynomial model =======')
    problem = systemID()
    print('\n======= Data info =======')
    data_dir=str(problem.config['VSI']['data_dir'])
    f=h5.File(data_dir,'r')
    data_list=list(f.keys())
    print('Data_list=',data_list)
    print('TRUE MODEL: ', f['true model'][()])
    x=f['x'][()]
    y=f['y'][()]

    #################
    theta_matrix=np.array([np.ones(10),x,x*x,x*x*x, x*x*x*x])
    theta_matrix=np.transpose(theta_matrix)

    #################
    print('\n======= SystemID by stepwise regression by specified_target =======')
    basis=np.append(theta_matrix, np.reshape(y,(-1,1)), 1)
    problem.identifying(basis)
    print('System identification results:\ngamma_matrix=\n',problem.results['model'].gamma_matrix,'\n\nloss=',problem.results['model'].loss)
    print('Final result:')
    theta_sr_mi=np.around(problem.results['prefactor'], decimals=2)
    print('y=',theta_sr_mi[0],'+',theta_sr_mi[1],'x','+',theta_sr_mi[2],'x^2','+',theta_sr_mi[3],'x^3','+',theta_sr_mi[4],'x^4')
    print('#The model is correctly identified!')
    
    
    #################
    print('\n======= SystemID by stepwise regression by confirmation_of_consistency =======')
    problem.confirmation_of_consistency(basis)
    print('results of confirmation_of_consistency :\n',problem.results['prefactor'])
    cos_similiarity=np.triu(problem.results['cos_similiarity'])  

    index=np.where(np.abs(np.abs(cos_similiarity)-1+np.identity(cos_similiarity.shape[0]))<1.0e-1)
    print('consistent pairs :\n',list(zip(index[0], index[1])))
