"""
Zhenlin Wang 2019
"""

import numpy as np
from sklearn import linear_model
from sklearn.metrics.pairwise import cosine_similarity

import h5py as h5
from mechanoChemML.src.stepwiseRegression import stepwiseRegression as ST

def getlist_str(option, sep=',', chars=None):
    """Return a list from a ConfigParser option. By default, 
     split on a comma and strip whitespaces."""
    list0 = [(chunk.strip(chars)) for chunk in option.split(sep)]
    list0 = [x for x in list0 if x]
    return list0


def getlist_int(option, sep=',', chars=None):
    """Return a list from a ConfigParser option. By default, 
     split on a comma and strip whitespaces."""
    list0 = option.split(sep)
    list0 = [x for x in list0 if x]
    if (len(list0)) > 0:
        return [int(chunk.strip(chars)) for chunk in list0]
    else:
        return []


def getlist_float(option, sep=',', chars=None):
    """Return a list from a ConfigParser option. By default, 
     split on a comma and strip whitespaces."""
    list0 = option.split(sep)
    list0 = [x for x in list0 if x]
    if (len(list0)) > 0:
        return [float(chunk.strip(chars)) for chunk in list0]
    else:
        return []

class systemID:
    """
  Class of system ID 
  """
    def __init__(self):
        self.parse_sys_args()
        self.read_config_file()
        self.results={
          'model':[0],
          'prefactor':np.zeros(0),
          'loss':np.zeros(0),
          'cos_similiarity':np.zeros(0),
          'sum_cos_similiarity':np.zeros(0)
        }

    def debugger(self):
        import logging
        logger = logging.getLogger('root')
        FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
        logging.basicConfig(format=FORMAT)
        logger.setLevel(logging.DEBUG)
        self.logger = logger

    def read_config_file(self):
        """ """
        from configparser import ConfigParser, ExtendedInterpolation
        config = ConfigParser(interpolation=ExtendedInterpolation())
        config.read(self.args.configfile)
        config['TEST']['FileName'] = self.args.configfile
        self.config = config

    def parse_sys_args(self):
        import argparse, sys, os
        parser = argparse.ArgumentParser(description='Run Variational System Identification', prog="'" + (sys.argv[0]) + "'")
        parser.add_argument('configfile', type=str, help='configuration file')

        args = parser.parse_args()
        self.args = args
    
    def setup_model(self):        
        F_criteria=getlist_float(self.config['StepwiseRegression']['F_criteria'])
        F_switch=[]
        if len(F_criteria)>1:
          F_switch=getlist_int(self.config['StepwiseRegression']['F_switch'] )
        basis_drop_strategy=self.config['StepwiseRegression']['basis_drop_strategy']
        regression_method=self.config['StepwiseRegression']['regression_method']
        anchor_index=getlist_int(self.config['StepwiseRegression']['anchor_index'] )
        if len(anchor_index)==0 :
          anchor_index=[-1]
        alpha_lasso=0
        alpha_ridge=0
        ridge_cv=[-1]
        n_jobs=1
        if regression_method=='ridge':
          alpha_ridge=self.config['StepwiseRegression'].getfloat('alpha_ridge')
        elif regression_method=='lasso':
          alpha_lasso=self.config['StepwiseRegression'].getfloat('alpha_lasso')
        elif regression_method=='ridge_cv':
          ridge_cv=getlist_float(self.config['StepwiseRegression']['ridge_cv'])
        elif regression_method!='linear_regression':
          print('bad regression_method')
          exit()
          
        
        
        return ST.stepwiseR(F_criteria=F_criteria,F_switch=F_switch,basis_drop_strategy=basis_drop_strategy,anchor_index=anchor_index,alpha_lasso=alpha_lasso,alpha_ridge=alpha_ridge, ridge_cv=ridge_cv,n_jobs=n_jobs)

    def identifying(self,basis):
        print('\n-------------- Identifying --------------  ')
        strategy=self.config['VSI']['identify_strategy']
        if strategy=='specified_target':
          target_index=self.config['VSI'].getint('target_index')
          self.results['model']=self.stepwise_regression(basis,target_index)
          self.results['prefactor']=self.results['model'].gamma_matrix[:,-1]
          self.results['loss']=self.results['model'].loss[-1]
        elif strategy=='confirmation_of_consistency':
          self.confirmation_of_consistency(basis)
    
    def stepwise_regression(self,basis,target_index):
      model=self.setup_model()
      y=basis[:,target_index]
      _,n_base_orign=basis.shape
      basis_index=np.delete(np.arange(n_base_orign),target_index)
      theta_matrix=basis[:,basis_index]
      model.stepwiseR_fit(theta_matrix,y)
      return model
      
    def confirmation_of_consistency(self,basis):
      print('\n-------------- running confirmation_of_consistency... --------------  ')
      _,n_base_orign=basis.shape
      
      self.results['model']=[self.stepwise_regression(basis,key) for key in range(n_base_orign)]
      self.results['prefactor']=np.zeros((n_base_orign,n_base_orign))
      self.loss=np.zeros(n_base_orign)
      for i in range(n_base_orign):
        basis_id_theta=np.delete(np.arange(n_base_orign),i)
        self.results['prefactor'][basis_id_theta,i]=self.results['model'][i].gamma_matrix[:,-1]
        self.results['prefactor'][i,i]=-1
        self.loss[i]=self.results['model'][i].loss[-1]
        
      self.results['cos_similiarity']=cosine_similarity(np.transpose(self.results['prefactor']))
      self.results['sum_cos_similiarity']=np.sum(np.abs(self.results['cos_similiarity']),0)

  
