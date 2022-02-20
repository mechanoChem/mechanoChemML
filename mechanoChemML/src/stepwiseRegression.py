"""
Zhenlin Wang 2019
"""

import numpy as np
from mechanoChemML.src import LeastR as LR 
#import LeastR as LR 


class stepwiseR(object):
  def __init__(self,F_criteria=[1],F_switch=[],basis_drop_strategy='aggressive',sigma_n=1.0e-20,anchor_index=[-1],alpha_lasso=0,alpha_ridge=0, ridge_cv=[-1],threshold_d=1.0e-14,n_jobs=1):
    self.F_criteria=F_criteria
    self.F_switch=F_switch
    self.sigma_n=sigma_n
    self.anchor_index=anchor_index
    self.alpha_lasso=alpha_lasso
    self.alpha_ridge=alpha_ridge
    self.ridge_cv=ridge_cv
    self.n_jobs=n_jobs
    self.threshold_d=threshold_d
    self.basis_drop_strategy=basis_drop_strategy
    self.last_F=0
    
  def test(self):
    print('test_pass')
    
  def stepwiseR_fit_aggressive(self, theta_matrix, X_matrix):
    _,n_base_orign=theta_matrix.shape
    self.anchor=np.zeros(n_base_orign)
    if self.anchor_index[0]!=-1:
      for key in self.anchor_index:
        self.anchor[key]=1

    self.loss=np.zeros(n_base_orign)
    self.score=np.zeros(n_base_orign)
    self.F_index=np.zeros(n_base_orign)
    self.gamma_matrix=np.zeros((n_base_orign,n_base_orign))
    alpha_sum=self.alpha_lasso+self.alpha_ridge+self.ridge_cv[0]
    threshold_d=self.threshold_d
    self.best_alpha=np.zeros(n_base_orign)
    
    # local_to_global_index
    local_to_global_index=np.arange(n_base_orign)
    F_threshold=self.F_criteria[0]
    
    #########
    #first LS_regression
    #########
    num_column=0
    if(alpha_sum==-1):
        [gamma_vector,self.loss[0]]=LR.fit(theta_matrix,X_matrix)
    if(self.alpha_lasso>0):
        [gamma_vector,self.loss[0]]=LR.fit_lasso(theta_matrix,X_matrix, alpha=self.alpha_lasso)
    if(self.alpha_ridge>0):
        [gamma_vector,self.loss[0],self.score[0]]=LR.fit_ridge(theta_matrix,X_matrix, alpha=self.alpha_ridge)
    if(self.ridge_cv[0]>-0.1):
        [gamma_vector,self.loss[0],self.score[0],self.best_alpha[0]]=LR.fit_ridge_cv(theta_matrix,X_matrix, alpha=self.ridge_cv)
        
    self.gamma_matrix[local_to_global_index,num_column]=gamma_vector
    
    #########
    #stepwise
    #########
    num_column=num_column+1;
    num_canditate_basis=n_base_orign;
    frozen_index=[]
    while num_canditate_basis>1 :
      #get current F_criteria
      for i in range(len(self.F_switch)):
        if num_column>self.F_switch[i]:
          F_threshold=self.F_criteria[i+1]
        else:
          break
          
      find_flag=False 
      # put anchor index into frozen_index
      for i in range(local_to_global_index.size) :
        if self.anchor[local_to_global_index[i]]==1 :
          frozen_index.append(i)
          
      # begin to do basis reduction
      for j in range(gamma_vector.size):
        # continue if j is in the frozen_index
        if j in frozen_index:
          continue
        # calculate the min of gamma_vector except the frozen_index
        gamma_vector_min=gamma_vector;
        gamma_vector_min=np.delete(gamma_vector_min, frozen_index)
        gamma_criteria=min(abs(gamma_vector_min) )+threshold_d;
        theta_matrix_try=theta_matrix;
        
        #tentative delete the basis
        if abs(gamma_vector[j])<gamma_criteria :
          frozen_index.append(j)
          find_flag=True
          # delete the corresponding column
          theta_matrix_try=np.delete(theta_matrix_try,j,1)
            
            
          if(alpha_sum==-1):
            [gamma_vector_try,loss_try]=LR.fit(theta_matrix_try,X_matrix)
          if(self.alpha_lasso>0):
            [gamma_vector_try,loss_try]=LR.fit_lasso(theta_matrix_try,X_matrix, alpha=self.alpha_lasso)
          if(self.alpha_ridge>0):
            [gamma_vector_try,loss_try,score_tem]=LR.fit_ridge(theta_matrix_try,X_matrix, alpha=self.alpha_ridge)
          if(self.ridge_cv[0]>-0.1):
              [gamma_vector_try,loss_try,score_tem,best_alpha_tem]=LR.fit_ridge_cv(theta_matrix_try,X_matrix, alpha=self.ridge_cv)
                
          F=(loss_try-self.loss[num_column-1])/self.loss[num_column-1]*(n_base_orign-local_to_global_index.size+1)   
          if(F>self.last_F):
            self.last_F=F
                   
          # do F_test
          if F<F_threshold or loss_try<self.sigma_n:
            theta_matrix=np.delete(theta_matrix,j,1)
            local_to_global_index=np.delete(local_to_global_index,j)
            self.F_index[num_column]=F
            if(len(self.ridge_cv)>2):
                self.best_alpha[num_column]=best_alpha_tem
            self.loss[num_column]=loss_try
            gamma_vector=gamma_vector_try
            self.gamma_matrix[local_to_global_index,num_column]=gamma_vector
            num_column=num_column+1
            num_canditate_basis=num_canditate_basis-1
            frozen_index=[]
            
        # break tentative deleting basis
        if find_flag==True:
          break
      #stop the algorithm   
      if find_flag==0 or gamma_vector_min.size<1:
        break
        
    self.gamma_matrix=np.delete(self.gamma_matrix,np.arange(num_column,n_base_orign), axis=1)
    self.loss=np.delete(self.loss,np.arange(num_column,n_base_orign))   
    self.F_index=np.delete(self.F_index,np.arange(num_column,n_base_orign))   
    self.best_alpha=np.delete(self.best_alpha,np.arange(num_column,n_base_orign))  
                    
  
  def stepwiseR_fit_most_insignificant(self, theta_matrix, X_matrix):
    _,n_base_orign=theta_matrix.shape
    self.anchor=np.zeros(n_base_orign)
    if self.anchor_index[0]!=-1:
      for key in self.anchor_index:
        self.anchor[key]=1

    self.loss=np.zeros(n_base_orign)
    self.score=np.zeros(n_base_orign)
    self.F_index=np.zeros(n_base_orign)
    self.gamma_matrix=np.zeros((n_base_orign,n_base_orign))
    alpha_sum=self.alpha_lasso+self.alpha_ridge+self.ridge_cv[0]
    threshold_d=self.threshold_d
    self.best_alpha=np.zeros(n_base_orign)
    
    # local_to_global_index
    local_to_global_index=np.arange(n_base_orign)
    F_threshold=self.F_criteria[0]
    
    #########
    #first LS_regression
    #########
    num_column=0
    if(alpha_sum==-1):
        [gamma_vector,self.loss[0]]=LR.fit(theta_matrix,X_matrix)
    if(self.alpha_lasso>0):
        [gamma_vector,self.loss[0]]=LR.fit_lasso(theta_matrix,X_matrix, alpha=self.alpha_lasso)
    if(self.alpha_ridge>0):
        [gamma_vector,self.loss[0],self.score[0]]=LR.fit_ridge(theta_matrix,X_matrix, alpha=self.alpha_ridge)
    if(self.ridge_cv[0]>-0.1):
        [gamma_vector,self.loss[0],self.score[0],self.best_alpha[0]]=LR.fit_ridge_cv(theta_matrix,X_matrix, alpha=self.ridge_cv)
        
    self.gamma_matrix[local_to_global_index,num_column]=gamma_vector
    
    #########
    #stepwise
    #########
    num_column=num_column+1;
    num_canditate_basis=n_base_orign;
    frozen_index=[]
    while num_canditate_basis>1 :
      #get current F_criteria
      for i in range(len(self.F_switch)):
        if num_column>self.F_switch[i]:
          F_threshold=self.F_criteria[i+1]
        else:
          break
          
      find_flag=False 
      # put anchor index into frozen_index
      for i in range(local_to_global_index.size) :
        if self.anchor[local_to_global_index[i]]==1 :
          frozen_index.append(i)
          
      # begin to do basis reduction
      loss_tem=np.ones(gamma_vector.size)*1.0e10
      best_alpha_tem=np.zeros(gamma_vector.size)
      score_tem=np.zeros(gamma_vector.size)
      gamma_matrix_try=np.zeros((gamma_vector.size-1,gamma_vector.size))
      for j in range(gamma_vector.size):
        # continue if j is in the frozen_index
        if j in frozen_index:
          continue
        
        theta_matrix_try=np.delete(theta_matrix,j,1)
        if(alpha_sum==-1):
          [gamma_matrix_try[:,j],loss_tem[j] ]=LR.fit(theta_matrix_try,X_matrix)
        if(self.alpha_lasso>0):
          [gamma_vector_try[:,j],loss_tem[j] ]=LR.fit_lasso(theta_matrix_try,X_matrix, alpha=self.alpha_lasso)
        if(self.alpha_ridge>0):
          [gamma_matrix_try[:,j],loss_tem[j] ,score_tem[j] ]=LR.fit_ridge(theta_matrix_try,X_matrix, alpha=self.alpha_ridge)
        if(self.ridge_cv[0]>-0.1):
          [gamma_matrix_try[:,j],loss_tem[j] ,score_tem[j],best_alpha_tem[j] ]=LR.fit_ridge_cv(theta_matrix_try,X_matrix, alpha=self.ridge_cv)
        
      drop_index=np.argmin(loss_tem)  
      loss_try=loss_tem[drop_index]  
      F=(loss_try-self.loss[num_column-1])/self.loss[num_column-1]*(n_base_orign-local_to_global_index.size+1) 
      if(F>self.last_F):
        self.last_F=F
      # do F_test
      if F<F_threshold or loss_try<self.sigma_n:
        find_flag=True
        theta_matrix=np.delete(theta_matrix,drop_index,1)
        local_to_global_index=np.delete(local_to_global_index,drop_index)
        self.F_index[num_column]=F
        if(len(self.ridge_cv)>2):
          self.best_alpha[num_column]=best_alpha_tem[drop_index]
        self.loss[num_column]=loss_try
        gamma_vector=gamma_matrix_try[:,drop_index]
        self.gamma_matrix[local_to_global_index,num_column]=gamma_vector
        num_column=num_column+1
        num_canditate_basis=num_canditate_basis-1
        frozen_index=[]
            
      #stop the algorithm of no operator can be eliminated or only one operator left.
      if find_flag==False:
        break
        
    self.gamma_matrix=np.delete(self.gamma_matrix,np.arange(num_column,n_base_orign), axis=1)
    self.loss=np.delete(self.loss,np.arange(num_column,n_base_orign))   
    self.F_index=np.delete(self.F_index,np.arange(num_column,n_base_orign))   
    self.best_alpha=np.delete(self.best_alpha,np.arange(num_column,n_base_orign))  
      
    
  ###################################################            
  def stepwiseR_fit(self, theta_matrix, X_matrix):
    if self.basis_drop_strategy=='aggressive':
      self.stepwiseR_fit_aggressive(theta_matrix, X_matrix)
    elif self.basis_drop_strategy=='most_insignificant':
      self.stepwiseR_fit_most_insignificant(theta_matrix, X_matrix)
    else:
      print('basis_drop_strategy is not well defined, using most_insignificant instead')
      self.stepwiseR_fit_most_insignificant(theta_matrix, X_matrix)


    

    
    
      
    
    