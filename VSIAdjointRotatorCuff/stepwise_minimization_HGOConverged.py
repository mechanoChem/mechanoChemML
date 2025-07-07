import numpy as np
import scipy.optimize as sp
import os
import sys

def iter_cb(m):
  print ("results = ", m)
  
def stepwise_minimization_HGOConverged(obj_f, x0, args_dict,F_threshold=1.0e16, method_options={}, grad_f=None, save_to_file=None):
  ##

  callback=None
  bounds=None
  current_activate_index=args_dict['activate_basis_index']
  target_index=args_dict['target_vector_index']
  num_theta=args_dict['num_theta']
  num_base_orign=len(current_activate_index)
  meshName = args_dict['meshName']
  tendonStamp = args_dict['tendonStamp']
  combination = args_dict['combination']
  lossFactor = args_dict['lossFactor']
  
  frozen_index=[]
  max_eliminate_step=num_base_orign-1
  if 'frozen_index' in args_dict.keys():
    frozen_index=args_dict['frozen_index']
  if 'max_eliminate_step' in args_dict.keys():
    max_eliminate_step=args_dict['max_eliminate_step']
  if 'method' in args_dict.keys():
    method=args_dict['method']
  if 'bounds' in args_dict.keys():
    bounds=args_dict['bounds']
    
  if 'disp' in method_options.keys():
    if method_options['disp']==True:
      callback=iter_cb
  gamma_matrix=np.empty((num_theta,max_eliminate_step+1))
  gamma_matrix[:]=np.NaN
  gamma_matrix[target_index,:]=1
  
  if save_to_file!=None:
    f=open(save_to_file,'ab')
  
  loss=np.zeros(max_eliminate_step+1)  
  res=sp.minimize(obj_f, x0,jac=grad_f, method=method, 
                  args=(tendonStamp, meshName, lossFactor, combination,
                        current_activate_index,target_index),
                        bounds=bounds, options=method_options,callback=callback )
  x0=res.x
  gamma_matrix[current_activate_index,0]=x0
  loss[0]=res.fun
  print('==============================================================')
  print('step=',0, ' current_activate_index=',current_activate_index,
        ' x0=',gamma_matrix[:,0],' loss=',loss[0] )
  if save_to_file!=None:
    info=np.reshape(np.append(gamma_matrix[:,0],(loss[0])),(1,-1))
    np.savetxt(f,info)
    f.flush()
    os.fsync(f.fileno())

  for step in range(len(current_activate_index)-1):
    if step==max_eliminate_step:
      break
    num_activate_index=len(current_activate_index)
    gamma_matrix_tem=np.zeros((num_activate_index-1,num_activate_index))
    loss_tem=np.ones(num_activate_index)*1.0e20 # why*1e20?
    for j in range(len(current_activate_index)):
      try_index=current_activate_index[j]
      # continue if j is in the frozen_index
      if try_index in frozen_index:
        continue
        
      current_activate_index_tem=np.delete(current_activate_index,j)
      x0_tem=np.delete(x0,j)
      bounds_tem=None
      if 'bounds' in args_dict.keys():
        bounds_tem=np.delete(bounds,j,0)
      res=sp.minimize(obj_f, x0_tem,jac=grad_f,  method=method, 
                      args=(tendonStamp, meshName, lossFactor, combination,
                            current_activate_index_tem,target_index),
                      bounds=bounds_tem, 
                      options=method_options,callback=callback)
      gamma_matrix_tem[:,j]=res.x
      loss_tem[j]=res.fun
    
    drop_index=np.argmin(loss_tem) 
    print('loss_try=',loss_tem)
    loss_try=loss_tem[drop_index]  
    F=(loss_try-loss[step])/loss[step]*(num_base_orign-num_activate_index+1)
    
    if F<F_threshold:
      current_activate_index=np.delete(current_activate_index,drop_index)
      x0=np.delete(x0,drop_index)
      if 'bounds' in args_dict.keys():
        bounds=np.delete(bounds,drop_index,0)
      gamma_matrix[current_activate_index,step+1]=gamma_matrix_tem[:,drop_index]
      loss[step+1]=loss_try
      
    else:
      break
    if save_to_file!=None:
      info=np.reshape(np.append(gamma_matrix[:,step+1],(loss[step+1])),(1,-1))
      np.savetxt(f,info)
      f.flush()
      os.fsync(f.fileno())
    print('==============================================================')
    print('step=',step+1, ' current_activate_index=',current_activate_index,' x0=',gamma_matrix[:,step+1],' loss=',loss[step+1] )
    
  return gamma_matrix, loss