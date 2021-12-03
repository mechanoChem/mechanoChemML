"""
Zhenlin Wang 2019
"""


from sklearn import linear_model
from sklearn.linear_model import RidgeCV
import numpy as np

def fit(theta_matrix,X_matrix, n_jobs=1):
    reg = linear_model.LinearRegression(fit_intercept=False,n_jobs=n_jobs)
    reg.fit(theta_matrix,X_matrix)
    gamma_vector=reg.coef_
    loss=np.mean(np.square(reg.predict(theta_matrix)-X_matrix))
    
    return gamma_vector, loss

def fit_lasso(theta_matrix,X_matrix,alpha=0):
    reg =linear_model.Lasso(alpha=alpha, fit_intercept=False)
    reg.fit(theta_matrix,X_matrix)
    gamma_vector=reg.coef_
    loss=np.mean(np.square(reg.predict(theta_matrix)-X_matrix))
    
    return gamma_vector, loss

def fit_ridge(theta_matrix,X_matrix,alpha=0):
    reg =linear_model.Ridge(alpha=alpha, fit_intercept=False)
    reg.fit(theta_matrix,X_matrix)
    gamma_vector=reg.coef_
    loss=np.mean(np.square(reg.predict(theta_matrix)-X_matrix))
    score=reg.score(theta_matrix,X_matrix)
    return gamma_vector, loss, score
    
def fit_ridge_cv(theta_matrix,X_matrix,alpha=[0]):
    reg =RidgeCV(alphas=alpha, fit_intercept=False)
    reg.fit(theta_matrix,X_matrix)
    gamma_vector=reg.coef_
    loss=np.mean(np.square(reg.predict(theta_matrix)-X_matrix))   
    score=reg.score(theta_matrix,X_matrix)
    return gamma_vector, loss, score, reg.alpha_
    
def multiple_fit_ridge(theta_matrix_list,X_matrix_list,map_index_list,x0,alpha=0,method='powell', options={}):
  num_eq=len(map_index_list)
  
  def loss_(x):
    fun=0
    for i in range(num_eq):
      fun+=np.mean((np.matmul(theta_matrix_list[i],x[map_index_list[i]])-X_matrix_list[i])**2)
    fun+=alpha*np.mean(x**2)
    return fun
    
  res = minimize(loss_, x0, method='powell',options={'xtol': 1e-8, 'disp': True})
  gamma_vector=res.x
  loss=res.fun
  return gamma_vector, loss
  