#!/usr/bin/env python


# Import python modules
import os,sys,copy,itertools,functools,inspect,timeit
import gc

import numpy as np
import scipy as sp
import scipy.stats,scipy.signal,scipy.cluster
import matplotlib as plt


from sklearn import linear_model,model_selection

# import multiprocess as mp
# import multithreading as mt
import joblib
import multiprocessing as multiprocessing
import multiprocessing.dummy as multithreading


# Logging
import logging
log = 'info'
logger = logging.getLogger(__name__)
#logger.setLevel(getattr(logging,log.upper()))


# Global Variables
DELIMITER='__'

# Import user modules
from .graph_utilities import set_loss,set_score,set_norm,set_scale,set_criteria
from .graph_utilities import Pool,nullPool,Parallelize,Parallel,nullParallel,nullcontext,nullfunc,wrapper
from .graph_utilities import delete,take,isin,rank,cond,gram,project
from .graph_utilities import call,shuffle,where,getattribute,setattribute
from .texify import scinotation
#from .plot import plot


class Base(object):
	__slots__ = ('_param_names')	
	def __init__(self, *args, **kwargs):
		self.set_params(*args,**kwargs)
		return

	def get_params(self,*args,deep=False,**kwargs):
		field = '_param_names'
		default = {}
		if not hasattr(self,field):
			getattr(self,'_set%s%s'%('_' if not field.startswith('_') else '',field))(default,*args,**kwargs)			
			self.set_params()
		return {k: getattr(self,k,None) for k in getattr(self,field)}	

	def set_params(self,*args,**kwargs):
		field = '_param_names'
		default = {}
		if not hasattr(self,field):
			getattr(self,'_set%s%s'%('_' if not field.startswith('_') else '',field))(default,*args,**kwargs)			
		
		params = getattr(self,field)
		params.update(kwargs)
		
		for param in params:
			value = params[param]			
			try:
				getattr(self,'set_%s'%param)(*args,**params)
			except (AttributeError,TypeError) as e:
				pass
			setattr(self,param,value)
			if param not in getattr(self,field):
				getattr(self,field)[param] = value
		return self	

	def _set_param_names(self,param_names,*args,**kwargs):
		field = '_param_names'
		param_names.update(kwargs)
		setattr(self,field,param_names)
		return

	def _get_param_names(self):
		field = '_param_names'
		default = {}
		if not hasattr(self,field):
			getattr(self,'_set%s%s'%('_' if not field.startswith('_') else '',field))(default,*args,**kwargs)						
		return getattr(self,field)	


class Estimator(Base):
	__slots__ = tuple(('%s%s'%(s,name) for name in 
			['loss_func','score_func','criteria_func',
			 'normalize','fit_intercept','parallel',
			 'stats']
			for s in ['','_']))	

	def __init__(self,*args,**kwargs):
		_kwargs = {
			'solver':'pinv',
			'loss_func':None,
			'score_func':None,
			'criteria_func':None,
			'normalize':None,
			'fit_intercept':False,
			'parallel':None,
			'n_jobs':1,
			'backend':'loky',
			'stats':None,
			'collect':True,
			'prioritize':False,
			'verbose':False,
		}
		kwargs.update({**_kwargs,**kwargs})
		call(super(),'__init__',*args,**kwargs)        


	def set_stats(self,stats,*args,**kwargs):
		field = '_stats'
		default = {}
		if hasattr(self,field):
			if stats is not None:
				getattr(self,field).update(stats)
			else:
				getattr(self,field).update(default)
		else:            
			if stats is not None:
				setattr(self,field,stats)
			else:      
				setattr(self,field,default)
				self.statistics(X=None,y=None,stats=getattr(self,field),index=None,append=True)  
		return


	def get_stats(self,*args,**kwargs):
		field = '_stats'
		default = None
		if not hasattr(self,field):
			getattr(self,'set%s%s'%('_' if not field.startswith('_') else '',field))(default,*args,**kwargs)
		return getattr(self,field)


	def statistics(self,X,y,stats,index,append,*args,**kwargs):

		def structure(obj):
			_structure = np.array
			if obj is None:
				return obj
			if not isinstance(obj,np.ndarray):
				obj = np.array(obj)
			if obj.ndim < 1:
				obj = obj.reshape(-1)
			return obj
		def join(key,obj,objs,index=None,append=True):
			obj = structure(obj)
			if key not in objs:
				objs[key] = None
			if index is not None:
				try:
					objs[key][index] = obj
				except:
					try:
						if obj.size > 1:
							objs[key] = obj.reshape((1,*obj.shape))
						else:
							raise Exception
					except:
						objs[key] = obj
			elif append:
				try:
					objs[key] = np.concatenate((objs[key],obj))
				except:
					try:
						if obj.size > 1:
							objs[key] = obj.reshape(1,*obj.shape)
						else:
							raise Exception
					except:
						objs[key] = obj
			elif obj is None:
				objs[key] = obj
			return
		fields = {'stats_modules': ['predict','loss','score','coef_','criteria'],
				  'stats_params':  ['index_','index','dim_','rank_','condition_','basis_','complexity_','coefficient'],
				  'stats_collect':['predict']}
		for field in fields:
			if not hasattr(self,field):
				setattr(self,field,fields[field])
			else:
				pass
		if stats is None:
			stats = self.get_stats()

		isXy = X is None and y is None
		for key in self.stats_modules:   
			try:
				value = kwargs.get(key,getattr(self,key))
			except:
				continue
			if key in kwargs:
				pass
			elif callable(value) and not isXy: 
				try:
					value = value(X,y,*args,**kwargs)
				except:
					value = None      
			else:
				value = None
			join(key,value,stats,index=index,append=append)

		for key in self.stats_params:             
			try:
				value = kwargs.get(key,self.get_stats().get(key)[index])
			except:
				value = None
			join(key,value,stats,index=index,append=append)

		return    

	def _set_estimator(self,estimator,*args,**kwargs):
		default = 'OLS'
		estimators = {
					 'CrossValidate':CrossValidate,
					 'Stepwise':Stepwise,
					 'OLS':OLS,
					 'Tikhonov':Tikhonov,
					 'Ridge':Ridge,
					 'Lasso':Lasso,
					 'Enet':Enet,
					 'TikhonovCV':TikhonovCV,					 
					 'RidgeCV':RidgeCV,
					 'LassoCV':LassoCV,
					 'EnetCV':EnetCV,
					}
		if isinstance(estimator,str):
			variables = {'args':args,'kwargs':kwargs}
			for variable in variables:
				_variable = 'estimator_%s'%(variable)
				if kwargs.get(_variable) is not None:
					if variable == 'args':
						variables[variable] = kwargs[_variable]
					elif variable == 'kwargs':
						variables[variable].update(kwargs[_variable])
			estimator = call(None,estimators.get(estimator,estimators[default]),*variables['args'],**variables['kwargs'])
		return estimator

	def set_estimator(self,estimator,*args,**kwargs):
		field = '_estimator'
		estimator = getattr(self,'_set%s%s'%('_' if not field.startswith('_') else '',field))(estimator,*args,**kwargs)
		setattr(self,field,estimator)
		return

	def get_estimator(self,*args,**kwargs):
		field = '_estimator'
		default = None
		if not hasattr(self,field):
			getattr(self,'set%s%s'%('_' if not field.startswith('_') else '',field))(default,*args,**kwargs)
		return getattr(self,field)

	def set_loss_func(self,loss_func,*args,**kwargs):
		field = '_loss_func'
		if callable(loss_func):
			setattr(self,field,loss_func)
		else:
			if isinstance(loss_func,str):
				kwds = {**kwargs,'loss_func':loss_func,'axis':0}
			else:
				kwds = {**kwargs,'loss_func':'rmse','axis':0}
			setattr(self,field,set_loss(**kwds))
		return

	def get_loss_func(self,*args,**kwargs):
		field = '_loss_func'
		default = None
		if not hasattr(self,field):
			getattr(self,'set%s%s'%('_' if not field.startswith('_') else '',field))(default,*args,**kwargs)
		return getattr(self,field) 

	def set_score_func(self,score_func,*args,**kwargs):
		field = '_score_func'
		if callable(score_func):
			setattr(self,field,score_func)            
		else:
			if isinstance(score_func,str):
				kwds = {**kwargs,'score_func':score_func,'axis':0}
			else:
				kwds = {**kwargs,'score_func':'rmse','axis':0}
			setattr(self,field,set_score(**kwds))
		return

	def get_score_func(self,*args,**kwargs):
		field = '_score_func'
		default = None
		if not hasattr(self,field):
			getattr(self,'set%s%s'%('_' if not field.startswith('_') else '',field))(default,*args,**kwargs)
		return getattr(self,field)     


	def set_criteria_func(self,criteria_func,*args,**kwargs):
		field = '_criteria_func'
		if callable(criteria_func):
			setattr(self,field,criteria_func)
		else:
			if isinstance(criteria_func,str):
				kwds = {**kwargs,'criteria_func':criteria_func}
			else:
				kwds = {**kwargs,'criteria_func':'F_test'}
			setattr(self,field,set_criteria(**kwds))
		return

	def get_criteria_func(self,*args,**kwargs):
		field = '_criteria_func'
		default = None
		if not hasattr(self,field):
			getattr(self,'set%s%s'%('_' if not field.startswith('_') else '',field))(default,*args,**kwargs)
		return getattr(self,field) 


	def set_normalize(self,normalize,*args,**kwargs):
		field = '_normalize'
		if callable(normalize):
			setattr(self,field,normalize)
		else:
			if isinstance(normalize,str):
				kwds = {**kwargs,'norm_func':normalize,'axis':0}
			else:
				kwds = {**kwargs,'norm_func':'l2','axis':0}
			setattr(self,field,set_scale(**kwds))
		return

	def get_normalize(self,*args,**kwargs):
		field = '_normalize'
		default = None
		if not hasattr(self,field):
			getattr(self,'set%s%s'%('_' if not field.startswith('_') else '',field))(default,*args,**kwargs)
		return getattr(self,field) 


	def set_solver(self,solver,*args,**kwargs):
		field = '_solver'
		setattr(self,field,Solver(solver))
		return

	def get_solver(self,*args,**kwargs):
		field = '_solver'		
		return getattr(self,field)


	def set_coef_(self,coef_,*args,**kwargs):
		field = 'coef_'
		if coef_ is None and hasattr(self.get_estimator(),'get%s%s'%('_' if not field.startswith('_') else '',field)):
			coef_ = getattr(self.get_estimator(),'get%s%s'%('_' if not field.startswith('_') else '',field))()
		if coef_ is None and hasattr(self.get_estimator(),field):
			coef_ = getattr(self.get_estimator(),field)

		setattr(self,field,coef_)

		if self.get_estimator():
			if hasattr(self.get_estimator(),'set%s%s'%('_' if not field.startswith('_') else '',field)):
				getattr(self.get_estimator(),'set%s%s'%('_' if not field.startswith('_') else '',field))(coef_)
			else:
				setattr(self.get_estimator(),field,coef_)

		return


	def get_coef_(self,*args,**kwargs):
		field = 'coef_'
		default = None
		if not hasattr(self,field):
			getattr(self,'set%s%s'%('_' if not field.startswith('_') else '',field))(default,*args,**kwargs)            
		return getattr(self,field) 


	def set_intercept_(self,X,y,*args,**kwargs):
		field = 'intercept_'
		default = 0.0
		if not self.fit_intercept:
			setattr(self,field,default)
		else:
			setattr(self,field,y.mean(axis=0) - (X.mean(axis=0)).dot(self.get_coef_()))

	def get_intercept_(self,*args,**kwargs):
		field = 'intercept_'
		default = None
		if not hasattr(self,field):
			getattr(self,'set%s%s'%('_' if not field.startswith('_') else '',field))(default,*args,**kwargs)            
		return getattr(self,field) 

	def set_predict(self,predict,*args,**kwargs):
		field = '_predict'
		setattr(self,field,predict)
		return

	def get_predict(self,*args,**kwargs):
		field = '_predict'
		default = None
		if not hasattr(self,field):
			getattr(self,'set%s%s'%('_' if not field.startswith('_') else '',field))(default,*args,**kwargs)            
		return getattr(self,field) 	


	def loss(self,X,y,*args,**kwargs):
		if self.prioritize:
			kwargs.update(self.get_params())				
		y_pred = self.predict(X,y,**kwargs)        
		return self.get_loss_func()(y_pred,y,*args,**kwargs)

	def score(self,X,y,*args,**kwargs):
		if self.prioritize:
			kwargs.update(self.get_params())				
		y_pred = self.predict(X,y,**kwargs)        
		return self.get_score_func()(y_pred,y,*args,**kwargs)    

	def criteria(self,X,y,*args,**kwargs):
		if self.prioritize:
			kwargs.update(self.get_params())				
		keys =  ['loss','complexity_','losses','complexities_']
		for k in keys:
			if k in kwargs:
				pass
			elif k in ['loss']:
				kwargs[k] = self.loss(X,y,*args,**kwargs)
			elif k in ['losses']:
				kwargs[k] = self.get_stats()['loss']
			elif k in ['complexity_']:
				kwargs[k] = getattr(self,k,self.get_coef_().size)
			elif k in ['complexities_']:
				kwargs[k] = self.get_stats()['complexity_']
		return self.get_criteria_func()(**kwargs)

	def normalized(self,X,y,*args,**kwargs):
		if self.prioritize:
			kwargs.update(self.get_params())				
		normalize_ = self.get_normalize()
		scale_X = normalize_(X)
		scale_y = normalize_(y)
		self.fit(X,y,*args,**kwargs)
		X /= scale_X
		y /= scale_y
		coef_ = self.get_coef()
		coef_ /= scale_X
		coef_ *= scale_y
		self.set_coef_(coef_)
		return self

	def plot(self,X,y,*args,**kwargs):
		self.fit(X,y,*args,**kwargs)
		
		parameters = {}
		parameters['name'] = r'\textrm{%s}'%(str(self.__class__.__name__).replace('_',r'\_')) #split('.')[-1].replace("'>",''))
		parameters['path'] = 'plot.pdf'        
		parameters['loss']  = self.loss(X,y,*args,**kwargs)       
		
		parameters['fig'] = kwargs.get('fig')
		parameters['axes'] = kwargs.get('axes')
		
		parameters['fit__y'] = y
		parameters['fit__x'] = np.arange(parameters['fit__y'].shape[0])
		parameters['fit__y_pred'] = self.predict(X,y,*args,**kwargs)
		parameters['fit__x_pred'] = np.arange(parameters['fit__y_pred'].shape[0])
		parameters['fit__label'] = r'$y_{}$'
		parameters['fit__label_pred'] = r'$y_{%s} - %s$'%(parameters['name'],scinotation(parameters['loss']))
		parameters['settings'] = {
			'fit':{
				'fig':{
					'set_size_inches':{'h':10,'w':15},
					'tight_layout':{},
					'savefig':{'fname':parameters['path'],'bbox_inches':'tight'},
				},
				'ax':{
					'plot':[
						*[{
						 'x':parameters['fit__x'],
						 'y':parameters['fit__y'],
						 'label':parameters['fit__label'],                         
						 'linestyle':'--',
						 'alpha':0.7,
						  } if parameters['fig'] is None else []],
						*[{
						 'x':parameters['fit__x_pred'],
						 'y':parameters['fit__y_pred'],
						 'label':parameters['fit__label_pred'],
						 'linestyle':'-',
						 'alpha':0.6,
						  }]
						],
					'set_xlabel':{'xlabel':r'$x_{}$'},
					'set_ylabel':{'ylabel':r'$y_{}$'},
					'legend':{'loc':'best','prop':{'size':15}}
					},
				'style':{
					'layout':{'nrows':1,'ncols':2,'index':1},
				},
			},                       
		}
		fig,axes = plot(settings=parameters['settings'],fig=parameters['fig'],axes=parameters['axes'])
		return fig,axes  

	
class KFolder(object):
	def __init__(self,n_splits,random_state,test_size,**kwargs):
		call(super(),'__init__')        
		self.n_splits = n_splits
		self.random_state = random_state
		self.test_size = test_size
		for k in kwargs:
			setattr(self,k,kwargs[k])
		self.set_Folder()

	def split(self,X,y,groups=None):		
		self.set_Folder(groups=groups)
		if groups is None:
			splits = self.get_Folder().split(X,y)
		else:
			splits = self.get_Folder().split(X,y,groups=groups)
		for train,test in splits:
			yield train,test

	def get_n_splits(self,*args,**kwargs):
		return self.n_splits

	def set_Folder(self,groups=None):
		if groups is None:
			self.Folder = model_selection.ShuffleSplit(n_splits=self.n_splits,
									   test_size=self.test_size,
									   random_state=self.random_state)
		else:
			self.Folder = model_selection.GroupShuffleSplit(n_splits=self.n_splits,
											test_size=self.test_size,
											random_state=self.random_state)
		return

	def get_Folder(self,*args,**kwargs):
		field = 'Folder'
		if not hasattr(self,field):
			getattr(self,'set%s%s'%('_' if not field.startswith('_') else '',field))(*args,**kwargs)
		return getattr(self,field)



class UniformFolder(object):
	def __init__(self,n_splits,random_state,test_size,**kwargs):
		self.n_splits = n_splits
		self.random_state = random_state
		self.test_size = test_size
		for k in kwargs:
			setattr(self,k,kwargs[k])

	def split(self,X,y,groups=None):		
		splits = ((slice(None),slice(None)),)*self.get_n_splits()
		for train,test in splits:
			yield train,test

	def get_n_splits(self,*args,**kwargs):
		return self.n_splits


class GridSearchCV(Base):
	def __init__(self,estimator,param_grid,cv,*args,**kwargs):
		_kwargs = {
			'estimator':estimator,
			'param_grid':param_grid,
			'cv':cv,
			'n_jobs':1,
			'backend':'loky',			
			'prefer':None,
			'parallel':None,
			'scoring':'score',
			'refit':True,
		}
		kwargs.update({**_kwargs,**kwargs})
		call(super(),'__init__',*args,**kwargs)       
		return


	def _fit(self,X,y,train,test,scoring,params,*args,**kwargs):
		cls = self.get_estimator().__class__
		params = {**self.get_estimator().get_params(),**kwargs,**params}
		estimator = cls(*args,**params)
		estimator.fit(X[train],y[train],*args,**kwargs)
		value = getattr(estimator,scoring)(X[test],y[test],*args,**kwargs) 
		del estimator
		return value


	def fit(self,X,y,*args,**kwargs):

		parallelize = Parallelize(self.n_jobs,self.backend,prefer=self.prefer,parallel=self.parallel,verbose=self.verbose)

		func = self._fit
		scoring = 'score'
		param_grid = self.get_param_grid()
		cv = self.get_cv()

		arguments = []
		keywords = {}

		arguments.extend(args)
		keywords.update(kwargs)
		keywords.update({
			'X':X,'y':y,'scoring':scoring,
			})

		iterables = ['train','test','params']		
		iterable = (dict(zip(iterables,(train,test,params)))					                     
					for params in param_grid
					for train,test in cv.split(X,y,*args,**kwargs))
		
		values = []

		parallelize(func,iterable,values,*arguments,**keywords)

		self.set_cv_results_(values)
		if self.refit:
			self.get_best_estimator_().fit(X,y,*args,**kwargs)

		return self


	# def predict(self,X,y=None,**kwargs):
	# 	if self.prioritize:
	# 		kwargs.update(self.get_params())			
	# 	try:
	# 		return self.get_estimator().predict(X,**kwargs)
	# 	except:
	# 		self.fit(X,y,**kwargs)
	# 		return self.get_estimator().predict(X,**kwargs)

	# def loss(self,X,y=None,**kwargs):
	# 	if self.prioritize:
	# 		kwargs.update(self.get_params())			
	# 	try:
	# 		return self.get_estimator().loss(X,y,*args,**kwargs)
	# 	except:
	# 		self.fit(X,y,**kwargs)
	# 		return self.get_estimator().loss(X,y,*args,**kwargs)

	# def score(self,X,y=None,**kwargs):
	# 	if self.prioritize:
	# 		kwargs.update(self.get_params())			
	# 	try:
	# 		return self.get_estimator().score(X,y,*args,**kwargs)
	# 	except:
	# 		self.fit(X,y,**kwargs)
	# 		return self.get_estimator().score(X,y,*args,**kwargs)

	# def criteria(self,X,y=None,**kwargs):
	# 	if self.prioritize:
	# 		kwargs.update(self.get_params())			
	# 	try:
	# 		return self.get_estimator().criteria(X,y,*args,**kwargs)
	# 	except:
	# 		self.fit(X,y,**kwargs)
	# 		return self.get_estimator().criteria(X,y,*args,**kwargs)


	def set_estimator(self,estimator,*args,**kwargs):
		field = '_estimator'
		setattr(self,field,estimator)
		return
	
	def get_estimator(self,*args,**kwargs):
		field = '_estimator'
		return getattr(self,field)
	
	def set_param_grid(self,param_grid,*args,**kwargs):
		field = '_param_grid'
		_field = '_param_grid_names'
		if isinstance(param_grid,dict):
			setattr(self,_field,param_grid)
			setattr(self,field,self._permute(param_grid))
		else:
			keys = list(set([k for params in param_grid for k in params]))
			values = {k: [params[k] for params in params_grid if k in params]
						 for k in keys}
			setattr(self,_field,values)
			setattr(self,field,param_grid)
		
		self.set_n_params(len(getattr(self,field)))
		return
	
	def get_param_grid(self,*args,**kwargs):
		field = '_param_grid'
		return getattr(self,field)

	def _get_cv(self,cv,n_splits,**kwargs):
		try:
			return cv(n_splits,**kwargs)
		except TypeError:
			return cv(n_splits)

	def get_cv(self,*args,**kwargs):
		field = '_cv'
		return getattr(self,field)

	def set_cv(self,cv,*args,**kwargs):
		default = 'KFold'
		cvs = {'KFold':model_selection.KFold,'RepeatedKFold':model_selection.RepeatedKFold,
			   'TimeSeriesSplit':model_selection.TimeSeriesSplit,
			   'KFolder':KFolder,
			   'UniformFolder':UniformFolder,
			   }
		if isinstance(cv,dict):
			field = 'cv'
			value = cv.get(field)
			if value is None:
				cv[field] = cvs[default]
			elif isinstance(value,str):
				cv[field] = cvs.get(value,cvs[default])
			kwargs.update(cv)
			cv = self._get_cv(**kwargs)
		field = '_cv'			
		setattr(self,field,cv)
		self.set_n_splits(getattr(cv,'n_splits',getattr(cv,'get_n_splits',nullfunc)()))
		return

	def set_n_params(self,n_params,*args,**kwargs):
		field = 'n_params'
		return setattr(self,field,n_params)

	def get_n_params(self,*args,**kwargs):
		field = 'n_params'
		return getattr(self,field)

	def set_n_splits(self,n_splits,*args,**kwargs):
		field = 'n_splits'
		return setattr(self,field,n_splits)

	def get_n_splits(self,*args,**kwargs):
		field = 'n_splits'
		return getattr(self,field)


	def set_cv_results_(self,cv_results_,*args,**kwargs):
		field = 'cv_results_'
		_cv_results_ = np.array(cv_results_).reshape((self.get_n_params(),self.get_n_splits()))
		cv_results_ = {}
		cv_results_.update({
			**{'param_%s'%(param): np.array([params[param] for params in self.get_param_grid()]) for param in self._param_grid_names},			
			**{'split%d_test_score'%(i):_cv_results_[:,i] for i in range(self.get_n_splits())},
			**{'mean_test_score':_cv_results_.mean(axis=1),'std_test_score':_cv_results_.std(axis=1)},
			**{'params':self.get_param_grid()},
			})
		setattr(self,field,cv_results_)

		best_index_ = np.argmax(cv_results_['mean_test_score'])
		best_score_ = cv_results_['mean_test_score'][best_index_]
		best_params_ = cv_results_['params'][best_index_]
		best_cls_ = self.get_estimator().__class__
		best_estimator_ = best_cls_(**best_params_)

		self.set_best_index_(best_index_)
		self.set_best_score_(best_score_)
		self.set_best_params_(best_params_)
		self.set_best_estimator_(best_estimator_)

		logger.log(self.verbose,"Best params: %s"%(' ,'.join(['%s: %r'%(k,self.get_best_params_()[k]) for k in self.get_best_params_()])))
		return
	
	def get_cv_results_(self,*args,**kwargs):
		field = 'cv_results_'        
		return getattr(self,field)

	def set_best_index_(self,best_index_,*args,**kwargs):
		field = 'best_index_'
		setattr(self,field,best_index_)
		return

	def get_best_index_(self,*args,**kwargs):
		field = 'best_index_'        
		return getattr(self,field)

	def set_best_estimator_(self,best_estimator_,*args,**kwargs):
		field = 'best_estimator_'
		setattr(self,field,best_estimator_)

	def get_best_estimator_(self,*args,**kwargs):
		field = 'best_estimator_'
		return getattr(self,field)

	def set_best_params_(self,best_params_,*args,**kwargs):
		field = 'best_params_'
		setattr(self,field,best_params_)
		return

	def get_best_params_(self,*args,**kwargs):
		field = 'best_params_'        
		return getattr(self,field)

	def set_best_score_(self,best_score_,*args,**kwargs):
		field = 'best_score_'
		setattr(self,field,best_score_)
		return

	def get_best_score_(self,*args,**kwargs):
		field = 'best_score_'        
		return getattr(self,field)
	
	
	def _permute(self,dictionary,_copy=False,_groups=None,_ordered=True):

		def _copier(key,value,_copy):
			# _copy is a boolean, or dictionary with keys
			if ((not _copy) or (isinstance(_copy,dict) and (not _copy.get(key)))):
				return value
			else:
				return copy.deepcopy(value)
		
		def indexer(keys,values,_groups):
			_groups = copy.deepcopy(_groups)
			if _groups is not None:
				inds = [[keys.index(k) for k in g] for g in _groups]
			else:
				inds = []
				_groups = []
			N = len(_groups)
			_groups.extend([[k] for k in keys if all([k not in g for g in _groups])])
			inds.extend([[keys.index(k) for k in g] for g in _groups[N:]])
			values = [[values[j] for j in i ] for i in inds]

			return _groups,values

		def zipper(keys,values,_copy): 
			return [{k:_copier(k,u,_copy) for k,u in zip(keys,v)} for v in zip(*values)]

		def unzipper(dictionary):
			keys, values = zip(*dictionary.items())	
			return keys,values

		def permuter(dictionaries): 
			return [{k:d[k] for d in dicts for k in d} for dicts in itertools.product(*dictionaries)]

		if dictionary in [None,{}]:
			return [{}]

		keys,values = unzipper(dictionary)

		keys_ordered = keys

		keys,values = indexer(keys,values,_groups)

		dictionaries = [zipper(k,v,_copy) for k,v in zip(keys,values)]

		dictionaries = permuter(dictionaries)

		if _ordered:
			for i,d in enumerate(dictionaries):
				dictionaries[i] = {k: dictionaries[i][k] for k in keys_ordered}    
		return dictionaries
	




class CrossValidate(Estimator):
	def __init__(self,*args,**kwargs):
		call(super(),'__init__',*args,**kwargs)

		fields = ['estimator','param_grid','cv']
		for field in fields:
			kwargs[field] = getattr(self,field,kwargs.get(field))

		self.gridsearch = call(None,GridSearchCV,*args,**kwargs)
		return

	def fit(self,X,y,**kwargs):
		if self.prioritize:
			kwargs.update(self.get_params())				
		self.gridsearch.fit(X,y)
		self.set_estimator(None)
		self.set_coef_(None)	
		return self
	
	def predict(self,X,y=None,**kwargs):
		if self.prioritize:
			kwargs.update(self.get_params())	
		try:
			return self.get_estimator().predict(X,**kwargs)
		except:
			self.fit(X,y,**kwargs)
			return self.get_estimator().predict(X,**kwargs)

	def loss(self,X,y,*args,**kwargs):
		if self.prioritize:
			kwargs.update(self.get_params())			
		try:
			return self.get_estimator().loss(X,y,*args,**kwargs)
		except:
			self.fit(X,y,**kwargs)
			return self.get_estimator().loss(X,y,*args,**kwargs)

	def score(self,X,y,*args,**kwargs):
		if self.prioritize:
			kwargs.update(self.get_params())			
		try:
			return self.get_estimator().score(X,y,*args,**kwargs)
		except:
			self.fit(X,y,**kwargs)
			return self.get_estimator().score(X,y,*args,**kwargs)

	def criteria(self,X,y,*args,**kwargs):
		if self.prioritize:
			kwargs.update(self.get_params())			
		try:
			return self.get_estimator().criteria(X,y,*args,**kwargs)
		except:
			# self.fit(X,y,**kwargs)
			return self.get_estimator().criteria(X,y,*args,**kwargs)

	def _get_coef_(self,*args,**kwargs):
		field = 'coef_'
		return getattr(self.get_estimator(),field,getattr(self,field))			


	def set_coef_(self,coef_,*args,**kwargs):
		field = 'coef_'		
		if coef_ is None and hasattr(self.get_estimator(),'get%s%s'%('_' if not field.startswith('_') else '',field)):
			coef_ = getattr(self.get_estimator(),'get%s%s'%('_' if not field.startswith('_') else '',field))()
		if coef_ is None and hasattr(self.get_estimator(),field):
			coef_ = getattr(self.get_estimator(),field)

		setattr(self,field,coef_)

		if self.get_estimator():
			if hasattr(self.get_estimator(),'set%s%s'%('_' if not field.startswith('_') else '',field)):
				getattr(self.get_estimator(),'set%s%s'%('_' if not field.startswith('_') else '',field))(coef_)
			else:
				setattr(self.get_estimator(),field,coef_)
		if self.get_best_estimator_():
			if hasattr(self.get_best_estimator_(),'set%s%s'%('_' if not field.startswith('_') else '',field)):
				getattr(self.get_best_estimator_(),'set%s%s'%('_' if not field.startswith('_') else '',field))(coef_)
			else:
				setattr(self.get_best_estimator_(),field,coef_)		
		return		


	def get_coef_(self,*args,**kwargs):
		field = 'coef_'
		return getattr(self,field,getattr(self,'_get%s%s'%('_' if not field.startswith('_') else '',field))(*args,**kwargs))


	def _get_best_estimator_(self,*args,**kwargs):
		field = 'best_estimator_'
		_field = 'estimator'
		default = getattr(self,'get%s%s'%('_' if not _field.startswith('_') else '',_field))(*args,**kwargs)
		return getattr(self.gridsearch,field,getattr(self,field,default))

	def set_best_estimator_(self,best_estimator_,*args,**kwargs):
		field = 'best_estimator_'
		if best_estimator_ is None:
			best_estimator_ = self.get_estimator()
		setattr(self,field,best_estimator_)
		try:
			setattr(self.gridsearch,field)
		except:
			pass
		return

	def get_best_estimator_(self,*args,**kwargs):
		field = 'best_estimator_'
		return getattr(self,'_get%s%s'%('_' if not field.startswith('_') else '',field))(*args,**kwargs)


	def set_estimator(self,estimator,*args,**kwargs):		
		field = '_estimator'
		_field = 'best_estimator_'
		if estimator is None:
			estimator = getattr(self,'get%s%s'%('_' if not _field.startswith('_') else '',_field))(*args,**kwargs)
		if estimator is None and hasattr(self,field):
			estimator = getattr(self,'get%s%s'%('_' if not field.startswith('_') else '',field))(*args,**kwargs)

		estimator = getattr(self,'_set%s%s'%('_' if not field.startswith('_') else '',field))(estimator,*args,**kwargs)
		setattr(self,field,estimator)
		getattr(self,'set%s%s'%('_' if not _field.startswith('_') else '',_field))(estimator,*args,**kwargs)
		return
	

	def get_best_params_(self,*args,**kwargs):
		field = 'best_params_'        
		default = self.get_params()
		try:
			return getattr(self.gridsearch,field,default)
		except:
			return default
	
	def set_best_params_(self,best_params_,*args,**kwargs):
		field = 'best_params_'
		setattr(self.gridsearch,field,best_params_)
		return

	def get_cv_results_(self,*args,**kwargs):
		field = 'cv_results_'        
		default = {}
		try:
			return getattr(self.gridsearch,field,default)
		except:
			return default
	
	def set_cv_results_(self,cv_results_,*args,**kwargs):
		field = 'cv_results_'
		setattr(self.gridsearch,field,cv_results_)
		return    


	def _get_cv(self,cv,n_splits,**kwargs):
		try:
			return cv(n_splits,**kwargs)
		except TypeError:
			return cv(n_splits)

	def get_cv(self,*args,**kwargs):
		field = '_cv'
		return getattr(self,field)

	def set_cv(self,cv,*args,**kwargs):
		default = 'KFold'
		cvs = {'KFold':model_selection.KFold,'RepeatedKFold':model_selection.RepeatedKFold,
			   'TimeSeriesSplit':model_selection.TimeSeriesSplit,
			   'KFolder':KFolder,
			   'UniformFolder':UniformFolder,
			   }
		if isinstance(cv,dict):
			field = 'cv'		
			value = cv.get(field)
			if value is None:
				cv[field] = cvs[default]
			elif isinstance(value,str):
				cv[field] = cvs.get(value,cvs[default])
			kwargs.update(cv)
			cv = self._get_cv(**kwargs)
		field = '_cv'				
		setattr(self,field,cv)
		self.set_n_splits(getattr(cv,'n_splits',getattr(cv,'get_n_splits',nullfunc)()))
		return


	def set_n_splits(self,n_splits,*args,**kwargs):
		field = 'n_splits'
		return setattr(self,field,n_splits)

	def get_n_splits(self,*args,**kwargs):
		field = 'n_splits'
		return getattr(self,field)


	def plot(self,X,y,*args,**kwargs):
		self.fit(X,y,*args,**kwargs)
		
		parameters = {}
		parameters['name'] = r'\textrm{%s}'%(str(self.__class__.__name__).replace('_',r'\_')) #split('.')[-1].replace("'>",''))
		parameters['path'] = 'plot.pdf'
		parameters['figsize'] = {'h':15,'w':30}
		parameters['fig'] = kwargs.get('fig')
		parameters['axes'] = kwargs.get('axes')

		parameters['param'] = kwargs.get('plot_param','alpha_')
		parameters['params'] = np.asarray(self.get_cv_results_().get('param_%s'%(parameters['param']), 
								   self.get_cv_results_()['param_%s'%('alpha_')]))
		parameters['result_mean'] = kwargs.get('plot_result','mean_test_loss')
		parameters['result_mean_'] = {'mean_test_loss': 'mean_test_score'}.get(parameters['result_mean'],parameters['result_mean'])

		parameters['result_std'] = kwargs.get('plot_result','std_test_loss')
		parameters['result_std_'] = {'std_test_loss': 'std_test_score'}.get(parameters['result_std'],parameters['result_std'])        
		
		parameters['score_mean'] = self.get_cv_results_().get(parameters['result_mean_'],self.get_cv_results_()['mean_test_score'])
		parameters['score_mean'] *= (-1 if 'loss' in parameters['result_mean'] else 1)
		parameters['score_std'] = self.get_cv_results_().get(parameters['result_std_'],self.get_cv_results_()['std_test_score'])
		parameters['score_mean_best'] = getattr(np,'min' if 'loss' in parameters['result_mean'] else 'max')(parameters['score_mean'])
		parameters['param_best'] = parameters['params'][getattr(np,'argmin' if 'loss' in parameters['result_mean'] else 'argmax')(parameters['score_mean'])]

		parameters['fit__y'] = y
		parameters['fit__x'] = np.arange(parameters['fit__y'].shape[0])
		parameters['fit__y_pred'] = self.predict(X,y,*args,**kwargs)
		parameters['fit__x_pred'] = np.arange(parameters['fit__y_pred'].shape[0])
		parameters['fit__label'] = r'$y_{}$'
		parameters['fit__label_pred'] = r'$y_{%s} - %s$'%(parameters['name'],scinotation(parameters['score_mean_best']))
		
		parameters['score__y'] = parameters['score_mean']
		parameters['score__x'] = parameters['params']
		parameters['score__y_error'] = parameters['score_std']        
		parameters['score__x_best'] = parameters['param_best']
		parameters['score__label'] = r'${%s}$'%(parameters['name'])
		parameters['score__label_best'] = r'$\lambda_{\textrm{%s}} : %s,~\textrm{Loss} : %s$'%(parameters['name'],scinotation(parameters['param_best']),scinotation(parameters['score_mean_best']))
		

		parameters['result_mean_manual'] = parameters['result_mean']
		parameters['result_mean_manual_'] = parameters['result_mean_']

		parameters['result_std_manual'] = parameters['result_std']
		parameters['result_std_manual_'] = parameters['result_std_']
		parameters['param_manual'] = parameters['param']
		parameters['params_manual'] = parameters['params']
		parameters['score_mean_manual'] = np.zeros(parameters['params_manual'].shape)
		parameters['score_mean_manual'] = np.zeros(parameters['params_manual'].shape)
		for i,param in enumerate(parameters['params_manual']):
			self.get_estimator().set_params(**{parameters['param_manual']:param})
			self.get_estimator().fit(X,y,*args,**kwargs)
			parameters['score_mean_manual'][i] = self.get_estimator().score(X,y,*args,**kwargs)
		parameters['score_mean_manual'] *= (-1 if 'loss' in parameters['result_mean_manual'] else 1)
		parameters['score_mean_best_manual'] = getattr(np,'min' if 'loss' in parameters['result_mean_manual'] else 'max')(parameters['score_mean_manual'])
		parameters['param_best_manual'] = parameters['params_manual'][getattr(np,'argmin' if 'loss' in parameters['result_mean_manual'] else 'argmax')(parameters['score_mean_manual'])]


		self.get_estimator().set_params(**{parameters['param_manual']:parameters['param_best_manual']})
		self.get_estimator().fit(X,y,*args,**kwargs)
		# self.set_estimator(None)
		parameters['fit__y_pred_manual'] = self.get_estimator().predict(X,y,*args,**kwargs)
		parameters['fit__x_pred_manual'] = np.arange(parameters['fit__y_pred'].shape[0])
		parameters['fit__label_pred_manual'] = r'$y_{%s_{\textrm{manual}}} - %s$'%(parameters['name'],scinotation(parameters['score_mean_best_manual']))

		parameters['score__y_manual'] = parameters['score_mean_manual']
		parameters['score__x_manual'] = parameters['params_manual']
		parameters['score__y_error_manual'] = 0        
		parameters['score__x_best_manual'] = parameters['param_best_manual']
		parameters['score__label_manual'] = r'${%s}_{\textrm{manual}}$'%(parameters['name'])
		parameters['score__label_best_manual'] = r'$\lambda_{\textrm{%s}_{\textrm{manual}}} : %s,~\textrm{Loss} : %s$'%(parameters['name'],scinotation(parameters['param_best_manual']),scinotation(parameters['score_mean_best_manual']))		
		



		parameters['settings'] = {
			'fit':{
				'fig':{
					'set_size_inches':parameters['figsize'],
					'tight_layout':{},
					'savefig':{'fname':parameters['path'],'bbox_inches':'tight'},
				},
				'ax':{
					'plot':[
						# *[{
						#  'x':parameters['fit__x'],
						#  'y':parameters['fit__y'],
						#  'label':parameters['fit__label'],                         
						#  'linestyle':'--',
						#  'alpha':0.7,
						#   } if parameters['fig'] is None else []],
						*[{
						 'x':parameters['fit__x_pred'],
						 'y':parameters['fit__y_pred']-parameters['fit__y'],
						 'label':parameters['fit__label_pred'],
						 'linestyle':'-',
						 'alpha':0.6,
						  }],
						*[{
						 'x':parameters['fit__x_pred_manual'],
						 'y':parameters['fit__y_pred_manual']-parameters['fit__y'],
						 'label':parameters['fit__label_pred_manual'],
						 'linestyle':'-',
						 'alpha':0.6,
						  }]						  
						],
					'set_xlabel':{'xlabel':r'$x_{}$'},
					'set_ylabel':{'ylabel':r'$y_{}$'},
					'legend':{'loc':'best','prop':{'size':15}}
					},
				'style':{
					'layout':{'nrows':1,'ncols':2,'index':1},
				},
			},
			'score':{
				'fig':{
					'set_size_inches':parameters['figsize'],
					'tight_layout':{},
					'savefig':{'fname':parameters['path'],'bbox_inches':'tight'},
				},
				'ax':{
					'plot':[
						*[{
						 'x':parameters['score__x'],
						 'y':parameters['score__y'],
						 # 'yerr':parameters['score__y_error'],
						 'label':parameters['score__label'],
						 'linestyle':'-',
						 'linewidth':1,						 						 
						 'alpha':0.7,
						  }],
						*[{
						 'x':parameters['score__x_manual'],
						 'y':parameters['score__y_manual'],
						 # 'yerr':parameters['score__y_error_manual'],
						 'label':parameters['score__label_manual'],
						 'linestyle':'-',
						 'linewidth':1,						 						 
						 'alpha':0.7,
						  }],						  
						],
					'axvline':[
						*[{
						 'x':parameters['score__x_best'],
						 'ymin':0,
						 'ymax':1,
						 'label':parameters['score__label_best'],
						 'linestyle':'--',
						 'linewidth':2,
						 'alpha':0.8,
						 # 'color':'__lines__',
						  }],
						*[{
						 'x':parameters['score__x_best_manual'],
						 'ymin':0,
						 'ymax':1,
						 'label':parameters['score__label_best_manual'],
						 'linestyle':'--',
						 'alpha':0.8,
						 'linewidth':2,						 
						 'color':'__cycle__',
						  }],						  
						],
					'set_xlabel':{'xlabel':r'${\lambda}_{}$'},
					'set_ylabel':{'ylabel':r'${\textrm{%s}}_{}$'%('Loss' if 'loss' in parameters['result_mean'] else 'Score')},                  
					'set_xscale':{'value':'log'},
					'set_yscale':{'value':'log'},  
					'set_xnbins':{'numticks':5},
					'set_ynbins':{'numticks':5},
					# "set_xmajor_formatter":{"ticker":"ScalarFormatter"},											
					# "set_ymajor_formatter":{"ticker":"ScalarFormatter"},											
					# "ticklabel_format":{"axis":"x","style":"sci","scilimits":[-1,2]},
					# "ticklabel_format":{"axis":"y","style":"sci","scilimits":[-1,2]},						         
					'legend':{'loc':'best','prop':{'size':15}}
					 },
				'style':{
					'layout':{'nrows':1,'ncols':2,'index':2},
				},                        
			}            
					
		}
		fig,axes = plot(settings=parameters['settings'],fig=parameters['fig'],axes=parameters['axes'])
#         fig,axes = parameters['fig'],parameters['axes']

		return fig,axes  


class Solver(object):
	def __init__(self,solver,*args,**kwargs):	

		self.solvers = {name: getattr(self,name) 
			for name in [
			'pinv',
			'lstsq',
			'solve',
			'svd',
			'normal',    
			'normalsolve',
			'normalpinv',    
			'normallstsq',
			'linearregression',
			'ridge',
			'lasso',
			]
		}
		self.default = 'lstsq'
		self.set_solver(solver,*args,**kwargs)
		return

	def set_solver(self,solver,*args,**kwargs):
		self.solver = self.solvers.get(solver,self.solvers[self.default])
		return

	def get_solver(self,*args,**kwargs):
		return self.solver

	def __call__(self,X,y,*args,**kwargs):
		return self.get_solver()(X,y,*args,**kwargs)

	def pinv(self,X,y,*args,**kwargs):
		return np.linalg.pinv(X).dot(y)

	def lstsq(self,X,y,*args,**kwargs):
		rcond = kwargs.get('rcond',None)
		return np.linalg.lstsq(X,y,rcond=rcond)[0]

	def solve(self,X,y,*args,**kwargs):
		return np.linalg.solve(X,y)

	def svd(self,X,y,*args,**kwargs):
		alpha_ = kwargs.get('alpha_',0.0)
		U,S,VT = np.linalg.svd(X,full_matrices=False)
		return (VT.T*(S/(S**2 + alpha_))).dot(U.T.dot(y))

	def normal(self,X,y,*args,**kwargs):
		alpha_ = kwargs.get('alpha_',0.0)
		A = gram(X) + alpha_*np.identity(X.shape[1])
		b = project(X,y)
		if alpha_ > 0:
			return self.solve(A,b,*args,**kwargs)
		else:
			return self.lstsq(X,y,*args,**kwargs)

	def normalsolve(self,X,y,*args,**kwargs):
		alpha_ = kwargs.get('alpha_',0.0)
		A = gram(X) + alpha_*np.identity(X.shape[1])
		b = project(X,y)
		return self.solve(A,b,*args,**kwargs)

	def normalpinv(self,X,y,*args,**kwargs):
		alpha_ = kwargs.get('alpha_',0.0)
		A = gram(X) + alpha_*np.identity(X.shape[1])
		b = project(X,y)
		return self.pinv(A,b,*args,**kwargs)		

	def normallstsq(self,X,y,*args,**kwargs):
		alpha_ = kwargs.get('alpha_',0.0)
		A = gram(X) + alpha_*np.identity(X.shape[1])
		b = project(X,y)
		return self.lstsq(A,b,*args,**kwargs)		

	def linearregression(self,X,y,*args,**kwargs):
		model = call(None,linear_model.LinearRegression,*args,**kwargs)
		model.fit(X,y)
		return model.coef_

	def ridge(self,X,y,*args,**kwargs):
		model = call(None,linear_model.Ridge,*args,**kwargs)
		model.fit(X,y)
		return model.coef_

	def lasso(self,X,y,*args,**kwargs):
		model = call(None,linear_model.Lasso,*args,**kwargs)
		model.fit(X,y)
		return model.coef_	    	    


class LinearRegression(Estimator):
	def __init__(self,*args,**kwargs):
		call(super(),'__init__',*args,**kwargs)        
		return

	def predict(self,X,y=None,**kwargs):
		try:
			coef_ = kwargs.get('coef_',self.get_coef_())
			return X.dot(coef_)
		except:
			self.fit(X,y,**kwargs)
			return X.dot(self.get_coef_())
		return

class OLS(LinearRegression):
	def __init__(self,*args,**kwargs):
		kwargs['solver'] = kwargs.get('solver','pinv')
		call(super(),'__init__',*args,**kwargs)
		return

	def fit(self,X,y,*args,**kwargs):
		if self.prioritize:
			kwargs.update(self.get_params())				
		self.set_coef_(self.get_solver()(X,y,*args,**kwargs))
		self.set_intercept_(X,y,*args,**kwargs)		
		return self

class Tikhonov(LinearRegression):
	def __init__(self,*args,**kwargs):
		kwargs['solver'] = kwargs.get('solver','solve')
		call(super(),'__init__',*args,**kwargs)
		return
	def fit(self,X,y,*args,**kwargs):
		if self.prioritize:
			kwargs.update(self.get_params())				
		alpha_ = getattr(self,'alpha_',kwargs.get('alpha_',kwargs.get('alpha',1e-30)))
		I = getattr(self,'constraint_',kwargs.get('constraint_',np.identity(X.shape[1])))	
		
		A = gram(X) + alpha_*gram(I)
		b = project(X,y)
		self.set_coef_(self.get_solver()(A,b,*args,**kwargs))
		self.set_intercept_(X,y,*args,**kwargs)				
		
		# logger.log(self.verbose,'alpha_: %r'%(alpha_))
		return self


class Ridge(LinearRegression):
	def __init__(self,*args,**kwargs):
		kwargs['solver'] = kwargs.get('solver','normal')
		call(super(),'__init__',*args,**kwargs)
		return
	def fit(self,X,y,*args,**kwargs):
		if self.prioritize:
			kwargs.update(self.get_params())				
		kwargs['alpha_'] = getattr(self,'alpha_',kwargs.get('alpha_',kwargs.get('alpha',1e-30)))
		self.set_coef_(self.get_solver()(X,y,*args,**kwargs))
		self.set_intercept_(X,y,*args,**kwargs)				
		
		# logger.log(self.verbose,'alpha_: %r'%(kwargs['alpha_']))
		return self


class Lasso(LinearRegression):
	def __init__(self,*args,**kwargs):
		kwargs['solver'] = kwargs.get('solver','normal')		
		call(super(),'__init__',*args,**kwargs)
		return
	def fit(self,X,y,*args,**kwargs):
		if self.prioritize:
			kwargs.update(self.get_params())				
		kwargs['alpha'] = getattr(self,'alpha_',kwargs.get('alpha_',kwargs.get('alpha',1e-30)))
		coef_, self.dual_gap_, self.eps_, self.n_iter_ = self.enet_coordinate_descent(X,y,**kwargs)
		self.set_coef_(np.asarray(coef_))
		self.set_intercept_(X,y,*args,**kwargs)				
		return self

	def enet_coordinate_descent(self,X,y,*args,**kwargs):
		# Enet coordinate descent
		settings = {
			'w':None,
			'alpha':1.0,
			'beta':0.0,
			'X':X,
			'y':y,
			'max_iter':1000,
			'tol':1e-4,
			'rng':None,
			'random':'random',
			'positive':False,
		}
		parsing = {
			'w': lambda **kwargs: np.asarray(kwargs['w'] if kwargs['w'] is not None else zeros(kwargs['X'].shape[1]),order='F'),
			'X': lambda **kwargs: np.asarray(X, order='F'),
			'y': lambda **kwargs: np.asarray(y, order='F'),
			'rng':lambda **kwargs:seed(kwargs['rng']),
			'random': lambda **kwargs: {'random':1,'cyclic':0}.get(kwargs['random'],0)
				  }
		settings.update({k: kwargs[k] for k in kwargs if k in settings})
		settings.update({k: parsing[k](**settings) for k in parsing})

		from sklearn import linear_model
		coef_,dual_gap_,eps_,n_iter_ = linear_model._cd_fast.enet_coordinate_descent(*[settings[k] for k in settings])

		return coef_,dual_gap_,eps_,n_iter_



class Enet(LinearRegression):
	def __init__(self):
		kwargs['solver'] = kwargs.get('solver','normal')		
		call(super(),'__init__',*args,**kwargs)        
		return
	def fit(self,X,y,*args,**kwargs):
		if self.prioritize:
			kwargs.update(self.get_params())				
		kwargs['alpha'] = kwargs.get('alpha_',kwargs.get('alpha',1.0))
		kwargs['beta'] = kwargs.get('beta_',kwargs.get('beta',1.0))
		coef_, self.dual_gap_, self.eps_, self.n_iter_ = self.enet_coordinate_descent(X,y,**kwargs)
		self.set_coef_(np.asarray(coef_))
		self.set_intercept_(X,y,*args,**kwargs)				
		return self

	def enet_coordinate_descent(self,X,y,*args,**kwargs):
		# Enet coordinate descent
		settings = {
			'w':None,
			'alpha':1.0,
			'beta':0.0,
			'X':X,
			'y':y,
			'max_iter':1000,
			'tol':1e-4,
			'rng':None,
			'random':'random',
			'positive':False,
		}
		parsing = {
			'w': lambda **kwargs: np.asarray(kwargs['w'] if kwargs['w'] is not None else zeros(kwargs['X'].shape[1]),order='F'),
			'X': lambda **kwargs: np.asarray(X, order='F'),
			'y': lambda **kwargs: np.asarray(y, order='F'),
			'rng':lambda **kwargs:seed(kwargs['rng']),
			'random': lambda **kwargs: {'random':1,'cyclic':0}.get(kwargs['random'],0)
				  }
		settings.update({k: kwargs[k] for k in kwargs if k in settings})
		settings.update({k: parsing[k](**settings) for k in parsing})

		from sklearn import linear_model
		coef_,dual_gap_,eps_,n_iter_ = linear_model._cd_fast.enet_coordinate_descent(*[settings[k] for k in settings])

		return coef_,dual_gap_,eps_,n_iter_


class TikhonovCV(CrossValidate):
	def __init__(self,*args,**kwargs):
		self.set_estimator(kwargs.pop('estimator','Ridge'),*args,**kwargs)
		kwargs['estimator'] = self.get_estimator()
		kwargs['param_grid'] = {
			**{'alpha_':
				kwargs.get('alphas',[*([0] if kwargs.get('alpha_zero') else []),
									 *np.logspace(*kwargs.get('alpha',[-10,-1,10]) if hasattr(kwargs.get('alpha',[-10,-1,10]),'__iter__') else [-10,1,10])
									]),
			},
			**kwargs.get('param_grid',{})
			}
		kwargs['cv'] = kwargs.get('cv')
		call(super(),'__init__',*args,**kwargs)        
		return
	
class RidgeCV(CrossValidate):
	def __init__(self,*args,**kwargs):
		self.set_estimator(kwargs.pop('estimator','Ridge'),*args,**kwargs)
		kwargs['estimator'] = self.get_estimator()
		kwargs['param_grid'] = {
			**{'alpha_':
				kwargs.get('alphas',[*([0] if kwargs.get('alpha_zero') else []),
									 *np.logspace(*kwargs.get('alpha',[-10,-1,10]) if hasattr(kwargs.get('alpha',[-10,-1,10]),'__iter__') else [-10,1,10])
									]),
			},
			**kwargs.get('param_grid',{})
			}
		kwargs['cv'] = kwargs.get('cv')
		call(super(),'__init__',*args,**kwargs)        
		return
	
	
class LassoCV(CrossValidate):
	def __init__(self,*args,**kwargs):
		self.set_estimator('Lasso',*args,**kwargs)
		kwargs['estimator'] = self.get_estimator()
		kwargs['param_grid'] = {
			**{'alpha_':
				kwargs.get('alphas',[*([0] if kwargs.get('alpha_zero') else []),
									 *np.logspace(*kwargs.get('alpha',[-10,-1,10]) if hasattr(kwargs.get('alpha',[-10,-1,10]),'__iter__') else [-10,1,10])
									]),
			},
			**kwargs.get('param_grid',{})
			}		
		kwargs['cv'] = kwargs.get('cv')
		call(super(),'__init__',*args,**kwargs)        
		return
	
class EnetCV(CrossValidate):
	def __init__(self,*args,**kwargs):
		self.set_estimator('Enet',*args,**kwargs)
		kwargs['estimator'] = self.get_estimator()
		kwargs['param_grid'] = {
			**{'alpha_':
				kwargs.get('alphas',[*([0] if kwargs.get('alpha_zero') else []),
									 *np.logspace(*kwargs.get('alpha',[-10,-1,10]) if hasattr(kwargs.get('alpha',[-10,-1,10]),'__iter__') else [-10,1,10])
									]),
				'beta_':
				kwargs.get('betas',[*([0] if kwargs.get('beta_zero') else []),
									 *np.logspace(*kwargs.get('beta',[-10,-1,10]) if hasattr(kwargs.get('beta',[-10,-1,10]),'__iter__') else [-10,1,10])
									]),
			},
			**kwargs.get('param_grid',{})
			}		
		kwargs['cv'] = kwargs.get('cv')
		call(super(),'__init__',*args,**kwargs)        
		return


class RegressionCV(CrossValidate):
	def __init__(self,*args,**kwargs):
		self.set_estimator(kwargs.get('estimator'),*args,**kwargs)
		kwargs['estimator'] = self.get_estimator()
		kwargs['param_grid'] = {'alpha_':kwargs.get('alphas',np.logspace(*kwargs.get('alpha',[-10,-1,10]) if hasattr(kwargs.get('alpha',[-10,-1,10]),'__iter__') else [-10,1,10])),
								'beta_':kwargs.get('betas',np.logspace(*kwargs.get('beta',[-10,-1,10])))}
		kwargs['cv'] = kwargs.get('cv')
		call(super(),'__init__',*args,**kwargs)        
		return




class Stepwise(Estimator):
	def __init__(self,*args,**kwargs):
		call(super(),'__init__',*args,**kwargs)        
		return

	def _fit(self,X,y,*args,**kwargs):
		self.get_estimator().fit(X,y,*args,**kwargs)
		self.set_coef_(None)
		self.set_intercept_(X,y,*args,**kwargs)
		return self 

	def fit(self,X,y,*args,**kwargs):
		if self.prioritize:
			kwargs.update(self.get_params())	

		Start = timeit.default_timer()

		# Get data shape
		if y.ndim < 2:
			y = y.reshape((*y.shape,*[1]*(2-y.ndim)))
		if X.ndim < 3:
			X = X.reshape((*X.shape,*[1]*(3-X.ndim)))
		Ndata,Ncoef,Ndim = X.shape
		Ndata,Ndim = y.shape

		# Get methods and included indices
		method = getattribute(self,'method','cheapest') # Method for applying statistical criteria
		threshold = getattribute(self,'threshold',1e20) # Threshold for criteria
		fixed = getattribute(self,'fixed',{}) # Fixed indices and coef_ in basis
		included = getattribute(self,'included',[]) # Included X indices in basis

		# Get mappings between stepwise indices, local indices of X', and global indices of X, 
		# where local position of index is X' index and mapped index is X index
		# i.e) indices_local = [0,2] : Stepwise index 1 -> X' index 2 (excludes fixed and included indices)
		# i.e) indices_basis = [0,1,2,3] : X' index 2 -> X' index 2 (excludes fixed indices)
		# i.e) indices_global = [1,2,4,5] : X' index 2: X index 4 (excludes fixed indices)

		local_global = np.arange(Ncoef) # X' indices to X indices
		
		local_fixed = np.array([i for i in fixed if i<Ncoef],dtype=int) # Fixed X' indices to X indices
		local_included = np.array([i for i in included if i<Ncoef and i not in local_fixed],dtype=int) # Included X' indices to X indices

		local_free = local_global[isin(local_global,local_fixed,invert=True)] # Free X' indices to X indices
		coef_fixed = np.array([fixed[i] for i in fixed if i<Ncoef]) # Fixed coef_ at X' indices

		indices_global = local_global[isin(local_global,local_fixed,invert=True)] # Free X' indices to X indices, without fixed indices
		indices_basis = np.arange(indices_global.size)
		indices_local = indices_basis[isin(indices_global,local_included,invert=True)]


		# Account for fixed coefficients in X,y
		X_fixed = X.copy()
		y_fixed = np.zeros((Ndata,Ndim))
		if local_fixed.size > 0:
			for dim in range(Ndim):
				y_fixed[:,dim] = self.predict(X[:,local_fixed,dim],y[...,dim],*args,coef_=coef_fixed,**kwargs)
				y[:,dim] -= y_fixed[:,dim]
			X = X[:,local_free]


		# Size of basis
		# Number of potential operators Nmin <= N <= Nmax, number of total operators G
		N = indices_local.size
		G = local_global.size
		Nmin = getattribute(self,'complexity_min',None)
		Nmax = getattribute(self,'complexity_max',None)
		Nmin = max(0 if N<Ncoef else 1 ,Nmin if isinstance(Nmin,(int,np.integer)) else 0)
		Nmax = min(N,Nmax if isinstance(Nmax,(int,np.integer)) else Ncoef)

		# Stats to be collected with shape, dtype, and default value of numpy array
		collect = getattribute(self,'collect',True)
		_stats = {}
		if collect:	
			_stats.update({'predict':{'shape':[Ncoef,Ndata],'dtype':float,'default':0},})

		_stats.update({
			'loss':{'shape':[Ncoef],'dtype':float,'default':0},
			'score':{'shape':[Ncoef],'dtype':float,'default':0},
			'coef_':{'shape':[Ncoef,Ncoef],'dtype':float,'default':0},
			'criteria':{'shape':[Ncoef],'dtype':float,'default':0},
			'index_':{'shape':[Ncoef],'dtype':int,'default':0},
			'index':{'shape':[Ncoef],'dtype':int,'default':0},
			'dim_':{'shape':[Ncoef],'dtype':int,'default':0},
			'rank_':{'shape':[Ncoef,Ndim],'dtype':int,'default':0},
			'condition_':{'shape':[Ncoef,Ndim],'dtype':float,'default':0},
			'basis_':{'shape':[Ncoef,Ncoef],'dtype':int,'default':-1},
			'complexity_':{'shape':[Ncoef],'dtype':int,'default':0},
			'size_':{'shape':[Ncoef],'dtype':int,'default':0},
			'iterations':{'shape':[Ncoef],'dtype':int,'default':1},
			'coefficient':{'shape':[Ncoef],'dtype':float,'default':0},
		})

		fields = {'stats_modules': [k for k in ['predict','loss','score','coef_','criteria'] if k in _stats],
				  'stats_params':  [k for k in ['index_','index','dim_','rank_','condition_','basis_','complexity_','size_','coefficient',] if k in _stats]}
		for field in fields:
			if not hasattr(self,field):
				setattr(self,field,fields[field])

		stats = {}
		stats.update({k:_stats[k].pop('default')*np.ones(**_stats[k])for k in _stats})
		if method in ['update']:
			stats.update({k: getattr(self,'stats',{}).get(k,stats[k]) for k in getattr(self,'stats_params',[])})

		field = 'coef_'
		stats[field][:,local_fixed] = coef_fixed

		indexes = np.arange(N)
		dims = np.arange(Ndim)
		values = np.zeros((Ndim,Ncoef))


		# Variables
		globs = globals()
		locs = locals()
		
		# Get/Set  kwargs
		fields = {
			**{k:'set' for k in []},
			**{k:'pop' for k in []}
			}
		for field in fields:
			if field in locs:
				if fields[field] in ['set']:
					kwargs[field] = locs[field]
				elif fields[field] in ['pop']:
					kwargs.pop(field)


		# Setup parallelization
		# parallelizer = Parallelize(self.n_jobs,self.backend)
		parallelizer = nullcontext()
		parallel = self.parallel
		processes = self.n_jobs
		def pooler(processes=processes,parallel=parallel,scope=globs): 
			return (getattribute(scope.get(parallel) if isinstance(scope,dict) else parallel,'Pool',nullPool)(processes=processes))


		logger.log(self.verbose,"Start Stepwise Regression")

		with parallelizer as parallelize:
			iteration = 0        
			while ((iteration == 0) or (N<=Nmax and N>Nmin)):
				indices = indexes[:N]
				values[:] = 0
				_values = []
				
				step_kwargs = {'index':None,'dim':None,
								'mapping':indices_local,'iteration':iteration,
								'X':X,'y':y,'fit':self._fit,'func':self.loss,'verbose':False,
								'kwargs':kwargs}
				callback_kwargs = {'index':None,'dim':None,'values':values}

				# Setup indices			
				if method in ['update']:
					indices = [stats['index'][iteration]]
				elif iteration == 0 and method in ['cheapest','smallest']:
					indices = [[]]
		
			
				start = timeit.default_timer()

				# func = self.step
				# iterables = ['index','dim']	
				# iterable = (dict(zip(iterables,(index,dim)))
				# 			for dim in dims
				# 			for index in indices)                     		

				# start = timeit.default_timer()	
				# parallelize(func,iterable,_values,**step_kwargs)
				# try:
				# 	values[:,:N] = np.asarray(_values).reshape((Ndim,N)) 
				# except:
				# 	values[:,-1] = np.asarray(_values).reshape((Ndim)) 
				# end = timeit.default_timer()

				
				pool = pooler()
				for dim in dims:			
					for index in indices:
						step_kwargs.update({'index':index,'dim':dim})
						callback_kwargs.update({'index':index,'dim':dim})
						
						func = wrapper(self.step,**step_kwargs)

						callback = wrapper(self.callback_step,**callback_kwargs)
						pool.apply_async(func=func,callback=callback)	
				pool.close()
				pool.join()
				
				end = timeit.default_timer()

				if iteration == 0:
					index = -1
					dim = dims[(values[:,index]).argmin(axis=0)]
					
					index_local = -1
					index_basis = -1
					index_free = -1
					index_global = -1
					index_local_global = -1
				else:
					criteria = where((np.abs(self.criteria(X,y,
														loss=values[:,indices],
														complexity_=N-1,
														losses=stats['loss'][:iteration],
														complexities_=stats['complexity_'][:iteration]))<=threshold
										).all(axis=0))
					if criteria.size == 0:
						break
					
					index = indices[criteria[((values[:,indices]).sum(axis=0)[criteria]).argmin(axis=0)]]
					dim = dims[(values[:,index]).argmin(axis=0)]


					index_local = indices_local[index]
					index_basis = indices_basis[index_local]
					index_global = indices_global[index_basis]
					index_local_global = local_global[index_global]
					X = delete(X,index_basis,axis=1)	
					
					indices_local = delete(indices_local,index,axis=0)
					indices_local[indices_local>index_local] -= 1								
					indices_basis = delete(indices_basis,index_local,axis=0)
					indices_basis[indices_basis>index_basis] -= 1				
					indices_global = delete(indices_global,index_basis,axis=0)				
					indices_global[indices_global>index_global] -= 1
					local_global = delete(local_global,index_global,axis=0)				

					
					N -= 1
					G -= 1
				
				_estimator = self._fit(X[:,:,dim],y[:,dim],**kwargs)


				if collect:
					stats['predict'][iteration] = self.predict(X[:,:,dim],y[:,dim],**kwargs) + y_fixed[:,dim]
				stats['loss'][iteration] = self.loss(X[:,:,dim],y[:,dim],**kwargs)
				stats['score'][iteration] = self.score(X[:,:,dim],y[:,dim],**kwargs)
				stats['criteria'][iteration] = self.criteria(X[:,:,dim],y[:,dim],
														loss=stats['loss'][iteration],
														complexity_=N,
														losses=stats['loss'][:iteration],
														complexities_=stats['complexity_'][:iteration])
				stats['coef_'][iteration][local_global[indices_global]] = self.get_coef_().copy()
				stats['index_'][iteration] = index_local_global
				stats['index'][iteration] = index
				stats['dim_'][iteration] = dim
				stats['rank_'][iteration] = [rank(X[:,:,dim]) for dim in dims]
				stats['condition_'][iteration] = [cond(X[:,:,dim]) for dim in dims]
				stats['basis_'][iteration][:G] = local_global
				stats['coefficient'][iteration] = np.abs(stats['coef_'][iteration-1][index_local_global])
				stats['complexity_'][iteration] = N
				stats['size_'][iteration] = G
				stats['iterations'][iteration] = iteration+1


				self.set_coef_(stats['coef_'][iteration])

				logger.log(self.verbose,'Iter: %d, Index: (%d,%d,%d), Loss: %0.4e, Rank: %r, Cond: %r, N: %d, Time: %0.4e'%(
									iteration,index,index_local_global,dim,
									stats['loss'][iteration],
									stats['rank_'][iteration].tolist()[slice(None) if len(dims)>1 else 0],
									stats['condition_'][iteration].tolist()[slice(None) if len(dims)>1 else 0],
									N,end-start))

				iteration += 1

		self.set_stats({k: stats[k][:iteration] for k in stats})
		

		End = timeit.default_timer()

		# Restore X,y
		X = X_fixed
		for dim in range(Ndim):
			y[:,dim] += y_fixed[:,dim]

		logger.log(self.verbose,"Done Stepwise Regression")

		return self		


	def predict(self,X,y=None,**kwargs):
		try:
			return self.get_estimator().predict(X,y,**kwargs)
		except:
			self.fit(X,y,**kwargs)
			return self.get_estimator().predict(X,y,**kwargs)

	# Stepwise iteration
	@staticmethod
	def step(iteration=None,index=None,dim=None,mapping=None,X=None,y=None,fit=None,func=None,verbose=False,kwargs={}):
		logger.log(verbose,'Iter: %r, Dim: %r, Index: %r'%(iteration,dim,index))			
		try:
			_X = delete(X[:,:,dim],mapping[index],axis=1)	
			_y = y[:,dim]
		except:
			_X = X[:,:,dim]
			_y = y[:,dim]
		fit(_X,_y,**kwargs)
		return func(_X,_y,**kwargs)

	# Stepwise callback
	@staticmethod
	def callback_step(value=None,index=None,dim=None,values=None):
		try:
			values[dim][index] = value
		except:
			values[dim] = value
		return

	def plot(self,X,y,*args,**kwargs):
		
		self.fit(X,y,*args,**kwargs)

		parameters = {}
		parameters['path'] = 'stepwise.pdf'
		parameters['name'] = str(self.get_estimator().__class__.__name__)
		parameters['nrows'] = 1
		parameters['ncols'] = 2 if 'CV' not in parameters['name'] else 3
		parameters['figsize'] = {'h':10,'w':20} if 'CV' not in parameters['name'] else {'h':10,'w':30}
		parameters['subplots_adjust'] = {"hspace":2,"wspace":2} if 'CV' not in parameters['name'] else {"hspace":2,"wspace":3}

		
		parameters['iterations'] = [i if i is not None else self.get_stats()['size_'].max() for i in [None,10,5,1] if i is None or (i > self.get_stats()['size_'].min())]
		parameters['iterations'] = list(sorted(list(set(parameters['iterations'])),key=lambda i: parameters['iterations'].index(i)))
		
		parameters['fig'] = kwargs.get('fig')
		parameters['axes'] = kwargs.get('axes')
		
		parameters['fit__x'] = X[:,0,0] if X.ndim>2 else X[:,0]
		parameters['fit__y'] = y[:,0] if y.ndim > 1 else y
		parameters['fit__label'] = r'$y_{}$'

		for i in parameters['iterations']:
			parameters['fit__x_pred_%d'%(i)] = X[:,0,0] if X.ndim>2 else X[:,0]
			parameters['fit__y_pred_%d'%(i)] = self.get_stats()['predict'][self.get_stats()['size_'].max()-i]
			parameters['fit__label_pred_%d'%(i)] = r'$y^{(%d)}_{\textrm{%s}}$'%(i,parameters['name'].replace('_',r'\_'))

		parameters['loss__x'] = self.get_stats()['complexity_']
		parameters['loss__y'] = self.get_stats()['loss']
		parameters['loss__label'] = r'$\textrm{%s}$'%(str(self.get_estimator().__class__.__name__).replace('_',r'\_'))


		if 'CV' in parameters['name']:
			parameters['param'] = kwargs.get('plot_param','alpha_')
			parameters['params'] = np.asarray(self.get_estimator().get_cv_results_().get('param_%s'%(parameters['param']), 
									   self.get_estimator().get_cv_results_()['param_%s'%('alpha_')]))
			parameters['result_mean'] = kwargs.get('plot_result','mean_test_loss')
			parameters['result_mean_'] = {'mean_test_loss': 'mean_test_score'}.get(parameters['result_mean'],parameters['result_mean'])

			parameters['result_std'] = kwargs.get('plot_result','std_test_loss')
			parameters['result_std_'] = {'std_test_loss': 'std_test_score'}.get(parameters['result_std'],parameters['result_std'])        
			
			parameters['score_mean'] = self.get_estimator().get_cv_results_().get(parameters['result_mean_'],self.get_estimator().get_cv_results_()['mean_test_score'])
			parameters['score_mean'] *= (-1 if 'loss' in parameters['result_mean'] else 1)
			parameters['score_std'] = self.get_estimator().get_cv_results_().get(parameters['result_std_'],self.get_estimator().get_cv_results_()['std_test_score'])
			parameters['score_mean_best'] = getattr(np,'min' if 'loss' in parameters['result_mean'] else 'max')(parameters['score_mean'])
			parameters['param_best'] = parameters['params'][getattr(np,'argmin' if 'loss' in parameters['result_mean'] else 'argmax')(parameters['score_mean'])]
			
			parameters['score__y'] = parameters['score_mean']
			parameters['score__x'] = parameters['params']
			parameters['score__y_error'] = parameters['score_std']        
			parameters['score__x_best'] = parameters['param_best']
			parameters['score__label'] = r'${%s}$'%(parameters['name'])
			parameters['score__label_best'] = r'$\lambda_{\textrm{%s}} : %s,~\textrm{Loss} : %s$'%(parameters['name'],scinotation(parameters['param_best']),scinotation(parameters['score_mean_best']))




		parameters['settings'] = {
			'fit':{
				'fig':{
					'set_size_inches':parameters['figsize'],
					'subplots_adjust':parameters['subplots_adjust'],
					'tight_layout':{},
					'savefig':{'fname':parameters['path'],'bbox_inches':'tight'},
				},
				'ax':{
					'plot':[
						*[{
						 'x':parameters['fit__x'],
						 'y':parameters['fit__y'],
						 'label':parameters['fit__label'],                         
						 'linestyle':'--',
						 'alpha':0.7,
						  } if parameters['fig'] is None else []],
						*[{
						 'x':parameters['fit__x_pred_%d'%(i)],
						 'y':parameters['fit__y_pred_%d'%(i)],
						 'label':parameters['fit__label_pred_%d'%(i)],
						 'linestyle':'-',
						 'alpha':0.6,
						  } for i in parameters['iterations']],
						],
					'set_xlabel':{'xlabel':r'$x_{}$'},
					'set_ylabel':{'ylabel':r'$y_{}$'},
					'legend':{'loc':'best','prop':{'size':15}}
					},
				'style':{
					'layout':{'nrows':parameters['nrows'],'ncols':parameters['ncols'],'index':1},
				},
			},
			'loss':{
				'fig':{
					'set_size_inches':parameters['figsize'],
					'subplots_adjust':parameters['subplots_adjust'],					
					'tight_layout':{},
					'savefig':{'fname':parameters['path'],'bbox_inches':'tight'},
				},
				'ax':{
					'plot':[
						*[{
						 'x':parameters['loss__x'],
						 'y':parameters['loss__y'],
						 'label':parameters['loss__label'],
						 'linestyle':'-',
						 'marker':'s',
						 'linewidth':1,						 						 
						 'alpha':0.7,
						  }],
						],
					'set_xlabel':{'xlabel':r'${N_{\textrm{Coefficients}}}$'},
					'set_ylabel':{'ylabel':r'${\textrm{Loss}}_{}$'},                  
					'set_xscale':{'value':'linear'},
					'set_yscale':{'value':'log'},  
					'set_xticks':{'ticks':np.linspace(min(parameters['loss__x'].min()-1,0),parameters['loss__x'].max()+3,max(4,int(parameters['loss__x'].max()-parameters['loss__x'].min())//5)).astype(int)},
					'set_ynbins':{'numticks':5},
					'set_xlim':{'xmin':parameters['loss__x'].max()+3,'xmax':min(parameters['loss__x'].min()-1,0)},
					# "set_xmajor_formatter":{"ticker":"ScalarFormatter"},											
					# "set_ymajor_formatter":{"ticker":"ScalarFormatter"},											
					# "ticklabel_format":{"axis":"x","style":"sci","scilimits":[-1,2]},
					# "ticklabel_format":{"axis":"y","style":"sci","scilimits":[-1,2]},						         
					'legend':{'loc':'best','prop':{'size':15}}
					 },
				'style':{
					'layout':{'nrows':parameters['nrows'],'ncols':parameters['ncols'],'index':2},
				},                        
			},
			**({'score':{
				'fig':{
					'set_size_inches':parameters['figsize'],
					'subplots_adjust':parameters['subplots_adjust'],					
					'tight_layout':{},
					'savefig':{'fname':parameters['path'],'bbox_inches':'tight'},
				},
				'ax':{
					'plot':[
						*[{
						 'x':parameters['score__x'],
						 'y':parameters['score__y'],
						 # 'yerr':parameters['score__y_error'],
						 'label':parameters['score__label'],
						 'linestyle':'-',
						 'linewidth':1,						 						 
						 'alpha':0.7,
						  }],
						],
					'axvline':[
						*[{
						 'x':parameters['score__x_best'],
						 'ymin':0,
						 'ymax':1,
						 'label':parameters['score__label_best'],
						 'linestyle':'--',
						 'linewidth':2,
						 'alpha':0.8,
						 # 'color':'__lines__',
						  }],
						],
					'set_xlabel':{'xlabel':r'${\lambda}_{}$'},
					'set_ylabel':{'ylabel':r'${\textrm{%s}}_{}$'%('Loss' if 'loss' in parameters['result_mean'] else 'Score')},                  
					'set_xscale':{'value':'log'},
					'set_yscale':{'value':'log'},  
					'set_xnbins':{'numticks':5},
					'set_ynbins':{'numticks':5},
					# "set_xmajor_formatter":{"ticker":"ScalarFormatter"},											
					# "set_ymajor_formatter":{"ticker":"ScalarFormatter"},											
					# "ticklabel_format":{"axis":"x","style":"sci","scilimits":[-1,2]},
					# "ticklabel_format":{"axis":"y","style":"sci","scilimits":[-1,2]},						         
					'legend':{'loc':'best','prop':{'size':15}}
					 },
				'style':{
					'layout':{'nrows':parameters['nrows'],'ncols':parameters['ncols'],'index':3},
				},                        
			}} if 'CV' in parameters['name'] else {}),
					
		}
		fig,axes = plot(settings=parameters['settings'],fig=parameters['fig'],axes=parameters['axes'])
		return fig,axes




if __name__ == '__main__':

	n = 100
	m = 5
	X = np.sort(np.random.rand(n,1),axis=0)
	# X = np.concatenate([np.ones((n,1)),*[np.linspace(0,1,n,endpoint=False)[:,None]]*(m-1)],axis=1)
	# X = np.concatenate([*[np.linspace(0,1,n,endpoint=False)[:,None]]*(m)],axis=1)
	coef_ = np.random.rand(m)
	# y = np.sin(X.dot(coef_))

	# func = lambda x: 1 - x + x**2 - x**3 + x**4
	# func = lambda x: x #+ x**2 - x**3 + x**4
	# func = lambda x: np.sin(x)
	# func = lambda x: x
	# yfunc = lambda x: np.concatenate([np.sin(x**i) for i in range(m)],axis=1)
	# Xfunc = lambda x: np.concatenate([x**i for i in range(m)],axis=1)
	# func = lambda x: np.sin(np.pi*X) #np.concatenate([np.sin(x[:,i:i+1]**i) for i in range(x.shape[1])],axis=1)

	yfunc = lambda x: np.sin(np.concatenate([x**i for i in range(1,m+1)],axis=1))
	Xfunc = lambda x: np.concatenate([x**i for i in range(1,m+1)],axis=1)
	X,y = Xfunc(X), yfunc(X).dot(coef_)
	# X = func(X + shuffle(X,axis=1,inplace=False))
	# X = X + shuffle(X,axis=1,inplace=False)


	args = []
	kwargs = {
		# 'estimator':'LassoCV',
		# 'estimator':'RidgeCV',
		# 'loss_func':'weighted',
		# 'score_func':'weighted',
		# 'criteria_func':None,
		# 'normalize':'l2',
		# 'fit_intercept':False,
		'n_jobs':1,
		# 'stats':{},
		# 'max_iter':10000,
		# 'tol':1e-8,
		'cv':{'cv':'UniformFolder','n_splits':1,'random_state':1235,'test_size':0.3},
		# 'cv':{'cv':'KFolder','n_splits':n,'random_state':1235,'test_size':0.3},
		# 'cv':{'cv':'KFold','n_splits':n,'random_state':1235,'test_size':0.3},
		'alphas':np.logspace(-12,-5,100),
		'alpha_zero':True,
		# # 'loss_weights':{'l2':0.5,'l1':0.5},
		# 'prioritize':True,
		'loss_func':'weighted',	
		'score_weights':{'l2':1.0,},
		'loss_weights':{'l2':1.0,},
		'score_l2_inds':slice(None),
		'loss_l2_inds':slice(None),
		"complexity_min": m, 
		'verbose':1,

		# 'estimator_kwargs':{
		# 	'loss_func':'weighted',
		# 	'loss_weights':{'l2':1.0,},
		# 	'alphas':np.logspace(-15,-1,5),
		# 	# 'loss_weights':{'l2':0.5,'l1':0.5},
		# 	'score_func':'weighted',
		# 	'score_weights':{'l2':1.0,},			
		# 	# 'score_weights':{'l2':0.5,'l1':0.5},			
		# 	'alpha_':1e-5,
		# 	'max_iter':10000,
		# 	'tol':1e-8,
		# 	'prioritize':True,
		# 	# 'cv':{'cv':'UniformFolder','n_splits':1,'random_state':1235,'test_size':0.3},
		# 	'cv':{'cv':'KFolder','n_splits':10,'random_state':1235,'test_size':0.3},

		# },
	}
	parameters = {
		'fig':None,
		'axes':None,
	}

	globs = globals()
	locs = locals()
	variables = {**globs,**locs}

	# _estimators = ['OLS','Ridge','Lasso']	
	# _estimators = ['OLS','Ridge','Lasso','RidgeCV','LassoCV']
	_estimators = ['RidgeCV']
	# _estimators = ['LassoCV']
	# _estimators = ['OLS']
	# _estimators = ['Stepwise']*1
	estimators = {}
	for i,name in enumerate(_estimators):
		for n_jobs in [1]:
			kwargs['estimator'] = {'Stepwise': ['OLS','RidgeCV','Ridge'][i+1],'RidgeCV':'Ridge','LassoCV':'Lasso'}.get(name)
			kwargs['n_jobs'] =  n_jobs
			estimators[name] = variables[name](*args,**kwargs)
			scale_X = estimators[name].get_normalize()(X)
			scale_y = estimators[name].get_normalize()(y)
			parameters['fig'],parameters['axes'] = estimators[name].plot(X,y,*args,**kwargs,**parameters)
