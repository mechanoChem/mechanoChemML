#!/usr/bin/env python

# Import python modules
import sys,os,glob,copy,itertools
from natsort import natsorted
import numpy as np
import pandas as pd


# Global Variables
DELIMITER='__'

# Import user modules
from .dictionary import _set,_get,_pop,_has,_update,_permute,_clone
from .load_dump import load,dump,path_split,path_join
from .graph_utilities import basis_size,ncombinations



# Logging
import logging,logging.handlers
log = 'info'

logger = logging.getLogger(__name__)
#logger.setLevel(getattr(logging,log.upper()))


def _identify(labels,values,prefix=None,postfix=None): 
	'''
	Get formatted identity string based on labels and values, with prefix and postfix amendments
	
	Args:
		labels (list): list of keys to get values for formatting into identity string
		values (dict): dictionary of values to be formatted with label keys in labels
		prefix (str): prefix string to be included at beginning of label
		postfix (str): postfix string to be included at end of label
	
	Returns:
		String with DELIMITER separated label_value strings
	'''
	def isnumber(string):
		try:
			string = int(number)
			string = '%d'%(string)
		except:
			try:
				string = float(number)
				string = '%0.1f'%(string)
			except:
				pass
		return string

	def stringify(obj,exceptions=[]):
		
		if isinstance(obj,dict):
			obj = '_'.join(['_'.join([k,str(obj[k])]) for k in obj])

		string = isnumber(str(obj))
		checks = {**{s:'_' for s in [os.path.sep]},
				  **{s:'-' for s in [', ']},
				  **{s:'_' for s in ['.']},
				  **{s:'' for s in ['[',']','(',')']},
				  **{s:'-' for s in [' ']}
				  }
		for e in exceptions:
			checks.pop(e);
		for s in checks:
			if not string.endswith(s):
				string = string.replace(s,checks[s])
			else:
				string = string[:-len(s)].replace(s,checks[s])
		return string

	delimeter=DELIMITER
	setter='_'
	ammendments = ['%s'%(stringify(a,exceptions=(['.'] if i==1 else [])) if a is not None else '') 
					for i,a in enumerate([prefix,postfix])]
	body = (delimeter.join(['%%s%s%%s'%(setter)]*(len(labels))))%(
				sum(zip([stringify(l[-1] if isinstance(l,list) else l.split(DELIMITER)[-1]) for l in labels],
						[stringify(_get(values,l,_split=DELIMITER)) for l in labels]),()))

	strings = [ammendments[0],body,ammendments[1]]
	strings = [string for string in strings if len(string)>0]
	if len(strings) > 1:
		strings = [*strings[:1],('' if strings[-1].startswith('.') else delimeter).join(strings[1:])]

	if len(strings) > 1:
		string = ('' if strings[-1].startswith('.') else delimeter).join(strings)
	else:
		string = delimeter.join(strings)

	return string


 # Default settings, can be overwritten in the main.py of the data folder
def _get_settings():
	'''
	Fixed default settings for graph theory library

	Returns:
		Dictionary of fixed default settings
	'''
	settings_grid = _permute({
		# System settings, inputs and outputs
		'sys__verbose':[True],		
		'sys__directories__cwd__load':['.'],
		'sys__directories__cwd__dump':[None],
		'sys__directories__src__load':[None],
		'sys__directories__src__dump':[None],		
		'sys__directories__directories__load':[['']],
		'sys__directories__directories__dump':[['']],
		
		'sys__files__files':[['data.csv']],
		'sys__files__settings':['settings.json'],

		'sys__ext__files':['csv'],
		'sys__ext__data':['csv'],
		'sys__ext__metadata':['pickle'],
		'sys__ext__plot':['pdf'],
		'sys__ext__settings':['json'],
		'sys__ext__model':['tex'],
		'sys__ext__analysis':['pickle'],
		'sys__ext__log':['log'],
		'sys__ext__mplstyle':['mplstyle'],

		'sys__read__files':['r'],
		'sys__read__data':['rb'],
		'sys__read__metadata':['rb'],
		'sys__read__plot':['r'],
		'sys__read__settings':['r'],
		'sys__read__model':['r'],
		'sys__read__analysis':['rb'],
		'sys__read__log':['r'],
		'sys__read__mplstyle':['r'],

		'sys__write__files':['w'],
		'sys__write__data':['w'],
		'sys__write__metadata':['wb'],
		'sys__write__plot':['w'],
		'sys__write__settings':['w'],
		'sys__write__model':['w'],
		'sys__write__analysis':['wb'],
		'sys__write__log':['w'],
		'sys__write__mplstyle':['w'],

		'sys__kwargs__load':[{}],
		'sys__kwargs__dump':[{}],
		
		'sys__label':[""],

		# Labels for output filenames
		'sys__labels':[['sys__label','model__basis','model__order']],	

		# Booleans to turn on/off code functionality
		'boolean__verbose':[True],		
		'boolean__load':[0],
		'boolean__dump':[1],
		'boolean__fit':[1],
		'boolean__plot':[1],
		'boolean__structure':[1],
		'boolean__terms':[1],
		'boolean__model':[1],
		'boolean__log': [1],
		'boolean__texify': [1],		
		'boolean__analysis':[1],
		'boolean__drop_duplicates':[0],					

		# Options for defining graph structure
		'structure__verbose':[True],
		'structure__seed':[0],
		'structure__index':[None],							
		'structure__directed':[True],
		'structure__label':['test'],
		'structure__conditions':[None],
		'structure__samples':[None],
		'structure__filters':[None],
		'structure__functions':[None],
		'structure__symmetries':[None],
		'structure__refinement':[None],
		'structure__scale':[None],
		'structure__rename':[None],
		'structure__drop_duplicates':[None],
		'structure__round':[None],
		'structure__groupby_filter':[None],
		'structure__groupby_equal':[None],
		'structure__kwargs':[{}],

		# Terms for model fitting
		'terms__verbose':[True],		
		'terms__inputs':[None],
		'terms__outputs':[None],
		'terms__constants':[None],
		'terms__orders':[None],
		'terms__weights':[None],
		'terms__adjacencies':[None],
		'terms__operations':[None],
		'terms__terms':[None],
		'terms__parameters':[{}],
		'terms__functions':[None],
		'terms__kwargs':[{}],
		
		# Operator basis used for fitting
		'model__verbose':[True],		
		'model__inputs':[['%s%d'%(x,i) for x in ['x'] for i in range(5)]],
		'model__outputs':[['%s%d'%(x,i) for x in ['y'] for i in range(3)]],	
		'model__constants':[None],	
		'model__basis':['taylorseries'],
		'model__order':[None],
		'model__size':[None],
		'model__p':[None],
		'model__iloc':[[0]],
		'model__accuracy':[None],
		'model__weights':[['stencil']],
		'model__adjacency':[['nearest']],
		'model__operations':[['partial']],
		'model__stencil':['vandermonde'],
		'model__metric':[2],
		'model__strict':[False],
		'model__format':['csr'],
		'model__dtype':['float'],
		'model__catch_iterations':[5],
		'model__catch_factor':[2],
		'model__chunk':[0.2],
		'model__unique':[True],
		'model__tol':[None],
		'model__atol':[None],
		'model__rtol':[None],
		'model__kind':['mergesort'],
		'model__n_jobs':[1],
		'model__samples':[None],
		'model__parameter':[None],
		'model__store':[{}],
		'model__normalization':['l2'],
		'model__scheme':['None'],
		'model__approach':['indiv'],
		'model__dtype':['float64'],
		'model__type':['function'],
		'model__selection':[None],
		'model__intercept_':[False],
		'model__expand':[False],
		'model__lhs':[{}],
		'model__rhs':[{}],
		'model__rhs_lhs':[{}],
		'model__kwargs':[{}],

		  
		# Regression operations and settings
		'fit__verbose':[True],		
		'fit__estimator':['Stepwise'],
		'fit__info':[None],	
		'fit__types':[['fit','predict','update','method','interpolate']],
		'fit__kwargs':[{}],
		'fit__kwargs__estimator':['OLS'],
		'fit__kwargs__loss_func':['rmse'],
		'fit__kwargs__score_func':['rmse'],
		'fit__kwargs__fit_intercept':[False],
		'fit__kwargs__njobs':[1],
		'fit__kwargs__parallel':[None],
		'fit__kwargs__method':['cheapest'],
		'fit__kwargs__threshold':[1e20],#1e-1
		'fit__kwargs__included':[None],
		'fit__kwargs__iterations':[[]],
		'fit__kwargs__complexity_max':[None],
		'fit__kwargs__complexity_min':[None],


		# Model Analysis
		'analysis__verbose':[True],
		'analysis__kwargs':[None],

		# Plots the results of the fits
		'plot__verbose':[True],
		'plot__names': [{
			**{
				'Loss':r'\textrm{Stepwise Loss}',
				'BestFit':r'\textrm{Fits}',				
				'Coef':r'\gamma ~\textrm{Coefficients}',
				'Error':r'\textrm{Model Error}',				
			},
			**{k: r'\textrm{%s}'%(k) 
				for k in ['Variables','Operators','Terms']},
			}],						
		'plot__groups':[{'fit':['Loss','BestFit','Coef','Error'],'plot':['Variables','Operators','Terms']}],
		'texify__usetex':[1],
		'texify__texargs':[{}],
		'texify__texstrings':[{}],
		})  
	return settings_grid[0]


# Settings that will be affected by the main input settings in the settings_grid
def _get_settings_dependent(settings,_keep=False):
	'''
	Settings dependent default settings for graph theory library

	Args:
		settings (dict): dictionary of settings to set settings-dependent default values
		_keep (bool): boolean of whether to keep original settings that may be altered during settings-dependent setting of settings

	Returns:
		Dictionary of settings with settings-dependent settings included
	'''
	settings_dependent = {

		**{'%s__verbose'%(k): (lambda settings: (
			{'notset':0,'debug':10,'info':20,'warning':30,'error':40,'critical':50}[settings['%s'%(k)].get('verbose').lower()] if (
			isinstance(settings['%s'%(k)].get('verbose'),str) and settings['boolean']['verbose']) else (
			{**{i:i for i in [10,20,30,40,50]},**{i:10*i for i in [2,3,4,5]},True:20,False:False}[settings['%s'%(k)].get('verbose')] if (				
			isinstance(settings['%s'%(k)].get('verbose'),(bool,int)) and settings['boolean']['verbose']) else (
			False))
			)) for k in ['structure','terms','model','fit','analysis','plot']},
		**{'%s__verbose'%(k): (lambda settings: (
			{'notset':0,'debug':10,'info':20,'warning':30,'error':40,'critical':50}[settings['%s'%(k)].get('verbose').lower()] if (
			isinstance(settings['%s'%(k)].get('verbose'),str) and settings['boolean']['verbose']) else (
			{**{i:i for i in [10,20,30,40,50]},**{i:10*i for i in [2,3,4,5]},True:20,False:False}[settings['%s'%(k)].get('verbose')] if (
			isinstance(settings['%s'%(k)].get('verbose'),(bool,int)) and settings['boolean']['verbose']) else (
			20 if settings['boolean']['verbose'] else False))
			)) for k in ['sys']},
		
		'sys__directories__cwd__load': lambda settings: (settings['sys']['directories']['cwd'].get('dump') if settings['boolean']['load'] or settings['sys']['directories']['cwd'].get('load') is None else settings['sys']['directories']['cwd'].get('load')),
		'sys__directories__cwd__dump': lambda settings: (settings['sys']['directories']['cwd'].get('load') if settings['sys']['directories']['cwd'].get('dump') is None else settings['sys']['directories']['cwd'].get('dump')),
		'sys__directories__src__dump': lambda settings: (settings['sys']['directories']['src'].get('dump') if (settings['sys']['directories']['src'].get('dump') is not None) 
														else settings['sys']['directories']['src'].get('load') if (settings['sys']['directories']['src'].get('load') is not None) 
														else (settings['sys']['directories']['cwd'].get('dump')) if (settings['sys']['directories']['cwd'].get('dump') is not None) 
														else (settings['sys']['directories']['cwd'].get('load'))),

		'sys__directories__src__load': lambda settings: (settings['sys']['directories']['src'].get('load') if (settings['sys']['directories']['src'].get('load') is not None) 
														else settings['sys']['directories']['src'].get('dump') if (settings['sys']['directories']['src'].get('dump') is not None) 
														else (settings['sys']['directories']['cwd'].get('load')) if (settings['sys']['directories']['cwd'].get('load') is not None) 
														else (settings['sys']['directories']['cwd'].get('dump'))),
		'sys__directories__directories__load': lambda settings: ([(path_join(settings['sys']['directories']['cwd']['load'],k) if not k.startswith(settings['sys']['directories']['cwd']['load']) else k) 
																	for k in (settings['sys']['directories']['directories']['load'] if not settings['boolean']['load'] else settings['sys']['directories']['directories']['dump'])]), 
		'sys__directories__directories__dump': lambda settings: ([(path_join(settings['sys']['directories']['cwd']['dump'],k) if not k.startswith(settings['sys']['directories']['cwd']['dump']) else k) 
																	for k in settings['sys']['directories']['directories']['dump']]),
		'sys__files__settings': lambda settings: (path_join(settings['sys']['directories']['src']['load'],settings['sys']['files']['settings']) if not settings['sys']['files']['settings'].startswith(settings['sys']['directories']['src']['load']) else settings['sys']['files']['settings']),

		'sys__read__files': lambda settings: (settings['sys']['read'].get('data','rb') if settings['boolean']['load'] else settings['sys']['read'].get('files','r')),


		**{'boolean__%s'%k: (lambda settings,k=k: ((not settings['boolean']['load']) and (settings['boolean'][k]))) for k in ['terms','fit']},


		'structure__seed': lambda settings: [settings['structure'].get('seed',0),np.random.seed(settings['structure'].get('seed',0))][0],
		'structure__drop_duplicates': lambda settings: (settings['structure'].get('drop_duplicates',{'drop':settings['model']['inputs']}) if settings['structure'].get('drop_duplicates') is not None else None),
		
		'structure__round': lambda settings: (settings['structure'].get('round',{'decimals':8,'labels':settings['model']['inputs']}) if settings['structure'].get('round') is not None else None),
		
		'structure__groupby_filter': lambda settings: (settings['structure'].get('groupby_filter',
										{
										# 'by':settings['model']['parameter'],
										'by':'BURNUP',
										'func':[i for i in settings['model']['inputs'] if i!=settings['model']['parameter']]
										}) if settings['structure'].get('groupby') is not None else None),
		
		
		'structure__groupby_equal': lambda settings: (settings['structure'].get('groubpy_equal',
															{'by':settings['model']['parameter'],'iloc':0,
															 'labels':[i for i in settings['model']['inputs'] if i!=settings['model']['parameter']]
															}) if settings['structure'].get('groupby') is not None else None),
		
		'structure__scale': lambda settings: (settings['structure'].get('scale',
															{'labels':[*settings['model']['inputs'],*settings['model']['outputs']]
															}) if settings['structure'].get('scale') is not None else None),

		'structure__refinement': lambda settings: (settings['structure'].get('refinement',{'base':2,'n':None,'p':None,'powers':[1,2,1],'settings':[],'keep':False}) if settings['structure'].get('refinement') is not None else None),


		'model__p': lambda settings: len(settings['model'].get('inputs',[])) if settings['model'].get('p') is None else settings['model']['p'],
		'model__manifold': lambda settings: settings['model'].get('inputs',[]) if settings['model'].get('manifold') is None else settings['model']['manifold'],
		'model__accuracy': lambda settings: settings['model']['accuracy'] if settings['model'].get('accuracy') is not None else settings['model']['order']+1,
		'model__neighbourhood': lambda settings: (settings['model']['neighbourhood'] if settings['model'].get('neighbourhood') is not None else {
					'vandermonde':ncombinations(settings['model']['p'],settings['model']['accuracy'],unique=settings['model']['unique'])-1
					}[settings['model']['stencil']]),
		'model__sparsity': lambda settings: (settings['model']['sparsity'] if settings['model'].get('sparsity') is not None else settings['model']['neighbourhood']*settings['model']['p']*settings['model']['order']),
		'model__adjacency': lambda settings: (settings['model']['adjacency'] if settings['model'].get('adjacency') is not None else None),

		# iloc can be passed as one of five types:
		# model fitting uses reference dataset df0 and given dataset df to form model in graph_fit and graph_models
		# 1) int : specific location in df0 to expand model and fit with data in df
		# 2) list[int]: several specific locations in df0 to expand model and fit with data in df
		# 3) float: proportion of locations in df0 to expand number, chooses randomly
		# 3) True: all allowed locations in ilocs of df0 
		# 4) None: Find closest point in ilocs of df0 for each point in df
		# 5) dict: dictionary with keys of datasets for different settings, and values of one of 1) to 4) iloc types above
		# Ensure iloc is either dictionary of {dataset:value} value, where value is 1) to 4) of above
		# If iloc type is an int,float,True then in graph_models, iloc parameter is preprocessed into list, depending on specific type and size of df0
		# If iloc is None, then kept as None and processed in graph_models when forming specific model/finding nearest points for fitting

		'model__iloc': lambda settings: (settings['model'].get('iloc',[0]) if not isinstance(settings['model'].get('iloc',[0]),dict) else (
										{k: settings['model']['iloc'][k] if not any([k in (settings['fit'].get('info',{}).get(t,{}) 
																					if settings['fit'].get('info',{}) is not None else {}) 
																			 		for t in settings['fit'].get('types',[]) if t in ['interpolate']])
																else None if (
																	not any([k in (settings['fit'].get('info',{}).get(t,{}) 
																			if settings['fit'].get('info',{}) is not None else {}) 
																			for t in settings['fit'].get('types',[]) if t not in ['interpolate']])) 
																else settings['model']['iloc'][k] 
																		if isinstance(settings['model']['iloc'][k],(list)) else (
																	settings['model']['iloc'][k])
																for k in settings['model']['iloc']})),

		'model__expand': lambda settings: settings['model'].get('expand',False) or settings['model']['basis'] in ['taylorseries'],
		'model__transform__features': lambda settings: {
								**{
								  **{k:{'order':None,'features':None,
										'dim':None,'intercept_':False,'samples':None,
								  		'iterate':False}
									 for k in [None,'taylorseries','default',
									 		   'linear','monomial','polynomial',
									 		   'chebyshev','legendre','hermite']},
								 }.get(settings['model']['basis'],{}),
								 **settings['model'].get('transform',{}).get('features',{})},
		'model__transform__normalization': lambda settings: {**{k:{'norm_func':k,'axis':0} for k in [None,'l2','uniform']}.get(settings['model']['normalization'],{}),
															 **settings['model'].get('transform',{}).get('normalization',{})},
		'model__transform__scheme': lambda settings: {**{str(k): {'scheme':k,'method':v} for k,v in {None:None,'euler':None}.items()}.get(settings['model']['scheme'],{}),
													  **settings['model'].get('transform',{}).get('scheme',{})},
		'model__transform__dtype': lambda settings: settings['model'].get('transform',{}).get('dtype',{str(k):k for k in ['float64']}.get(settings['model']['dtype'])),
		# 'model__lhs__order': lambda settings: (settings['model'].get('lhs',{}).get('order',settings['model']['order'])),
		# 'model__lhs__label': lambda settings: (settings['model'].get('lhs',{}).get('label',settings['model']['outputs'])),
		# 'model__lhs__function': lambda settings: (settings['model'].get('lhs',{}).get('function',settings['model']['outputs'])),
		# 'model__lhs__variable': lambda settings: (settings['model'].get('lhs',{}).get('variable',settings['model']['inputs'])),
		# 'model__lhs__operations': lambda settings: (settings['model'].get('lhs',{}).get('operations',settings['terms']['operations'])),
		# 'model__rhs__order': lambda settings: (settings['model'].get('rhs',{}).get('order',settings['model']['order'])),
		# 'model__rhs__label': lambda settings: (settings['model'].get('rhs',{}).get('label',settings['model']['outputs'])),
		# 'model__rhs__function': lambda settings: (settings['model'].get('rhs',{}).get('function',settings['model']['outputs'])),
		# 'model__rhs__variable': lambda settings: (settings['model'].get('rhs',{}).get('variable',settings['model']['inputs'])),
		# 'model__rhs__operations': lambda settings: (settings['model'].get('rhs',{}).get('operations',settings['terms']['operations'])),



		'terms__inputs': lambda settings: settings['model']['inputs'] if  settings['terms'].get('inputs') is None else settings['terms']['inputs'],
		'terms__outputs': lambda settings: settings['model']['outputs'] if  settings['terms'].get('outputs') is None else settings['terms']['outputs'],
		'terms__constants': lambda settings: settings['model']['constants'] if  settings['terms'].get('constants') is None else settings['terms']['constants'],
		'terms__orders': lambda settings: (list(range(settings['model']['order']+1)) if settings['terms'].get('orders') is None else list(range(settings['terms']['orders']+1)) if isinstance(settings['terms']['orders'],(int,np.integer)) else settings['terms']['orders']),
		'terms__accuracies': lambda settings: ([settings['model']['order']+1] if settings['terms'].get('accuracies',settings['model'].get('accuracies',settings['model'].get('accuracy',settings['model']['order']+1))) is None else [settings['terms'].get('accuracies',settings['model'].get('accuracies',settings['model'].get('accuracy',settings['model']['order']+1)))]
		 										if isinstance(settings['terms'].get('accuracies',settings['model'].get('accuracies',settings['model'].get('accuracy',settings['model']['order']+1))),(int,np.integer)) else settings['terms'].get('accuracies',settings['model'].get('accuracies',[settings['model'].get('accuracy',settings['model']['order']+1)]))),
		'terms__parameters': lambda settings: ({
								**settings['model'],
								**(settings['terms'].get('parameters') if settings['terms'].get('parameters') is not None else {}),
								}),		
		'terms__functions': lambda settings: ([
								*(settings['structure'].get('functions') if settings['structure'].get('functions') is not None else []),
								*(settings['terms'].get('functions') if settings['terms'].get('functions') is not None else []),
								]),
		'terms__weights': lambda settings: settings['model'].get('weights',[settings['model'].get('weight')]) if settings['terms'].get('weights') is None else settings['terms']['weights'],		
		'terms__adjacencies': lambda settings: settings['model'].get('adjacencies',[settings['model'].get('adjacency')]) if settings['terms'].get('adjacencies') is None else settings['terms']['adjacencies'],		
		'terms__operations': lambda settings: settings['model']['operations'] if  settings['terms'].get('operations') is None else settings['terms']['operations'],		
		'terms__unique': lambda settings: settings['model']['unique'] if  settings['terms'].get('unique') is None else settings['terms']['unique'],		
		'fit__kwargs__method': lambda settings: (settings['fit']['kwargs'].get('method','cheapest') if settings['fit']['estimator'] == 'Stepwise' else None),
		'fit__kwargs__included': lambda settings: ([*(settings['fit']['kwargs'].get('included',[]) if settings['fit']['kwargs'].get('included') is not None else [])]),
		'fit__kwargs__fixed': lambda settings: ({**({0: 1} if settings['model']['expand'] else {}),**settings['fit']['kwargs'].get('fixed',{})}),


		'plot__fig': lambda settings: ({k:settings['plot'].get('fig',{}).get(k,{}) for k in settings['plot'].get('fig',settings['plot']['names'])}),
		'plot__axes': lambda settings: ({k:settings['plot'].get('axes',{}).get(k,{}) for k in settings['plot']['fig']}),
		'plot__names': lambda settings: ({k: settings['plot'].get('names',{}).get(k,k)
										 for k in settings['plot']['fig']}),	
		
		'plot__dims': lambda settings: (settings['plot'].get('dims',slice(None))),
		'plot__rescale': lambda settings: ({k:settings['plot'].get('rescale',{}).get(k,False) if isinstance(settings['plot'].get('rescale'),dict) else settings['plot'].get('rescale',False)
							  				for k in settings['plot'].get('fig',settings['plot']['names'])}),
		'plot__retain__fig': lambda settings: ({k:settings['plot'].get('retain',{}).get('fig',{}).get(k,False) for k in settings['plot']['fig']}),
		'plot__retain__axes': lambda settings: ({k:settings['plot'].get('retain',{}).get('axes',{}).get(k,False) for k in settings['plot']['fig']}),
		'plot__retain__label': lambda settings: ({k:settings['plot'].get('retain',{}).get('label',{}).get(k,False) for k in settings['plot']['fig']}),
		'plot__retain__dim': lambda settings: ({k:settings['plot'].get('retain',{}).get('dim',{}).get(k,False) for k in settings['plot']['fig']}),
		'plot__retain__key': lambda settings: ({k:settings['plot'].get('retain',{}).get('key',{}).get(k,False) for k in settings['plot']['fig']}),

		'plot__file': lambda settings: (settings['plot'].get('file',{k: path_join(settings['sys']['directories']['src']['load'],'plot',ext=settings['sys']['ext']['settings'])
					 	for k in settings['plot']['fig']}) if isinstance(settings['plot'].get('file'),dict) else (
					 	{k: settings['plot'].get('file',path_join(settings['sys']['directories']['src']['load'],'plot',ext=settings['sys']['ext']['settings']))
					 	for k in settings['plot']['fig']})),
		'plot__mplstyle': lambda settings: (settings['plot'].get('mplstyle',{k: path_join(settings['sys']['directories']['src']['load'],'plot' if settings['texify'].get('usetex') else 'plot_notex',ext=settings['sys']['ext']['mplstyle'])
					 	for k in settings['plot']['fig']}) if isinstance(settings['plot'].get('mplstyle'),dict) else (
					 	{k: settings['plot'].get('mplstyle',path_join(settings['sys']['directories']['src']['load'],'plot' if settings['texify'].get('usetex') else 'plot_notex',ext=settings['sys']['ext']['mplstyle']))
					 	for k in settings['plot']['fig']})),					 		
		'plot__settings': lambda settings: ({k:{
							**{name:{
								'style':{'layout':{'nrows':2,'ncols':5},
										**settings['plot'].get('settings',{}).get(name,{}).get('style',{})},
								'other':{'fontsize':45,'constant':[],
										**settings['plot'].get('settings',{}).get(name,{}).get('other',{})},
								**{k: settings['plot'].get('settings',{}) .get(name,{})[k] for k in settings['plot'].get('settings',{}).get(name,{}) if k not in ['style','other']}}
								for name in ['Coef']},
							**{name:{
								'other':{
									'iterations': [None,100,30,20,10,5,3,1],
									'x':settings['model']['inputs'][:1],
									'fontsize':55,
									'constant':[],
									'data':True,
									'sort':False,
									**settings['plot'].get('settings',{}).get(name,{}).get('other',{})} ,
								**{k: settings['plot'].get('settings',{}).get(name,{})[k] for k in settings['plot'].get('settings',{}).get(name,{}) if k not in ['other']}}
								for name in ['BestFit','Error']},
							**{name:{
							 	'other':{'fontsize':100,'constant':[],**settings['plot'].get('settings',{}).get(name,{}).get('other',{})},
								**{k: settings['plot'].get('settings',{}) .get(name,{})[k] for k in settings['plot'].get('settings',{}).get(name,{}) if k not in ['other']}}
								for name in ['Loss']},
							**{name:{
								'other':{'subplots':True,
										 'x':[],
										 'y':[],
										 'fontsize':32,
										 'terms': [{'function':settings['model']['outputs'],
												    'variable':settings['model']['inputs']}],
										 'constant':[],
										  **settings['plot'].get('settings',{}).get(name,{}).get('other',{})},
	 							'style':{'layout':{
									'nrows':min(2,max(1,min(len(settings['plot'].get('settings',{}).get(name,{}).get('other',{}).get('x',[])),
		 											      len(settings['plot'].get('settings',{}).get(name,{}).get('other',{}).get('y',[]))))
		 										if settings['plot'].get('settings',{}).get(name,{}).get('other',{}).get('subplots') else 1),
		 							'ncols':min(5,max(1,min(len(settings['plot'].get('settings',{}).get(name,{}).get('other',{}).get('x',[])),
		 											      len(settings['plot'].get('settings',{}).get(name,{}).get('other',{}).get('y',[]))))
		 										if settings['plot'].get('settings',{}).get(name,{}).get('other',{}).get('subplots') else 1)},
		 							**settings['plot'].get('settings',{}).get(name,{}).get('style',{})},
								**{k: settings['plot'].get('settings',{}).get(name,{})[k] for k in settings['plot'].get('settings',{}).get(name,{}) if k not in ['other','style']}}
								for name in ['Variables']},
							**{name:{
								'other':{'subplots':True,
										 'x':[],
										 'y':[],
										 'fontsize':32,
										 'terms': [{'function':settings['model']['outputs'],
										 		    'variable':settings['model']['inputs']}],
										  'constant':[],
										  **settings['plot'].get('settings',{}).get(name,{}).get('other',{}),										  
										},
	 							'style':{'layout':{
									'nrows':min(4,max(1,min(len(settings['plot'].get('settings',{}).get(name,{}).get('other',{}).get('x',[])),
		 											      len(settings['plot'].get('settings',{}).get(name,{}).get('other',{}).get('y',[]))))
		 										if settings['plot'].get('settings',{}).get(name,{}).get('other',{}).get('subplots') else 1),
		 							'ncols':min(8,max(1,min(len(settings['plot'].get('settings',{}).get(name,{}).get('other',{}).get('x',[])),
		 											      len(settings['plot'].get('settings',{}).get(name,{}).get('other',{}).get('y',[]))))
		 										if settings['plot'].get('settings',{}).get(name,{}).get('other',{}).get('subplots') else 1)},
		 							**settings['plot'].get('settings',{}).get(name,{}).get('style',{})},
								**{k: settings['plot'].get('settings',{}).get(name,{})[k] for k in settings['plot'].get('settings',{}).get(name,{}) if k not in ['other','style']}}
								for name in ['Operators','Terms']},
							}[k] 
						 for k in settings['plot']['fig']}  if ((settings['plot'].get('settings') is None) or any([k in settings['plot'].get('settings',{}) or k in settings['plot']['names'] 
						 for k in settings['plot']['fig']])) else (
						 {l: {k:{
							**{name:{'style':{'layout':{'nrows':2,'ncols':5},
										**settings['plot'].get('settings',{}).get(l,{}).get(name,{}).get('style',{})},
									'other':{'fontsize':45,'constant':[],
											**settings['plot'].get('settings',{}).get(l,{}).get(name,{}).get('other',{})},
									**{k: settings['plot'].get('settings',{}).get(l,{}).get(name,{})[k] for k in settings['plot'].get('settings',{}).get(l,{}).get(name,{}) if k not in ['style','other']}}
								for name in ['Coef']},									
							**{name:{
								'other':{
									'iterations': [None,100,30,20,10,5,3,1],
									'x':settings['model']['inputs'][:1],
									'fontsize':55,
									'constant':[],
									'data': True,
									'sort':False,
									**settings['plot'].get('settings',{}).get(l,{}).get(name,{}).get('other',{})} ,
								**{k: settings['plot'].get('settings',{}).get(l,{}).get(name,{})[k] for k in settings['plot'].get('settings',{}).get(l,{}).get(name,{}) if k not in ['other']}}
								for name in ['BestFit','Error']},
							**{name:{
							 	'other':{
							 		'fontsize':100,'constant':[],
							 		**settings['plot'].get('settings',{}).get(l,{}).get(name,{}).get('other',{})},
								**{k: settings['plot'].get('settings',{}).get(l,{}).get(name,{})[k] for k in settings['plot'].get('settings',{}).get(l,{}).get(name,{}) if k not in ['other']}}
								for name in ['Loss']},								
							**{name:{
								'other':{'subplots':True,
										'x':[],
										'y':[],
	 								    'fontsize':32,
										'terms': [{'function':settings['model']['outputs'],
												    'variable':settings['model']['inputs']}],
										'constant':[],
										  **settings['plot'].get('settings',{}).get(name,{}).get('other',{}),	
										}, 								    
	 							'style':{'layout':{
									'nrows':min(2,max(1,min(len(settings['plot'].get('settings',{}).get(l,{}).get(name,{}).get('other',{}).get('x',[])),
		 											        len(settings['plot'].get('settings',{}).get(l,{}).get(name,{}).get('other',{}).get('y',[]))))
		 										if settings['plot'].get('settings',{}).get(l,{}).get(name,{}).get('other',{}).get('subplots') else 1),
		 							'ncols':min(5,max(1,min(len(settings['plot'].get('settings',{}).get(l,{}).get(name,{}).get('other',{}).get('x',[])),
		 											        len(settings['plot'].get('settings',{}).get(l,{}).get(name,{}).get('other',{}).get('y',[]))))
		 										if settings['plot'].get('settings',{}).get(l,{}).get(name,{}).get('other',{}).get('subplots') else 1)},
		 							**settings['plot'].get('settings',{}).get(l,{}).get(name,{}).get('style',{})},
								**{k: settings['plot'].get('settings',{}).get(l,{}).get(name,{})[k] 
									for k in settings['plot'].get('settings',{}).get(l,{}).get(name,{}) if k not in ['other','style']}}
								for name in ['Variables']},
							**{name:{
								'other':{'subplots':True,
										'x':[],
										'y':[],
	 								    'fontsize':32,'constant':[],**settings['plot'].get('settings',{}).get(l,{}).get(name,{}).get('other',{})},
	 							'style':{'layout':{
									'nrows':min(4,max(1,min(len(settings['plot'].get('settings',{}).get(l,{}).get(name,{}).get('other',{}).get('x',[])),
		 											        len(settings['plot'].get('settings',{}).get(l,{}).get(name,{}).get('other',{}).get('y',[]))))
		 										if settings['plot'].get('settings',{}).get(l,{}).get(name,{}).get('other',{}).get('subplots') else 1),
		 							'ncols':min(8,max(1,min(len(settings['plot'].get('settings',{}).get(l,{}).get(name,{}).get('other',{}).get('x',[])),
		 											        len(settings['plot'].get('settings',{}).get(l,{}).get(name,{}).get('other',{}).get('y',[]))))
		 										if settings['plot'].get('settings',{}).get(l,{}).get(name,{}).get('other',{}).get('subplots') else 1)},
		 							**settings['plot'].get('settings',{}).get(l,{}).get(name,{}).get('style',{})},
								**{k: settings['plot'].get('settings',{}).get(l,{}).get(name,{})[k] 
									for k in settings['plot'].get('settings',{}).get(l,{}).get(name,{}) if k not in ['other','style']}}
								for name in ['Operators','Terms']},								
							}[k] 
						 for k in settings['plot']['fig']}
						 for l in settings['plot'].get('settings',{})})),
		'texify__texargs__order': lambda settings: (settings['model']['order'] if settings['model']['basis'] in ['taylorseries','default',None,'polynomial','monomial'] else settings['texify'].get('texargs',{}).get('order')),
		'texify__texargs__basis': lambda settings: (settings['model']['order'] if settings['model']['basis'] in [None,'default','linear','monomial','polynomial','chebyshev','legendre','hermite']  else settings['texify'].get('texargs',{}).get('order')),
		'texify__texargs__iloc': lambda settings: (settings['texify'].get('texargs',{}).get('iloc',settings['model']['iloc'])),
		'texify__texargs__unique': lambda settings: (settings['texify'].get('texargs',{}).get('unique',settings['model']['unique'])),
		'texify__texargs__weights': lambda settings: (settings['texify'].get('texargs',{}).get('weights',settings['model'].get('weights',['diff','frobenius','gauss','poly','decay','stencil']))),
		'texify__texargs__inputs': lambda settings: (settings['texify'].get('texargs',{}).get('inputs',{'%s'%(x):r'{%s_{%d}}'%(u,i) for u in ['x'] for i,x in enumerate(settings['model']['inputs'])})),
		'texify__texargs__outputs': lambda settings: (settings['texify'].get('texargs',{}).get('outputs',{'%s'%(x):r'{%s_{%d}}'%(u,i) for u in ['y'] for i,x in enumerate(settings['model']['outputs'])})),
		'texify__texargs__groups': lambda settings: (settings['texify'].get('texargs',{}).get('groups',({**{'%s'%(x):r'{%s}'%(x) for u in ['x'] for i,x in enumerate(settings['model']['inputs'])}}))),
		'texify__texargs__constants': lambda settings: settings['model'].get('constants',{}),
		'texify__texargs__bases': lambda settings: settings['texify'].get('texargs',{}).get('bases',{None:1,'default':1,'monomial':1,'polynomial':1,'taylorseries':1,'derivative':1,
																								   'chebyshev':1,'legendre':1,'hermite':1}),
		'texify__texstrings': lambda settings: ({**settings['texify'].get('texstrings',{})}),
		'sys__label': lambda settings: (settings['sys']['labeler'](settings) if callable(settings['sys'].get('labeler')) else settings['sys'].get('label','template')),
		'sys__identity': lambda settings: _identify(settings['sys']['labels'],settings),
		**{'sys__files__%s'%(k):(lambda settings,k=k: (settings['sys'].get('files',{}).get(k) if (not settings['boolean']['load']) else (
						[_identify(settings['sys']['labels'],settings,'data','.%s'%settings['sys']['ext']['data'])])))	
				for k in ['files']},
		**{'sys__files__%s'%(k): (lambda settings,k=k: (settings['sys'].get('files',{}).get(k,path_join('%s%s%s'%(k,DELIMITER if len(settings['sys']['identity'])>0 else '',settings['sys']['identity']),ext=settings['sys']['ext'][k]))
												if (not settings['boolean']['load']) else  (
										_identify(settings['sys']['labels'],settings,k,'.%s'%settings['sys']['ext'][k]))))
										for k in ['metadata','data']},									
		**{'sys__files__%s'%(k):(lambda settings,k=k: (settings['sys'].get('files',{}).get(k) if (settings['sys'].get('files',{}).get(k) is not None) else  (
										_identify(settings['sys']['labels'],settings,'%s%s%%s'%(k,DELIMITER),'.%s'%settings['sys']['ext'][k]))))
										for k in ['model']},
		**{'sys__files__%s'%(k):(lambda settings,k=k: (settings['sys'].get('files',{}).get(k) if (settings['sys'].get('files',{}).get(k) is not None) else  (
										_identify(settings['sys']['labels'],settings,'%s%s%%s'%(k,DELIMITER),'.%s'%settings['sys']['ext'][k]))))
										for k in ['analysis']},
		**{'sys__files__%s'%(k): (lambda settings,k=k: (settings['sys'].get('files',{}).get(k,path_join('%s%s%s'%(k,DELIMITER if len(settings['sys']['identity'])>0 else '',settings['sys']['identity']),ext=settings['sys']['ext'][k]))
												if (not settings['boolean']['load']) else  (
										_identify(settings['sys']['labels'],settings,k,'.%s'%settings['sys']['ext'][k]))))
										for k in ['log']},										
		**{'sys__files__%s'%(k):(lambda settings,k=k: (settings['sys'].get('files',{}).get(k) if (settings['sys'].get('files',{}).get(k) is not None) else  (
										_identify(settings['sys']['labels'],settings,'%s%s%%s%s%%s'%(k,DELIMITER,DELIMITER),'.%s'%settings['sys']['ext'][k]))))
										for k in ['plot']},		

		}



	_settings = {k: copy.deepcopy(settings_dependent[k]) if k in settings_dependent else settings[k] for k in settings}
	for key,value in settings_dependent.items():
		_set(_settings,key,value(_settings),_split=DELIMITER,_copy=False)
	
	if _keep:
		for key in copy.deepcopy(_settings):
			if _has(settings,key,_split=DELIMITER):
				_set(_settings,key,_get(settings,key,_split=DELIMITER),_split=DELIMITER,_copy=False)
	
	return _settings


def set_settings(settings,path='settings.json',_copy=False,_dump=False):
	''' 
	Set dictionary of settings

	Args:
		settings (dict,str): dictionary of settings, either with delimiter-separated strings or nested dictionary entries; or string path to dictionary to be loaded; to be modified in place
		path (str): string of path to save dictionary in JSON format (='settings.json')
		_copy (bool): boolean to copy data in settings dictionary (=False)
		_dump (bool): boolean to save dictionary to path (=False)
	'''	

	# Check if settings are passed as atring to be loaded as dictionary
	if not isinstance(settings,dict):
		settings = load(settings,default={})

	assert isinstance(settings,dict), "Settings not Dict"

	# Check if settings are present in path, in case of empty settings dictionary
	if settings == {}:
		settings.update(load(path,default={}))


	# Copy settings to _settings, to be modified before replacing settings
	_settings = {}
	_clone(settings,_settings)

	# Get keys of _settings and sort by place in nested structure (based on number of DELIMITER)
	keys = list(_settings)
	keys = sorted(keys,key=lambda key: (key,-len(key.split(DELIMITER))))

	# Replace DELIMITER separated keys with proper nested dictionary entries
	for key in keys:
		value = _pop(settings,key,_split=False,_copy=_copy)
		_set(settings,key,value,_split=DELIMITER,_copy=_copy)


	# Get fixed default settings
	_settings = _get_settings()

	# Set settings based on DELIMITER separated string keys in _settings
	for key,value in _settings.items():
		if not _has(settings,key,_split=DELIMITER):
			_set(settings,key,value,_split=DELIMITER,_copy=False,_reset=False)


	# Set settings-dependent settings
	for key,value in _get_settings_dependent(settings).items():
		_set(settings,key,value,_split=DELIMITER,_copy=False,_reset=False)

	# Save settings to path if _dump is True
	if _dump:
		dump(settings,path)
	return


def get_settings(settings,path='settings.json',_dump=False):
	'''	
	Get dictionary of settings

	Args:
		settings (dict,str): dictionary of settings, either with delimiter-separated strings or nested dictionary entries; or string path to dictionary to be loaded; to be modified in place
		path (str): string of path to save dictionary in JSON format (='settings.json')
		_dump (bool): boolean to save dictionary to path (=False)
	'''	


	assert isinstance(settings,dict), "Settings not Dict"

	if _dump:
		dump(settings,path)

	return settings


def permute_settings(settings,path='settings.json',_copy=False,_dump=False,_groups=None,_set_settings=True):
	""" 
	Permute dictionary of settings

	Args:
		settings (dict): dictionary of settings with values of lists of values to be permuted
		path (str): string of path to save dictionary in JSON format (='settings.json')
		_dump (bool): boolean to save dictionary to path (=False)
		_copy (bool): boolean to copy data in settings dictionary (=False)
		_groups (list): list of groups of dictionary keys to group when permuting
		_set_settings (bool): boolean to update settings with default values after permutations

	Returns:
		List of settings with all permutations in original settings
	"""	
	_path = lambda i,N,path=path: '%s%s.%s'%(path_split(path,file=True,directory_file=True),'%s%d'%(DELIMITER,i) if N>1 else '',path_split(path,ext=True))
	_settings = _permute(settings,_copy=_copy,_groups=_groups)



	if _set_settings:
		N = len(_settings)
		for i in range(N):
			set_settings(_settings[i],_path(i,N),_dump=_dump,_copy=_copy)
	return _settings



