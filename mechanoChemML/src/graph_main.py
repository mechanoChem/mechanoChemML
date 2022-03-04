#!/usr/bin/env python

# Import python modules
import sys,os,glob,copy,itertools
from natsort import natsorted
import numpy as np
import pandas as pd
import numexpr as ne

# Global Variables
DELIMITER='__'
MAX_PROCESSES = 7
PARALLEL = 0

ne.set_vml_num_threads(MAX_PROCESSES)

from .graph_settings import set_settings, permute_settings#,get_settings,

from .graph_functions import structure,terms,save#,analysis

from .texify import Texify#,scinotation

from .dictionary import _set,_get,_pop,_has,_update,_permute

from .load_dump import setup,path_join #, load,dump,path_split
	


# Logging
import logging,logging.handlers
log = 'info'

rootlogger = logging.getLogger()
rootlogger.setLevel(getattr(logging,log.upper()))
stdlogger = logging.StreamHandler(sys.stdout)
stdlogger.setLevel(getattr(logging,log.upper()))
rootlogger.addHandler(stdlogger)	


logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging,log.upper()))


def decorator_func(func):
	# Choose basic load, dump, plot, usetex settings
	load = 0
	dump = 1
	plot = 1
	usetex = 1
	display = 0
	verbose = 1
	
	def inner(*args, **kwargs):
		#Create dummy sys settings
		sys_settings = {
			'sys__directories__cwd__load':[kwargs['settings']['cwd']],
			'sys__directories__cwd__dump':[kwargs['settings']['cwd']],
			'sys__directories__src__load':[kwargs['settings']['cwd']],
			'sys__directories__src__dump':[kwargs['settings']['cwd']],
			'sys__directories__directories__load':[kwargs['settings']['directories_load']],
			'sys__directories__directories__dump':[kwargs['settings']['directories_dump']],
			'sys__files__files':[[kwargs['settings']['data_filename']]],			
			'sys__label':[None],
			'sys__read__data':['r'],
			'sys__read__metadata':['rb'],
			'sys__write__data':['w'],
			'sys__write__metadata':['wb'],
			'sys__labels':[[]], #To change output filenames
			}		
		#Create dummy structure settings
		structure_settings = {
			'structure__index':[None],
			'structure__seed':[1234556789],
			'structure__filters':[None],
			'structure__conditions':[None],
			'structure__refinement':[None],
			'structure__functions': kwargs['settings']['algebraic_operations']
			}
		#Create dummy flags
		boolean_settings = {
			'boolean__load':[load],
			'boolean__dump':[dump],
			'boolean__verbose':[verbose],
			'boolean__texify':[usetex],
			'boolean__display':[display],
			'boolean__plot':[plot],
			}
		#Create dummy model settings
		model_settings = {
			'model__order':kwargs['settings']['model_order'],
			'model__p':kwargs['settings']['model_p'],
			'model__basis':[None],
			'model__intercept_':[0],
			'model__inputs':[[]],
			'model__outputs':[[]],
			'model__selection':[[]],		
			'model__normalization':['l2'],
			'model__rhs_lhs': [{
				'model_label':{
					'lhs': [],
					'rhs': []}
					}]
			}
		terms_settings = {'terms__terms': kwargs['settings']['differential_operations']}
		settings = {**kwargs['settings'], **sys_settings, **model_settings, **boolean_settings, **structure_settings, **terms_settings}
		#print(settings)
		settings_grid = permute_settings(settings,_copy = True)
		func(data={}, metadata={} ,settings=settings_grid[0])
	
	return inner

@decorator_func
def main(data={},metadata={},settings={}):
	""" 
	Main program for graph theory library

	Args:
		data (dict): dictionary of {key:df} string keys and Pandas Dataframe datasets
		metadata (dict): dictionary of {key:{}} string keys and dictionary metadata about datasets
		settings (dict): settings in JSON format for library execution
	"""

	# Set Settings
	set_settings(settings,
				path=path_join(settings.get('sys',{}).get('src',{}).get('dump',''),
						  settings.get('sys',{}).get('settings')) if (
					(isinstance(settings.get('sys',{}).get('settings'),str)) and (
					not (settings.get('sys',{}).get('settings').startswith(settings.get('sys',{}).get('src',{}).get('dump',''))))) else (
					settings.get('sys',{}).get('settings')),
				_dump=True,_copy=False)


	# Import datasets
	if any([k in [None] for k in [data,metadata]]):
		data = {}
		metadata = {}
	if any([k in [{}] for k in [data,metadata]]):
		setup(data=data,metadata=metadata,
			  files=settings['sys']['files']['files'],
			  directories__load=settings['sys']['directories']['directories']['load'],# if ((not settings['boolean']['load']) or (not settings['boolean']['dump'])) else settings['sys']['directories']['directories']['dump'],
			  directories__dump=settings['sys']['directories']['directories']['dump'],
			  metafile=settings['sys']['files']['metadata'],
			  wr=settings['sys']['read']['files'],
			  flatten_exceptions=[],
			  **settings['sys']['kwargs']['load'])

	verbose = settings['sys']['verbose'] 
	models = {}

	# Set logger and texifying
	if settings['boolean']['log']:
		filelogger = logging.handlers.RotatingFileHandler(path_join(
				settings['sys']['directories']['cwd']['dump'],
				settings['sys']['files']['log'],
				ext=settings['sys']['ext']['log']))
		fileloggerformatter = logging.Formatter(
			fmt='%(asctime)s: %(message)s',
			datefmt='%Y-%m-%d %H:%M:%S')
		filelogger.setFormatter(fileloggerformatter)
		filelogger.setLevel(getattr(logging,log.upper()))
		if len(rootlogger.handlers) == 2:
			rootlogger.removeHandler(rootlogger.handlers[-1])
		rootlogger.addHandler(filelogger)
		


	logger.log(verbose,'Start')
	logger.log(verbose,'Set Settings')

	# Show Directories
	logger.log(verbose,'Imported Data: %s'%(settings['sys']['identity']) if len(data)>0 else "NO DATA")
	logger.log(verbose,'Datasets: %s'%('\n\t'.join(['',*[r'%s: %r'%(key,data[key].shape) for key in data]])))
	logger.log(verbose,'Load paths: %s'%('\n\t'.join(['',*settings['sys']['directories']['directories']['load']])))
	logger.log(verbose,'Dump paths: %s'%('\n\t'.join(['',*settings['sys']['directories']['directories']['dump']])))


	# Texify operation
	if settings['boolean']['texify']:
		tex = Texify(**settings['texify'])
		texify = tex.texify
	else:
		tex = None
		texify = None


	logger.log(verbose,'Setup Texify')

	# Define Structure of graph and perform pre-processing on datasets
	if settings['boolean']['structure']:
		structure(data,metadata,settings,verbose=settings['structure']['verbose'])
	logger.log(verbose,'Defined Structure')


	# Calculate terms
	if settings['boolean']['terms']:
		terms(data,metadata,settings,verbose=settings['terms']['verbose'],texify=texify)
	logger.log(verbose,'Calculated Operators')
	
	# Save Data
	if settings['boolean']['dump']:
		save(settings,paths={key: metadata[key]['directory']['dump'] for key in data},data=data,metadata=metadata)
	logger.log(verbose,'Saved Data') 
		
	Debug_Flag = False
	
	if Debug_Flag: 
		#Standalone fitting and saving models
	
		# Save Data
		if settings['boolean']['dump']:
			save(settings,paths={key: metadata[key]['directory']['dump'] for key in data},data=data,metadata=metadata)
		logger.log(verbose,'Saved Data')  


		# Calculate Model
		if settings['boolean']['model']:
			model(data,metadata,settings,models,verbose=settings['model']['verbose'])
		logger.log(verbose,'Setup Model')


		# Save Data
		if settings['boolean']['dump']:
			save(settings,paths={key: metadata[key]['directory']['dump'] for key in data},data=data,metadata=metadata)
		logger.log(verbose,'Saved Data') 


		# Fit Data     
		if settings['boolean']['fit']:
			for label in models:	
				fit(data,metadata,
					{key: metadata[key]['rhs_lhs'].get(label,{}).get('rhs') for key in data},
					{key: metadata[key]['rhs_lhs'].get(label,{}).get('lhs') for key in data},
					label,
					settings['fit']['info'],
					models[label],
					settings['fit']['estimator'],
					{
						**settings['fit']['kwargs'],
						**settings['model'],
						**{'modelparams':settings['analysis']}
					},				
					verbose=settings['fit']['verbose']
					)
		logger.log(verbose,'Fit Data')   


		# Save Data
		if settings['boolean']['dump']:
			save(settings,paths={key: metadata[key]['directory']['dump'] for key in data},data=data,metadata=metadata)
		logger.log(verbose,'Saved Data')  

		# Analyse Fit Results and Save Texified Model
		if settings['boolean']['analysis']:
			analysis(data,metadata,settings,models,texify,verbose=settings['analysis']['verbose'])
		logger.log(verbose,'Analysed Results')  

		# Plot Fit Results
		if settings['boolean']['plot']:
			plotter(data,metadata,settings,models,texify,verbose=settings['plot']['verbose'])
		logger.log(verbose,'Plotted Data')    

		logger.log(verbose,'Done\n')
	 

	
		return
	else: 
		return

