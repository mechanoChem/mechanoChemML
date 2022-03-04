#!/usr/bin/env python

# Import python modules
import os,sys,copy,warnings,functools,itertools
from natsort import natsorted
import numpy as np
import scipy as sp
import scipy.stats,scipy.signal
import pandas as pd

# import multiprocess as mp
# import multithreading as mt
# import multiprocessing as mp

warnings.simplefilter("ignore", (UserWarning,DeprecationWarning,FutureWarning))

# Global Variables
DELIMITER='__'

# Import user modules
from .load_dump import load,dump,path_split,path_join
from .dictionary import _set,_get,_pop,_has,_update,_permute,_clone,_find,_replace
from .graph_utilities import invert,add,outer,broadcast,convert,norm,repeat,where,delete,rank,isin,issparse,issingular,isfullrank,isarray,isnone
from .graph_utilities import iscallable,array_series,series_array,masked_index,wrapper,nonzeros,isiterable,subarray,round,explicit_zeros,catch,icombinations
from .graph_utilities import zeros,ones,arange
from .graph_utilities import refinement,neighbourhood,adjacency_matrix,similarity_matrix,combinations,ncombinations,nearest_neighbourhood,boundaries,stencils,twin
from .graph_utilities import add,subtract,prod,multiply,sqrt
from .graph_utilities import ge,gt,lt,le,eq
from .texify import scinotation
#from .estimator import OLS

# Logging
import logging
log = 'info'
logger = logging.getLogger(__name__)
#logger.setLevel(getattr(logging,log.upper()))


# Define dataset class
class dataset(pd.DataFrame):
	'''
	Class of dataset, with numerical data, metadata, and settings
	Args:
		key (str): Label for dataset
		data (panda.DataFrame): Dataframe class for numerical data
		metadata (dict): dictionary of metadata
		settings (dict): dictionary of dataset model settings
	'''
	def __init__(self,key,data,metadata,settings):
		'''
		Initialize class
		'''
		self.set_data(data)
		self.set_metadata(metadata)
		self.set_settings(settings)

		return

	# def __getitem__(self,key):		
	# 	return self.get_data()[key]

	# def __setitem__(self,key,value):
	# 	self.get_data()[key] = values
	# 	return

	def get(self,string):
		return getattr(self,'get_%s'%(string))()

	def set(self,string,*args,**kwargs):
		getattr(self,'set_%s'%(string))(*args,**kwargs)
		return		

	def set_data(self,data):
		super(dataset,self).__init__(data=data)
		return

	def get_data(self):
		return self.values

	def set_metadata(self,metadata):
		self.metadata = metadata
		return

	def get_metadata(self):
		return self.metadata

	def set_settings(self,settings):
		self.settings = settings
		fields = {'model':'model'}
		for field in fields:
			if field in settings:
				self.set(fields[field],setttings[field])


	def get_settings(self):
		return self.settings

	def set_model(self,model):
		self.model = model

	def get_model(self):
		return self.model




# Define nodes and edges and perform pre-processing of graph
def structure(data,metadata,settings,verbose=False):

	locs = locals()
	locs.update(settings['structure'])
	fields = [
		'directed','rename','round','drop_duplicates','groupby_filter','groupby_equal',
		'index','conditions','samples','functions','filters','symmetries',
		'scale','processed','refinement',]
	defaults = {'directed':False,'index':None,'refinement':None,'scale':{},'processed':False,}

	while not all([metadata[key].get('processed') for key in data]):
		keys = list(data)
		for key in keys:
			if metadata[key].get('processed'):
				continue

			for field in fields:

				try:
					value = locs[field][key]
					if value is None and field in defaults:
						locs[field][key] = defaults[field]
						value = defaults[field]
				except:
					value = locs.get(field)
					if value is None and field in defaults:
						locs[field] = defaults[field]
						value = defaults[field]
				if value is None:
					if field in defaults:
						metadata[key][field] = value
					continue

				elif field == 'directed':
					metadata[key][field] = value

				elif field == 'rename':
					data[key].rename(columns=value,inplace=True)

				elif field == 'round':
					labels = value['labels']
					decs = value['decimals']
					data[key][labels] = data[key][labels].round(decs)

				elif field == 'drop_duplicates':
					data[key] = data[key].drop_duplicates(subset=value['subset'],keep='first')

				elif field == 'groupby_filter':
					try:
						by = value['by']
						df_groupby = data[key].groupby(by)
						_functions = {
							'min_size':lambda func,by,df_groupby: lambda group: group.shape[0] == min(df_groupby.size()),
							'max_size':lambda func,by,df_groupby: lambda group: group.shape[0] == max(df_groupby.size()),
							'min':lambda func,by,df,df_groupby: lambda group: (group[by] == df[by].min()).all(),
							'max':lambda func,by,df,df_groupby: lambda group: (group[by] == df[by].max()).all(),
							'default': lambda func,by,df,df_groupby: lambda group: func(group,df_groupby)}
						func = value['func']
						func = _functions.get(func,_functions['default'])(func,by,data[key],df_groupby)
						data[key] = df_groupby.filter(func=func).copy()
					except:
						pass

				elif field == 'groupby_equal':
					try:
						df_groupby = data[key].groupby(value['by'])				
						ilocs = value['ilocs']
						labels = value['labels']
						if labels is None:
							labels = [labels]
							values = df_groupby.nth(ilocs).values
						else:
							if not isinstance(labels,list):
								labels = [labels]
							values = df_groupby.nth(ilocs)[labels].values
						assert equal_axis(values,0), "Node %d: not equal in all groups %s"%(ilocs,','.join(labels))
					except:
						pass

				elif field == 'index':
					_option = 'functions'
					if locs.get(_option) is not None:
						try:
							_variable = locs[field][key]
						except:
							_variable = locs[field]
						for _var in _variable:
							if value == _var['labels'] or value in _var['labels']:
								data[key][value] = _variable[value](data[key])
					try:

						data[key] = data[key].groupby(value).mean().reset_index()
						data[key].sort_values(value, axis='index', ascending=True, inplace=True, kind='quicksort')
					#     data[key] = data[key].reindex(index=sorter(data[key][value].values,data[key][value].values))
						data[key].set_index(value,drop=False, append=False, inplace=True)
					except:
						pass
					metadata[key][field] = value

				elif field == 'conditions':
					_functions = ['isin','eq','ne','gt','ge','lt','le']
					boolean = conjunction(*[getattr(data[key][condition['labels']],condition['func'])(value['other']) 
											for condition in value if condition['func'] in _functions])
					if boolean is not None:
						data[key].drop(data[key].loc[~boolean].index,inplace=True)


				elif field == 'samples':
					n = data[key].shape[0]
					if isinstance(value,(int,np.integer,float,np.float)):
						inds = np.random.choice(arange(n),size=int(n*(1-value)),replace=False)
						inds = data[key].index[inds]
					elif isinstance(value,np.ndarray):
						inds = delete(arange(n),value[value<n])
						inds = data[key].index[inds]
					elif isinstance(value,(list,slice)):
						inds = data[key].index[value]
					else:
						inds = []

					data[key].drop(inds,axis=0,inplace=True)


				elif field == 'functions':
					for function in value:
						labels = function['labels']
						try:
							data[key][labels] = function['func'](data[key])
						except:
							pass


				elif field == 'filters':
		
					for f in value:
						labels = f['labels']
						if f['type'] in ['rolling']:
							if labels is None:
								labels = list(data[key].columns)
							passes = iscallable(f['passes'],data[key])
							for n in range(passes):
								rolling_kwargs = iscallable(f['rolling_kwargs'],data[key],n)
								mean_kwargs = iscallable(f['mean_kwargs'],data[key],n)
								data[key][labels] = data[key][labels].rolling(**rolling_kwargs).mean(**mean_kwargs)
						elif f['type'] in ['filtering']:
							for label in labels:
								data[key][label],_,_ = filtering(data[key][label].values,**f['filtering_kwargs'])

						n = data[key].shape[0]
						inds = f.get('dropinds')
						if inds is not None:
							if isinstance(inds,(int,np.integer,float,np.float)):
								inds = np.random.choice(arange(n),size=int(n*(1-inds)),replace=False)
								inds = data[key].index[inds]
							elif isinstance(inds,np.ndarray):
								inds = delete(arange(n),inds[inds<n])
								inds = data[key].index[inds]
							elif isinstance(inds,(list,slice)):
								inds = data[key].index[inds]
							else:
								inds = []
							data[key].drop(inds,axis=0,inplace=True)								

				elif field == 'symmetries':
					df_ = [df.copy()]
					for symmetry in value:
						df_.append(data[key].copy())
						df_[-1].rename(columns=symmetry['labels'],inplace=True)
					data[key] = pd.concat(df_,ignore_index=True,axis=0)

				elif field == 'scale':
					_labels = data[key].select_dtypes(exclude=(object,int)).columns.values.tolist()
					if value.get('labels') is True:
						labels = _labels
					else:
						labels = value.get('labels',[])
					labels = [label for label in labels if label in _labels]

					if len(labels)>0:
						scales = data[key][labels].abs().max(axis=0).values
						scales[scales==0] = 1
						data[key][labels] /= scales
						scales = {k:v for k,v in zip(labels,scales)}
					else:
						scales = {}
					metadata[key][field] = scales
				

				elif field == 'processed':
					metadata[key][field] = True					


				elif field == 'refinement':

					if metadata[key]['type'] in ['refined']:
						continue



					# Get indices of refined meshes, with training and testing sets for each mesh refinement
					n = data[key].shape[0]
					powers = refinement(int(n**(1/value['p'])),value['p'],value['base'],powers=value['powers'],string=key,boundary=False)

					# Get list of settings keys for updating settings dictionary with new dataset meshes
					_settings = value['settings']
					_settings = [] if _settings is None else _settings 
					_settings.extend([_setting for _setting in ['sys__files','terms','model','terms__parameters','fit__info','plot__settings'] 
									if _setting not in _settings])


					# Update data, metadata, settings with new datasets and keys based on refinement keys	
					for j,power in enumerate(powers):

						# Update powers for training and testing data with new _key, where _key = key__k__i where data size has been refined n -> n/k 
						# and i represents the offset for the datasets, either indices {0,k,2k,...,n} (training) or {k/2,3k/2,...,n-k/2} (testing)
						for i,_key in enumerate(powers[power]):

							data[_key] = data[key].iloc[powers[power][_key]].copy()
							_replace(metadata,key,_key,_append=True,_values=True,_copy=True)				
							metadata[_key]['processed'] = False
							metadata[_key]['type'] = 'refined'
							metadata[_key]['refinement'] = powers[power][_key] 

						# Update each of _settings based on new keys from refinement
						for _setting in _settings:

							setting = _get(settings,_setting,_split=DELIMITER)							
							
							# Update specific (possibly key-dependent) settings

							# Update system label for saving data,metadata
							if _setting in ['sys__files']:
								# Add _key to any key dependent settings with the setting associated with key
								for i,_key in enumerate(powers[power]):

									_replace(setting,key,_key,_append=(value['keep'] or i < (len(powers[power])-1) or j < (len(powers)-1)),_values=True,_copy=True)				

									# Update certain parameters based on whether training and testing data

									# For data,metadata:
									#   Append _key postfix to file names to prevent overwriting

									params = {param:setting[param].replace(param,'%s%s'%(param,_key.replace(key,''))) if isinstance(setting[param],str) else setting[param][_key].replace(param,'%s%s'%(param,_key.replace(key,''))) 
												for param in ['data','metadata']}
									update = {param: True for param in params}

									for param in params:
									
										if not update[param] or param not in setting:
											continue

										# Make settings key-dependent
										if (not isinstance(setting[param],dict) or _key not in setting[param]):
											setting[param] = {k: copy.deepcopy(setting[param]) for k in data}											


										setting[param].update({_key:copy.deepcopy(params[param])})

							# Update terms and model and terms__parameters that may be key-dependent
							elif _setting in ['terms','model','terms__parameters']:
				
								# Add _key to any key dependent settings with the setting associated with key
								for i,_key in enumerate(powers[power]):

									_replace(setting,key,_key,_append=(value['keep'] or i < (len(powers[power])-1) or j < (len(powers)-1)),_values=True,_copy=True)				

									# Update certain parameters based on whether training and testing data

									# For iloc:
									# 	True if training data (allows all ilocs to be used for fitting)
									#	None if testing data (allows closest iloc to be found in training data for predicting)

									params = {'iloc':{0:True,1:None}.get(i,None)}
									update = {'iloc':{0:False,1:True}.get(i,False)}
									for param in params:
										if not update[param] or param not in setting:
											continue

										# Make settings key-dependent
										if (not isinstance(setting[param],dict) or _key not in setting[param]):
											setting[param] = {k: copy.deepcopy(setting[param]) for k in data}											


										setting[param].update({_key:copy.deepcopy(params[param])})
										
							# Update fit type based on training (fit) or testing (interpolate) datasets
							elif _setting in ['fit__info']:

								params = dict(zip(['fit','interpolate'],powers[power]))
								for param in params:
									if param not in setting:
										setting[param] = {}
									setting[param].update({
										params[param]:{
										'type':{'fit':'fit','interpolate':'fit'}.get(param,'fit'),
										'key':params[{'fit':'fit','interpolate':'fit'}.get(param,'fit')],
										'keys':[params[param]],								
										} for i,_key in enumerate(powers[power])
									})
									if not value['keep'] and key in setting[param]:
										setting[param].pop(key)


							# Update plot (possibly key-dependent) [lot settings]
							elif _setting in ['plot__settings']:
								for i,_key in enumerate(powers[power]):
									_replace(setting,key,_key,_append=(value['keep'] or i < (len(powers[power])-1) or j < (len(powers)-1)),_values=True,_copy=True)
							
							# Update other settings and add new _key settings
							else:
								for i,_key in enumerate(powers[power]):
									_replace(setting,key,_key,_append=(value['keep'] or i < (len(powers[power])-1) or j < (len(powers)-1)),_values=True,_copy=True)

					if len(powers)>0 and not value['keep']:
						data.pop(key);
						metadata.pop(key);

						continue	



	return



# Save data
def save(settings,paths,**objs):
	for name in objs:
		obj = objs[name]
		files = {key: settings['sys']['files'][name].get(key,name) if isinstance(settings['sys']['files'][name],dict) else settings['sys']['files'][name]
				 for key in paths}
		exts = {key: settings['sys']['ext'][name].get(key,'pickle') if isinstance(settings['sys']['ext'][name],dict) else settings['sys']['ext'][name]
				 for key in paths}
		wrs = {key: settings['sys']['write'][name].get(key,'w') if isinstance(settings['sys']['write'][name],dict) else settings['sys']['write'][name]
		 for key in paths}
		kwargs = {key: settings['sys']['kwargs']['dump'].get(key,settings['sys']['kwargs']['dump']) if isinstance(settings['sys']['kwargs']['dump'],dict) else settings['sys']['kwargs']['dump']
				 for key in paths}
		for key in paths:
			path = path_join(paths[key],files[key],ext=exts[key])
			try:
				dump(obj[key],path,wr=wrs[key],**kwargs[key])				
			except Exception as e:
				pass



def analysis(data,metadata,settings,models,texify=None,verbose=False):
	'''
	Analyse results of model
	Args:
		data (dict): Dictionary of pandas dataframes datasets
		metadata (dict): Dictionary of metadata for each dataset
		settings (dict): Dictionary of library settings
		texify (Texify,callable,None): Texify function or callable for rendering strings in Latex
		verbose (str): Print out additional information
	'''
	def replace(string,replacements,conditions=None):
		'''
		Replace substrings in string
		Args:	
			string (str): String to be updated
			replacements (dict): Dictionary of replacements
			conditions (list,None): List of conditions on substrings to perform replacements
		Returns:
			string with replacements
		'''
		for replace in replacements:
			if conditions is None or any([condition in string for condition in conditions]):
				string = string.replace(replace,replacements[replace])
		return string


	labels = list(models)
	keys = list(data)
	types = settings['fit']['types']



	if texify is None:
		texify = lambda string: str(string).replace(DELIMITER,' ').replace('$','')

	for ilabel,label in enumerate(labels):
		dims = arange(max([len(metadata[key]['rhs_lhs'].get(label,{}).get('lhs',[])) for key in data]))[settings['plot'].get('dims',slice(None))]
		for idim,dim in enumerate(dims):	
			for key in keys:
				r = metadata[key]['rhs_lhs'][label]['rhs']
				l = metadata[key]['rhs_lhs'][label]['lhs']
				if verbose == 2:
					logger.log(verbose,'LABEL: %s',label)
					logger.log(verbose,'RHS: %r'%r)
					logger.log(verbose,'LHS: %r\n'%l)
				for ityped,typed in enumerate(types):
					label_dim = DELIMITER.join([label,typed,'%d'%(dim)])							
					if (typed not in metadata[key]) or (label_dim not in metadata[key][typed]):
						continue
					
					stats = metadata[key][typed][label_dim]['stats']			

					iloc = {
						'indiv':metadata[key]['iloc'][dim%len(metadata[key]['iloc'])] if metadata[key]['iloc'] not in [None] else [metadata[key]['iloc']],
						'multi':metadata[key]['iloc'][dim%len(metadata[key]['iloc'])] if metadata[key]['iloc'] not in [None] else [metadata[key]['iloc']]
						}[settings['model']['approach']]
					Niterations = stats['iterations'].max()
					iterations = [None,*list(range(10-1,0,-1))]
					iterations = [Niterations-(Niterations if i is None else i) for i in iterations if i is None or i<=Niterations]
					iterations = list(sorted(list(set(iterations)),key=lambda i: iterations.index(i)))
					slices = slice(-1,-10-1,-1)
					



					# Latex model with symbolic coefficients
					values = {}



					for i in iterations:

						values['model_%d'%(i)] = r'%s = %s'%(
													texify(l[dim].replace('partial','delta')).replace('partial','delta'),
													r'\\'.join([r' + '.join(['%s%s'%(texify(o.replace('partial','delta')
																if 'taylorseries' in o else DELIMITER.join(['coefficient',str(None),o.replace('partial','delta')]) if 'constant' not in o else '').replace('partial','delta'),
																texify(o.replace('partial','delta')).replace('partial','delta') if 'taylorseries' not in o else '') 
																for o in stats['ordering'][slices] if stats['coef_'][o][i] != 0.0]) 
																])
													)			


						values['model_%d'%(i)] = r'$%s$'%(values['model_%d'%(i)].replace('$','').replace('\n','').replace(' + -',' - '))



					# values['elbow'] = np.array([*(np.abs((((stats['loss'][:-1]-stats['loss'][1:])/stats['loss'][:-1])))>0.04),0],dtype=bool)


					# Latex model with numeric coefficients					
					for i in iterations:
						values['coef_%d'%(i)] = r'%s = %s'%(
													texify(l[dim].replace('partial','delta')).replace('partial','delta'),
													r'\\'.join([r' + '.join(['%s%s'%(texify(stats['coef_'][o][i]) if stats['coef_'][o][i] != 1 else '',texify(o.replace('partial','delta')).replace('partial','delta')) 
																for o in stats['ordering'][slices] if stats['coef_'][o][i] != 0.0]) 
																])
													)
						values['coef_%d'%(i)] = r'$%s$'%(values['coef_%d'%(i)].replace('$','').replace('\n','').replace(' + -',' - '))

					fields = {'model':[*['model_%d'%(i) for i in iterations],*['coef_%d'%(i) for i in iterations]]}

					if verbose in ['info','warning','error','critical']:
						logger.log(verbose,'Fit: %s %s'%(key,l[dim]))
						logger.log(verbose,'Loss: %s\n'%(stats['loss'][::-1]))
						for i,k in enumerate(values):
							if all([k not in fields[field] for field in fields]):
								logger.log(verbose,'%s: %r%s'%(k.capitalize(),values[k],'' if i<(len(values)-1) else '\n\n\n\n'))

					if settings['boolean']['dump']:
						for field in fields:
							file = settings['sys']['files'][field]
							file = file%(field,DELIMITER.join([l[dim],'iloc',str(iloc)])) if file.count('%')==2 else file%(l[dim]) if file.count('%')==1 else file
							path = path_join(metadata[key]['directory']['dump'],file,ext=settings['sys']['ext'][field])
							for i,f in enumerate(fields[field]):
								wr = settings['sys']['write'][field] if i==0 else 'a'
								dump(values[f],path,wr=wr,**settings['sys']['kwargs']['dump'])
								dump('\n',path,wr='a',**settings['sys']['kwargs']['dump'])




					# Analyze derivatives and error
					field = 'analysis'
					subfield = 'parameters'
					parameters = settings.get(field,{}).get(subfield)

					if typed not in ['interpolate','predict'] or parameters is None:
						continue


					iteration = 0

					N = data[key].shape[0]
					p = parameters['p']
					n = int(N**(1/p))
					q = parameters['q']
					r = parameters['accuracy']
					unique = parameters['unique']
					order = parameters['order']
					K = parameters['K']
					h = np.mean(stats['distances'])/(p**(1/2))
					L = h*(2*n)
					inputs = parameters['inputs']
					outputs = parameters['outputs']
					refine = stats['refinement_inherited']
					manifold = {i:''.join([r'{%s^{%s}}'%(inputs[k] if j>0 else '',str(j) if j>1 else '') for k,j in enumerate(u)]) 
								for i,u in enumerate(combinations(p,r,unique=unique))}

					rhs = stats['rhs'][dim]
					lhs = stats['lhs'][dim]

					rhs_pred = list(stats['coef_'])
					lhs_pred = '%s%s%d'%(label_dim,DELIMITER,stats['size_'].max()-iteration)


					y = data[key][lhs].values

					if lhs_pred in data[key]:
						y_pred = data[key][lhs_pred].values
					else:
						y_pred = np.array([])

					X = data[key][inputs].values


					field = 'derivative'
					if parameters.get(field) is not None:
						derivative = parameters[field][refine][:,:,dim]
						derivative_pred = data[stats['key']][rhs].values
					else:
						derivative = np.array([])
						derivative_pred = np.array([])

					gamma = np.array([stats['coef_'][r][iteration] for r in stats['coef_']]).T

					field = 'indices'
					if parameters.get(field) is not None:
						indices = parameters[field]
					else:
						indices = np.array([])
					
					field = 'alpha'
					if parameters.get(field) is not None:
						alpha = parameters[field]
						alpha = alpha[:,dim]
						# alpha = alpha.ravel().tolist() if len(set(alpha.ravel().tolist()))>1 else (alpha.ravel()[0] if not int(alpha.ravel()[0])==alpha.ravel()[0] else int(alpha.ravel()[0]))					
					else:
						alpha = np.array([])

					# slices = boundaries(X,size=settings['model']['neighbourhood'],excluded=[0])
					slices = slice(None)


					def localerror(y,y_pred): 
						try:
							out = np.abs(y - y_pred)
						except:
							out = np.array([])
						return out
					def error(y,y_pred,axis=None,ord=2,slices=slice(None)): 
						try:
							out = norm(localerror(y,y_pred)[slices],axis=axis,ord=ord)/((max(1,y[slices].size))**(1./ord))
						except:
							out = np.array([])
						return out

					name = parameters['funcstring']
					params = {
						'h':h,
						'n':n,
						'p':p,
						'q':q,
						'r':r,
						'K':K,
						'order':order,
						'L':L,
						'manifold':manifold,
						'funcstring': parameters['funcstring'],
						'modelstring': parameters['modelstring'],
						'operator':parameters['operator'],
						'indices':parameters['indices'],
						'error': error(y,y_pred,ord=2,slices=slices),
						'localerror': localerror(y,y_pred),
						'gamma':gamma,
						'alpha':alpha,
						'derivativeerror': error(derivative,derivative_pred,axis=0,ord=2,slices=slices).T,
						'localderivativeerror':localerror(derivative,derivative_pred).T,
						'derivative':derivative.T,
						 }



					field = 'analysis'
					file = settings['sys']['files'][field]
					file = file%('%s_p%d_n%d_K%d_k%d'%(stats['lhs'][0],p,parameters['N'],K,order))
					path = path_join(metadata[key]['directory']['dump'],file,ext=settings['sys']['ext'][field])
					rw = settings['sys']['read'][field] 
					wr = settings['sys']['write'][field] 




					values = load(path,wr=rw,default={})


					if name in values:
						values[name].append(params)
					else:
						values[name] = [params]
					
					dump(values,path,wr=wr)



					continue
					# Plot Operators

					import matplotlib
					import matplotlib.pyplot as plt
					matplotlib.use('TkAgg')
					matplotlib.rcParams['text.usetex'] = True
					matplotlib.rcParams['font.size'] = 40


					slices = slice(None)


					if p == 1:
						fig,ax = plt.subplots()
					else:
						fig,ax = plt.figure(),plt.axes(projection='3d')

					
					if p == 1:
						O = [[0],[1],[2]]
						O = [[1]]
					else:
						_O_ = [0 for i in range(p)]
						O = [[1,0],[0,1],[1,1]]
						# O = [[1,0],[0,1]]
						O = [[1,0],[0,1],[1,1],[2,0],[0,2]]#[2,1],[1,2],[3,0],[0,3]]
						O = [[1,1]]#,[0,1],[1,1],[2,0],[0,2]]#[2,1],[1,2],[3,0],[0,3]]



					plots = []
					for o in O:

						if o not in indices:
							continue

						i = indices.index(o)	
						j = inputs.index(variable)

						x = X[slices]

						partial = derivative[slices,i]
						delta = derivative_pred[slices,i]

						partiallabel = r'$\frac{\partial^{%s}u}{\partial x^{%s}}$'%(str(sum(o)) if sum(o)>1 else '',str(''.join([r'{%s}_{%s}^{%s}'%(r'\mu' if len(o)>1 else u,l if len(o)>1 else '',u if len(o)>1 and u>1 else '') for l,u in enumerate(o) if u>0])) if (len(o)==1 and sum(o)>1) or (len(o)>1) else str(o[0]) if o[0]>1 else '') if i>0 else r'$u_{}$'
						deltalabel = (r'$\frac{\delta^{%s}u}{\delta x^{%s}}$'%(str(sum(o)) if (sum(o)>1) else '',str(''.join([r'{%s}_{%s}^{%s}'%(r'\mu' if len(o)>1 else u,l if len(o)>1 else '',u if len(o)>1 and u>1 else '') for l,u in enumerate(o) if u>0])) if (len(o)==1 and sum(o)>1) or (len(o)>1) else str(o[0]) if (o[0]>1) else '')) if i>0  else r'$u_{}$'


						if p == 1:

							plotter = 'plot'

							shape = {'plot':(-1)}[plotter]

							x,z,w = x[:,0].reshape(shape),partial.reshape(shape),delta.reshape(shape)

							if o == [0]:
								func = lambda x: 1 + x + x**2 + x**3 + x**4 + x**5 + x**6 + x**7 + x**8
							elif o == [1]:
								func = lambda x: 1 + 2*x + 3*x**2 + 4*x**3 + 5*x**4 + 6*x**5 + 7*x**6 + 8*x**7
							elif o == [2]:
								func = lambda x: 2 + 6*x + 12*x**2 + 20*x**3 + 30*x**4 + 42*x**5 + 56*x**6

							# plots.append(ax.plot(x,w,label=r'$\frac{\delta^{%s}u}{\delta x^{%s}}$'%(str(i) if i>1 else '',str(i) if i>1 else '') if i>0 else r'$u_{}$'))
							# plots.append(ax.plot(x,z,label=r'$\frac{\partial^{%s}u}{\partial x^{%s}}$'%(str(i) if i>1 else '',str(i) if i>1 else '') if i>0 else r'$u_{}$'))
							plots.append(getattr(ax,plotter)(x,np.abs(func(x)-w),label=r'%s - %s'%(deltalabel,partiallabel)))



						else:
							plotter = 'plot_surface'
							m = int(np.sqrt(x.shape[0]))
							shape = {'plot_surface':(m,-1),'plot_trisurf':(-1)}[plotter]

							x,y,z,w = x[:,0].reshape(shape),x[:,1].reshape(shape),partial.reshape(shape),delta.reshape(shape)

							
							if o == [0,0]:
								func = lambda x,y: x**3 + y**3 + x**2*y + x*y**2 + x**2 + y**2 + x*y + x + y
							elif o == [1,0]:
								func = lambda x,y: 3*x**2 + 2*x*y + y**2 + 2*x + y + 1
							elif o == [0,1]:
								func = lambda x,y: 3*y**2 + 2*x*y + x**2 + 2*y + x + 1
							elif o == [1,1]:
								func = lambda x,y: 2*x + 2*y + 1
							elif o == [2,0]:
								func = lambda x,y: 6*x + 2*y + 2
							elif o == [0,2]:
								func = lambda x,y: 6*y + 2*x + 2
							elif o == [3,0]:
								func = lambda x,y: 6
							elif o == [0,3]:
								func = lambda x,y: 6
							elif o == [2,1]:
								func = lambda x,y: 2
							elif o == [1,2]:
								func = lambda x,y: 2

							# ax.plot(x,partial,label=r'$\frac{\partial^{%s}u}{\partial x^{%s}}$'%(str(i) if i>1 else '',str(i) if i>1 else '') if i>0 else r'$u_{}$')
							# ax.plot(x,delta,label=r'$\frac{\delta^{%s}u}{\delta x^{%s}}$'%(str(i) if i>1 else '',str(i) if i>1 else '') if i>0 else r'$u_{}$')
							# plots.append(ax.plot_trisurf(x[:,0],x[:,1],np.abs(delta-partial),label=r'$\frac{\delta^{%s}u}{\delta x^{%s}}$'%(str(i) if i>1 else '',str(i) if i>1 else '') if i>0 else r'$u_{}$'))
							# plots.append(ax.plot_surface(x,y,z,label=r'$\frac{\delta^{%s}u}{\delta x^{%s}}$'%(str(i) if i>1 else '',str(i) if i>1 else '') if i>0 else r'$u_{}$'))
							# plots.append(getattr(ax,plotter)(x,y,w,label=deltalabel))
							# plots.append(getattr(ax,plotter)(x,y,z,label=partiallabel))
							# plots.append(getattr(ax,plotter)(x,y,w-z,label=r'%s - %s'%(deltalabel,partiallabel)))
							# plots.append(getattr(ax,plotter)(x,y,z-func(x,y),label=r'%s - %s'%(deltalabel,partiallabel)))
							plots.append(getattr(ax,plotter)(x,y,func(x,y),label=r'%s - %s'%(deltalabel,partiallabel)))


					for plot in plots:
						try:
							plot._facecolors2d = plot._facecolor3d 
							plot._edgecolors2d = plot._edgecolor3d
							ax._facecolors2d = ax._facecolor
						except:
							pass

					ax.legend(loc=(1.55,0.15),prop={'size':20})
					ax.set_title(('\n').join([
							r'$u_{} = \sum_{q=0}^{%d}\sum_{\mu = \{\mu_{1}\cdots\mu_{p=%d}\} : |\mu| = q} \alpha_{q_{\mu_{1}\cdots\mu_{p}}} x^{\mu_{1}\cdots\mu_{p}}$'%(K,p),
							r'$\alpha_q = %s$'%('[%s]'%(', '.join([r'%s~%s'%(scinotation(a,scilimits=[-2,1]),''.join([r'{x}^{{%s}^{%s}}'%(str(l) if len(j)>1 else str(u),str(u) if u>1 and len(j)>1 else '') if u>0 else '' for l,u in enumerate(j)])) + ('\\' if t==(len(indices)//2) else '')
							 for t,(j,a) in enumerate(zip(indices,alpha)) if a!= 0])) if isinstance(alpha,list) else str(alpha))
							]),pad=50)

					if p == 1:
						ax.set_xlabel(r'$x_{}$')
						ax.set_ylabel(r'$u_{}$')
						ax.set_yscale('log')

						ax.grid(True,alpha=0.6)
						
						fig.set_size_inches(10,10)
						fig.tight_layout()

						file = DELIMITER.join(['.'.join(path.split('.')[:-1]),'derivatives'])
						ext = 'pdf'
						path = '%s.%s'%(file,ext)
						fig.savefig(path,bbox_inches='tight')
					
					else:
						ax.set_xlabel(r'$x_{}$',labelpad=50)
						ax.set_ylabel(r'$y_{}$',labelpad=50)
						ax.set_zlabel(r'$u_{}$',labelpad=50)
						ax.set_zscale('linear')
						ax.grid(True,alpha=0.6)
						
						fig.set_size_inches(10,10)
						# fig.tight_layout()
						# fig.show()
						# plt.pause(100)
						# file = '.'.join(path.split('.')[:-1])
						# ext = 'pdf'
						# path = '%s.%s'%(file,ext)
						# fig.savefig(path,bbox_inches='tight')						




# Metrics
def metrics():

	def euclidean(x,y,ord=2,axis=-1):
		out = similarity_matrix(x,y,metric=ord,sparsity=None,directed=False)
		return out 

	locs = locals()
	funcs = {k:f for k,f in locs.items() if callable(f) and not k.startswith('_')}	

	return funcs


# Adjacencies
def adjacencies():
	'''
	Returns dictionary with functions for adjacency matrices, based on weights function and conditions for adjacency matrix depending on connectivity
	Args:
		data (ndarray): data to compute adjacency matrix of shape (n,p)
		parameters (dict): parameters for adjacency conditions
	Returns:
		weights (callable): callable weight function that accepts ndarray as data
		conditions (int,tuple,list,None): integer for k nearest neighbours, or tuple of (open) bounds on nearest neighbours, list of functions to make adjacency elements non-zero		
		out (int,float,ndarray,None): Return type of adjacency matrix of either out value, or weights value if None
		a (ndarray): adjacency matrix of shape (n,n)
	'''

	def _wrapper(func):
		@functools.wraps(func)
		def _func(data,parameters):
			weights,conditions = func(data,parameters)
			n = parameters['n']
			tol,atol,rtol = parameters['tol'],parameters['atol'],parameters['rtol']
			kind = parameters['kind']
			format = parameters['format']
			dtype = parameters['dtype']
			verbose = parameters['verbose']
			
			adjacency = adjacency_matrix(n=n,weights=weights,conditions=conditions,
							tol=tol,atol=atol,rtol=rtol,kind=kind,diagonal=False,format=format,dtype=dtype,verbose=verbose)
			
			return adjacency,weights
		return _func


	@_wrapper
	def default(data,parameters):
		def weights(data,parameters):
			metric = parameters['metric']
			format = parameters['format']
			dtype = parameters['dtype']
			chunk = parameters['chunk']
			n_jobs = parameters['n_jobs']
			verbose = parameters['verbose']				
			weights = similarity_matrix(data,data,metric=metric,directed=False,format=format,dtype=dtype,verbose=verbose,chunk=chunk,n_jobs=n_jobs)
			return weights
		def conditions(data,parameters):
			condition = None
			return condition
		returns = (weights(data,parameters),conditions(data,parameters))
		return returns

	@_wrapper
	def forward_all(data,parameters):
		def weights(data,parameters):
			p = parameters['p']
			n = parameters['n']
			sparsity = n
			metric = parameters['metric']			
			chunk = parameters['chunk']			
			format = parameters['format']
			dtype = parameters['dtype']
			n_jobs = parameters['n_jobs']
			verbose = parameters['verbose']
			weights = similarity_matrix(data,data,metric=metric,directed=True,sparsity=sparsity,format=format,dtype=dtype,verbose=verbose,chunk=chunk,n_jobs=n_jobs)
			if issparse(weights):				
				fmt = 'csr'
				format = weights.getformat()
				weights = weights.asformat(fmt)
				mask = lt(weights,0)
				for i in range(n):
					if mask[i].indices.size == weights[i].indices.size-1:
						weights[i] = weights[i].multiply(-1)
				weights = weights.asformat(format)
			else:
				mask = where(prod(lt(weights,0),axis=1))				
				weights[mask] *= -1
			weights += 0
			weights = explicit_zeros(weights,indices=[np.array([i]) for i in range(n)])			
			return weights
		def conditions(data,parameters):
			def condition(weights,argsort,counts):
				return gt(weights,0)
			return condition
		returns = (weights(data,parameters),conditions(data,parameters))
		return returns

	@_wrapper
	def forward_nearest(data,parameters):
		def weights(data,parameters):
			p = parameters['p']
			n = parameters['n']
			sparsity = 2*p+1
			metric = parameters['metric']			
			chunk = parameters['chunk']			
			format = parameters['format']
			dtype = parameters['dtype']
			n_jobs = parameters['n_jobs']
			verbose = parameters['verbose']			
			weights = similarity_matrix(data,data,metric=metric,directed=True,sparsity=sparsity,format=format,dtype=dtype,verbose=verbose,chunk=chunk,n_jobs=n_jobs)
			if issparse(weights):				
				fmt = 'csr'
				format = weights.getformat()
				weights = weights.asformat(fmt)
				mask = lt(weights,0)
				for i in range(n):
					if mask[i].indices.size == weights[i].indices.size-1:
						weights[i] = weights[i].multiply(-1)
				weights = weights.asformat(format)
			else:
				mask = where(prod(lt(weights,0),axis=1))				
				weights[mask] *= -1
			weights += 0
			weights = explicit_zeros(weights,indices=[np.array([i]) for i in range(n)])			
			return weights
		def conditions(data,parameters):
			def condition(weights,argsort,counts):
				return multiply(gt(weights,0),eq(argsort,1))
			return condition
		returns = (weights(data,parameters),conditions(data,parameters))
		return returns

	@_wrapper
	def backward_all(data,parameters):
		def weights(data,parameters):
			p = parameters['p']
			n = parameters['n']
			sparsity = n			
			metric = parameters['metric']			
			chunk = parameters['chunk']			
			format = parameters['format']
			dtype = parameters['dtype']
			n_jobs = parameters['n_jobs']
			verbose = parameters['verbose']			
			weights = similarity_matrix(data,data,metric=metric,directed=True,sparsity=sparsity,format=format,dtype=dtype,verbose=verbose,chunk=chunk,n_jobs=n_jobs)
			mask = where(prod(ge(weights,0),axis=1))
			if issparse(weights):				
				fmt = 'csr'
				format = weights.getformat()
				weights = weights.asformat(fmt)
				mask = gt(weights,0)
				for i in range(n):
					if mask[i].indices.size == weights[i].indices.size-1:
						weights[i] = weights[i].multiply(-1)
				weights = weights.asformat(format)
			else:
				mask = where(prod(gt(weights,0),axis=1))				
				weights[mask] *= -1
			weights += 0
			weights = explicit_zeros(weights,indices=[np.array([i]) for i in range(n)])			
			return weights
		def conditions(data,parameters):
			def condition(weights,argsort,counts):
				return lt(weights,0)
			return condition
		returns = (weights(data,parameters),conditions(data,parameters))
		return returns

	@_wrapper
	def backward_nearest(data,parameters):
		def weights(data,parameters):
			p = parameters['p']
			n = parameters['n']
			sparsity = 2*p+1
			metric = parameters['metric']			
			format = parameters['format']
			dtype = parameters['dtype']
			chunk = parameters['chunk']
			n_jobs = parameters['n_jobs']
			verbose = parameters['verbose']			

			weights = similarity_matrix(data,data,metric=metric,directed=True,sparsity=sparsity,format=format,dtype=dtype,verbose=verbose,chunk=chunk,n_jobs=n_jobs)
			
			if issparse(weights):				
				fmt = 'csr'
				format = weights.getformat()
				weights = weights.asformat(fmt)
				mask = gt(weights,0)
				for i in range(n):
					if mask[i].indices.size == weights[i].indices.size-1:
						weights[i] = weights[i].multiply(-1)
				weights = weights.asformat(format)
			else:
				mask = where(prod(gt(weights,0),axis=1))
				weights[mask] *= -1
			weights += 0
			weights = explicit_zeros(weights,indices=[np.array([i]) for i in range(n)])
			return weights
		def conditions(data,parameters):
			def condition(weights,argsort,counts):
				return multiply(lt(weights,0),eq(argsort,1))
			return condition
		returns = (weights(data,parameters),conditions(data,parameters))
		return returns

	@_wrapper
	def nearest(data,parameters):
		def weights(data,parameters):
			p = parameters['p']
			n = parameters['n']
			accuracy = parameters['accuracy']
			metric = parameters['metric']			
			chunk = parameters['chunk']
			format = parameters['format']
			dtype = parameters['dtype']		
			unique = parameters['unique']
			neighbourhood = parameters.get('neighbourhood',ncombinations(p,accuracy,unique=unique)-1)			
			sparsity = parameters.get('sparsity',2*neighbourhood*p)
			n_jobs = parameters['n_jobs']
			verbose = parameters['verbose']
			weights = similarity_matrix(data,data,metric=metric,sparsity=sparsity,directed=False,chunk=chunk,format=format,dtype=dtype,verbose=verbose,n_jobs=n_jobs)
			return weights	
		def conditions(data,parameters):
			n,p = data.shape[:2]
			n = min(parameters['n'],n)
			p = min(parameters['p'],p)
			chunk = parameters['chunk']
			accuracy = parameters['accuracy']
			unique = parameters['unique']			
			size = parameters.get('neighbourhood',ncombinations(p,accuracy,unique=unique)-1)
			strict = parameters['strict']
			atol = parameters['atol']
			rtol = parameters['rtol']
			kind = parameters['kind']
			argsortbounds = [None,None,None]
			weightbounds = [None,None,[0]]

			# condition = nearest_neighbourhood(size=size,
			# 								strict=strict,
			# 								atol=atol,rtol=rtol,kind=kind,
			# 								argsortbounds=argsortbounds,weightbounds=weightbounds)
			# def condition(weights,argsort,counts):
			# 	out = weights.astype(bool)
			# 	out.eliminate_zeros()
			# 	return out

			condition = None

			return condition
		returns = (weights(data,parameters),conditions(data,parameters))
		return returns


	@_wrapper
	def complete(data,parameters):
		def weights(data,parameters):
			metric = parameters['metric']			
			format = parameters['format']
			dtype = parameters['dtype']
			chunk = parameters['chunk']		
			n_jobs = parameters['n_jobs']
			verbose = parameters['verbose']			
			weights = similarity_matrix(data,data,metric=metric,directed=False,format=format,dtype=dtype,verbose=verbose,chunk=chunk,n_jobs=n_jobs)
			return weights
		def conditions(data,parameters):
			condition = None
			return condition
		returns = (weights(data,parameters),conditions(data,parameters))
		return returns


	@_wrapper
	def custom(data,parameters):
		returns = tuple((parameters[f](data,parameters) for f in ['weights','conditions']))
		return returns



	locs = locals()
	funcs = {k: f for k,f in locs.items() if callable(f) and not k.startswith('_')}	
	funcs[None] = default
	return funcs



# Operators
def operations():
	
	def _wrapper(func):
		@functools.wraps(func)		
		def _func(x,y,weights,adjacency):

			# Update adjacency to have identical sparsity structure as weights
			adjacency = twin(adjacency,weights)

			out = func(x,y,weights,adjacency)
			if issparse(out):
				out = out.A
			return out
		return _func

	def _dy(x,y,weights,adjacency):
		out = subtract(y,y,where=adjacency)
		return out

	def _dx(x,y,weights,adjacency):
		out = subtract(x,x,where=adjacency)
		return out
	
	@_wrapper
	def constant(x,y,weights,adjacency):
		return y

	@_wrapper	
	def partial(x,y,weights,adjacency):
		out = add(multiply(_dy(x,y,weights,adjacency),_dx(x,y,weights,adjacency),weights),axis=1,size=True)
		return out

	@_wrapper	
	def divergence(x,y,weights,adjacency):
		out = add(multiply(_dy(x,y,weights,adjacency),sqrt(weights)),axis=1,size=True)
		return out

	@_wrapper	
	def laplacian(x,y,weights,adjacency):
		out = add(multiply(_dy(x,y,weights,adjacency),weights),axis=1,size=True)
		return out

	locs = locals()
	funcs = {k:f for k,f in locs.items() if callable(f) and not k.startswith('_')}	

	return funcs
	


# Operator weight - i.e. weight calculation
def weightings():

	def _wrapper(func):
		@functools.wraps(func)
		def _func(data,dimension,accuracy,adjacency,manifold,parameters):

			def update(exception,*args,**kwargs):
				kwargs['parameters']['sparsity'] *= parameters['catch_factor']
				return

			exceptions = (AssertionError,)
			raises = (ValueError,TypeError)
			iterations = parameters['catch_iterations']



			@catch(update,exceptions,raises,iterations)
			def results(data=None,adjacency=None,parameters={}):

				# Get adjacency
				if callable(adjacency):
					adj,wgts = adjacency(data,parameters)
				elif adjacency is None or isinstance(adjacency,str):
					adj,wgts = adjacencies()[adjacency](data,parameters)
				else:
					adj,wgts = adjacency
				if not (isnone(adj) or isarray(adj)):
					raise ValueError("Error - adjacency matrix is not array or None")

				# Get weights
				weights = func(data,adj,wgts,parameters)	

				return weights,adj


			# Update parameters
			assert data.ndim == 2, "Error - incorrect weights data shape %r"%(tuple(data.shape))

			parameters['n'] = data.shape[0]
			parameters['p'] = data.shape[1]
			parameters['dimension'] = dimension
			parameters['accuracy'] = accuracy
			parameters['manifold'] = manifold
			parameters['neighbourhood'] = min(parameters['neighbourhood'],
											ncombinations(parameters['p'],parameters['accuracy'],
															unique=parameters['unique'])-1)	
			constants = {}
			for const in constants:
				value = parameters.get(const,constants[const])
				if callable(value):
					parameters[const] = value(**parameters)
				else:
					parameters[const] = value


			# Get weights and adjacency matrices, catching exceptions and updating parameters until no exceptions
			weights,adjacency = results(data=data,adjacency=adjacency,parameters=parameters)

			return weights,adjacency

		return _func


	@_wrapper
	def decay(data,adjacency,weights,parameters):
		parameters['r'] = metrics()['euclidean'](data,data)
		parameters['R'] = parameters.get('R',np.max(parameters['r'])/2)
		parameters['sigma'] = parameters.get('sigma',0.01*parameters['R'])
		parameters['epsilon'] = parameters.get('epsilon',parameters['p']/2.0)
		parameters['z'] = parameters.get('z',parameters['R']*invert(parameters['sigma'],constant=1.0))
		parameters['W'] = 1
		parameters['C'] = 1

		weights = parameters['C']*func((parameters['r']*invert(parameters['sigma'],constant=0.0)))
		return weights

	@_wrapper	
	def diff(data,adjacency,weights,parameters):
		func = lambda x,eps: invert(x**(2.0+eps),constant=0.0)
		parameters['r'] = metrics()['euclidean'](data,data)
		parameters['R'] = parameters.get('R',np.max(parameters['r'])/2)
		parameters['sigma'] = parameters.get('sigma',0.01*parameters['R'])
		parameters['epsilon'] = parameters.get('epsilon',parameters['p']/2.0)
		parameters['z'] = parameters.get('z',parameters['R']*invert(parameters['sigma'],constant=1.0))
		parameters['W'] = 1
		parameters['C'] = 1
		parameters['sigma'] = 1
		weights = parameters['C']*func((parameters['r']*invert(parameters['sigma'],constant=0.0)))
		return weights

	@_wrapper
	def gauss(data,adjacency,weights,parameters):  
		func = lambda x: np.exp(-(1/2)*(x**2))
		parameters['r'] = metrics()['euclidean'](data,data)
		parameters['R'] = parameters.get('R',np.max(parameters['r'])/2)
		parameters['sigma'] = parameters.get('sigma',0.01*parameters['R'])
		parameters['epsilon'] = parameters.get('epsilon',parameters['p']/2.0)
		parameters['z'] = parameters.get('z',parameters['R']*invert(parameters['sigma'],constant=1.0))
		parameters['W'] = sp.integrate.quad(func,0,parameters['z'],args=(parameters['p']))[0]
		parameters['C'] = ((parameters['z']**parameters['p'])*invert((parameters['sigma']**2)*(parameters['W']),constant=1.0))
		weights = parameters['C']*func(parameters['r']*invert(parameters['sigma'],constant=0.0))
		return weights

	@_wrapper
	def poly(data,adjacency,weights,parameters):
		func = lambda x,eps: invert(x**(2.0+eps),constant=0.0)
		parameters['r'] = metrics()['euclidean'](data,data,where=adjacency)
		parameters['R'] = parameters.get('R',np.max(parameters['r'])/2)
		parameters['sigma'] = parameters.get('sigma',0.01*parameters['R'])
		parameters['epsilon'] = parameters.get('epsilon',parameters['p']/2.0)
		parameters['z'] = parameters.get('z',parameters['R']*invert(parameters['sigma'],constant=1.0))
		parameters['epsilon'] = parameters.get('epsilon',parameters['p']/2)
		parameters['W'] = (parameters['z']**(parameters['p']-parameters['epsilon']))*(1/(parameters['p']-parameters['epsilon']))
		parameters['C'] = ((parameters['z']**(parameters['p']))*invert((parameters['sigma']**2)*(parameters['W']),constant=1.0))
		weights = parameters['C']*func((parameters['r']*invert(parameters['sigma'],constant=0.0)))
		return weights

	@_wrapper
	def stencil(data,adjacency,weights,parameters):

		# Get order of stencil and dimension of derivative
		dimension = parameters['dimension']
		accuracy = parameters['accuracy']
		size = parameters['neighbourhood']

		# Get sort algorithm parameters
		tol,atol,rtol = parameters['tol'],parameters['atol'],parameters['rtol']
		kind = parameters['kind']

		# Get verbose
		verbose = parameters['verbose']

		# Basis of points
		basis = parameters['stencil']

		# Get weights for stencil
		weights = stencils(data,accuracy,size,basis,adjacency=adjacency,weights=weights,dimension=dimension,tol=tol,atol=atol,rtol=rtol,kind=kind,verbose=verbose)

		return weights

	@_wrapper
	def adjacency(data,adjacency,weights,parameters):

		# Get weights as adjacency matrix
		weights = adjacency

		return weights

	@_wrapper
	def default(data,adjacency,weights,parameters):

		# Get weights as adjacency matrix
		weights = adjacency

		return weights


	locs = locals()
	funcs = {k:f for k,f in locs.items() if callable(f) and not k.startswith('_')}	
	funcs[None] = default

	return funcs


# Operator labels
def labelings():
	def label(operation,function,variable,manifold,weight,adjacency,order,dimension,accuracy,**kwargs):
		return [DELIMITER.join([DELIMITER.join(operation[:j]),'%d'%j,function,DELIMITER.join(variable[:j]),
						   DELIMITER.join(weight[:j])]) 
				for j in range(1,order+1)] if order>0 else [function]

	def weighting(operation,function,variable,manifold,weight,adjacency,order,dimension,accuracy,**kwargs):
		return [DELIMITER.join([weight[j-1],manifold[j-1][dimension[j-1]],str(dimension[j-1])])
				for j in range(1,order+1)] if order > 0 else []
		# return [DELIMITER.join([weight[j-1],DELIMITER.join(manifold[j-1]),str(dimension[j-1])])
		# 		for j in range(1,order+1)] if order > 0 else []

	def adjacencies(operation,function,variable,manifold,weight,adjacency,order,dimension,accuracy,**kwargs):
		return [DELIMITER.join([adjacency[j-1],manifold[j-1][dimension[j-1]],str(dimension[j-1])])
				for j in range(1,order+1)] if order > 0 else []		
		# return [DELIMITER.join([adjacency[j-1],DELIMITER.join(manifold[j-1]),str(dimension[j-1])])
		# 		for j in range(1,order+1)] if order > 0 else []

	locs = locals()
	funcs = {k:f for k,f in locs.items() if callable(f) and not k.startswith('_')}	

	return funcs




# Operator function wrapper
def operators(funcs=None):


	if funcs is None:
		funcs = operations()


	# Operators
	def _wrapper(func):
		@functools.wraps(func)
		def _func(x,y,weights,dimension,accuracy,weight,adjacency,manifold,parameters={}):

			if callable(weight):
				pass
			elif weight is None or isinstance(weight,str):
				weight = weightings()[weight]
			else:
				_weight = weight
				weight = lambda weights,dimension,accuracy,adjacency,manifold,parameters,_weight=_weight: (_weight,adjacency)

			if not callable(weight):
				raise ValueError("Error - weight function is not callable")

			parameters = copy.deepcopy(parameters)			
			weights,adjacency = weight(weights,dimension,accuracy,adjacency,manifold,parameters) 

			d = func(x,y,weights,adjacency)

			return d,weights,adjacency

		return _func

	funcs = {func: _wrapper(funcs[func]) for func in funcs}		

	return funcs





# Compute Terms in model
def terms(data,metadata,settings,verbose=False,texify=None):
	'''
	Compute graph operators based on terms and append to data dataframes in place
	Args:
		data (dict): Dictionary of pandas dataframes datasets
		metadata (dict): Dictionary of metadata for each dataset
		settings (dict):
		verbose (str): Print out additional information
	'''
	def get(df,obj,*args,**kwargs):
		try:
			return df[obj].values
		except:
			return obj

	# Dictionaries of functions for different operators (i.e) partial for partial derivatives) and weights (i.e) poly for polynomial weights)
	operator = operators()

	# Setup and parameters are all present
	setup(data,metadata,settings,verbose=verbose)


	# Iterate through each dataset
	for key in data:

		# Get dataframe and list of terms dictionaries for each operator to compute
		df = data[key]
		terms = metadata[key]['terms']
		functions = metadata[key].pop('functions',[])

		if terms is None:
			continue


		# Private variables to reuse labels weights and adjacencies amongst operators and avoid recomputing operator data
		_is = {k:False for k in ['label','weighting','adjacencies']} # Booleans whether variable has been already computed
		_var = {k: None for k in ['label','weighting','adjacencies']} # Current type of each variable (i,e) partial)
		_vars = {k: {} for k in ['label','weighting','adjacencies']} # Dictionary for all values for each type of each variable


		# Iterate over terms and order of term to compute operators
		for term in terms:
			funcs = [term['function'],*term['label']]
			parameters = copy.deepcopy(metadata[key]['parameters'])

			for j in range(term['order']):

				_var['label'] = funcs[j+1]
				_is['label'] = _var['label'] in df
				if _is['label']:
					continue

				_var['weighting'] = term['weighting'][j]
				_is['weighting'] = _var['weighting'] in _vars['weighting']
				
				_var['adjacencies'] = term['adjacencies'][j]
				_is['adjacencies'] = _var['adjacencies'] in _vars['adjacencies']

				# Get values of term fields
				function = get(df,funcs[j])
				variable = get(df,term['variable'][j])
				variables = get(df,term['manifold'][j])
				manifold = term['manifold'][j]
				dimension = term['dimension'][j]
				accuracy = term['accuracy'][j]
				weight = _vars['weighting'][_var['weighting']] if _is['weighting'] else term['weight'][j]
				adj = _vars['adjacencies'][_var['adjacencies']] if _is['adjacencies'] else term['adjacency'][j]
				operation = operator[term['operation'][j]]

				logger.log(verbose,'Computing %s Operator %d with shape %r'%(_var['label'],dimension,variables.shape))

				df[_var['label']],_weight,_adjacencies = operation(variable,function,variables,dimension,accuracy,weight,adj,manifold,parameters)

				if not _is['weighting']:	
					_vars['weighting'][_var['weighting']] = _weight
					if parameters['store'].get('weighting'):
						metadata[key][_var['weighting']] = _weight
				if not _is['adjacencies']:
					_vars['adjacencies'][_var['adjacencies']] = _adjacencies
					if parameters['store'].get('adjacency'):
						metadata[key][_var['adjacencies']] = _adjacencies

		# Compute functions on operators
		for function in functions:
			labels = function['labels']
			if ((isinstance(labels,(list,tuple)) and any([l not in df for l in labels])) or (labels not in df)):
				try:
					df[labels] = function['func'](data[key])
				except:
					pass

	return



# Setup operator terms
def setup(data,metadata,settings,verbose=False):

	defaults = {
		'inputs':[],'outputs':[],'operations':[],'orders':range(1+1),'accuracies':[1],'constants':None,
		'weights':[None],'adjacencies':[None],'terms':[],'parameters':{},'unique':True,'kwargs':{}}


	field = 'terms'
	parameters = {}
	parameters.update({parameter: (defaults.get(parameter,settings[field].get(parameter)) if settings[field].get(parameter) is None else settings[field][parameter]) 
						for parameter in settings[field]})


	# Add parameters to metadata for each dataset, under keyword key in metadata dictionary
	for key in data:

		# Iterate through parameters and add parameter values to metadata, depending on value types
		for parameter in parameters:
			value = copy.deepcopy(parameters[parameter])
			try:
				metadata[key][parameter] = value[key]
			except:
				metadata[key][parameter] = value

			if isinstance(metadata[key][parameter],dict):
				for field in metadata[key][parameter]:
					value = copy.deepcopy(metadata[key][parameter][field])
					try:
						metadata[key][parameter][field] = value[key]
					except:
						metadata[key][parameter][field] = value


		# Handle terms parameter in metadata and check if fields are present and labelled
		parameter = 'terms'
		labelor = labelings()
		fields = ['label','weighting','adjacencies']	

		# Setup list of dictionaries of terms parameter in metadata parameters
		if metadata[key].get(parameter) is None:
			metadata[key][parameter] = {}
		_setup(data[key],**metadata[key])

		# Get label formatters for fields
		for value in metadata[key][parameter]:
			for field in fields:
				if field not in value:
					if field in labelor:
						value[field] = labelor[field](**{**metadata[key],**value})
					else:
						value[field] = metadata[key][field]

	



	return


# Get default terms
def _setup(data,inputs,outputs,operations,orders,accuracies,constants,weights,adjacencies,terms,unique,**kwargs):
	
	# Return if terms already contains variables
	if len(terms)>0:
		return

	# Check variables are correct type
	_terms = [
			 *[{'function':y,
				'variable': list(x),
				'manifold':[[*inputs]]*max(1,j),
				'weight':[w]*max(1,j),
				'adjacency':[a]*max(1,j),
				'accuracy': [accuracy]*max(j,1),		
				'dimension':[inputs.index(u) for u in x],	
				'order':j,
				'operation':[o]*max(1,j),
			   }
			   for j in orders
			   for o in operations
			   for w in weights
			   for a in adjacencies
			   for accuracy in accuracies
			   for x in icombinations(inputs,[j],unique=unique)
			   for y in outputs   
			   if (
				  (constants is None) or 
				  (not any([(v in constants.get(y,[])) for v in x]))
				   )
			  ],
			]

	terms.extend(_terms)

	return
