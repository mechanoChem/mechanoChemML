#!/usr/bin/env python

# Import python modules
import os,sys,copy,warnings,itertools,inspect
import glob,json,jsonpickle,h5py,pickle,dill
import numpy as np
import pandas as pd


from natsort import natsorted, ns,index_natsorted,order_by_index

# Logging
import logging
log = 'info'
logger = logging.getLogger(__name__)
# logger.setLevel(getattr(logging,log.upper()))


# Split path into directory,file,ext
def path_split(path,directory=False,file=False,ext=False,directory_file=False,file_ext=False,ext_delimeter='.'):
	if not (directory or file or ext):
		return path
	returns = {'directory':directory,'file':file or directory_file or file_ext,'ext':ext}
	paths = {}
	paths['directory'] = os.path.dirname(path)
	paths['file'],paths['ext'] = os.path.splitext(path)
	if paths['ext'].startswith(ext_delimeter):
		paths['ext'] = ext_delimeter.join(paths['ext'].split(ext_delimeter)[1:])
	if not directory_file:
		paths['file'] = os.path.basename(paths['file'])
	if file_ext and paths['ext'].startswith(ext_delimeter):
		paths['file'] = ext_delimeter.join([paths['file'],paths['ext']])
	paths = [paths[k] for k in paths if returns[k]] 
	return paths if len(paths)>1 else paths[0]

# Join path by directories, with optional extension
def path_join(*paths,ext=None,abspath=False,ext_delimeter='.'):
	path = os.path.join(*paths)
	if ext is not None and not path.endswith('%s%s'%(ext_delimeter,ext)):
		path = ext_delimeter.join([path,ext])
	if abspath:
		path = os.path.abspath(path)
	return path


# glob path
def path_glob(path,**kwargs):
	return glob.glob(os.path.abspath(os.path.expanduser(path)),**kwargs)


# Class wrapper for functions
class funcclass(object):
	def __init__(self,func=lambda x:x):
		self.func = func
	def __call__(self,*args,**kwargs):
		return self.func(*args,**kwargs)

# Serialize object to JSON
def serialize(obj,key='py/object'):
	if callable(obj) or isinstance(obj,(slice,range)):
		if callable(obj) and not inspect.isclass(obj):            
			obj = funcclass(obj)
		obj = jsonpickle.encode(obj)
	elif isinstance(obj,np.ndarray):
		obj = obj.tolist()
	return obj

# Deserialize object from JSON
def deserialize(obj,key='py/object'):
	if isinstance(obj,dict) and key in obj:
		obj = pickle.loads(str(obj[key]))
	# return  jsonpickle.decode(str(obj))
	return obj

# Load data - General file import
def load(path,wr='r',default=None,verbose=False,**kwargs):
	loaders = {**{ext: (lambda obj,ext=ext,**kwargs:getattr(pd,'read_%s'%ext)(obj,**kwargs)) for ext in ['csv']},
			   **{ext: (lambda obj,ext=ext,**kwargs:getattr(pd,'read_%s'%ext)(obj,**kwargs) if wr=='r' else (pickle.load(obj,**kwargs))) for ext in ['pickle']},
			   **{ext: (lambda obj,ext=ext,**kwargs: json.load(obj,**{'object_hook':deserialize,**kwargs})) for ext in ['json']},
			  }
	if not isinstance(path,str):
		return default
	
	if path is None:
		return default

	ext = path.split('.')[-1]
	if ('.' in path) and (ext in loaders):
		paths = {ext: path}
	else:
		paths = {e: '%s.%s'%(path,e) for e in loaders}
	loaders = {paths[e]: loaders[e] for e in paths}
	for path in loaders:
		loader = loaders[path]
		for wr in [wr,'r','rb']:
			try:
				data = loader(path,**kwargs)
				logger.log(verbose,'Loading path %s'%(path))
				return data
			except Exception as e:
				try:
					with open(path,wr) as obj:
						data = loader(obj,**kwargs)
						logger.log(verbose,'Loading obj %s'%(path))
						return data
				except:
					pass

	return default			
		
# Dump data - General file save/export
def dump(data,path,wr='w',verbose=False,**kwargs):

	dumpers = {**{ext: (lambda data,obj,ext=ext,**kwargs:getattr(data,'to_%s'%ext)(obj,**{'index':False,**kwargs})) for ext in ['csv']},
			   **{ext: (lambda data,obj,ext=ext,**kwargs:getattr(data,'to_%s'%ext)(obj,**kwargs) if isinstance(data,pd.DataFrame) else pickle.dump(data,obj,protocol=pickle.HIGHEST_PROTOCOL,**kwargs)) for ext in ['pickle']},
			   **{ext: (lambda data,obj,ext=ext,**kwargs: json.dump(data,obj,**{'default':serialize,'ensure_ascii':False,'indent':4,**kwargs})) for ext in ['json']},
			   **{ext: (lambda data,obj,ext=ext,**kwargs: obj.write(data,**kwargs)) for ext in ['tex']},
			  }

	if path is None:
		return
	ext = path.split('.')[-1]
	if ('.' in path) and (ext in dumpers):
		paths = {ext: path}
	else:
		paths = {e: '%s.%s'%(path,e) for e in dumpers}
		return
	dumpers = {paths[e]: dumpers[e] for e in paths}

	for path in dumpers:
		dirname = os.path.abspath(os.path.dirname(path))
		if not os.path.exists(dirname):
			os.makedirs(dirname)

	for path in dumpers:
		dumper = dumpers[path]
		for _wr in [wr,'w','wb']:		
			with open(path,_wr) as obj:
				try:
					dumper(data,path,**kwargs)
					logger.log(verbose,'Dumping path %s'%(path))
					return
				except Exception as e:
					try:
						dumper(data,obj,**kwargs)
						logger.log(verbose,'Dumping obj %s'%(path))
						return
					except Exception as e:
						try:
							# dumper(pickleable(copy.deepcopy(data),_return=True),path,**kwargs)
							dumper(data,path,**kwargs)
						except:	
							try:					
								# dumper(pickleable(copy.deepcopy(data),_return=True),obj,**kwargs)
								dumper(data,obj,**kwargs)
							except:
								pass

	return			



# Check if object can be written to file
# Check if object can be pickled
def pickleable(obj,path=None,_return=False):
	if isinstance(obj,dict):
		pickleables = {k: pickleable(obj[k],path,_return=False) for k in obj} 
		for k in pickleables:
			if not pickleables[k]:
				obj.pop(k);
				pickleables[k] = True
		if _return:
			return obj
		else:
			return all([pickleables[k] for k in pickleables])
	ispickleable = False
	if path is None:
		path  = '__tmp__.__tmp__.%d'%(np.random.randint(1,int(1e8)))
	with open(path,'wb') as fobj:
		try:
			pickle.dump(obj,fobj)
			ispickleable = True
		except Exception as e:
			pass
	if os.path.exists(path):
		os.remove(path)
	return ispickleable

# Import Data as pandas dataframe
def importer(files,directory,wr='rb',verbose=False):

	paths = []
	for file in files:
		path = path_join(directory,file,abspath=True)
		paths.extend(path_glob(path,recursive=True))
	paths = natsorted(paths)
	data = [load(path,wr=wr,verbose=verbose) for path in paths]
	if len(data) > 0:
		df = pd.concat(data,axis=0,ignore_index=True)
		return df
	else:
		return None


# Sort Data
def sorter(seq,index,multiple=False,wrapper=lambda arr:arr):
	index = index_natsorted(index)
	if multiple:
		return [wrapper(order_by_index(s,index)) for s in seq]
	else:
		return wrapper(order_by_index(seq,index))


# Flattening multi-dimensional data
def _flatten(df,exceptions=[]):
	if df is None:
		return
	for label in df:
		if label in exceptions:
			continue
		data = np.array([i for i in df[label].values]) #.astype('float64')
		if data.ndim<=1:
			continue

		shape = data.shape
		labels = itertools.product(*[range(i) for i in shape[1:]])
		for _label in labels:
			d = data.reshape(*shape[1:],shape[0])
			for i in _label:
				d = d[i]
			_label = '_'.join([label,*[str(i) for i in _label]])
			df[_label] = d
		df.drop(columns=label,inplace=True)
	return	

# Setup Data - Global function to call specific setup functions
def setup(data,metadata,files,directories__load,directories__dump=None,metafile=None,wr='rb',flatten_exceptions=[],verbose=False,**kwargs):
	_setups = _setup()
	_setup_ = _setups['default']
	
	# Get loader depending on extension of files
	for exts in _setups:
		if any([path_split(file,ext=True) == exts for file in files]):
			_setup_ = _setups[exts]
			break

	# Get regexed directories
	if directories__dump is None:
		directories__dump = directories__load.copy()
	_directories(directories__load,directories__dump)

	# Load Data
	_setup_(data,metadata,files,directories__load,metafile,wr,flatten_exceptions,verbose,**kwargs)


	for key in data:
		directory_load = metadata[key].pop('directory')
		for dir_load,dir_dump in zip(directories__load,directories__dump):
			if dir_load in directory_load and len(dir_load)>=len(directory_load):
				directory_dump = dir_dump
				break
		metadata[key]['directory'] = {'load':directory_load,'dump':directory_dump}
		metadata[key]['type'] = 'imported'

	return


def _directories(directories__load,directories__dump):

	def replace(strings,patterns,threshold=None,delimeter=''):
		if threshold is None:
			threshold = len(patterns)
		matches = []
		nonmatches = []
		for pattern in patterns:
			if patttern in strings:
				matches.append(pattern)
			else:
				nonmatches.append(pattern)
		if ((isinstance(threshold,(int,np.integer)) and (len(matches)>=threshold)) or (callable(threshold) and threshold(matches))):
			for i,(substring,pattern) in ennumerate(zip(strings,nonmatches)):
				strings[i] = patterns

		string = strings.join(delimeter)
		return string




	# Glob directories patterns
	# Ensure directories exist

	for directories in 	[directories__load,directories__dump]:
		directories_split = [directory.split('*') for directory in directories]
		directories_glob = natsorted([d for directory in directories 
								for d in path_glob(directory)])		
		if len(directories_glob) <= len(directories):
			for directory in directories:
				if not os.path.exists(directory):
					os.makedirs(directory)

		directories_glob = natsorted([d for directory in directories 
						for d in path_glob(directory)])		

		directories.clear()
		directories.extend(directories_glob)
	
	if len(directories__dump) < len(directories__load):
		directories__dump.clear()
		directories__dump.extend([directory for directory in directories__load])

	for directories in [directories__load,directories__dump]:
		for directory in directories:
			if not os.path.exists(directory):
				os.makedirs(directory)

	return


def _setup():	

	# Default CSV/Pickle file input

	def _default(data,metadata,files,directories,metafile=None,wr='rb',flatten_exceptions=[],verbose=False,**kwargs):
		
		for directory in directories:
			key = directory #os.path.basename(directory) 
			data[key] = importer(files,directory,wr=wr,verbose=verbose)
			if data[key] is None:
				data.pop(key)
				continue
			_flatten(data[key],flatten_exceptions)
			metadata[key] = {}
			if isinstance(metafile,str):
				metadata[key] = load(path_join(directory,metafile),default=metadata[key],wr=wr,verbose=verbose)

			metadata[key]['directory'] = directory
			logger.log(verbose,'Importing: %s %r'%(key,data[key].shape if data[key] is not None else None))

		return


	# csv file input
	def _csv(*args,**kwargs):
		return _default(*args,**kwargs)

	# csv file input
	def _pickle(*args,**kwargs):
		return _default(*args,**kwargs)

	
	# HDf5 file input
	def _h5(data,metadata,files,directories,metafile=None,wr='rb',flatten_exceptions=[],verbose=False,**kwargs):

		def zero_check(x,val=0):
			try:
				return x[~np.equal(x,val)][0]
			except IndexError:
				return val
		join = lambda *args,seperator = '/':seperator.join(["",*args])
		name = lambda state: int(state.split('_')[-1])
		
		for directory in directories:
			paths = []
			for file in files:
				path = path_join(directory,file,abspath=True)
				paths.extend(path_glob(path,recursive=True))
			paths = natsorted(paths)
			for path in paths:

				h = h5py.File(path,'r')

				keys = kwargs.get('keys',[{},{}])
				groups = kwargs.get('groups',[]) 
				states = [k for k in h.keys() if groups[0] in k]
				views = {}

				for s in states[:]:

					labels = {**{keys[0][key]:join(s,key) for key in keys[0]}, 
							  **{keys[1][key]:join(s,*groups[1:],key) 
							     for key in keys[1]}}

					s = name(s)
					views[s] = {}

					for label in labels:
						try:
							dataset = h[labels[label]]
							shape = dataset.shape
							ndim = dataset.ndim
							if ndim == 0:
								views[s][label] = float(dataset[...])
							elif ndim == 3:
								views[s][label] = zero_check(dataset[...])
							elif ndim == 4:
								for i in range(shape[0]):
									views[s]['%s_%d_%d'%(label,i,0)] = zero_check(dataset[i])
							elif ndim == 5:
								for i in range(shape[0]):
									for j in range(shape[1]):                    
										views[s]['%s_%d_%d'%(label,i,j)] = zero_check(dataset[i][j])                      
						except KeyError:
							pass

				key = directory

				df = pd.DataFrame.from_dict(views, orient='index')
				df.index.name = join(*groups).split('_')[0].replace('/','')
				df.reset_index(drop=False,inplace=True)
				df.to_pickle(path.replace('h5','pickle'))
				if data.get(key) is not None:
					data[key] = pd.concat([data[key],df],axis=0,ignore_index=True)
				else:
					data[key] = df.copy()


				if metadata.get(key) is None:
					metadata[key] = {}
					if isinstance(metafile,str):
						metadata[key] = load(path_join(directory,metafile),default=metadata[key],wr='rb',verbose=verbose)
					metadata[key]['directory'] = directory

				logger.log(verbose,'Importing: %s [%d]'%(key,len(data[key])))
		return



	def _mat(data,metadata,files,directories,metafile=None,wr='r',flatten_exceptions=[],verbose=False,**kwargs):

		loader = sp.io.loadmat

		for directory in directories:
			paths = []
			for file in files:
				path = path_join(directory,file,abspath=True)
				paths.extend(path_glob(path,recursive=True))
			paths = natsorted(paths)
			for path in paths:
				try: 
					_data = loader(path)
				except:
					return
				labels = [k for k in _data if ((not k.startswith('__')) and (not k.endswith('__')))]
				shapes = {k: np.array(_data[k].shape) for k in labels if all([s>1 for s in list(_data[k].shape)])}
				labels = list(shapes)
				dims = min([shapes[k].size for k in shapes])
				shape = [max([shapes[k][i] for k in shapes]) for i in range(dims)]

				keys = ['_'.join(str(j) for j in i) for i in itertools.product(*[range(i) for i in shape])]
				data.update({k:pd.DataFrame() for k in keys if k not in data})
				for key in keys:
					_shape = [int(i) for i in key.split('_')]
					for label in labels:
						d = _data[label]
						for i in _shape:
							d = d[i]
						d = np.squeeze(d.reshape(sorted(d.shape,reverse=True)))
						if d.ndim>1:
							d = [i for i in d]
						data[key][label] = d
						metadata[key] = {}
						if isinstance(metafile,str):
							metadata[key] = load(path_join(directory,metafile),metadata[key],wr='rb',verbose=verbose)
						metadata[key]['directory'] = directory
		return



	# xarray file input
	def _nc(data,metadata,files,directories,metafile=None,wr='r',flatten_exceptions=[],verbose=False,**kwargs):
		try:
			import xarray as xr
		except:
			return

		for directory in directories:
			paths = []
			for file in files:
				path = path_join(directory,file,abspath=True)
				paths.extend(path_glob(path,recursive=True))
			paths = natsorted(paths)
			for path in paths:

				df = xr.open_dataset(path).to_dataframe()

				df.reset_index(drop=False,inplace=True)
				df.rename(columns=lambda x:x.upper(),inplace=True)


				inputs = ['BURNUP',  'FUELTEMP', 'MODTEMP', 'MODDENS', 'BORON','BANK_POS',
				          'FUELTEMP_AVG', 'MODTEMP_AVG', 'MODDENS_AVG', 'BANK_POS_AVG']
				outputs = ['NXSF', 'XSF', 'XSRM', 'XSTR', 'XSS']
				groupby = ['INGROUP', 'OUTGROUP']
				label = 'SAMPLE'

				values = itertools.product(*[df[g].unique() for g in groupby])
				df_ = None
				for value in values:
				    _df = df.copy()
				    _df = _df[(_df[groupby]==value).all(axis=1)]
				    _df.rename(columns={y: '%s_%s'%(y,'_'.join([str(x) for x in value]))  for y in outputs},inplace=True)
				    if df_ is None:
				        df_ = _df.copy().drop(groupby,axis=1).reset_index(drop=True)
				    else:
				        df.reset_index()
				        df_ = pd.merge(df_,_df.drop(groupby,axis=1).reset_index(drop=True),on=[label,*inputs] if label in df_ else inputs,how='outer',copy=False).reset_index(drop=True)
				    df_ = df_.T.drop_duplicates().T

				df = df_


				if label in df and len(df[label].unique()) > 1:
					df_groupby = df.groupby(label)
					for key in df_groupby.groups:
						key = '%s_%s'%(path,key)
						data[key] = df_groupby.get_group(key).drop(label,axis=1).reset_index(drop=True)
						metadata[key]['directory'] = directory
				else:
					key = path
					data[key] = df.reset_index(drop=True)
					metadata[key]['directory'] = directory


		return

	locs = locals()

	funcs = {k[1:]:locs[k] for k in locs if callable(locs[k]) and k.startswith('_')}
	return funcs
