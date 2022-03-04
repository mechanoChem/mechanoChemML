#!/usr/bin/env python

# Import python modules
import os,sys,copy,warnings,functools,itertools,inspect,timeit
from natsort import natsorted
import numpy as np
import scipy as sp
import scipy.stats,scipy.signal
import pandas as pd
import sparse as sparray
# import numba as nb



# import multiprocess as mp
# import multithreading as mt
import joblib
import multiprocessing as multiprocessing
import multiprocessing.dummy as multithreading

# warnings.simplefilter("ignore", (UserWarning,DeprecationWarning,FutureWarning))
# warnings.simplefilter("ignore", (sp.sparse.SparseEfficiencyWarning))
# warnings.filterwarnings('error',category=sp.sparse.SparseEfficiencyWarning)

DELIMITER='__'
MAX_PROCESSES = 8

# Logging
import logging
log = 'info'
logger = logging.getLogger(__name__)
#logger.setLevel(getattr(logging,log.upper()))		
from progress.bar import Bar





# Sparse array class
class sparsearray(sparray.COO):
	def __init__(self,shape,data=None,coords=None,dtype=None,fill_value=None):
		self.shape = shape
		self.ndim = len(self.data)		
		self.data = data if data is not None else np.array([])
		self.coords = coords if coords is not None else (np.array([]),)*self.ndim
		self.dtype = dtype if dtype is not None else self.data.dtype
		self.fill_value = fill_value if fill_value is not None else 0
		
		self.nnz = self.data.size

		return


# Quasi array class
class ndarray(object):
	def __init__(self,data,row,col):
		self.data = data
		self.row = row
		self.col = col
	def eliminate_zeros(self):
		inds = where(self.data==0)
		self.data = delete(self.data,inds)
		arr.row = delete(self.row,inds)
		arr.col = delete(self.col,inds)
		return


def timing(verbose):
	''' 
	Timing function wrapper
	
	'''
	def decorator(func):
		@functools.wraps(func)
		def wrapper(*args,**kwargs):
			if verbose:
				time = 0
				time = timeit.default_timer() - time
				value = func(*args,**kwargs)
				time = timeit.default_timer() - time
				logger.log(verbose,'%r: %r s'%(repr(func),time))
			else:
				value = func(*args,**kwargs)				
			return value
		return wrapper
	return decorator


# Wrapper class for function, with
# class args and kwargs assigned after
# classed args and kwrags
class wrapper(object):
	def __init__(self,_func,*args,**kwargs):
		self.func = _func
		self.args = args
		self.kwargs = kwargs
		functools.update_wrapper(self, _func)
		return

	def __call__(self,*args,**kwargs):
		args = [*args,*self.args]
		kwargs = {**self.kwargs,**kwargs}
		return self.func(*args,**kwargs)

	def __repr__(self):
		return self.func.__repr__()

	def __str__(self):
		return self.func.__str__()


# Decorator with optional additional arguments
def decorator(*ags,**kwds):
	def wrapper(func):
		@functools.wraps(func)
		def function(*args,**kwargs):
			args = list(args)
			args.extend(ags)
			kwargs.update(kwds)
			return func(*args,**kwargs)
		return function
	return wrapper


# Context manager
class context(object):
	def __init__(self,func,*args,**kwargs):
		self.obj = func(*args,**kwargs)		
	def __enter__(self):
		return self.obj
	def __exit__(self, type, value, traceback):
		self.obj.__exit__(type,value,traceback)

# Empty Context manager
class emptycontext(object):
	def __init__(self,func,*args,**kwargs):
		self.obj = func	
	def __call__(self,*args,**kwargs):
		return self
	def __enter__(self,*args,**kwargs):
		return self.obj
	def __exit__(self, type, value, traceback):
		try:
			self.obj.__exit__(type,value,traceback)
		except:
			pass
		return


# Null Context manager
class nullcontext(object):
	def __init__(self,*args,**kwargs):
		return
	def __call__(self,*args,**kwargs):
		return self
	def __enter__(self,*args,**kwargs):
		return self
	def __exit__(self, type, value, traceback):
		return


def nullfunc(*args,**kwargs):
	return


class nullclass(object):
	pass


# Call function with proper signature
def call(cls,func,*args,**kwargs):
	try:
		func = getattr(cls,func,func)	
	except:
		pass
	assert callable(func), "Error - cls.func or func not callable"

	params = inspect.signature(func).parameters.values()
	arguments = []
	keywords = {}    

	for param in params:
		name = param.name
		default = param.default
		kind = str(param.kind)
		if kind in ['VAR_POSITIONAL']:
			if name in kwargs:
				keywords[name] = kwargs.get(name,default)
			arguments.extend(args)
		elif kind in ['VAR_KEYWORD']:
			keywords.update(kwargs)
		elif kind not in ['POSITIONAL_OR_KEYWORD'] and default is param.empty:
			pass
		else:
			keywords[name] = kwargs.get(name,default)

	return func(*arguments,**keywords)


def empty(obj,*attrs):
	class Empty(obj.__class__):
		def __init__(self): pass
	newobj = Empty()
	newobj.__class__ = obj.__class__
	for attr in inspect.getmembers(obj):
		attr = attr[0]
		if attr in attrs:
			setattr(newobj,attr,copy.deepcopy(getattr(obj,attr)))
	newobj.__dict__.update({attr: obj.__dict__.get(attr) 
						   for attr in attrs if not getattr(newobj,attr,False)})
	return newobj


# Pool class, similar to multiprocessing.Pool
class Pooler(object):
	def __init__(self,processes,pool=None,initializer=None,initargs=(),maxtasksperchild=None,context=None):
		self.set_processes(processes)
		self.set_pool(pool)
		self.set_initializer(initializer)
		self.set_initargs(initargs)
		self.set_maxtasksperchild(maxtasksperchild)
		self.set_context(context)
		return
	@timing(False)	
	def __call__(self,module,func,iterable,args=(),kwds={},callback_args=(),callback_kwds={},callback=nullfunc,error_callback=nullfunc):
		with self.get_pool(
			processes=self.get_processes(),
			initializer=self.get_initializer(),
			initargs=self.get_initargs(),
			maxtasksperchild=self.get_maxtasksperchild(),
			context=self.get_context()) as pool:

			self.set_iterable(iterable)
			jobs = (getattr(pool,module)(
					func=wrapper(func,*args,**{**kwds,**i}),
					**(dict(callback=wrapper(callback,*callback_args,**{**callback_kwds,**i}),
					error_callback=wrapper(error_callback,*callback_args,**{**callback_kwds,**i}))
					if 'async' in module else dict()))
					for i in self.get_iterable())
						

			start = timeit.default_timer()	
			pool.close()
			pool.join()
			end = timeit.default_timer()

			if not self.get_null():
				logger.log(self.get_verbose(),"processes: %d, time: %0.3e"%(self.get_processes(),end-start))							

		return

	def set_pool(self,pool):
		attr = 'pool'  
		self.set_null()
		if self.get_null():
			value = nullPool
		elif value in [False]:
			value = nullPool
		elif pool in [None,True]:
			value = Pool 
		elif callable(pool):
			value = emptycontext(pool)	
		else:
			value = pool		
		setattr(self,attr,value)
		return 
	def get_pool(self,default=None):
		attr = 'pool'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value

	def set_processes(self,processes):
		attr = 'processes'
		default = 1
		processes = default if processes is None else processes
		processes = min(processes,MAX_PROCESSES-1)
		setattr(self,attr,processes)
		self.set_null()
		return
	def get_processes(self,default=None):
		attr = 'processes'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value

	def set_initializer(self,initializer):
		attr = 'initializer'
		value = initializer
		setattr(self,attr,value)
		return
	def get_initializer(self,default=None):
		attr = 'initializer'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value

	def set_initargs(self,initargs):
		attr = 'initargs'
		value = initargs
		setattr(self,attr,value)
		self.initargs = initargs
		return
	def get_initargs(self,default=None):
		attr = 'initargs'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value

	def set_maxtasksperchild(self,maxtasksperchild):
		attr = 'maxtasksperchild'
		value = maxtasksperchild
		setattr(self,attr,value)
		self.initargs = initargs
		return
	def get_maxtasksperchild(self,default=None):
		attr = 'maxtasksperchild'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value

	def set_context(self,context):
		attr = 'context'
		value = context
		setattr(self,attr,value)
		self.initargs = initargs
		return
	def get_context(self):
		attr = 'context'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value

	def set_verbose(self,verbose):  
		attr = 'verbose'
		if verbose is None:
			value = False 
		else:
			value = verbose
		setattr(self,attr,value)
		return 
	def get_verbose(self,default=None):
		attr = 'verbose'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value

	def set_iterable(self,iterable):
		attr = 'iterable'
		if isinstance(iterable,dict):
			keys = list(iterable)
			value = (dict(zip(keys,values)) 
							for values in itertools.product(*[iterable[key] 
															for key in keys]))
		elif isinstance(iterable,int):
			value = ({'i':i} for i in range(iterable))
		else:
			value = iterable
		setattr(self,attr,value)
		return
	def get_iterable(self,default=None):
		attr = 'iterable'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value		

	def set_null(self):
		attr = 'null'
		min_processes = 2
		value = self.get_processes() < min_processes
		setattr(self,attr,value)
		return
	def get_null(self,default=None):
		attr = 'null'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value	

# nullPool class, similar to multiprocessing.Pool
class Pool(multiprocessing.pool.Pool):
	pass

# nullPool class, similar to multiprocessing.Pool
class nullPool(object):
	def __init__(self,processes=None,initializer=None,initargs=(),maxtasksperchild=None,context=None):
		return
	@timing(False)	
	def apply(self,func,args=(),kwds={}):
		return func(*args,**kwds)
	@timing(False)
	def apply_async(self,func,args=(),kwds={},callback=nullfunc,error_callback=nullfunc):
		try:
			callback(func(*args,**kwds))
		except:
			error_callback(func(*args,**kwds))
		return
	@timing(False)	
	def map(self,func,iterable,chunksize=None):
		return list(map(func,iterable))
	@timing(False)		
	def map_async(self,func,iterable,chunksize=None,callback=nullfunc,error_callback=nullfunc):
		try:
			map(callback,list(map(func,iterable)))
		except:
			map(error_callback,list(map(func,iterable)))
		return 
	@timing(False)		
	def imap(self,func,iterable,chunksize=None):
		return list(map(func,iterable))
	@timing(False)	
	def imap_unordered(self,func,iterable,chunksize=None):
		return list(map(func,iterable))
	@timing(False)	
	def starmap(self,func,iterable,chunksize=None):
		return list(map(func,iterable))
	@timing(False)	
	def starmap_async(self,func,iterable,chunksize=None,callback=nullfunc,error_callback=nullfunc):
		try:
			map(callback,list(map(func,iterable)))
		except:
			map(error_callback,list(map(func,iterable)))
		return 		
	def close(self):
		pass
	def join(self):
		pass
	
	def set_processes(self,processes):
		attr = 'processes'
		default = 1
		processes = default if processes is None else processes
		processes = min(processes,MAX_PROCESSES-1)
		setattr(self,attr,processes)
		self.set_null()
		return
	def get_processes(self,default=None):
		attr = 'processes'
		if not hasattr(self,attr):
			self.set_processes(default)
		return getattr(self,attr)

	def set_verbose(self,verbose):  
		attr = 'verbose'
		if verbose is None:
			value = False 
		else:
			value = verbose
		setattr(self,attr,value)
		return 
	def get_verbose(self,default=None):
		attr = 'verbose'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value

	def set_iterable(self,iterable):
		attr = 'iterable'
		if isinstance(iterable,dict):
			keys = list(iterable)
			value = (dict(zip(keys,values)) 
							for values in itertools.product(*[iterable[key] 
															for key in keys]))
		elif isinstance(iterable,int):
			value = ({'i':i} for i in range(iterable))
		else:
			value = iterable
		setattr(self,attr,value)
		return
	def get_iterable(self,default=None):
		attr = 'iterable'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value	

	def set_null(self):
		attr = 'null'
		min_processes = 2
		value = self.get_processes() < min_processes
		setattr(self,attr,value)
		return
	def get_null(self,default=None):
		attr = 'null'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value	


# Parallelize iterations, similar to joblib
class Parallelize(object):
	def __init__(self,n_jobs,backend=None,parallel=None,delayed=None,prefer=None,verbose=False):
		self.set_n_jobs(n_jobs)
		self.set_backend(backend)
		self.set_parallel(parallel)
		self.set_delayed(delayed)
		self.set_prefer(prefer)
		self.set_verbose(verbose)
		return
	@timing(False)
	def __call__(self,func,iterable,values,*args,**kwargs):
		with self.get_parallel()(n_jobs=self.get_n_jobs(),backend=self.get_backend(),prefer=self.get_prefer()) as parallel:           
			self.set_iterable(iterable)
			jobs = (self.get_delayed()(func)(*args,**{**kwargs,**i}) 
					for i in self.get_iterable())

			start = timeit.default_timer()	
			values.extend(parallel(jobs))
			end = timeit.default_timer()

			if not self.get_null():
				logger.log(self.get_verbose(),"n_jobs: %d, time: %0.3e"%(self.get_n_jobs(),end-start))
		return 
	def __enter__(self,*args,**kwargs):
		return self
	def __exit__(self, type, value, traceback):
		return 

	def set_n_jobs(self,n_jobs):  
		attr = 'n_jobs'
		if n_jobs is None:
			n_jobs = 1  
		value = max(1,min(joblib.effective_n_jobs(n_jobs),MAX_PROCESSES-1))
		setattr(self,attr,value)
		self.set_null()		
		return 
	def get_n_jobs(self,default=None):
		attr = 'n_jobs'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value	

	def set_backend(self,backend):  
		attr = 'backend'
		if backend is None:
			value = 'loky'  
		else:
			value = backend
		setattr(self,attr,value)
		return 
	def get_backend(self,default=None):
		attr = 'backend'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value	

	def set_parallel(self,parallel):
		attr = 'parallel'  
		self.set_null()
		if self.get_null():
			value = nullParallel
		elif parallel in [False]:
			value = nullParallel
		elif parallel in [None,True]:
			value = Parallel 
		elif callable(parallel):
			value = emptycontext(parallel)	
		else:
			value = parallel
		setattr(self,attr,value)
		return 
	def get_parallel(self,default=None):
		attr = 'parallel'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value	

	def set_delayed(self,delayed):
		attr = 'delayed'  
		if delayed is None:
			value = Delayed 
		else:
			value = delayed
		setattr(self,attr,value)
		return 
	def get_delayed(self,default=None):
		attr = 'delayed'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value	

	def set_prefer(self,prefer):  
		attr = 'prefer'
		if prefer is None:
			value = None 
		else:
			value = prefer
		setattr(self,attr,value)
		return 
	def get_prefer(self,default=None):
		attr = 'prefer'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value			

	def set_verbose(self,verbose):  
		attr = 'verbose'
		if verbose is None:
			value = False 
		else:
			value = verbose
		setattr(self,attr,value)
		return 
	def get_verbose(self,default=None):
		attr = 'verbose'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value			

	def set_iterable(self,iterable):
		attr = 'iterable'
		if isinstance(iterable,dict):
			keys = list(iterable)
			value = (dict(zip(keys,values)) 
							for values in itertools.product(*[iterable[key] 
															for key in keys]))
		elif isinstance(iterable,int):
			value = ({'i':i} for i in range(iterable))
		else:
			value = iterable
		setattr(self,attr,value)
		return
	def get_iterable(self,default=None):
		attr = 'iterable'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value	

	def set_null(self):
		attr = 'null'
		min_n_jobs = 2
		value = self.get_n_jobs() < min_n_jobs
		setattr(self,attr,value)
		return
	def get_null(self,default=None):
		attr = 'null'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value			


# Parallel class using joblib
class Parallel(joblib.Parallel):
	pass


# null Parallel class using joblib
class nullParallel(joblib.Parallel):
	@timing(False)
	def __call__(self,jobs):
		return [func(*args,**kwargs) for func,args,kwargs in jobs]


# Delayed function call for parallelization using joblib
class Delayed(object):
	def __init__(self,function, check_pickle=None):
		self.function = joblib.delayed(function)
		return
	def __call__(self,*args,**kwargs):
		return self.function(*args,**kwargs)



# TODO: Modify to match class API of concurrent futures 
# (can be also similar to structure of __call__ and get_exucator_pool in place of get_pool, with a Future() and nullFuture() class as the context)
# Futures class, similar to concurrent.futures
class Futures(object):
	def __init__(self,processes,pool=None,initializer=None,initargs=(),maxtasksperchild=None,context=None):
		self.set_processes(processes)
		self.set_pool(pool)
		self.set_initializer(initializer)
		self.set_initargs(initargs)
		self.set_maxtasksperchild(maxtasksperchild)
		self.set_context(context)
		return
	@timing(False)	
	def __call__(self,module,func,iterable,args=(),kwds={},callback_args=(),callback_kwds={},callback=nullfunc,error_callback=nullfunc):
		with self.get_pool(
			processes=self.get_processes(),
			initializer=self.get_initializer(),
			initargs=self.get_initargs(),
			maxtasksperchild=self.get_maxtasksperchild(),
			context=self.get_context()) as pool:

			self.set_iterable(iterable)
			jobs = (getattr(pool,module)(
					func=wrapper(func,*args,**{**kwds,**i}),
					**(dict(callback=wrapper(callback,*callback_args,**{**callback_kwds,**i}),
					error_callback=wrapper(error_callback,*callback_args,**{**callback_kwds,**i}))
					if 'async' in module else dict()))
					for i in self.get_iterable())
						

			start = timeit.default_timer()	
			pool.close()
			pool.join()
			end = timeit.default_timer()

			logger.log(self.get_verbose(),"processes: %d, time: %0.3e"%(self.get_processes(),end-start))							

		return

	def set_pool(self,pool):
		attr = 'pool'  
		self.set_null()
		if self.get_null():
			value = nullPool
		elif value in [False]:
			value = nullPool
		elif pool in [None,True]:
			value = Pool 
		elif callable(pool):
			value = emptycontext(pool)	
		else:
			value = pool		
		setattr(self,attr,value)
		return 
	def get_pool(self,default=None):
		attr = 'pool'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value

	def set_processes(self,processes):
		attr = 'processes'
		default = 1
		processes = default if processes is None else processes
		processes = min(processes,MAX_PROCESSES-1)
		setattr(self,attr,processes)
		self.set_null()
		return
	def get_processes(self,default=None):
		attr = 'processes'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value

	def set_initializer(self,initializer):
		attr = 'initializer'
		value = initializer
		setattr(self,attr,value)
		return
	def get_initializer(self,default=None):
		attr = 'initializer'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value

	def set_initargs(self,initargs):
		attr = 'initargs'
		value = initargs
		setattr(self,attr,value)
		self.initargs = initargs
		return
	def get_initargs(self,default=None):
		attr = 'initargs'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value

	def set_maxtasksperchild(self,maxtasksperchild):
		attr = 'maxtasksperchild'
		value = maxtasksperchild
		setattr(self,attr,value)
		self.initargs = initargs
		return
	def get_maxtasksperchild(self,default=None):
		attr = 'maxtasksperchild'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value

	def set_context(self,context):
		attr = 'context'
		value = context
		setattr(self,attr,value)
		self.initargs = initargs
		return
	def get_context(self):
		attr = 'context'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value

	def set_verbose(self,verbose):  
		attr = 'verbose'
		if verbose is None:
			value = False 
		else:
			value = verbose
		setattr(self,attr,value)
		return 
	def get_verbose(self,default=None):
		attr = 'verbose'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value

	def set_iterable(self,iterable):
		attr = 'iterable'
		if isinstance(iterable,dict):
			keys = list(iterable)
			value = (dict(zip(keys,values)) 
							for values in itertools.product(*[iterable[key] 
															for key in keys]))
		elif isinstance(iterable,int):
			value = ({'i':i} for i in range(iterable))
		else:
			value = iterable
		setattr(self,attr,value)
		return
	def get_iterable(self,default=None):
		attr = 'iterable'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value		

	def set_null(self):
		attr = 'null'
		min_processes = 2
		value = self.get_processes() < min_processes
		setattr(self,attr,value)
		return
	def get_null(self,default=None):
		attr = 'null'
		if not hasattr(self,attr):
			getattr(self,'set_%s'%(attr))(default)
		value = getattr(self,attr) 
		return value




def catch(update,exceptions,raises,iterations=1000):
	'''
	Wrapper to loop through function and catch exceptions, updating args and kwargs until no exceptions
	
	Args:
		update (callable): function with signature update(exception,*args,**kwargs) to update *args and **kwargs after exceptions
		exceptions (tuple): exceptions that invoke updating of *args and **kwargs
		raises (tuple): exceptions that raise exception and do not update *args and **kwargs
		iterations (int): maximum number of iterations before exiting
	Returns:
		func (callable): wrapped function for catching exceptions
	'''
	def wrap(func):
		@functools.wraps(func)
		def wrapper(*args,**kwargs):
			result = None
			exception = Exception
			iteration = 0
			while (exception is not None) and (iteration < iterations):
				try:
					result = func(*args,**kwargs)
					exception = None
				except Exception as e:
					exception = e
					if isinstance(exception,exceptions):
						update(exception,*args,**kwargs)
					elif isinstance(exception,raises):
						raise exception
				iteration += 1
			if exception is not None:
				raise exception
			return result
		return wrapper
	return wrap



def getmethod(obj,attr,default=None):
	if default is None:
		def default(*args,**kwargs): 
			return obj
	value = (getattr(obj,attr) if (getattr(obj,attr,None) is not None) else default)
	return value

def getattribute(obj,attr,default=None): 
	value = (getattr(obj,attr) if (getattr(obj,attr,None) is not None) else default)
	return value

def setattribute(obj,attr,value):
	setattr(obj,attr,value)
	return 

def chunkfunc(index,data,shape,chunk,sparsity,where,function,wrapper,serialize,format,dtype,n_jobs,verbose=False):
	'''
	Chunk operation on N arrays
	
	Args:
		index (list[int]): list of N integers for indices of each chunk
		shape (list[int]): list of N integers for length of each data array
		chunk (list[int]): list of N integers for size of each chunk for each data array
		sparsity (int,float,list): Sparsity of chunk. If int, sparsity number of smallest elements along that axis of the chunk, if float, all elements with abs(chunk[axis]) < sparsity		
		where (ndarray,sparse_matrix): array of shape shape, of where to compute chunks
		data (list[array]): list of N data arrays		
		function (callable): callable function that has signature function(data,slices,shape,where)
							 for sliced arrays based on index and chunk
		wrapper (callable): callable function on chunked return values and slices from function with signature wrapper(shape,values)		
		serialize (list): list of integers of arrays that are processed in serial
		format (str): output format (sparse types 'csr','csc', etc. or 'array' for dense numpy array)		
		dtype (str,data-type): data type of output
		n_jobs (int): Number of parallel jobs
		verbose (bool,int): Print out details of chunking							 
	Returns:
		value: Return value of function
		bounds (list): list of [start,stop,step] bounds of chunk	
	'''

	def loop(index,data,shape,chunk,where,format,dtype,verbose):
		# Slices for index and chunk size
		bounds = [[index[i]*chunk[i],min(shape[i],(index[i]+1)*chunk[i]),1] for i in range(N)]
		slices = [slice(*bounds[i]) for i in range(N)]
		lengths = [min(chunk[i],shape[i]-index[i]*chunk[i]) for i in range(N)]

		# Compute chunk value
		value = function(data,slices,shape,where)
		# value = asdtype(asformat(value,format),dtype)
		value = asdtype(value,dtype)

		# Print out chunking details		
		# logger.log(verbose,'Chunk %s with shape %r'%(','.join(['%d/%d'%(index[i],shape[i]//chunk[i]) for i in range(N)]),chunk))

		return value,bounds

	# Number of arrays
	N = min(len(index),len(shape),len(chunk),len(data))

	# Serialize any part of chunk for index that is null
	serialize = [] if serialize is None else serialize
	indexes = (ind for ind in itertools.product(*[range(int(shape[i]//chunk[i]+1)) if i in serialize else [index[i]] for i in range(N)]))

	values = []
	for index in indexes:
		value = loop(index,data,shape,chunk,where,format,dtype,verbose)
		values.append(value)

	value = wrapper(values,shape,chunk,sparsity,where,format,dtype)

	return value



def chunkify(data,shape,chunk,sparsity,where,function,wrapper=None,index=None,serialize=None,format=None,dtype=None,n_jobs=1,verbose=False):
	'''
	Chunk operation on N arrays
	
	Args:
		data (list[array]): list of N data arrays of lengths
		shape (list[int]): list of N integers for length of each data array		
		chunk (int, list[int]): list of N integers for size of each chunk for each data array
		sparsity (int,float,list): Sparsity of chunk. If int, sparsity number of smallest elements along that axis of the chunk, if float, all elements with abs(chunk[axis]) < sparsity		
		where (ndarray,sparse_matrix): array of shape shape, of where to compute chunks
		function (callable): callable function that has signature function(data,slices,shape,where)
							 for sliced arrays based on index and chunk
		wrapper (list,callable): callable function on chunked return values and slices from function,
			with signature wrapper(values,shape,chunk,sparsity,where,format,dtype). List if nested chunkify calls for parallel/serial calls
		index (list[int]): list of N integers for indices of each chunk		
		serialize (list): list of integers of arrays that are processed in serial, or list of lists if nested if nested chunkify calls for parallel/serial calls
		format (str): output format (sparse types 'csr','csc', etc. or 'array' for dense numpy array)				
		dtype (str,data-type): data type of output
		n_jobs (int): Number of parallel jobs
		verbose (bool,int): Print out details of chunking
	Returns:
		values: Return value of wrapper
		slices: Slices of arrays
		
	'''
	
	# Serial Wrapper
	def wrapper_serial(values,shape,chunk,sparsity,where,format,dtype):

		# values = [(bounds_0,values_0),...,(bounds_n-1,values_n-1)]
		# where bounds are bounds of N dimensional data, and 
		# bounds_j = [[start_j_0,stop_j_0,step_j_0]...,[start_j_n-1,stop_j_n-1,step_j_n-1]], 
		# with slice lengths lengths_j = [lengths_j_0,....,lengths_j_N-1] and total size size_j = prod(lengths_j)
		# values_j = [value_j_0,...,value_j_size_j-1] of length size_j
		
		# Get number of values, where 
		n = len(values)			

		# Get whether sparse values
		sparse = issparse(where)

		# Get bounds = [[bounds_0_0,...,bounds_0_N-1],...,[bounds_n-1_0,...,bounds_n-1_N-1]]
		bounds = [values[i][1] for i in range(n)]

		# Get out = [[values_0_0,...,values_0_length_0-1],...,[values_n-1_0,...,values_n-1_length_n_1-1]]
		out = [values[i][0] for i in range(n)]
		values = out

		# Get array of bounds and values
		out = concatenate(out,axis=-1)

		bounds = asndarray(bounds)
		bounds = [[bounds[:,i,0].min(),bounds[:,i,1].max(),bounds[:,i,2].max()] for i in range(bounds.shape[1])]

		# Sparsify array
		out = sparsify(out,sparsity,axis=0,format=format)

		return out,bounds

	# Parallel Wrapper
	def wrapper_parallel(values,shape,chunk,sparsity,where,format,dtype):

		# values = [(bounds_0,values_0),...,(bounds_n-1,values_n-1)]
		# where bounds are bounds of N dimensional data, and 
		# bounds_j = [[start_j_0,stop_j_0,step_j_0]...,[start_j_n-1,stop_j_n-1,step_j_n-1]], 
		# with slice lengths lengths_j = [lengths_j_0,....,lengths_j_N-1] and total size size_j = prod(lengths_j)
		# values_j = [value_j_0,...,value_j_size_j-1] of length size_j
		
		# Get number of values, where 
		n = len(values)			

		# Get whether sparse values
		sparse = issparse(where)

		# Get bounds = [[bounds_0_0,...,bounds_0_N-1],...,[bounds_n-1_0,...,bounds_n-1_N-1]]
		bounds = [values[i][1] for i in range(n)]

		# Get values = [[values_0_0,...,values_0_length_0-1],...,[values_n-1_0,...,values_n-1_length_n_1-1]]
		values = [values[i][0] for i in range(n)]

		# Get array of bounds and values
		out = concatenate(values,axis=0)
		out = asformat(out,format=format)

		return out



	# Number of arrays
	N = min(len(data),len(shape))

	# Chunk sizes
	if not isinstance(chunk,list):
		chunk = [chunk,chunk]
	chunk = [max(1,min(shape[i],shape[i] if chunk[i] is None else int(shape[i]*chunk[i]) if isinstance(chunk[i],(float,np.float64)) else chunk[i])) 
			 for i in range(N)]


	# Default wrapper (concatenation of array chunks)
	wrapper = [wrapper_parallel,wrapper_serial] if wrapper is None else wrapper

	# Get wrapper and serial (list if serial is list for nested chunkify calls)
	wrapper = [wrapper] if not isinstance(wrapper,list) else wrapper
	serialize = [] if serialize is None else serialize
	
	# Check if chunkify is a parent loop (wrapper has multiple items for nested calls)
	# and if chunkify is a child loop (index is not None), and adjust wrapper accordingly
	parent = len(wrapper[1:]) > 1
	child = index is not None
	

	# Parallel
	parallel = Parallelize(n_jobs=n_jobs,verbose=verbose)

	# Loop
	func = chunkify if parent else chunkfunc
	wrappers = wrapper[1:] if parent else wrapper[1]
	wrapper = wrapper[0]
	iterable = ({'index':index} for index in itertools.product(*[range(int(shape[i]//chunk[i]+1)) if i not in serialize else [index[i] if child else -1] for i in range(N)]))
	values = []
	args = []
	kwargs = {
		'data':data,'shape':shape,'chunk':chunk,'sparsity':sparsity,'where':where,
		'function':function,'wrapper':wrappers,'serialize':serialize,
		'format':format,'dtype':dtype,'n_jobs':n_jobs,'verbose':verbose
		}

	parallel(func,iterable,values,*args,**kwargs)


	# Wrapped value
	value = wrapper(values,shape,chunk,sparsity,where,format,dtype)

	return value


# Return tuple of returns
def returnargs(returns):
	if isinstance(returns,tuple) and len(returns) == 1:
		return returns[0]
	else:
		return returns


# Get range from slice
def slice_range(slices,format=None,zero=False):
	def get(obj,default):
		return obj if obj is not None else default
	if isinstance(slices,slice):
		if not zero:
			ranges = range(get(slices.start,0),slices.stop,get(slices.step,1))
		else:
			ranges = range(0,slices.stop - get(slices.start,0),get(slices.step,1))
	else:
		ranges = slices
	ranges = format(ranges) if ranges is not None else ranges
	return ranges

# Get slices index with indices, where slice has maximum value of n
def slice_index(slices,indices,n):
	if isinstance(slices,slice):
		defaults = (0,n,1)
		slices = slices.indices(n)
		slices = tuple([s if s is not None else d for s,d in zip(slices,defaults)])
		arr = np.array(list(range(*slices)))
	else:
		arr = np.array(slices)
	arr = arr[indices]
	return arr


# Set diagonal of array in place
def fill_diagonal(arr,values,offset=0):
	'''
	Set diagonal of ndarray with values in place
	
	Args:
		arr (ndarray): Array to be set in place with shape (...,n,...m,...) at axis1 and axis2
		values (ndarray) Array of values of size (n-abs(offset))*(m-abs(offset))
		offset (int): Offset from diagonal of array
		
	'''
	assert arr.ndim == 2, "Error - ndim != 2 and array is not matrix"
	n,m = arr.shape
	arr.ravel()[max(offset,-m*offset):max(0,(m-offset))*m:m+1] = values.ravel()
	return


def sortarr(arr,axis=0):
	'''
	Sort array along axis, with key by all elements along other axes
	
	Args:
		arr (ndarray): array to be sorted
		axis (int): axis to sort along
	Returns
		arr (ndarray): sorted array
	'''
	shape = list(arr.shape)
	size = shape.pop(axis)
	arr = [a for a in np.swapaxes(arr,axis,0).reshape((size,-1))]
	arr = np.swapaxes(np.array(list(sorted(arr,key=lambda a: tuple(a)))).reshape((size,*shape)),axis,0)
	return arr


def within(arr,indices,axis,bounds):
	'''
	Return arr where unique indices along axis are all within bounds (open bounds)
	
	Args:
		arr (ndarray): array to be searched of shape (n0,n1,...naxis,...,nndim)
		indices (int,list,ndarray): indices along axis to search
		axis (int): axis to search
		bounds (list): upper and lower bounds that arr must be within
	Returns:
		mask (ndarray): mask of elements within bounds of shape (n0,n1,...naxis-1,naxis+1...,nndim)
		
	'''
	if not isinstance(indices,(list,np.ndarray)):
		indices = [indices]
	bounds = [np.array(b) for b in bounds]
	
	arr = take(arr,indices,axis)
	mask = ((arr>bounds[0]) & (arr<bounds[1])).all(axis=axis)
	return mask


def similarity_matrix(x,y,adjacency=None,metric=2,sparsity=None,directed=False,chunk=None,format='csr',dtype=None,eliminate_zeros=False,kind='mergesort',n_jobs=None,verbose=False):
	'''
	Get comparison of ndarrays y-x, either by euclidean p-norm for p>0 or positive sign if p=0 or negative sign if p=-1
	
	Args:
		x (ndarray): Array to be measured of shape (n,(d))
		y (ndarray): Array to be measured of shape (m,(d))
		adjacency (ndarray,sparse_matrix): array of shape (n,m) of where to compute differences between elements		
		metric (int,float,str,callable): Type of metric, or order of euclidean norm over all axis. 
			If metric>0, then euclidean metric-norm of y-x, if metric=0, then sign of y-x, if metric<0, then signed metric-norm of y-x. 
			If callable, is function which accepts broadcasted x and y of shapes (n,(d),m,(d)) and returns similarity metric of shape (n,m)		
		sparsity (int,float,list): Sparsity of similarity matrix. If int, sparsity number of smallest elements along that axis of the similarity matrix, if float, all elements with \vert similarity matrix[axis]\vert < sparsity				
		directed (bool): Whether data is directed between and x and y 		
		chunk (int,float,list,None): Size of chunks along first axes of x,y to perform distance calculation
		format (str): Matrix format (sparse types 'csr','csc', etc. or 'array' for dense numpy array)		
		dtype (str,data-type): data type of output		
		eliminate_zeros (bool): eliminate elements with explicit zero similarity
		kind (str): Sort algorithm
		n_jobs (int): Number of parallel jobs
		verbose (bool,int): Print out details of computation		
	Returns:
		out (sp.sparse.format_matrix): p-norm distance between pairwise points in x and y of shape	
	'''

	# Get size of arrays and chunk sizes

	def formatter(dtype,format):
		def decorator(func):
			@functools.wraps(func)
			def wrapper(*args,**kwargs):
				fmt = 'csr'
				out = func(*args,**kwargs)
				out = out.astype(dtype)
				out = out.asformat(format) if issparse(out) else getattr(sp.sparse,'%s_matrix'%(fmt))(out).asformat(format)
				return out


	n,m = x.shape[0],y.shape[0]	
	shape = (n,m)

	# Get sparsity
	sparse = issparse(adjacency)
	format = adjacency.getformat() if (sparse and (format is None)) else 'array' if format is None else format

	# Update norm order depending on directedness of graph
	if directed:	
		if isinstance(metric,(int,np.integer,float,np.float64)):
			metric = -abs(metric)
		elif isinstance(metric,str):
			metric = '_'.join([metric,'directed'])
		else:
			metric = metric
	else:
		metric = metric
	metric = 'euclidean' if metric is None else metric

	# Comparisons
	if adjacency is None:
		def func(x,y,where=None):
			def func_x(x,y):
				out = broadcast(x,axes=[x.ndim//2 + dim for dim in range(y.ndim//2)],shape=y.shape[y.ndim//2:],newaxis=False)
				return out
			def func_y(x,y):
				out = broadcast(y,axes=[dim for dim in range(x.ndim//2)],shape=x.shape[:x.ndim//2],newaxis=False)
				return out
			out = [outer(func,x,y,where=where) for func in [func_x,func_y]]
			return out
	else:
		def func(x,y,where=None):
			def func_x(x,y):
				out = x
				return out
			def func_y(x,y):
				out = y
				return out
			out = [outer(func,x,y,where=where) for func in [func_x,func_y]]
			return out			

	# metric function takes arrays with shape (n,m,(d)) and returns comparison with shape (n,m)
	if callable(metric):
		_metric = metric
		def function(data,slices,shape,where):
			return _metric(*func(data[0][slices[0]],data[1][slices[1]],where=where[slices[0],:][:,slices[1]] if where is not None else None))
	elif metric == 'euclidean':
		ord = 2
		dtype = 'float'
		def function(data,slices,shape,where):
			out = None
			N = min(len(data),len(slices),len(shape))
			full = not any([data[i].shape[0] == 0 for i in range(N)])
			if full:
				out = norm(getdiag(data[0][slices[0]][...,None,None] - data[1][slices[1]][None,None,...],axis1=1,axis2=-1),ord=ord,axis=-1)
				out = where[slices[0],:][:,slices[1]].multiply(out) if where is not None else out
			return out
	elif metric == 0:
		# Sign of cos(x,y) for all higher dimensions
		ord = 2
		dtype = 'int'
		def function(data,slices,shape,where):
			return normsigndiff(*func(data[0][slices[0]],data[1][slices[1]],where=where[slices[0],:][:,slices[1]] if where is not None else None),
								axis=-1,ord=ord,normed=False,signed=True)
	elif metric > 0:        
		# Euclidean p norm of x - y for all higher dimensions
		ord = abs(metric)
		dtype = 'float'
		def function(data,slices,shape,where):
			return normsigndiff(*func(data[0][slices[0]],data[1][slices[1]],where=where[slices[0],:][:,slices[1]] if where is not None else None),
								axis=-1,ord=ord,normed=True,signed=False)
	elif metric < 0:
		# Euclidean p norm and Sign of cos(x,y) for all higher dimensions
		ord = abs(metric)        
		dtype = 'float'
		def function(data,slices,shape,where):
			return normsigndiff(*func(data[0][slices[0]],data[1][slices[1]],where=where[slices[0],:][:,slices[1]] if where is not None else None),
								axis=-1,ord=ord,normed=True,signed=True)
	

	# Comparisons array
	# broadcast function returns x with shape (n,(d),1,...,1) to (n,(d),m,(d)) or y with shape (1,...,1,m,(d)) to (n,(d),m,(d))
	# outer returns x or y with shape (n,m,(d))
	# metric returns comparison with shape (n,m)
	
	N = 2
	data = [x,y]
	shape = [len(data[i]) for i in range(N)]
	where = adjacency
	index = None
	serialize = [1]
	wrapper = None

	# Loop
	out = chunkify(data,shape,chunk,sparsity,where,function,wrapper,index,serialize,format,dtype,n_jobs,verbose)

	return out




def adjacency_matrix(n,weights=None,conditions=None,format=None,dtype=int,tol=None,atol=None,rtol=None,kind='mergesort',diagonal=False,return_argsort=False,return_counts=False,verbose=False):
	'''
	Create adjacency matrix for n vertices with weights depending on condition
	
	Args:
		n (int): size of adjacency matrix rows
		weights (ndarray,sparse_matrix,None): weights matrix of vertices, possibly sparse
		conditions (int,tuple,callable,None): integer for k nearest neighbours, or tuple of (open) bounds on nearest neighbours, or callable function with weights,argsort,counts arguments to make adjacency elements non-zero
		format (str): Matrix format (sparse types 'csr','csc', etc. or 'array' for dense numpy array)		
		dtype (str,data-type): data type of output		
		tol (float): Tolerance for rank and singularity of matrix
		atol (float): Absolute tolerance of difference between unique elements
		rtol (float): Relative tolerance of difference between unique elements
		kind (str): Sort algorithm                
		diagonal (bool): explicitly include diagonal of adjacency
		return_argsort (bool): Return sort order of elements equal within tolerance
		return_counts (bool): Return number of occurrences of unique elements within tolerance
		verbose (bool,int): Print out details of adjacency calculation							 
	Returns:
		adjacency (ndarray,sparse_matrix,sparse_array): adjacency matrix with shape (n,n), possibly sparse
		argsort (ndarray): Sort order of elements that are equal within tolerance along higher dimensions
		counts (list): list of ndarrays of counts of unique elements if return_counts is True				
	'''


	# Size of adjacency matrix
	if n is None and weights is not None:
		n = weights.shape[0]
	elif n is None and (weights is None):
		raise "Error - adjacency size and weights arguments are None"
		return



	# Default weights as nearest neighbours in vertices
	if weights is None:
		offsets = [0,1,-1]
		diags = [0,1,1]
		weights = sp.sparse.diags(diags,offsets,shape=(n,n),dtype=float)


	# Get weights shape
	n,m = weights.shape

	# Get if weights are sparse
	sparse = issparse(weights)

	# Default conditions as nearest neighbours
	isint = isinstance(conditions,(int,np.integer))
	istuple = isinstance(conditions,tuple)
	isnone = conditions is None

	if isnone:
		def conditions(weights,argsort,counts):
			out = weights.astype(bool)
			out.eliminate_zeros()
			return out
	elif isint or istuple:
		argsortbounds = [0-1 if (isnone or isint or conditions[0] is None) else conditions[0],
						 n if (isnone or conditions[1] is None) else conditions+1 if (isint) else conditions[1],
						 []]
		def conditions(weights,argsort,counts):
			sparse = issparse(weights) or issparse(argsort)
			condition = [gt(argsort,argsortbounds[0]) ,lt(argsort,argsortbounds[1])]
			condition = multiply(*condition)
			return condition

	# Sort weights
	if not isnone:
		argsort,counts = unique_argsort(np.abs(weights),return_counts=True,atol=atol,rtol=rtol,signed=False,kind=kind)
	else:
		argsort,counts = None,None


	# Create adjacency matrix
	adjacency = conditions(weights,argsort,counts)


	# Set adjacency format and dtype
	adjacency = asformat(adjacency,format)
	adjacency = asdtype(adjacency,dtype)


	# Explicitly include diagonal in adjacency
	if diagonal:
		adjacency = setdiag(adjacency,diagonal)		

	# Returns
	returns = ()

	returns += (adjacency,)

	if return_argsort:
		returns += (argsort,)
	if return_counts:
		returns += (counts,)

	return returnargs(returns)

			  
def unique_argsort(arr,return_counts=False,atol=None,rtol=None,signed=False,kind='mergesort'):             
	'''
	Get sort indices of unique elements within absolute and relative tolerances
	
	Args:
		arr (ndarray,sparse_matrix,sparse_array): Array of shape(n,m) to be searched, possibly sparse
		return_counts (bool): Return number of occurrences of unique elements within tolerance		
		atol (float): Absolute tolerance of difference between unique elements
		rtol (float): Relative tolerance of difference between unique elements
		signed (bool): Sort by absolute value and then return argsort by sign of grouped unique elements		
		kind (str): Sort algorithm
	Returns:
		argsort (ndarray,sparse_matrix,sparse_array): Sort order of elements that are equal within tolerance along higher dimensions, possibly sparse
		counts (list): list of ndarrays of counts of unique elements if return_counts is True	
	'''


	# Get shape of arr
	ndim = arr.ndim
	if ndim == 1:
		arr = arr.reshape((1,-1))
	n,m = arr.shape

	# Check if arr is sparse
	sparse = issparse(arr)
	if sparse:
		format = arr.getformat()
		argsort = arr.astype(int,copy=True)
		counts = [[] for i in range(n)]
	else:
		format = 'array'
		argsort = zeros((n,m),dtype=int)
		counts = [[] for i in range(n)]

	# Sort array along axis dimension
	for i in range(n):
		result = unique_tol(arr[i],return_unique=False,return_counts=return_counts,return_argsort=True,atol=atol,rtol=rtol,signed=signed,kind=kind)
		if return_counts:
			counts[i] = result[0]
			result = result[1]

		argsort[i] = result

	if ndim == 1:
		argsort = argsort[-1]
		counts = counts[-1]

	returns = ()

	returns += (argsort,)

	if return_counts:
		returns += (counts,)

	return returnargs(returns)



def unique_tol(arr,
			  return_unique=True,return_index=False,return_counts=False,return_argsort=False,return_sets=False,
			  atol=None,rtol=None,
			  signed=False,
			  kind='mergesort'):             
	'''
	Get unique elements within absolute and relative tolerances
	
	Args:
		arr (ndarray,sparse_matrix,sparse_array): Array to be searched, possibly sparse
		return_unique (bool): Return unique elements
		return_index (bool): Return indices of unique elements
		return_counts (bool): Return number of occurrences of unique elements within tolerance
		return_argsort (bool): Return sort order of elements equal within tolerance
		return_sets (bool): Return indices grouped by their argsort
		atol (float): Absolute tolerance of difference between unique elements
		rtol (float): Relative tolerance of difference between unique elements
		signed (bool): Sort by absolute value and then return argsort by sign of grouped unique elements
		kind (str): Sort algorithm                
	Returns:
		unique (ndarray): Unique elements of array
		indices (ndarray): Indices of unique elements if return_index is True
		counts (ndarray): Counts of unique elements if return_counts is True
		argsort (ndarray,sparse_matrix,sparse_array): Sort order of elements that are equal within tolerance if return_argsort is True
		sets (list): Sets of indices grouped by their sort order
	Example:
		arr = [-4,2,4,5,1,-2,-2] 
		has absolute value sorting rank (if signed is False)
		argsort = [2,1,2,3,0,1,1]
		and signed sorting rank
		argsort = [4,3,5,6,0,1,2] 
		since negative numbers are sorted before positive numbers of equal absolute value.	
	'''



	# Check if arr is sparse
	sparse = issparse(arr)
	if sparse:
		format = arr.getformat()
		size,dtype = arr.shape[-1],arr.dtype 
		inds = arr.indices
		indptr = arr.indptr
		arr = arr.data.ravel()
	else:
		format = 'array'
		size,dtype = arr.shape[-1],arr.dtype 		
		inds = arange(size)
		arr = arr.ravel()

	# Get absolute value of array if signed is True
	if not signed:
		_arr = arr
	else:
		_arr = np.abs(arr)

	# Get mask of unique elements
	n = arr.size
	mask = np.empty(n,dtype=bool)

	# Sort array
	indices = _arr.argsort(kind=kind)

	# Get difference between nearest elements
	diff = np.diff(_arr[indices])

	# Get indices where adjacent sorted elements differ by greater than tolerances
	mask[:1] = True
	mask[1:] = ~isclose(diff,rtol=rtol,atol=atol)

	# Get return values
	returns = ()

	# Get unique elements
	if return_unique:
		unique = _arr[indices][mask]
		returns += (unique,)

	# Get index of unique elements
	if return_index:
		index = inds[indices[mask]]
		returns += (index,)

	# Get counts of unique elements
	# Get argsort order of equal elements within tolerance
	# Get indices grouped by their argsort
	if return_counts or return_argsort or return_sets:
		counts = np.diff(concatenate(np.nonzero(mask) + ([n],)))

		if return_counts:
			returns += (counts,)

		counts = [0,*cumsum(counts)]		
		argsort = zeros(n,dtype=int)
		sets = []
		for j in range(len(counts)-1):
			if not signed:
				i = j
			else:
				i = counts[j] + arr[indices[counts[j]:counts[j+1]]].argsort(kind=kind).argsort(kind=kind)
			sets.append(indices[counts[j]:counts[j+1]])
			argsort[sets[-1]] = i


		if return_argsort:
			if sparse:
				argsort = getattr(sp.sparse,'%s_matrix'%(format))((argsort,inds,indptr),shape=(1,size),dtype=int)

			returns += (argsort,)

		if return_sets:
			sets = [inds[s].tolist() for s in sets]
			returns += (sets,)

	return returnargs(returns)



def unique_int(arr,kind='mergesort'):
	'''
	Get indices of unique integers in integer array
	
	Args:
		arr (array): dimensional array of size n, flattened if multidimensional
		kind (str): Sort algorithm
	Returns:
		indices (dict): dictionary of unique elements and arrays of indices of each unique element
	'''

	# Get array shape an dtype
	arr = arr.ravel()
	n,dtype = arr.size,arr.dtype

	# if n == 0:
	# 	return 

	assert dtype in [int,np.integer], "Error - Array is not integer dtype"

	# Get unique elements and their indices of sorted array
	argsort = arr.argsort(kind=kind)
	unique,inds = np.unique(arr[argsort],return_index=True)
	inds = np.insert(inds,obj=inds.size,values=n)
	
	# Get indices of each unique element based on indices of unique sorted elements
	indices = {unique[i]: np.sort(argsort[inds[i]:inds[i+1]]) for i in range(inds.size-1)}

	return indices




def neighbourhood(x,y,size,basis,order=None,adjacency=None,indices=None,metric=2,variable=None,unique=True,diagonal=False,argsortbounds=None,weightbounds=None,
					strict=False,tol=None,atol=None,rtol=None,chunk=None,format=None,dtype=None,kind='mergesort',
					verbose=False,n_jobs=None,return_weights=False):
	'''
	Get indices of neighbourhood of indices of array y that are nearest to x
	
	Args:
		x (ndarray,DataFrame): Array to be measured of shape (n,(d))
		y (ndarray,DataFrame): Array to be measured of shape (m,(d))
		size (int): Size of neighbourhood
		basis (str,callable): function to yield basis, or string in ['vandermonde']
		order (int): order of basis matrix for linearly independent points
		adjacency (ndarray,sparse_matrix): array of shape (n,m) of where to compute differences between elements				
		indices (array,None): Allowed indices of y to compare to x
		metric (int,float,str,callable): Type of metric, or order of euclidean norm over all axis. 
		ord (int,float): Order of euclidean norm
		variable (str,list): dataframe column labels to be used for data	
		unique (bool): Include unique basis terms q = choose(p+n,n) else q = (p^(n+1)-1)/(p-1)				
		diagonal (bool): explicitly include diagonal of adjacency		
		argsortbounds (list): Minimum and maximum ranked distance between neighbors (open boundaries), and disallowed values, between -1 and m+1, where None is replaced by -1 or m+1
		weightbounds (list): Minimum and maximum weight distance between neighbors (open boundaries), and disallowed values, between -np.inf and np.inf, where None is replaced by -np.inf or np.inf
		strict (bool): Whether size of neighborhood is exactly size, or rounded to minimum set of nearest neighbors within bounds
		tol (float): Tolerance for rank and singularity of basis matrix of points
		atol (float): Absolute tolerance of difference between unique elements
		rtol (float): Relative tolerance of difference between unique elements
		chunk (int,float,None): Size of chunks along 0th axis to perform distance calculation		
		format (str): Matrix format (sparse types 'csr','csc', etc. or 'array' for dense numpy array)		
		dtype (str,data-type): data type of output		
		kind (str): Sort algorithm        
		verbose (bool,int): Print out details of neighbourhood
		n_jobs (int): Number of parallel jobs		
		return_weights (bool): Return weights of neighbourhood points in y that are nearest to x        
	Returns:
		indices (list,ndarray): Indices of neighborhood of points in y (x) that are closest to each x (y). ndarray if each x (y) has same size of neighborhood.
		weights (list,ndarray): If return_weights is True, Weights of neighborhood between closest y (x) values to each x (y). ndarray if each x has same size of neighborhood.
	Example:
		Arguments of:
		x = [[1,0],[4,-4],[5,2],[2,-3]]
		y = [[0,0],[4,3],[3,-2]]        
		indices = None
		size = 2
		ord = 2
		argsortbounds = (0,2,None)
		
		returns the indices of y that is closest to each point in x.
		This example will return
		indices = [0,2,1,2] since x[0] is closest to y[0], x[1] is closest to y[2] etc.
	'''
	
	# Get sparsity
	exists = isarray(adjacency)
	sparse = issparse(adjacency)

	# Get array if dataframe
	isvariable = variable is not None
	if isdataframe(x):
		if isvariable:
			x = x[variable]
		x = x.to_numpy()
	if isdataframe(y):
		if isvariable:
			y = y[variable]
		y = y.to_numpy()


	# Get number of elements in x and y
	n,m = x.shape[0],y.shape[0]
	p,d = x.shape[1:],y.shape[1:]
	p = prod(p,dtype=int)


	# Get size and order of neighbourhood
	if order is None:
		if size is None:
			size = n
		order = 1
	else:
		if size is None:
			size = ncombinations(p,order,unique=unique)
	if size in [1]:
		order = 0
	sparsity = min(m,4*p*size*max(order,1))

	# Get allowed indices of y to compare to x
	if indices is None:
		indices = arange(m)
	indices = np.array(indices)

	y = y[indices]

	adjacency = adjacency[:,indices] if exists else adjacency


	# Get euclidean distance weight function
	weights = similarity_matrix(x,y,adjacency=adjacency,metric=metric,sparsity=sparsity,directed=True,chunk=chunk,kind=kind,format=format,dtype=dtype,n_jobs=n_jobs)

	# Set conditions to choose nearest neighbours such that neighborhood is at least size
	conditions = nearest_neighbourhood(size=size,strict=strict,atol=atol,rtol=rtol,argsortbounds=argsortbounds,weightbounds=weightbounds)


	# Get adjacency matrix, based on indices of nearest neighbours as per weights, as per conditions on neighborhood bounds and size and strict
	adjacency = adjacency_matrix(n,weights,conditions,tol=tol,atol=atol,rtol=rtol,format=format,dtype=dtype)

	# Get indices of linearly independent
	inds = linearly_independent(x,y,order,size,basis,
		adjacency=adjacency,dimension=None,ones=True,unique=unique,
		tol=tol,atol=atol,rtol=rtol,kind=kind,diagonal=diagonal,verbose=verbose,return_differences=False)

	# Get sparsity
	sparse = sparse or issparse(weights)

	# Get indices nearest neighbours in y for each x
	# Get weights of nearest neighbours
	if sparse:
		indices = [indices[i] for k,i in enumerate(inds)]
		weights = [weights[k,i].A.ravel() for k,i in enumerate(inds)]
	else:
		indices = [indices[i] for k,i in enumerate(inds)]
		weights = [weights[k,i] for k,i in enumerate(inds)]

	# Sort indices by absolute value of weights
	argsort = [np.abs(w).argsort(kind=kind) for i,w in zip(indices,weights)]
	indices = [i[a] for a,i,w in zip(argsort,indices,weights)]
	weights = [w[a] for a,i,w in zip(argsort,indices,weights)]

	# Format as ndarray if each x has same size of neighborhood
	squeeze = len(set([i.size for i in indices])) == 1
	if squeeze:
		indices = np.array(indices).squeeze()
		weights = np.array(weights).squeeze()


	# Get return values
	returns = ()
	returns += (indices,)
	if return_weights:
		returns += (weights,)

	return returnargs(returns)




def nearest_neighbourhood(size,strict=False,argsortbounds=None,weightbounds=None,atol=None,rtol=None,kind='mergesort'):
	'''
	Wrapper function to return conditions for nearest neighborhood of size, with neighbors not along dimension where x,y are equal
	
	Args:
		size (int): minimum size of neighborhood
		strict (bool): whether neighborhood must be exactly of size
		argsortbounds (list): Minimum and maximum ranked distance between neighbors (open boundaries), and disallowed values, between -1 and m+1, where None is replaced by -1 or m+1
		weightbounds (list): Minimum and maximum weight distance between neighbors (open boundaries), and disallowed values, between -np.inf and np.inf, where None is replaced by -np.inf or np.inf		
		atol (float): Absolute tolerance of difference between unique elements
		rtol (float): Relative tolerance of difference between unique elements
		kind (str): Sort algorithm        
	Returns:
		conditions (callable): function for adjacency function, which accepts weights, argsort,counts as arguments
	'''

	q = size

	def conditions(weights,argsort,counts):


		# Mask function based on weights,argsort,counts,z and bounds that returns allowed indices
		def ismask(i,m,weights,argsort,counts,size,sums,argsortbounds,weightbounds):
			
			sparse = issparse(weights) or issparse(argsort)


			_mask = [
				gt(argsort,argsortbounds[0]) if argsortbounds[0] > 0 else 1,
				lt(argsort,argsortbounds[1]) if argsortbounds[1] < m else 1,
				gt(weights,weightbounds[0]) if weightbounds[0] > -np.inf else 1,
				lt(weights,weightbounds[1]) if weightbounds[1] < np.inf else 1,
				isin(weights,weightbounds[2],invert=True) if weightbounds[2].size > 0 else 1
				]
			_mask = multiply(*_mask) if not all([i is 1 for i in _mask]) else 1
			
			mask = [_mask] if _mask is not 1 else 1

			indices = argsort.indices if sparse else arange(m)
			indices = where(multiply(*mask)) if mask is not 1 else indices


			if indices.size < size:
				nulllength = 0
				lengths = (sums - nulllength)>=(size)
				length = where(lengths)[0] if lengths.any() else sums.max()

				mask = [le(argsort,length), _mask]
				indices = where(multiply(*mask))

			return indices


		# Check if arr is sparse
		sparse = issparse(weights) or issparse(argsort)
		format = weights.getformat() if sparse else 'array'


		# Get conditions shape and data
		n,m = weights.shape
		shape = (n,m)
		size = min(q,m)
		data,indices,indptr = [],[],[0]
		fmt = 'csr'
		value = True


		# Get bounds of neighborhood ranked distances
		isargsortbounds = argsortbounds is not None
		isweightbounds = weightbounds is not None
		if not isargsortbounds:
			_argsortbounds = [-1,m,np.array([],dtype=int)]
		else:
			_argsortbounds = [min(m,max(-1,argsortbounds[0])) if isinstance(argsortbounds[0],(int,np.integer)) else -1,
						 max(0,min(m,argsortbounds[1])) if isinstance(argsortbounds[1],(int,np.integer)) else m,
						 np.array(argsortbounds[2],dtype=int).reshape(-1)
							if len(argsortbounds)>2 and argsortbounds[2] is not None else np.array([],dtype=int).reshape(-1)]

		if not isweightbounds:
			_weightbounds = [-np.inf,np.inf,np.array([])]
		else:
			_weightbounds = [min(np.inf,max(-np.inf,weightbounds[0])) if isinstance(weightbounds[0],(int,np.integer,float,np.float64)) else -np.inf,
						max(-np.inf,min(np.inf,weightbounds[1])) if isinstance(weightbounds[1],(int,np.integer,float,np.float64)) else np.inf,
						np.array(weightbounds[2]).reshape(-1)
							if len(weightbounds)>2 and weightbounds[2] is not None else np.array([],dtype=float).reshape(-1)]



		excluded = [argsort[i,where(isin(weights[i],_weightbounds[2]))] for i in range(n)]
		excluded = [np.unique([*_argsortbounds[2],*(excluded[i].data if sparse else excluded[i])]) for i in range(n)]
		nullsums = [np.array([add(counts[i][excluded[i][excluded[i]<=j]]) for j,c in enumerate(counts[i])]) if excluded[i].size>0 else 0 for i in range(n)]
		sums = [(cumsum(counts[i]) - nullsums[i]) for i in range(n)]

		sizes = -ones(n,dtype=int)
		_sizes = -ones(n,dtype=int)

		iteration = 0
		while (iteration==0) or (iteration < m) and ((sizes<(size)).any() and strict) and ((sizes==-1).any() or (((strict) and (sizes<sizes.max())).any() and ((sizes != _sizes).all()))):
			size = max(size,np.min([0,*sizes[sizes>size]]))

			for i in range(n):
				abounds = copy.deepcopy(_argsortbounds)
				wbounds = copy.deepcopy(_weightbounds)

				if (sizes[i] >= size):
					continue

				inds = ismask(i,m,weights[i],argsort[i],counts[i],size,sums[i],abounds,wbounds)

				while ((strict and (inds.size < size)) or (inds.size == 0)) and ((abounds[0]>=-1) and (abounds[1] <= m)):
					abounds[0] -= 1
					abounds[1] += 1 
					inds = ismask(i,m,weights[i],argsort[i],counts[i],size,sums[i],abounds,wbounds)

				if (strict and (inds.size>size)):
					mask = lt(unique_tol(weights[i,inds],return_unique=False,return_argsort=True,signed=True,atol=atol,rtol=rtol,kind=kind),size)
					inds = inds[mask.data if sparse else mask]

				sizes[i] = inds.size

				data.extend([value]*sizes[i])
				indices.extend(inds)
				indptr.append(indptr[-1]+sizes[i])

			iteration += 1


		condition = getattr(sp.sparse,'%s_matrix'%(fmt))((data,indices,indptr),shape=shape)
		condition = condition.asformat(format)

		return condition


	return conditions




def explicit_zeros(arr,indices,shape=None,format=None,eliminate_zeros=False):
	'''
	Ensure array has explicit zeros at indices 
	
	Args:
		arr (array,sparse_matrix,sparse_array): array to be made with explicit zeros with shape (n,m)
		indices (list): list of arrays indices with possible explicit zeros in arr
		shape (iterable): shape of full array
		format (str): Format of output sparse_matrix
		eliminate_zeros (bool): eliminate explicit zeros
	Returns:
		out (sparse_matrix): sparse_matrix with explicit zeros	
	'''

	# Get array shape and temporary format of array
	n,m = arr.shape
	shape = (n,m) if shape is None else shape


	# Convert arr to class with fmt data,row,col attributes for appending to datum,row,col and handle explicit zeros
	if isndarray(arr):
		fmt = 'coo'
		data = arr.ravel()
		row = repeat(arange(n),repeats=m,axis=0).ravel()
		col = repeat(arange(m)[None,:],repeats=n,axis=0).ravel()
		out = getattr(sp.sparse,'%s_matrix'%(fmt))((data,(row,col)),shape=shape)
		format = 'array' if format is None else format
	elif issparsearray(arr):
		format = arr.getformat() if format is None else format

		fmt = 'coo'	
		out = arr.tocsr().asformat(fmt)
		data,row,col = out.data.tolist(),out.row.tolist(),out.col.tolist()
		
		fmt = 'csr'	
		out = arr.asformat(fmt)

		for i in range(n):					
			data.extend(out[i,indices[i]].A[0])
			row.extend(i*ones(indices[i].size,dtype=int))
			col.extend(indices[i])
		fmt = 'coo'
		out = getattr(sp.sparse,'%s_matrix'%(fmt))((data,(row,col)),shape=shape)
		format = fmt if format is None else format
	elif issparsematrix(arr):
		format = arr.getformat() if format is None else format

		fmt = 'coo'	
		out = arr.asformat(fmt)
		data,row,col = out.data.tolist(),out.row.tolist(),out.col.tolist()
		
		fmt = 'csr'	
		out = arr.asformat(fmt)

		for i in range(n):					
			data.extend(out[i,indices[i]].A[0])
			row.extend(i*ones(indices[i].size,dtype=int))
			col.extend(indices[i])
		fmt = 'coo'		
		out = getattr(sp.sparse,'%s_matrix'%(fmt))((data,(row,col)),shape=shape)
		format = fmt if format is None else format


	if eliminate_zeros:
		out.eliminate_zeros()

	out = out.asformat(format)

	return out

def sparsify(arr,sparsity,axis=0,func=None,format=None):
	'''
	Sparsify array with only sparsity number of elements or absolute value of elements less than sparsity along axes other than axis
	
	Args:
		arr (array,sparse_matrix): array to make sparse
		sparsity (int,float): integer or float to define sparsity, or float to define maximum absolute value of elements
		axis (int): axis along which to make sparse
		func (callable): func(arr,sparsity) to sort elements and choose sparsity, returning indices of sparse array.
			defaults to sorting by minimum absolute value
		format (str): Matrix format (sparse types 'csr','csc', etc. or 'array' for dense numpy array)				
	Returns:
		out (array,sparse_matrix): sparsified array
	'''

	# @nb.njit(fastmath=True,cache=True,parallel=True)	
	def _sparsify(data,indices,indptr,sparsity,func):
		_data,_indices,_indptr = [],[],[0]
		# for i in nb.prange(indptr.size-1):
		for i in np.arange(indptr.size-1):
			inds = func(data[indptr[i]:indptr[i+1]],sparsity)
			_data.extend(data[indptr[i]:indptr[i+1]][inds])
			_indices.extend(indices[indptr[i]:indptr[i+1]][inds])
			_indptr.append(_indptr[i]+inds.size)
		return _data,_indices,_indptr


	# If sparsity is None, return arr
	if sparsity is None:
		return arr


	# Get type of arr
	shape = arr.shape
	sparse = issparse(arr)

	# Get arr data
	if sparse:
		format = arr.getformat() if format is None else format
		fmt = 'csr' if axis in [0,None] else 'csc'
		arr = arr.asformat(fmt)
		indices = arr.indices
		indptr = arr.indptr
		arr = arr.data
	else:
		format = 'array' if format is None else format
		fmt = 'csr' if axis in [0,None] else 'csc'		
		indices = tile(arange(shape[1-axis]),shape[axis])
		indptr = shape[1-axis]*arange(shape[axis]+1)
		arr = arr.ravel()



	# Get func
	if func is None:

		# @nb.njit(fastmath=True,cache=True)
		def func(arr,sparsity):
			if int(sparsity) == float(sparsity):
				indices = np.argsort(np.abs(arr))[:sparsity]
			else:
				indices = np.where(np.abs(arr)<sparsity)[0]
			return indices
				


	# Sparsify arr
	arr,indices,indptr = _sparsify(arr,indices,indptr,sparsity,func)

	out = getattr(sp.sparse,'%s_matrix'%(fmt))((arr,indices,indptr),shape=shape)
	out = out.asformat(format)
	return out


def twin(arr,like):
	'''
	Make array arr have same sparse structure as like
	Sparsify array with only sparsity number of elements in rows or absolute value of elements less than sparsity
	
	Args:
		arr (array,sparse_matrix): array of shape (n,m) to match sparsity to
		like (array,sparse_matrix): reference array of shape (n,m) of sparsity
	Returns:
		out (array,sparse_matrix): array with data of arr and sparsity of like	
	'''

	# Get sparsity
	sparse = issparse(arr)
	likesparse = issparse(like)
	shape = arr.shape
	format = arr.getformat() if sparse else 'array'

	assert (arr.ndim==2) and (arr.ndim == like.ndim) and all([i==j for i,j in zip(arr.shape,like.shape)]),"Error - arr and like are not identical shapes"

	# Convert arrays to fmt sparse arrays
	fmt = 'csr'	
	if sparse:
		arr = arr.asformat(fmt)
	else:
		arr = getattr(sp.sparse,'%s_matrix'%(fmt))(arr)

	fmt = 'coo'
	if likesparse:
		like = like.asformat(fmt)
	else:
		like = getattr(sp.sparse,'%s_matrix'%(fmt))(like)

	# Match sparsity structure
	data,row,col = [],like.row,like.col
	for i,j in zip(row,col):
		data.append(arr[i,j])


	fmt = 'coo'
	arr = getattr(sp.sparse,'%s_matrix'%(fmt))((data,(row,col)),shape=shape)
	arr = arr.asformat(format)


	out = arr

	return out






def powerset(p,n):
	''' 
	Get all powerset of p non-negative integers less than or equal to n
	
	Args:
		p (int): Number of integers
		n (int): Maximum value of integers
	Returns:
		integers (ndarray): powerset of integers of shape ((n+1)^p,p)	
	'''
	integers = np.array(list(itertools.product(range(n+1),repeat=p)))
	return integers

def combinations(p,n,unique=False):
	''' 
	Get all combinations of p number of non-negative integers that sum up to at most n
	
	Args:
		p (int): Number of integers
		n (int): Maximum sum of integers
		unique (bool): Return unique combinations of integers and q = choose(p+n,n) else q = (p^(n+1)-1)/(p-1)
	Returns:
		combinations (ndarray): All combinations of integers of shape (q,p)
	'''
	combinations = []
	iterable = range(p)
	for i in range(n+1):
		combos = list((tuple((j.count(k) for k in iterable)) for j in itertools.product(iterable,repeat=i)))
		if unique:
			combos = sorted(set(combos),key=lambda i:combos.index(i))
		combinations.extend(combos)
		
	combinations = np.vstack(combinations)
	return combinations


def ncombinations(p,n,unique=False):
	'''
	Number of all combinations of p number of non-negative integers that sum up to at most n
	
	Args:
		p (int): Number of integers
		n (int): Maximum sum of integers
		unique (bool): Return unique combinations of integers and q = choose(p+n,n) else q = (p^(n+1)-1)/(p-1)		
	Returns:
		q (int): Number of all combinations of integers    
	'''
	if p > 1:
		if unique:
			q = sp.special.comb(p+n,p,exact=True)
		else:
			q = int((p**(n+1)-1)/(p-1))
	else:
		q = n + 1
	return q


def icombinations(iterable,n,unique=False):
	''' 
	Get all combinations of p number of non-negative integers that sum up to at most n
	
	Args:
		iterable (int,iterable): Number of integers or iterable of length p
		n (int,iterable): Maximum number of elements, or allowed number of elements
		unique (bool): Return unique combinations of integers and q = choose(p+n,n) else q = (p^(n+1)-1)/(p-1)
	Returns:
		combinations (list): All combinations of iterable with q list of lists of length up to n, or lengths in n
	'''
	iterable = list(iterable) if not isinstance(iterable,int) else range(iterable)
	p = len(iterable)
	n = range(n+1) if isinstance(n,(int,np.integer)) else n
	combinations = []
	for i in n:
		combos = list((tuple(sorted(j,key=lambda i:iterable.index(i))) for j in itertools.product(iterable,repeat=i)))
		if unique:
			combos = sorted(set(combos),key=lambda i:combos.index(i))
		combinations.extend(combos)
	return combinations
	
	
def polynomials(X,order,derivative=None,selection=None,commutativity=False,intercept_=True,
			   variables=None,
			   return_poly=True,return_derivative=False,
			   return_coef=False,return_polylabel=False,return_derivativelabel=False,
			   return_polyindices=False,return_derivativeindices=False,
			   **kwargs):
	r'''
	Get matrix of multivariate polynomial V, where each term is product of monomials up to order order.
	
	A polynomial can be constructed with a vector of coefficients alpha, where
	p(x) = V \dot \alpha = \sum_{q=0}^{order} \sum_{\lambda : \sum \lambda = q} \alpha_{\lambda} x^{\lambda},
	where x^{\lambda} = \prod_{\mu} x^{\mu}^{\lambda_{\mu}} and V can be written as a block matrix, 
	where the block with q order terms has elements
		{V_q}_\lambda = x^{\lambda} 
	for all \lambda such that \sum \lambda = q.
	
	The derivative of this polynomial, for a p-length derivative index \nu is
	{d}_{\nu}(x) = D_{\nu} \dot \alpha = \sum_{q=0}^{order} \sum_{\lambda : \sum \lambda = q} \alpha_{\lambda} x^{\lambda}/x^{\nu} \prod_{\mu} (\lambda_{\mu}!/(\lambda_{\mu}-\nu_{\mu})!),
	where x^{\lambda} = \prod_{\mu} x^{\mu}^{\lambda_{\mu}} and D can be written as a block matrix, 
	where the block with q order terms has elements
		D_\nu_k_\nu_q_\lambda = dx^{\lambda}/dx^{\nu} 
	for all p-length \nu such that \sum \nu = k, and p-length \lambda such that \sum \lambda = q.
	
	If selection is None, then V has only unique terms of polynomial expansion and V has shape (n, p+order Choose order).
	If selection is 'unique', then V has only unique terms of polynomial expansion and V has shape (n, p+order Choose order).
	If unique is 'nonunique, then V has all terms of polynomial expansion and V has shape (n, (p^(order+1)-1)/(p-1)).
	If selection is 'powerset', then each monomial in a term can have up to maximum order order and V has shape (n, (order+1)^p)
	If intercept _ is False, then do not have constant term all 0 powers in monomials
	
	D has is matrix of all derivatives (assuming non-commutativity, q = (p^(derivative+1)-1)/(p-1)  or non-commutativity q = p+derivative choose p ) of shape (n,q,V.shape[1])
	
	Args:
		X (ndarray): matrix of data of shape (n,p) for n points of p dimensional data
		order (int): maximum order of polynomial
		derivative (int): maximum order of derivatives, order if None
		selection (str,ndarray,None): If None, include all terms with each term up to order order, 
			if 'unique', only unique monomial terms, 
			if 'powerset', each sub-monomial in term can have up to order order,
			if ndarray, keep only powers that are present in selection.
		commutativity (bool): Whether derivatives commute.
		intercept _ (bool): Keep term with all 0 powers in monomials
		variables (iterable,None): p dimensional variable names
		return_poly (bool): Return matrix of polynomial terms
		return_derivative (bool): Return matrix of derivative of polynomial terms
		return_coef (bool): Return vector of coefficients for polynomial terms
		return_polylabel (bool): Return dictionary of labels and indices of coefficients for polynomial terms        
		return_derivativelabel (bool): Return dictionary of labels and indices of derivative orders for polynomial terms        
		return_polyindices (bool): Return powers for matrix of polynomial terms
		return_derivativeindices (bool): Return powers for matrix of derivative of polynomial terms
	Returns:
		V (ndarray): matrix of data of polynomial terms
		D (ndarray): matrix of data of derivatives of polynomial terms
		alpha (ndarray): vector of coefficients for matrix of polynomial terms
		polylabel (dict): dictionary of labels and indices for matrix of polynomial terms
		derivativelabel (dict): dictionary of labels and indices for matrix of derivatives of polynomial terms
		polyindices (ndarray): powers for matrix of polynomial terms
		derivativeindices (ndarray): powers for matrix of derivative of polynomial terms
	'''
	
	n,p = X.shape
	order = 0 if None else order
	derivative = order if None else derivative

	if selection is None:
		indices = combinations(p,order,unique=True)
	elif selection in ['unique']:		
		indices = combinations(p,order,unique=True)
	elif selection in ['nonunique']:
		indices = combinations(p,order,unique=False)		
	elif selection in ['powerset']:
		indices = powerset(p,order)
	else:
		indices = np.array(selection)
		intercept_ = False

	derivatives = combinations(p,derivative,unique=commutativity)	


	X = broadcast(X)
	indices = broadcast(indices)
	derivatives = broadcast(derivatives)


	returns = ()
	
	if return_poly:
		V = (X**indices.transpose(2,1,0)).prod(axis=1)
		if not intercept_:
			V = V[:,1:]
		returns += (V,)
	
	if return_derivative:     
		
		numerator = sp.special.factorial(indices,exact=True)
		denominator = sp.special.factorial(indices-derivatives.transpose(2,1,0),exact=True)
		factorials = broadcast((numerator*invert(denominator,constant=0.0)).transpose(1,2,0),axes=0)

		numerator = broadcast(X**indices.transpose(2,1,0),axes=-2)
		denominator = broadcast(X**derivatives.transpose(2,1,0),axes=-1) 
		powers = numerator*invert(denominator,constant=0.0)

		# Make sure 0/0 -> 1 and 0^i/0^j -> 0 where i>j cases are handled
		zeroes = broadcast(X)**ones(powers.shape[2:])

		mask = (
			((broadcast((indices-derivatives.transpose(2,1,0)).transpose(1,2,0),0))>0)
			& (zeroes==0)
			)
		powers[mask] = 0

		mask = (
			((broadcast((indices-derivatives.transpose(2,1,0)).transpose(1,2,0),0))==0)
			& (zeroes==0)
			)		
		powers[mask] = 1

		D = (powers*factorials).prod(axis=1)
 
		if not intercept_:
			D = D[:,:,1:]
			
		returns += (D,)
		
	if return_coef:
		alpha = zeros(indices.shape[0])
		if not intercept_:
			alpha = alpha[1:]
			
		returns += (alpha,)
		
	if return_polylabel:   
		if variables is None:
			variables = ['x%d'%(i) for i in range(p)]
		name = ''
		delimeter = ''        
		splitter = '-'
		separator = '_'
		polylabel = {int(i-(1-intercept_)):'%s%s%s'%(name,delimeter,splitter.join(
							['%s%s%d'%(variable,separator,ind) for variable,ind in zip(variables,inds)]))
							for i,inds in enumerate(indices) if intercept_ or i>0}
				
		returns += (polylabel,)
		
	if return_derivativelabel:   
		
		if variables is None:
			variables = ['x%d'%(i) for i in range(p)]
		name = ''
		delimeter = ''        
		splitter = '-'
		separator = '_'
		derivativelabel = {int(i-(1-intercept_)):'%s%s%s'%(name,delimeter,splitter.join(
							['%s%s%d'%(variable,separator,ind) for variable,ind in zip(variables,inds)]))
							for i,inds in enumerate(derivatives) if intercept_ or i>0}
				
		returns += (derivativelabel,)        
	
	
	if return_polyindices:
		indices = indices[...,0]
		if not intercept_:
			indices = indices[1:]
		returns += (indices,)

	if return_derivativeindices:
		derivatives = derivatives[...,0]
		if not intercept_:
			derivatives =  derivatives[1:]
		returns += (derivatives,)

	
	return returnargs(returns)



def mesh(n,d=1,bounds=(0,1),distribution='uniform',shape=None,dtype=None):
	'''
	Generate d dimensional mesh with n points per dimension along bounds
	
	Args:
		n (int,iterable): Size of mesh along each dimension. If iterable, is different mesh size along each dimension
		d (int): Number of dimensions
		bounds (iterable): Bounds of data. Either iterable of [start,end] or iterable of iterables of [start,end] is different bounds along each dimension
		distribution (str,callable): Type of distribution to generate points, string in 'uniform',linspace','logspace','rand','randn','randint','chebyshev'
		shape (tuple): Shape if mesh is embedded in sparse array. Shape must have total number of elements N >= n**{d}
		dtype (str,data-type): Data type of mesh
	Returns:
		grid (ndarray): mesh of n^{d}, d dimensional points
	'''
	def wrap(func): 
		@functools.wraps(func)
		def wrapper(*args,**kwargs):
			out = func(*args,**kwargs)
			if out.ndim > 1:
				shape = out.shape
				out = sorted([tuple(o.ravel().tolist()) for o in out],key=lambda x:x)
				out = np.array([list(o) for o in out]).reshape(shape)
			else:
				out.sort()
			return out
		return wrapper
	
	types = {
		'constant':{
			'distributions':['uniform','linspace','logspace'],
			'type':'meshgrid',
			'func': lambda dim,n,d,bounds,points: points[:,dim]},
		'random':{
			'distributions':['random','rand','randn',],
			'type':'meshgrid',
			'func': lambda dim,n,d,bounds,points: points[:,dim]},
		'custom':{
			'distributions':['randint','randuniform','randintuniform'],
			'type':'meshgrid',
			'func': lambda dim,n,d,bounds,points: points(dim,n,d,bounds)},
		'mesh':{
			'distributions':['randommesh','randmesh','randnmesh','randintmesh','randuniformmesh'],
			'type':'mesh',
			'func': lambda dim,n,d,bounds,points: points(dim,n,d,bounds)},
	}

	assert callable(distribution) or any([distribution in types[typed]['distributions'] for typed in types]), "Error - distribution %r not permitted"%(distribution)

	if not isinstance(n,(list,np.ndarray)):
		n = [n]*d
	if not all([isiterable(bound) for bound in bounds]):
		bounds = [bounds]*d


	if callable(distribution):
		name = distribution
		@wrap
		def func(dim,n,d,bounds,name=name):
			return distribution(dim,n,d,bounds)
		func = distribution
	if distribution in ['uniform','linspace']:
		name = distribution
		@wrap		
		def func(dim,n,d,bounds,name=name):
			return np.linspace(*bounds,n,endpoint=True)			
	elif distribution in ['logspace']:        
		name = distribution
		@wrap		
		def func(dim,n,d,bounds,name=name):
			return np.logspace(*bounds,n)
	elif distribution in ['random','rand','randn']:
		name = 'rand' if distribution in ['random'] else distribution
		@wrap		
		def func(dim,n,d,bounds,name=name):
			return bounds[0] + (bounds[1]-bounds[0])*(getattr(np.random,name)(n)-0)/(1-0)
	elif distribution in ['randint']:
		name = 'randint' if distribution in [] else distribution		
		@wrap		
		def func(dim,n,d,bounds,name=name):
			return getattr(np.random,name)(bounds[0],bounds[1],n)
	elif distribution in ['randuniform']:
		name = 'randint' if distribution in ['randuniform'] else distribution		
		@wrap		
		def func(dim,n,d,bounds,name=name):
			return getattr(np.random,name)(bounds[0],bounds[1],n)/(bounds[1]-bounds[0])

	elif distribution in ['randintuniform']:
		name = 'randint' if distribution in ['randintuniform'] else distribution		
		@wrap		
		def func(dim,n,d,bounds,name=name):
			return getattr(np.random,name)(bounds[0],bounds[1],n)

	elif distribution in ['randommesh','randmesh','randnmesh']:
		name = 'rand' if distribution in ['randommesh','randmesh'] else 'randn' if distribution	['randnmesh'] else distribution
		@wrap		
		def func(dim,n,d,bounds,name=name):
			return bounds[0] + (bounds[1]-bounds[0])*(getattr(np.random,name)(n**d,d)-0)/(1-0)
	elif distribution in ['randintmesh']:
		name = 'randint' if distribution in ['randintmesh'] else distribution		
		@wrap		
		def func(dim,n,d,bounds,name=name):
			return getattr(np.random,name)(bounds[0],bounds[1],(n**d,d))
	elif distribution in ['randuniformmesh']:
		name = 'randint' if distribution in ['randuniformmesh'] else distribution		
		@wrap		
		def func(dim,n,d,bounds,name=name):
			return getattr(np.random,name)(bounds[0],bounds[1],(n**d,d))/(bounds[1]-bounds[0])

	if any([distribution in types[typed]['distributions'] for typed in types if types[typed]['type'] in ['meshgrid']]):
		grid = [u.reshape(-1,1) for u in np.meshgrid(*[func(dim,m,d,bound,name) for dim,(m,bound) in enumerate(zip(n,bounds))])]
	
	elif any([distribution in types[typed]['distributions'] for typed in types if types[typed]['type'] in ['mesh']]):
		dim = -1
		grid = func(dim,n[dim],d,bounds[dim],name)

	
	def wrapper(grid,d,axis=-1): 
		# return grid[:,[1,0,*arange(2,d)] if d>2 else slice(None)]
		if isinstance(grid,(list,tuple)):
			grid = concatenate(grid,axis=-1)
		return grid
	grid = wrapper(grid,d,axis=-1)

	if shape is not None:
		out = grid.ravel()
		size,dtype = out.size,out.dtype
		density = max(0,min(size/(shape[0]*shape[1]),1))
		grid = sp.sparse.rand(*shape,density=density,format='csr').astype(dtype)
		grid.data[:] = out[:grid.nnz]		
	

	dtype = grid.dtype if dtype is None else dtype
	grid = grid.astype(dtype)

	return grid



def zeros(shape,dtype=None,order='C',format=None):
	'''
	Create array of zeros
	
	Args:
		shape (int,tuple): Shape of array
		dtype (str,data-type): Data type of array
		order (str): 'C' for row-major or 'F' for column-major ordering
		format (str): Format of array, sparse_matrix or sparse_array
	Returns:
		out (ndarray,sparse_matrix,sparse_array): array of zeros
	'''

	if format in [None,'array']:
		out = np.zeros(shape,dtype=dtype,order=order)
	elif format in ['csr','csc','coo','lil','bsr','dia','dok']:
		out = getattr(sp.sparse,'%s_matrix'%(format))(shape,dtype=dtype)		
	elif format in ['COO']:
		format = format.lower()
		out = sparray.zeros(shape,dtype=dtype,format=format)
	else:
		out = np.zeros(shape,dtype=dtype,order=order)

	return out


def ones(shape,dtype=None,order='C',format=None):
	'''
	Create array of ones
	
	Args:
		shape (int,tuple): Shape of array
		dtype (str,data-type): Data type of array
		order (str): 'C' for row-major or 'F' for column-major ordering
		format (str): Format of array, sparse_matrix or sparse_array
	Returns:
		out (ndarray,sparse_matrix,sparse_array): array of ones
	'''

	if format in [None,'array']:
		out = np.ones(shape,dtype=dtype,order=order)
	elif format in ['csr','csc','coo','lil','bsr','dia','dok']:
		out = getattr(sp.sparse,'%s_matrix'%(format))(np.ones(shape,dtype=dtype,order=order),shape,dtype=dtype)
	elif format in ['COO']:
		format = format.lower()
		out = sparray.ones(shape,dtype=dtype,format=format)		
	else:
		out = np.ones(shape,dtype=dtype,order=order)

	return out


def arange(start,stop=None,step=None,dtype=None):
	'''
	Create array of zeros. Create range between start and stop, in increments of step
	
	Args:
		start (int,float): starting value of range, or range from 0 to start if stop and stop is None
		stop (int,float): stopping value of range
		step (int,float): step value of range
		dtype (str,data-type): Data type of range
	Returns:
		out (ndarray): array of range
	'''

	out = np.arange(start,stop,step,dtype=dtype)

	return out


def boundaries(data,size,adjacency=None,reverse=False,excluded=None,atol=None,rtol=None,kind='mergesort',n_jobs=None):
	'''
	Get indices of points in data that have at least a neighborhood of size within the data and are outside boundary region
	
	Args:
		data (ndarray): data array of shape (n,p) to be searched
		size (int): size of neighborhood
		adjacency (ndarray,sparse_matrix): array of shape (n,n) of where to compute differences between elements						
		reverse (bool): return indices inside boundary region
		excluded (ndarray,list): excluded argsort indices
		atol (float): Absolute tolerance of difference between unique elements
		rtol (float): Relative tolerance of difference between unique elements
		kind (str): Sort algorithm                		
		n_jobs (int): Number of parallel jobs		
	Returns:
		indices (ndarray): indices of data that are inside or outside of boundary region
	'''
	if data.ndim != 2:
		data = data.reshape((data.shape[0],-1))
	n,p = data.shape

	size = min(size,n)
	metric = 2
	if excluded is None:
		excluded = []

	
	indices = arange(n)

	sparsity = size+1

	weights = similarity_matrix(data,data,adjacency=adjacency,metric=metric,sparsity=sparsity,directed=False,kind=kind,n_jobs=n_jobs)

	argsort,counts = unique_argsort(np.abs(weights),return_counts=True,atol=atol,rtol=rtol,signed=False,kind=kind)


	excluded = np.array(excluded)
	excluded = [excluded[excluded<=i] for i in range(n)]
	nullsums = [np.array([add(counts[i][excluded[i][excluded[i]<=j]]) for j,c in enumerate(counts[i])]) if excluded[i].size>0 else 0 for i in range(n)]
	sums = [(cumsum(counts[i]) - nullsums[i]) for i in range(n)]

	nulllength = 0
	lengths = [(s - nulllength)>=(size) for s in sums]
	lengths = [where(l)[0] if l.any() else s.max() for s,l in zip(sums,lengths)]
	length = np.min(lengths)

	if not reverse:
		mask = lengths <= length
	else:
		mask = lengths > length

	indices = indices[where(mask)]

	return indices


def min_eps(arr,is_sorted=False,kind='mergesort'):
	'''	
	Get minimum distance between elements of array
	
	Args:
		arr (ndarray): Array to be searched
		is_sorted (bool): If array is previously sorted, do not sort when computing epsilon
		kind (str): Sort algorithm                
	Returns:
		eps (float): Minimum distance between elements of arr
	'''
	if not is_sorted or arr.ndim > 1:
		arr = np.diff(np.sort(arr.ravel(),kind=kind))
	else:
		arr = np.diff(arr.ravel())

	eps = (arr[arr>0]).min()
	return eps

def decimals(arr,return_eps=False):
	'''
	Get minimum number of decimals that when rounded, elements of array can still be distinguished
	
	Args:
		arr (ndarray): array of elements
		return_eps (bool): Return smallest relative difference between elements of arr
	Returns:
		decs (int): decimal places of smallest difference between elements of arr
		eps (float): smallest difference between elements of arr
	'''
	
	# Get minimum difference and base-10 decimals
	eps = min_eps(arr)
	decs = np.abs(np.int(np.floor(np.log10(eps))))

	# Get return values
	returns = ()
	returns += (decs,)
	if return_eps:
		returns += (eps,)

	return returnargs(returns)


# Safely invert x
def invert(x,constant=1,copy=True):
	# Safely divide by zero 
	# error state ignored due to unknown numpy error:
	# RuntimeWarning: divide by zero encountered in true_divide
	with np.errstate(divide='ignore'):
		if isarray(x):
			x = x.copy() if copy else x
			sparse = issparse(x)
			if sparse:
				isx = (np.isnan(x.data) | np.isinf(x.data) | eq(x.data,0))
				x.data[isx] = 1
				x.data[:] = 1/x.data[:]
				x.data[isx] = constant
			else:
				y = x.copy()
				isx = (np.isnan(x) | np.isinf(x) | eq(x,0))
				x[isx] = 1
				x = 1/x
				x[isx] = constant
		else:		
			isx = (np.isnan(x) | np.isinf(x) | eq(x,0))
			x = 1 if isx else x
			x = 1/x
			x = constant if isx else x
	return x


# Safely divide y/x and account for y==x==0 -> zero
def divide(y,x,constant=0,zero=1,copy=True):
	d = multiply(y,invert(x,constant=constant,copy=copy))
	d[eq(y,0) & eq(x,0)] = zero
	return d

# Remove diagonal from array
def nodiagonal(arr):
	n,m = arr.shape[:2]
	return arr[~np.eye(n,dtype=bool)].reshape(n,-1)


# Multiply along axis
def _multiply_along(arr,multiple,axis):
	shape = ones((1,arr.ndim),(int,np.integer)).ravel()
	shape[axis] = -1
	multiple = multiple.reshape(shape)
	return arr*multiple  


# As numpy ndarray function
def asndarray(arr,dtype=None,order=None):
	if issparse(arr):
		arr = arr.todense()
	return np.array(arr)

# As array function
def asarray(arr,dtype=None,order=None,format=None,like=None):
	format = like.getformat() if issparse(like) else None
	dtype = getattr(arr,'dtype',type(arr)) if dtype is None else dtype
	if format in ['csr','csc','coo','lil','bsr','dia','dok']:
		typed = 'sparse'
		format = format
		arraylike = lambda arr,dtype,order,format,like: assparse(arr,like,dtype,order,format)
	elif format in ['array',None]:
		typed = 'dense'
		format = 'array'
		arraylike = lambda arr,dtype,order,format,like: np.asarray(arr,dtype,order)
	else:
		typed = 'dense'
		format = 'array'
		arraylike = lambda arr,dtype,order,format,like: np.asarray(arr,dtype,order)

	scalar = isscalar(arr)
	listtuple = islisttuple(arr)

	if scalar:
		shape = like.shape if like is not None else 1
		arr = arr*ones(shape,dtype,order) 
	elif listtuple:
		shape = len(arr)
		arr = arr

	sparse = issparse(arr)	

	arr = arr.A if sparse and typed not in ['sparse'] else arr

	return arraylike(arr,dtype,order,format,like)


# As sparse array function
def assparse(arr,like,dtype=None,order=None,format=None):
	dtype = getattr(arr,'dtype',type(arr)) if dtype is None else dtype
	if like is None:
		like = getattr(sp.sparse,'%s_matrix'%(format))(arr.shape,dtype=dtype)
	else:
		like = like.asformat(format,copy=True).astype(dtype)
	sparse = issparse(arr)
	scalar = isscalar(arr)
	if sparse:
		like.data[:] = arr.data[:]
	elif scalar:
		like.data[:] = arr
	else:
		like.data[:] = arr.ravel()[:like.nnz]
	return like


# As format function
def asformat(arr,format):
	out = arr
	if format is None:
		return out
	if issparsematrix(arr):
		out = arr.asformat(format)
	elif issparsearray(arr):
		fmt = 'csr'
		out = getattr(arr,'to%s'%(fmt))().asformat(format)
	elif format not in ['array']:
		fmt = 'csr'
		out = getattr(sp.sparse,'%s_matrix'%(fmt))(arr).asformat(format)
	return out


# As dtype function
def asdtype(arr,dtype):
	if dtype is None:
		out = arr
	else:
		out = arr.astype(dtype)
	return out


def broadcast(arr,axes=None,shape=None,axis=None,newaxis=True):
	'''
	Broadcast array  to have expanded shape, depending on shape and axes
	
	Args:
		arr (ndarray,sparse_matrix,sparse_array): array of shape (n_0,...,n_ndim-1) to be broadcasted
		axes (int,tuple,list,ndarray): axes on which to expand array. Defaults to -1 axis
		shape (int,tuple,list,ndarray): sizes of dimensions to expand or repeat array. Defaults to 1
		axis (int): single axis to broadcast, default if axes is None
		newaxis (bool,tuple,list,ndarray): booleans on whether to expand dimension at axis, or repeat existing axis
	Returns:
		out (ndarray,sparse_matrix,sparse_array): array with expanded dimensions
	'''
	
	# Get if matrix is sparse
	sparse = issparse(arr)
	
	# Ensure arguments are iterables
	if shape is None:
		shape = 1
	if axes is None and axis is None:
		axes = -1
	elif axes is None and isinstance(axis,(int,np.integer)):
		axes = axis
	if isinstance(shape,(int,np.integer)):
		shape = [shape]
	if isinstance(axes,(int,np.integer)):
		axes = [axes]*len(shape)
	if isinstance(newaxis,bool):
		newaxis = [newaxis]*len(shape)

	out = arr

	# Expand shape of arr
	for size,axis,new in zip(shape,axes,newaxis):
		if new:
			out = expand_dims(out,axis)
		out = repeat(out,size,axis)

	return out
  


def outer(func,a,b,where=None,squeeze=False):
	'''
	Apply binary function on broadcasted arrays
	
	Args:
		func (callable): binary function that takes 2 broadcasted arrays as arguments of shapes (n,(d),m,(d)) and returns broadcasted function of shape (n,(d),m,(d))
		a (ndarray): array of shape (n,(d))
		b (ndarray): array of shape (m,(d))
		where (ndarray,sparse_matrix): array of shape (n,m) of where to compute differences between elements
		squeeze (bool): If a and b are 1 dimensional, (d) = (), then remove expanded dimensions that have size 1 after function
	Returns:
		out (ndarray,sparse_matrix,sparse_array): array of shape (n,m,(d)) of binary function between elements in array
	'''

	# Reshape arrays
	isndim = a.ndim == 1
	ismdim = b.ndim == 1
	if isndim:
		a = a.reshape((-1,1))
	if ismdim:
		b = b.reshape((-1,1))





	# Get array shapes	
	n,d,ndim = a.shape[0],a.shape[1:],a.ndim
	m,p,mdim = b.shape[0],b.shape[1:],b.ndim


	assert (ndim==mdim) and (all([i==j for i,j in zip(d,p)])), "Error - unequal a,b array shapes"

	shape = (n,m)


	# Get booleans for where
	exists = where is not None
	sparse = issparse(where) and exists

	# Get format of where
	format = 'array' if not sparse else where.getformat()



	# Get booleans for dimensionality of where	
	ndsparse = exists and ndim>1


	if ndsparse:
		fmt = 'COO'
		where = getattr(sparray,fmt)(where).reshape((*where.shape,*[1]*(ndim-1)))

	onedsparse = not ndsparse and sparse




	# Expand array dimensions for broadcasted operations
	# a shape (n,d) -> (n,(d),1,(1)) with ndim = 2(1+|d|)
	# b shape (m,d) -> (1,(1),m,(d)) with mdim = 2(1+|d|)


	if not exists:
		a = broadcast(a,axes=(-1,)*ndim,shape=(1,)*ndim)
		b = broadcast(b,axes=(0,)*ndim,shape=(1,)*ndim)
	elif onedsparse:
		a = where.T.multiply(a).T
		b = where.multiply(b)
	elif ndsparse:
		a = (where.swapaxes(0,1)*a).swapaxes(0,1) 
		b = where*b
	else:
		a = (where.T*a).T 
		b = where*b


	# Compute outer function on arrays
	# func shape (n,(d),m,(d)) if not exists else (n,m,(d))
	out = func(a,b)

	# Get diagonal elements of out such that out is elementwise function 
	# out shape (n,m,(d))
	if not exists:
		for dim in range(ndim-1):
			out = getdiag(out,offset=0,axis1=1,axis2=ndim-dim+1)

	# Squeeze last dimensions if original arrays are 1 dimensional
	if squeeze and (isndim and ismdim):
		out = out.reshape((n,m))

	fmt = 'csr'
	if not sparse and ndsparse and exists:
		out = out.todense()
	elif onedsparse or (isndim and ismdim):
		out = getattr(out,'to%s'%(fmt))().asformat(format)
	elif ndsparse:
		pass

	return out





# Invert sparse boolean array
def invert_sparse_bool(arr):
	shape,dtype,format = arr.shape,arr.dtype,arr.getformat()
	arr = getattr(sp.sparse,'%s_matrix'%(format))(~arr.A)
	return arr

# Reshape data
def reshaper(X,shape,axis={1:-1},order='C'):
	shape = [size for size in shape]
	for ax in axis:
		shape[ax] = axis[ax]
	return X.reshape(shape,order=order)


# Delete obj indices along axis from arr
def delete(arr,obj,axis=None):
	try:
		out = np.delete(arr,obj,axis)
	except (AttributeError,TypeError):
		_obj = arange(arr.shape[axis])
		_obj = _obj[isin(_obj,obj,invert=True)]
		out = take(arr,_obj,axis)
	return out

# Concatenate arrays along axis
def concatenate(arrs,axis=None):
	sparse = any([issparse(arr) for arr in arrs])
	if sparse:
		sparsematrix = any([issparsematrix(arr) for arr in arrs])
		sparsearray = all([issparsearray(arr) for arr in arrs])
		if sparsematrix:
			if axis in [0,None]:
				out = sp.sparse.vstack(arrs)
			elif axis in [1,-1]:
				out = sp.sparse.hstack(arrs)
		elif sparsearray:
			out = sparray.concatenate(arrs,axis=axis)
	else:
		out = np.concatenate(arrs,axis=axis)
	return out

# Take indices along axis from arr
def take(arr,indices,axis):
	sparse = issparse(arr)
	single = isinstance(axis,(int,np.integer))
	if not single:
		axes = np.array(axis,dtype=int)
		indexes = np.array(indices,dtype=object)

		argsort = axes.argsort()[::-1]
		axes = axes[argsort]
		indexes = indexes[argsort]
	else:
		axes = [axis]
		indexes = [indices]

	out = arr
	for axis,indices in zip(axes,indexes):
		integer = isinstance(indices,(int,np.integer))
		shape = out.shape
		if sparse:
			sparsematrix = issparsematrix(arr)
			sparsearray = issparsearray(arr)
			if sparsematrix:
				if axis in [0]:
					out = out[indices,:]
				elif axis in [1,-1]:
					out = out[:,indices]			
			elif sparsearray:
				if integer:
					out = out.reshape((1,*shape))
				out = arr.swapaxes(axis,0)[indices]
				if integer:
					out = out.reshape((*shape[:axis],*shape[axis+1:]))
				else:
					out = out.swapaxes(axis,0).reshape((*shape[:axis],len(indices),*shape[axis+1:]))
		else:
			out = np.take(out,indices,axis)
	return out


# Expand dimensions along axis
def expand_dims(arr,axis):
	shape,ndim = list(arr.shape),arr.ndim
	axis = ndim+1+axis if axis<0 else axis
	shape.insert(axis,1)

	out = arr.reshape(shape)
	return out

# Indices in one dimensional array
def isin(arr,elements,assume_unique=False,invert=False):
	sparse = issparse(arr)
	if sparse:

		sparsematrix = issparsematrix(arr)
		sparsearray = issparsearray(arr)

		if sparsematrix:
			shape = arr.shape
			format = arr.getformat()

			fmt = 'csr'
			out = arr.asformat(fmt,copy=True).astype(bool)
			indices = out.indices
			indptr = out.indptr

			fmt = 'lil'
			out = out.asformat(fmt)

			for i in range(shape[0]):
				out[i,indices[indptr[i]:indptr[i+1]]] = np.isin(arr[i].data,elements,assume_unique=assume_unique,invert=invert)
			out = out.asformat(format)

		elif sparsearray:
			shape = arr.shape
			format = 'COO'
			arr = arr.reshape(-1)
			coords = arr.nonzero()
			data = np.isin(arr[indices].todense(),elements,assume_unique=assume_unique,invert=invert)
			out = getattr(sparray,format)(coords=coords,data=data,shape=shape)
	else:
		out = np.isin(arr,elements,assume_unique=assume_unique,invert=invert)
	return out


# Find where elements in array are true
def where(condition,x=None,y=None):
	# Find where sparse elements in array are true
	def wheresparse(condition,x=None,y=None):
		def wheretrue(rows,cols,data,like,x):
			x = asarray(x,like=like)
			shape,dtype,format = x.shape,x.dtype,x.getformat()

			fmt = 'coo'
			data = ones(data.shape,dtype=dtype)
			data = getattr(sp.sparse,'%s_matrix'%(fmt))((data,(rows,cols)),shape=shape,dtype=dtype)

			fmt = 'csr'
			try:
				x = x.asformat(fmt)[rows,cols][0]
			except:
				pass
			data.data[:] = x
			data = data.asformat(format)
			data.eliminate_zeros()        
			return data

		def wherefalse(rows,cols,data,like,y):
			like = invert_sparse_bool(like)
			y = asarray(y,like=like)
			shape,dtype,format = y.shape,y.dtype,y.getformat()

			fmt = 'lil'
			data = y.asformat(fmt,copy=True)
			data[rows,cols] = 0
			data = data.asformat(format)
			data.eliminate_zeros()
			return data

		shape = condition.shape
		
		is1d = condition.shape[0] == 1

		isxy = (x is not None) and (y is not None)
		   
		rows,cols,data = sp.sparse.find(condition)

		inds = rows.argsort()
		rows,cols,data = rows[inds],cols[inds],data[inds]

		if isxy:
			data = wheretrue(rows,cols,data,condition,x) + wherefalse(rows,cols,data,condition,y)
			return data
		else:
			returns = ()

			if is1d:
				returns += (cols,)
			else:
				returns += (rows,cols,)
			return returnargs(returns)
		
		return
		

	# Find where dense elements in array are true
	def wheredense(condition,x=None,y=None):
		
		isxy = (x is not None) and (y is not None)
		
		if isxy:
			return np.where(condition,x,y)
		else:
			return returnargs(np.where(condition))



	sparse = issparse(condition)

	if sparse:
		return wheresparse(condition,x,y)
	else:
		return wheredense(condition,x,y)



# Get cumulative sum of array
def cumsum(arr,axis=None):
	sparse = issparse(arr)
	if sparse:
		shape,dtype = arr.shape,arr.dtype
		sparsematrix = issparsematrix(arr)
		sparsearray = issparsearray(arr)
		if sparsematrix:
			format = arr.getformat()
			fmt = 'lil'
			out = getattr(sp.sparse,'%s_matrix'%(fmt))((shape[1-axis],shape[axis]),dtype=dtype)
			arr = arr.T if axis in [0] else arr
			for i in range(shape[1-axis]):
				out[i,arr.indices[arr.indptr[i]:arr.indptr[i+1]]] = np.cumsum(arr[i].data)
			out = out.T if axis in [0] else out
			out = out.asformat(format)
		elif sparsearray:
			raise ValueError("Not implemented for sparse_array")
	else:
		out = np.cumsum(arr,axis=axis)
	return out


# Returns a boolean array where two arrays are element-wise equal within a tolerance.
def isclose(a, b=None, rtol=None, atol=None, equal_nan=False):

	sparse = issparse(a) or issparse(b)
	format = a.getformat() if sparse else None
	shape,dtype = a.shape,a.dtype

	if b is None:
		b = zeros(shape,dtype=dtype,format=format)


	# Get tolerances based on minimum differences
	tol = 1e-10
	absa = np.abs(a)	
	try:
		tol = max(tol,(absa.min())/max(1,(absa.max()-absa.min()))*(tol**(1/2)))
	except:
		tol = tol
	if atol is None:
		atol = tol
	if rtol is None:
		rtol = tol

	if sparse:
		out = le(np.abs(a-b),atol + rtol*absa)
	else:
		out = np.isclose(a,b,rtol=rtol,atol=atol,equal_nan=equal_nan)
	return out


# Number of non-zero elements
def nonzeros(arr,axis=-1):
	sparse = issparse(arr)
	scalar = isscalar(arr)
	if sparse:
		counts = arr.astype(bool).sum(axis).astype(int)
	elif scalar:
		counts = int(arr != 0)
	else:
		counts = np.count_nonzero(arr,axis=axis)
	return counts


# Single Tiles (array only)
def tile(arr,reps,axis=-1):
	sparse = issparse(arr)
	if sparse:
		raise
		shape = arr.shape
		size = shape[axis]
		
		out = stack([arr]*repeats,axis)
		
		indices = np.array([arange(i,size*repeats,size) for i in range(size)]).reshape(-1)
		permutation = np.array([arange(i*repeats,(i+1)*repeats,1) for i in range(size)]).reshape(-1)
		
		out = out.reshape((*shape[:axis],-1,*shape[axis+1:]))
		out = out.swapaxes(0,axis)
		out[permutation] = out[indices]
		out = out.swapaxes(0,axis)
	else:
		out = np.tile(arr,reps=reps)
	return out


# Single Repeats
def repeat(arr,repeats,axis=-1):
	sparse = issparse(arr)
	if sparse:
		shape = arr.shape
		size = shape[axis]
		
		out = stack([arr]*repeats,axis)
		
		indices = np.array([arange(i,size*repeats,size) for i in range(size)]).reshape(-1)
		permutation = np.array([arange(i*repeats,(i+1)*repeats,1) for i in range(size)]).reshape(-1)
		
		out = out.reshape((*shape[:axis],-1,*shape[axis+1:]))
		out = out.swapaxes(0,axis)
		out[permutation] = out[indices]
		out = out.swapaxes(0,axis)
	else:
		out = np.repeat(arr,repeats=repeats,axis=axis)
	return out


# Single Stack along new axis
def stack(arrs,axis=-1):
	sparse = any([issparse(arr) for arr in arrs])
	if sparse:
		fmt = 'COO'
		sparsematrix = any([issparsematrix(arr) for arr in arrs])
		sparsearray = all([issparsearray(arr) for arr in arrs])
		if sparsematrix:
			format = arr.getformat()
			out = sparray.stack([getattr(sparray,fmt)(arr) for arr in arrs],axis)
		elif sparsearray:
			out = sparray.stack(arrs,axis)
	else:
		out = np.stack(arrs,axis=axis)
	return out





def reduce(func,arr,axis,dtype=None,where=True,**kwargs):
	'''
	Reduce elements in arr along axis at locations where
	
	Args:
		arr (ndarray,sparse_matrix,sparse_array): array to be reduced
		axis (int): axis along which to reduce
		dtype (str,data-type): datatype of resulting output
		where (bool,ndarray): where to reduce elements of arr
	Returns:
		out (ndarray,sparse_matrix,sparse_array): output reduced array
	'''

	sparse = issparse(arr)

	dtype = arr.dtype if dtype is None else dtype
	where = True if where is None else where


	if sparse:

		sparsearray = issparsearray(arr)			
		sparsematrix = issparsematrix(arr)			

		fmt = 'csr'
		shape = list(arr.shape)
		dim = shape.pop(axis)
		size = prod(np.array(shape))

		
		if sparsematrix:
			format = arr.getformat()
			if axis in [0]:
				arr = arr.T.asformat(fmt) # fmt sparse matrix of shape (size,dim)
			elif axis in [1,-1]:
				arr = arr.asformat(fmt) # fmt sparse matrix of shape (size,dim)

		elif sparsearray:
			format = 'COO'
			arr = getattr(arr.swapaxes(axis,-1).reshape((-1,dim)),'to%s'%(fmt))() # fmt sparse matrix of shape (size,dim)			
		
		if issparsematrix(where):
			where = where.asformat(fmt).reshape((-1,1))
		elif issparsearray(where):
			where = getattr(where,'to%s'%(fmt))().asformat(fmt).reshape((-1,1))

		inds,iptr = arr.indices,arr.indptr

		fmt = 'csr'
		data,indices,indptr = [],[],[0]

		arr = multiply(arr,where)

		for i in range(size):
			if iptr[i+1]>iptr[i]:
				data.append(func(arr[i,inds[iptr[i]:iptr[i+1]]].data,axis=-1,dtype=dtype,**kwargs))
				indices.append(0)
				indptr.append(indptr[-1]+1)
			else:
				indptr.append(indptr[-1])



		shape = (*shape,1) if len(shape) < 2 else shape
		out = getattr(sp.sparse,'%s_matrix'%(fmt))((data,indices,indptr),shape=(size,1),dtype=dtype).reshape(shape)

		if sparsematrix:
			out = out.asformat(format)
		elif sparsearray:
			out = getattr(sparray,format)(out)
	else:
		out = func(arr,axis=axis,dtype=dtype,where=where,**kwargs)

	return out



def add(arr,axis=-1,dtype=None,where=True,size=False):
	'''
	Sum elements in arr along axis at locations where, dividing by size
	
	Args:
		arr (ndarray,sparse_matrix,sparse_array): array to be summed
		axis (int): axis along which to sum
		dtype (str,data-type): datatype of resulting output
		where (bool,ndarray): where to sum elements of arr
		size (bool): Normalize sum by number of sum
	Returns:
		out (ndarray,sparse_matrix,sparse_array): output summed array	
	'''

	# Get sparsity
	sparse = issparse(arr)
	exists = not isscalar(where)
	sparsearray = issparsearray(arr)			
	sparsematrix = issparsematrix(arr)				
	sparsewhere = issparse(where)

	if size:
		if exists:
			size = nonzeros(where,axis=axis)
		elif sparse:
			fmt = 'csr'
			if sparsematrix:
				size = arr.getnnz(axis)
			elif sparsearray:
				size = getattr(sp.sparse,'%s_matrix'%(fmt))(getattr(arr.swapaxes(0,axis).reshape((arr.shape[axis],-1)).T,'to%s'%(fmt)).getnnz(-1))
		else:
			size = arr.shape[axis]

	else:
		size = 1

	def func(arr,axis=None,dtype=None,where=True):
		out = arr.sum(axis=axis,dtype=dtype,where=where)

		return out

	out = reduce(func,arr,axis=axis,dtype=dtype,where=where)

	sparsearray = issparsearray(out)			
	sparsematrix = issparsematrix(out)				

	if sparsematrix:
		format = out.getformat()
	elif sparsearray:
		format = 'COO'
	else:
		format = 'array'

	out = (multiply(out.T,invert(size,constant=0.0))).T

	if sparsematrix:
		out = out.asformat(format)
	elif sparsearray:
		pass
	else:
		pass

	return out


def subtract(a,b,where=None,dtype=None,squeeze=True):
	'''
	Differences elements in b-a along first axis with shape (n,m,(d)) (b-a)[i,j] = b[j]-a[i]

	Args:
		a (ndarray): array with shape (n,(d)) to be subtracted by
		b (ndarray): array with shape (m,(d)) to be subtracted from
		where (bool,ndarray,sparse_matrix,sparse_array): where to compute differences in arrays
		dtype (str,data-type): datatype of resulting output		
		squeeze (bool): If a and b are 1 dimensional, (d) = (), then remove expanded dimensions that have size 1 after function		
	Returns:
		out (ndarray): output of differences of arrays, of shape (n,m,(d))
	'''

	def func(a,b):
		return -np.subtract(a,b,dtype=dtype) + np.array(0,dtype=dtype)

	out = outer(func,a,b,where=where,squeeze=squeeze)

	return out


def prod(arr,axis=-1,dtype=None,where=True):
	'''
	Multiply elements in arr along axis at locations where

	Args:
		arr (ndarray): array to be summed
		axis (int): axis along which to sum
		dtype (str,data-type): datatype of resulting output
		where (bool,ndarray): where to sum elements of arr
	Returns:
		out (ndarray): output multiplied array		
	'''


	def func(arr,axis=None,dtype=None,where=True):
		try:
			out = arr.prod(axis=axis,dtype=dtype,where=where)
		except:
			out = np.prod(arr,dtype=dtype,where=where)
		return out

	out = reduce(func,arr,axis=axis,dtype=dtype,where=where)

	return out



def multiply(*arrs):
	'''
	Multiply many arrays elementwise in place

	Args:
		arrs: Arrays to multiply
	Returns:
		out (ndarray) if out argument is not None		
	'''
	
	out = arrs[0]

	sparse = issparse(out)


	for arr in arrs[1:]:
		if not sparse:
			out = out*arr
		else:
			out = out.multiply(arr)
	
	return out

def mean(arr,axis=None,ddof=0):
	return np.mean(arr,axis=axis)

def var(arr,barr=None,axis=None,ddof=0):
	return np.covar(arr,barr,axis=axis,ddof=ddof)

def sqrt(arr):
	sparse = issparse(arr)
	if sparse:
		out = arr.sqrt()
	else:
		out = np.sqrt(arr)
	return out


# Get ord-norm of y-x, with sign of cos(x,y) = x.y/|x|.|y| along axis
def normsign(x,y,ord=2,axis=-1,normed=True,signed=True):
	shape = list(x.shape)
	size = shape.pop(axis)
	z = ones(shape)
	if normed:
		z = norm(y-x,axis=axis,ord=ord)
	if signed:
		z *= sign(x,y,ord=ord,axis=axis)
	return z

# Get ord-norm of y-x, with sign of cos(y-x,1) = y-x-1/|y-x|.|1| along axis
def normsigndiff(x,y,ord=2,axis=-1,normed=True,signed=True):
	shape = list(x.shape)
	size = shape.pop(axis)
	z = ones(shape)
	if normed:
		z = norm(y-x,axis=axis,ord=ord)
	if signed:
		z *= signdiff(x,y,ord=ord,axis=axis)
	return z

# Get sign of cos(x,y) = x.y/|x|.|y| along axis
def sign(x,y,ord=2,axis=-1):
	isx = eq(x,0).all(axis=axis)
	isy = eq(y,0).all(axis=axis)

	s = np.sign(cosine(x,y,ord=ord,axis=axis))

	s[multiply(isx,ne(isy,True))] = np.sign(add(y,axis=axis))[multiply(isx,ne(isy,True))]
	s[multiply(ne(isx,True),isy)] = np.sign(add(x,axis=axis))[multiply(ne(isx,True),isy)]

	return s

# Get sign of cos(y-x,1) = (y-x).1/|y-x|.|1| along axis
def signdiff(x,y,ord=2,axis=-1):
	o = ones(x.shape[axis],dtype=x.dtype)
	s = np.sign(cosine(y-x,o,ord=ord,axis=axis))
	iszero = (~(x==y).all(axis=axis)) & (s==0)
	s[iszero] = -1
	return s

# Get cos(x,y) = x.y/|x|.|y| along axis
def cosine(x,y,ord=2,axis=-1):
	xy = multiply(x,y)**(ord/2)
	return multiply(add(xy,axis=axis),invert(norm(x,axis=axis,ord=ord)*norm(y,axis=axis,ord=ord),constant=1.0))

# Get acos(x,y) = acos(x.y/|x|.|y|) along axis
def acosine(x,y,ord=2,axis=-1):
	return np.acos(cosine(x,y,ord=ord,axis=axis))


def gt(arr,value):
	'''
	Greater than function for explicit elements of array
	
	Args:
		arr (ndarray,sparse_matrix,sparse_array): array for comparision
		value (ndarray,int,float): value for comparison
	Returns:
		out (ndarray,sparse_matrix,sparse_array): boolean array of comparison of arr to value	
	'''
	sparse = issparse(arr)
	scalar = isscalar(value)
	if sparse:
		if scalar:
			if value < 0:
				out = arr.astype(bool,copy=True)
				out.data = arr.data > value				
			else:
				out = arr > value
		else:
			out = arr > value
	else:
		out = arr > value
	return out

def lt(arr,value):
	'''
	Less than function for explicit elements of array
	
	Args:
		arr (ndarray,sparse_matrix,sparse_array): array for comparision
		value (ndarray,int,float): value for comparison
	Returns:
		out (ndarray,sparse_matrix,sparse_array): boolean array of comparison of arr to value	
	'''
	sparse = issparse(arr)
	scalar = isscalar(value)
	if sparse:
		if scalar:
			if value > 0:
				out = arr.astype(bool,copy=True)
				out.data = arr.data < value
			else:
				out = arr < value
		else:
			out = arr < value
	else:
		out = arr < value
	return out

def ge(arr,value):
	'''
	Greater than or equal function for explicit elements of array
	
	Args:
		arr (ndarray,sparse_matrix,sparse_array): array for comparision
		value (ndarray,int,float): value for comparison
	Returns:
		out (ndarray,sparse_matrix,sparse_array): boolean array of comparison of arr to value
	'''
	sparse = issparse(arr)
	scalar = isscalar(value)
	if sparse:
		if scalar: 
			if value <= 0:
				out = arr.astype(bool,copy=True)
				out.data = arr.data >= value				
			else:
				out = arr >= value
		else:
			out = arr >= value
	else:
		out = arr >= value
	return out

def le(arr,value):
	'''
	Less than or equal function for explicit elements of array
	
	Args:
		arr (ndarray,sparse_matrix,sparse_array): array for comparision
		value (ndarray,int,float): value for comparison
	Returns:
		out (ndarray,sparse_matrix,sparse_array): boolean array of comparison of arr to value
	'''
	sparse = issparse(arr)
	scalar = isscalar(value)
	if sparse:
		if scalar: 
			if value >= 0:
				out = arr.astype(bool,copy=True)
				out.data = arr.data <= value				
			else:
				out = arr <= value
		else:
			out = arr <= value
	else:
		out = arr <= value
	return out


def eq(arr,value):
	'''
	Equal function for explicit elements of array
	
	Args:
		arr (ndarray,sparse_matrix,sparse_array): array for comparision
		value (ndarray,int,float): value for comparison
	Returns:
		out (ndarray,sparse_matrix,sparse_array): boolean array of comparison of arr to value
	'''
	sparse = issparse(arr)
	scalar = isscalar(value)
	if sparse:
		if scalar: 
			if value == 0:
				out = arr.astype(bool,copy=True)
				out.data = arr.data == value
			else:
				out = arr == value
		else:
			out = arr == value
	else:
		out = arr == value
	return out

def ne(arr,value):
	'''
	Not equal function for explicit elements of array
	
	Args:
		arr (ndarray,sparse_matrix,sparse_array): array for comparison
		value (ndarray,int,float): value for comparison
	Returns:
		out (ndarray,sparse_matrix,sparse_array): boolean array of comparison of arr to value
	'''
	sparse = issparse(arr)
	scalar = isscalar(value)
	if sparse:
		if scalar: 
			if value in [1,0,True,False]:
				out = arr.astype(bool,copy=True)
				out.data = arr.data != value
			else:
				out = arr != value
		else:
			out = arr != value
	else:
		out = arr != value
	return out





# Get indices,indptr from row,col of sparse matrix
def indices_rowcol(row,col,shape,format):
	argsort = row.argsort()
	row,col = row[argsort],col[argsort]

	if format in ['csr','csc','coo','lil','bsr','dia','dok']:
		indices = col
		indptr = zeros(shape[0]+1,dtype=int)
		row,counts = unique_tol(row,return_unique=True,return_counts=True)
		indptr[row+1] = counts
		indptr = cumsum(indptr)
	else:
		indices,indptr = row,col

	return indices,indptr



# Get row,col from indices,indptr of sparse matrix
def rowcol_indices(indices,indptr,shape,format):
	if format in ['csr','csc','coo','lil','bsr','dia','dok']:
		diffs = np.diff(indptr)
		inds = where(diffs != 0)
		diffs = diffs[inds]
		nnz,size,counts = diffs.sum(),inds.size,cumsum([0,*diffs])
		row = zeros(nnz,dtype=int)
		for i in range(size):
			row[counts[i]:counts[i+1]] = inds[i]
		col = indices
	else:
		row,col = indices,indptr
	return row,col


def subarray(arr,indices,dtype=None,format=None):
	'''
	Get submatrix of array along indices, with explicit zeros if sparse
	
	Args:
		arr (ndarray,sparse_matrix,sparse_array): array of shape (n,m) to get submatrix
		indices (ndarray): arr row indices of length size <= n
		dtype (str,data-type): data type of submatrix, defaults to data type of arr if None
		format (str): format of submatrix ('array','csr','csc','coo','lil','bsr','dia','dok','COO'), defaults to format of arr if None
	Returns:
		out (ndarray,sparse_matrix,sparse_array): submatrix of shape (n,m) with size elements from arr at indices
	'''

	shape,size = arr.shape,indices.size
	rows,cols = repeat(indices,repeats=shape[1],axis=0).ravel(),repeat(arange(shape[1])[None,:],repeats=size,axis=0).ravel()
	subrows,subcols = rows,cols
	out = submatrix(arr,shape=shape,rows=rows,cols=cols,subrows=subrows,subcols=subcols,dtype=dtype,format=format)

	return out



def submatrix(arr,shape,rows,cols,subrows,subcols,dtype=None,format=None):
	'''
	Get submatrix of array, with explicit zeros if sparse
	
	Args:
		arr (ndarray,sparse_matrix,sparse_array): array of shape (n,m) to get submatrix
		shape (iterable): shape of submatrix (s,t)
		rows (ndarray): arr row indices of length k <= n*m
		cols (ndarray): arr column indices of length k <= n*m
		subrows (ndarray): submatrix row indices of length k <= n*m
		subcols (ndarray): submatrix column indices of length k <= n*m
		dtype (str,data-type): data type of submatrix, defaults to data type of arr if None
		format (str): format of submatrix ('array','csr','csc','coo','lil','bsr','dia','dok','COO'), defaults to format of arr if None
	Returns:
		out (ndarray,sparse_matrix,sparse_array): submatrix of shape (s,t) with k elements from arr at rows and cols
	'''

	n,m = arr.shape
	k = min(rows.size,cols.size,subrows.size,subcols.size)	

	sparse = issparse(arr)
	sparsematrix = issparsematrix(arr)
	sparsearray = issparsearray(arr)

	if dtype is None:
		dtype = arr.dtype
	if format is None:
		if not sparse:
			format = 'array'
		elif sparsematrix:
			format = arr.getformat()
		elif sparsearray:
			format = 'COO'


	data = zeros(k,dtype=dtype)
	for l,(i,j) in enumerate(zip(rows,cols)):
		data[l] = arr[i,j]

	if format in ['array']:
		out = zeros(shape,dtype=dtype,format=format)
		for l,(i,j) in enumerate(zip(subrows,subcols)):
			out[i,j] = data[l]
	elif format in ['csr','csc','coo','lil','bsr','dia','dok']:
		out = getattr(sp.sparse,'%s_matrix'%(format))((data,(subrows,subcols)),shape=shape,dtype=dtype)
	elif format in ['COO']:
		coords = np.array([subrows,subcols])
		out = getattr(sparray,format)(coords,data=data,shape=shape)		


	return out



def vandermonde(X,order,ones=True,unique=True,dims=None,along=False):
	'''
	Vandermonde matrix of data matrix of unique polynomial terms up to order of shape (n,q), where for d = len(dims) and o = 1-ones
	If unique and along - q = choose(p+order-d,p) + o
	If unique and not along - q = choose(p+order,p) - o
	If not unique and along - q = (p^(order)-1)/(p-1) - o
	If not unique not along - q = q = (p^(order+1)-1)/(p-1) - o
	
	Args:
		X (ndarray,sparse_matrix,sparse_array): data matrix of shape (n,p) for n p-dimensional data points
		order (int): order of powers of data
		ones (bool): Include constant ones column in vandermonde
		unique (bool): Return unique combinations of integers
		dims (int,list): dimensions of column vectors of data X to divide vandermonde by
		along (bool): keep polynomials only with powers along dims
	Returns:
		V (ndarray,sparse_matrix,sparse_array) vandermonde of data of shape (n,q = choose(p+n,p) - ones) (if not along else (n,q = choose(p+n-1,p) + ones))
	'''	

	sparse = issparse(X)

	if not sparse:
		X = asarray(X)


	X = X.reshape((-1,1)) if X.ndim != 2 else X
	n,p = X.shape
	dims = [dims] if isinstance(dims,(int,np.integer)) else dims if isinstance(dims,list) else []

	# Powers of X with shape (p,q = choose(p+n,p))
	I = combinations(p,order,unique=unique).T


	# Reduce powers along dims to safely divide by columns of X along dims
	masks = {}
	for dim in dims:
		masks[dim] = where(I[dim] == 0)

		I[dim] -= 1

		if along:
			I = delete(I,masks[dim],1)
		else:
			I[dim,masks[dim]] += 1

	# Get expanded vandermonde of shape(p,q,n)	
	V = (broadcast(X)**(I)).transpose(1,2,0)


	# Safely divide by columns of X along dims
	for dim in dims:
		if along:
			pass
		else:
			V[dim,masks[dim]] *= invert(X[:,dim],constant=0.0)
			I[dim,masks[dim]] -= 1
		I[dim] += 1

	# Get product for polynomial V of shape (n,q)
	V = prod(V,axis=0).transpose(1,0)


	if not ones:
		mask = ~(I==0).all(0)
		V = V[:,mask]

	return V


def linearly_independent(x,y,order,size,basis,adjacency=None,weights=None,dimension=None,ones=False,unique=True,tol=None,atol=None,rtol=None,kind='mergesort',diagonal=False,verbose=False,return_differences=False):
	'''
	Get indices of p-dimensional points in y that are linearly independent, in terms of the multi-dimensional basis and closest to points in x. Computes differences between points, as per adjacency, and finds set of nearest neighbouring points that are independent.
	
	Args:
		x (ndarray): data array of shape (n,p)
		y (ndarray): data array of shape (m,p)
		order (int,iterable): order of basis
		size (int,iterable): number of basis constraints
		basis (callable): function to yield basis, or string in ['vandermonde']
		adjacency (array,sparse_matrix): adjacency matrix of shape (n,m) of which data points are adjacent
		weights (array,sparse_matrix): weights matrix of shape (n,m) of pairwise weights of data points
		dimension (int): dimension direction of basis, between 0 ... p-1
		ones (bool): Include constant ones column in basis
		unique (bool): Include unique basis terms q = choose(p+n,n) else q = (p^(n+1)-1)/(p-1)		
		tol (float): Tolerance for rank and singularity of basis
		atol (float): Absolute tolerance of difference between unique elements
		rtol (float): Relative tolerance of difference between unique elements		
		kind (str): Sort algorithm
		diagonal (bool): explicitly include diagonal of adjacency
		verbose (int): Print out linear dependent findings
		return_differences (bool): return differences of data points
	Returns:
		indices (list): list of arrays of indices of linearly independent points of length n
		z (ndarray,sparse_array): array of differences in data points of shape (n,m,p)
	'''

	# Get basis
	default = 'vandermonde'
	bases = {'vandermonde':vandermonde}
	if isinstance(basis,str):
		basis = bases.get(basis,bases.get(default))
	assert callable(basis), "Error - basis function is not callable"

	# Reshape data to ensure p>0 dimensional data
	isdimension = dimension is not None

	if x.ndim < 1:
		x = x.reshape((-1,1))
	if y.ndim < 1:
		y = y.reshape((-1,1))

	# Get shape of data
	ndim,dtype = x.ndim,x.dtype
	n,p = x.shape
	m,d = y.shape
	shape = (n,m)

	assert ndim == 2, "Error - data is not of shape (n,p)"
	assert (p==d), "Error - x,y not of shape shape (n/m,p)"


	# Get sparsity
	exists = adjacency is not None
	sparse = issparse(adjacency)
	distances = weights is not None
	

	# Include diagonal of adjacency
	if diagonal and exists:		
		adjacency = setdiag(adjacency,diagonal)		

	# Get adjacency properties
	format = adjacency.getformat() if sparse else 'array'

	# Get differences in data of shape (n,m,p)
	z = subtract(x,y,where=adjacency)	

	if not distances:

		# Get radius of differences of shape (n,m)
		fmt = 'csr'
		r = norm(z,ord=2,axis=-1,where=adjacency)
		r = getattr(r,'to%s'%(fmt))().asformat(format) if sparse else r
	else:
		r = np.abs(weights)


	# Include explicit zero differences along diagonal
	if diagonal:
		indices = [adjacency[i].indices for i in range(n)]
		r = explicit_zeros(setdiag(r,0),indices,format=format)
	elif sparse:
		r.eliminate_zeros()

	# Setup system of constraints for basis
	order = repeat(order,n) if isinstance(order,(int,np.integer)) else order	
	size = repeat(size,n) if isinstance(size,(int,np.integer)) else size	

	# Indices output
	indices = [None for i in range(n)]

	with Bar("Linear %d independent points..."%(size[0]),max=n) as bar:

		# Iterate through data points and get q points to form stencil
		for i in range(n):

			# logger.log(verbose,"Point %d"%(i))
			bar.next()

			# Get indices of adjacent data points
			if sparse:
				indices[i] = r[i].indices			
				indices[i] = indices[i][r[i].data.argsort(kind=kind)]
			elif exists:
				indices[i] = arange(m)[r[i].astype(bool)]
				indices[i] = indices[i][r[i][indices[i]].argsort(kind=kind)]
			else:
				indices[i] = arange(m)
				indices[i] = indices[i][r[i].argsort(kind=kind)]

			# Get size of neighbourhood, order, number of constraints, and number of removed data points
			s = indices[i].size
			o = order[i]
			q = size[i]
			k = 0

			# Iterate through neighbourhood data points
			v = None
			for j in range(s):
				
				# Check if neighbourhood size equals number of constraints
				if (j-k+1) > q:
					indices[i] = indices[i][:q]
					break

				# Check if difference in data points is zero along dimension
				if isdimension and (z[i,indices[i][j-k],dimension] == 0):
					indices[i] = delete(indices[i],j-k)
					k += 1
					continue


				# Check if basis of constraints if full rank
				inds = indices[i][:j+1-k]
				u = subarray(z[i],indices=inds,format=format)
				u = u[inds].todense() if sparse else u[inds]
				v = basis(u,order=o,ones=ones,dims=dimension,unique=unique)
				if not isfullrank(v,tol=tol):
					indices[i] = delete(indices[i],j-k)
					k += 1
					continue

			# Assert basis of constraints is full rank
			assert v is not None, "Error - less data points than constraints"
			assert isfullrank(v,tol=tol), "Error - basis for point %d, with shape %r is singular: %r\n%r\n%r"%(i,v.T.shape,rank(v,tol=tol),v.T,indices[i])

	# Returns
	returns = ()

	returns += (indices,)

	if return_differences:
		returns += (z,)

	return returnargs(returns)



def stencils(x,order,size,basis,adjacency=None,weights=None,dimension=None,tol=None,atol=None,rtol=None,kind='mergesort',verbose=False):
	'''
	Get weights of stencil of order accurate derivative along dimension for n p-dimensional data points
	Finds reduced weights a of shape (n,n) such that the derivatives at x0 are:
	df(x)/dx^{dimension}(x0) = sum_{adjacency(x0)}((f(x)-f(x0)/(x^{dimension}-x0^{dimension})*a(x0,x))
	The stencil weights of shape (n,n) are then computed as w(x0,x) = adjacency(x0).size*a(x0,x)/(x^{dimension}-x0^{dimension})^2
	The weights are found by computing the multi-dimensional basis of the data v, and for order accurate stencil and {d} points,
	this matrix has q=choose(p+order,p)-1 terms and shape ( d,q ), and so d=q points must be chosen such that the terms are linearly dependent and
	v is full rank.
	
	Args:
		x (ndarray): data array of shape (n,p)
		order (int,iterable): order of accuracy of stencil
		size (int,iterable): number of constraints of stencil
		basis (str,callable): function to yield basis, or string in ['vandermonde']
		adjacency (array,sparse_matrix): adjacency matrix of shape (n,n) of which data points are used in stencil
		weights (array,sparse_matrix): weights matrix of shape (n,n) of weights of data points used in stencil
		dimension (int): dimension of derivative, between 0 ... p-1
		tol (float): Tolerance of rank and singularity of basis
		atol (float): Absolute tolerance of difference between unique elements
		rtol (float): Relative tolerance of difference between unique elements		
		kind (str): Sort algorithm
		verbose (bool,int): Print out details of stencils							 
	Returns:
		out (ndarray,sparse_matrix): weights of stencil of shape (n,n) 
	'''

	from .estimator import OLS

	# Get basis
	default = 'vandermonde'
	bases = {'vandermonde':vandermonde}
	name = basis
	if isinstance(basis,str):
		basis = bases.get(basis,bases.get(default))
	assert callable(basis), "Error - basis function is not callable"

	# Get sparsity
	n,p = x.shape
	shape = (n,n)
	dtype = x.dtype
	sparse = issparse(adjacency)
	format = adjacency.getformat() if sparse else 'array'

	# Setup system of constraints for basis
	order = repeat(order,n) if isinstance(order,(int,np.integer)) else order			
	size = repeat(size,n) if isinstance(size,(int,np.integer)) else size			
	solver = 'lstsq' if (order!=n).any() else 'lstsq'
	estimator = OLS(solver=solver)
	unique = True


	# Get linearly independent points and differences in points
	indices,z = linearly_independent(x,x,order,size,basis,adjacency=adjacency,weights=weights,dimension=dimension,ones=False,
								   tol=tol,atol=atol,rtol=rtol,kind=kind,diagonal=False,verbose=verbose,return_differences=True)
	

	# Get number of nnz indices in each row
	sizes = np.array([i.size for i in indices])

	# Setup output data,rows,cols
	fmt = 'coo'
	data,rows,cols = [],[],[]


	# Iterate through data points and get q points to form stencil
	for i in range(n):

		# Set up linear system of constraints with chosen stencil of points
		u = subarray(z[i],indices=indices[i],format=format)
		u = u[indices[i]].todense() if sparse else u[indices[i]]
		v = basis(u,order=order[i],ones=False,dims=dimension,unique=unique)
		e = zeros(sizes[i],dtype=dtype)

		e[dimension] = 1

		estimator.fit(v.T,e)
		coef = estimator.get_coef_() + 0.0

		data.extend(coef)
		rows.extend(i*ones(indices[i].size,dtype=int))
		cols.extend(indices[i])


	# Convert data,rows,cols to array
	data,rows,cols = np.array(data),np.array(rows),np.array(cols)

	# Eliminate zeros within tolerance
	data[isclose(data,atol=atol,rtol=rtol)] = 0

	# Get output array
	out = getattr(sp.sparse,'%s_matrix'%(fmt))((data,(rows,cols)),shape=shape,dtype=dtype)

	# Eliminate zeros
	out.eliminate_zeros()

	# Get number of nnz indices in each row
	sizes = out.getnnz(-1)

	if sparse:
		fmt = 'csr'
		out = out.asformat(fmt)
		u = getattr(z[...,dimension],'to%s'%(fmt))()
		out = multiply(invert(u.power(2),constant=0.0),out,sizes)
		out = out.asformat(format)
	else:
		out = out.A
		u = z[...,dimension]
		out = multiply(invert(u**2,constant=0.0),out,sizes)





	return out


# Grammian of array
def gram(X,*args,**kwargs):
	return X.T.dot(X)

# Projection of vector onto array
def project(X,y,*args,**kwargs):
	return X.T.dot(y)

# Array rank
def rank(arr,tol=None,**kwargs):
	try:
		return np.linalg.matrix_rank(arr,tol=tol)
	except:
		return 0


# Round array
def round(arr,decimals):
	sparse = issparse(arr)
	if sparse:
		sparsematrix = issparsematrix(arr)
		sparsearray = issparsearray(arr)
		if sparsematrix:
			out = arr.copy()
			out.data[:] = np.round(asarray(out.data),decimals)
		elif sparsearray:
			out = arr.round(decimals)
	elif isndarray(arr):
		out = arr.round(decimals)
	else:
		try:
			out = arr.round(decimals)
		except:
			out = np.round(arr,decimals)
	out += 0.0
	return out

# Array condition number
def cond(arr,ord=None,**kwargs):
	try:
		return np.linalg.cond(arr,p=ord)
	except np.linalg.LinAlgError:
		return 0


def getdiag(arr,offset=0,axis1=0,axis2=1):
	'''
	Get offset diagonal of array along axis1 and axis2
	
	Args:
		arr (array): array to extract diagonal
		offset (int): offset from diagonal to extract
		axis1 (int): first axis to extract diagonal
		axis2 (int): second axis to extract diagonal
	Returns:
		out (array): array with axis1 and axis2 condensed to only diagonal elements along last axis
	'''

	out = np.diagonal(arr,offset=offset,axis1=axis1,axis2=axis2)

	return out

def setdiag(arr,value=None,k=0):
	'''
	Set k-diagonal of array with value, including explicit zeros
	
	Args:
		arr (array,sparse_matrix): array to add diagonal elements to
		value (object): value to add to array
		k (int): index of diagonal from center
	Returns:
		out (array,sparse_matrix): array with value at k-diagonal
	'''

	# Get sparsity
	shape,dtype = arr.shape,arr.dtype
	n = min(shape)
	sparse = issparse(arr)
	format = arr.getformat() if sparse else 'array'

	# Get value
	if value is None:
		value = (arr.diagonal(k=k) if sparse else getdiag(arr,offset=k))
	scalar = isscalar(value)

	# Get if value is zero for explicit zeros
	zero = (scalar and value==0) or ((not scalar) and (value==0).any())

	# Set value as array
	value = value*ones(n,dtype=type(value)) if scalar else value

	if sparse:
		if not zero:
			fmt = 'lil'
			out = arr.asformat(fmt)
			out.setdiag(value)
		else:
			fmt = 'coo'
			out = arr.asformat(fmt)
			data,row,col = out.data,out.row,out.col
			if k < 0:
				data,row,col = np.array([*data,*value[:n+k]]),np.array([*row,*arange(-k,n)]),np.array([*col,*arange(0,n+k)])
			else:
				data,row,col = np.array([*data,*value[:n-k]]),np.array([*row,*arange(0,n-k)]),np.array([*col,*arange(k,n)])
			out = getattr(sp.sparse,'%s_matrix'%(fmt))((data,(row,col)),shape=shape,dtype=dtype)
		out = out.asformat(format)
	else:
		out = adjacency.copy()
		np.fill_diagonal(adjacency,diagonal or getdiag(adjacency))

	return out


# Get if array is singular
def issingular(arr,tol=None,**kwargs):
	ndim = arr.ndim
	assert ndim==2, "Error - array is not 2 dimensional"
	n,m = arr.shape
	r = rank(arr,tol=tol,**kwargs)
	return not ((n==m) and (r==n))

# Get if array is full rank
def isfullrank(arr,tol=None,**kwargs):
	ndim = arr.ndim
	assert ndim==2, "Error - array is not 2 dimensional"
	n = min(*arr.shape)
	r = rank(arr,tol=tol,**kwargs)
	return  (r==n)




# Get if array is sparse
def issparse(arr,*args,**kwargs):
	return issparsematrix(arr) or issparsearray(arr)

# Get if array is sparse matrix
def issparsematrix(arr,*args,**kwargs):
	return sp.sparse.issparse(arr)

# Get if array is sparse array
def issparsearray(arr,*args,**kwargs):
	return isinstance(arr,sparray.SparseArray)

# Get if array is numpy array
def isndarray(arr,*args,**kwargs):
	return isinstance(arr,(np.ndarray))

# Get if array is pandas dataframe
def isdataframe(arr,*args,**kwargs):
	return isinstance(arr,(pd.DataFrame))

# Get if array is array
def isarray(arr,*args,**kwargs):
	return isndarray(arr) or issparse(arr)

# Get if array is scalar
def isscalar(arr,*args,**kwargs):
	return (not isarray(arr) and not islisttuple(arr)) or (isarray(arr) and (arr.ndim<1) and (arr.size<2))

# Get if array is None
def isnone(arr,*args,**kwargs):
	return arr is None

# Get if array is python list
def islist(arr,*args,**kwargs):
	return isinstance(arr,(list))

# Get if array is python tuple
def istuple(arr,*args,**kwargs):
	return isinstance(arr,(tuple))

# Get if array is python list,tuple
def islisttuple(arr,*args,**kwargs):
	return islist(arr) or istuple(arr)

# Random seed
def seed(seed):
	if seed is None or seed is np.random:
		return np.random.mtrand._rand
	if isinstance(seed, int):
		return np.random.RandomState(seed)
	if isinstance(seed, np.random.RandomState):
		return seed
	raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
					 ' instance' % seed)


# Shuffle array along axis
def shuffle(arr,axis,inplace=True):
	axes = [axis] if isinstance(axis,(int,np.integer)) else axis
	for axis in axis:
		n = arr.shape[axis]
		indices = np.random.permutation(n)
		if inplace:
			arr[:] = take(arr,indices,axis)
		else:
			arr = take(arr,indices,axis)
	if inplace:
		return
	else:
		return arr


def checksum(objs,attrs,alls=None):
	'''
	Check attributes of objects are equal
	
	Args:
		objs (iterable): objects to check
		attrs (str,iterable): string attributes to test
	Returns:
		checksum (bool): boolean of all pairwise checks of attributes
	'''

	attrs = [attrs] if isinstance(attrs,str) else attrs
	checksum = all([all([getmethod(getattr(a,attr,a)==getattr(b,attr,b),'all')() for attr in attrs]) 
			for a in objs for b in objs])
	return checksum


# Soft threshold function for a <,=,> b
def soft_threshold(a,b):
	if a < - b:
		return (a + b)
	elif a >  b:
		return (a - b)
	else: 
		return 0


# Get norm of array along axis
def norm(arr,axis,ord,*args,where=True,**kwargs):
	def func(arr,axis=-1,dtype=None,where=True,ord=2):
		out = np.linalg.norm(arr,axis=axis,ord=ord)
		return out
	try:
		out = reduce(func,arr,axis,dtype=None,where=where,ord=ord)
	except ValueError:		
		shape = list(arr.shape)
		shape.pop(axis)
		out = ones(shape)		
	return out

def set_norm(norm_func,*args,**kwargs):
	default = 'l2'
	field = 'norm'
	norm_func = str(norm_func)
	globs = globals()
	func = globs.get('_'.join([field,norm_func]),globs['_'.join([field,default])])
	return wrapper(func,*args,**kwargs)

def norm_norm(arr,axis,ord,*args,**kwargs):
	ords = {'l1':1,'l2':2,'linf':np.inf,'uniform':'uniform'}
	ord = ords.get(ord,ord)
	return norm(arr,axis,ord,*args,**kwargs)

def norm_l2(arr,*args,**kwargs):
	ord = kwargs.pop('ord','l2')
	axis = kwargs.pop('axis',0)
	return norm_norm(arr,axis,ord,*args,**kwargs)

def norm_l1(arr,*args,**kwargs):
	ord = kwargs.pop('ord','l1')
	axis = kwargs.pop('axis',0)
	return norm_norm(arr,axis,ord,*args,**kwargs)

def norm_linf(arr,*args,**kwargs):
	ord = kwargs.pop('ord','linf')
	axis = kwargs.pop('axis',0)
	return norm_norm(arr,axis,ord,*args,**kwargs)

def norm_uniform(arr,*args,**kwargs):
	ord = kwargs.pop('ord','uniform')
	axis = kwargs.pop('axis',0)
	return norm_norm(arr,axis,ord,*args,**kwargs)

def norm_None(arr,*args,**kwargs):
	ord = kwargs.pop('ord','uniform')
	axis = kwargs.pop('axis',0)
	return norm_norm(arr,axis,ord,*args,**kwargs)


# Set array scale
def scale(arr,axis,norm_func,*args,**kwargs):

	def func(arr,axis,ord): 
		return invert(norm(arr,axis=axis,ord=ord),constant=1.0)

	ords = {'l1':1,'l2':2,'linf':np.inf}
	ord = ords.get(norm_func,norm_func)

	ndim,shape = arr.ndim,list(arr.shape)
	dims = shape.pop(-1)
	shape.pop(axis)

	if ndim < 2:
		out = func(arr,axis,ord)
		arr[...] *= out
	else:
		out = zeros((dims,*shape))		
		for dim in range(dims):
			out[dim] = func(arr[...,dim],axis,ord)
			arr[...,dim] *= out[dim]

	return out


def set_scale(norm_func,*args,**kwargs):
	return wrapper(scale,*args,norm_func=norm_func,**kwargs)



# Set array loss
def loss(y_pred,y,loss_func,*args,**kwargs):
	field = 'loss'
	default = 'rmse'
	loss_func = str(loss_func)
	globs = globals()
	func = globs.get('_'.join([field,loss_func]),globs['_'.join([field,default])])
	return func(y_pred,y,*args,**kwargs)


def set_loss(loss_func,*args,**kwargs):
	field = 'loss'
	default = 'rmse'
	loss_func = str(loss_func)
	globs = globals()
	func = globs.get('_'.join([field,loss_func]),globs['_'.join([field,default])])
	return wrapper(func,*args,**kwargs)

def loss_norm(y_pred,y,axis,ord,*args,**kwargs):
	field = 'loss'
	ords = {'l1':1,'l2':2,'linf':np.inf,'uniform':np.inf}
	ord = ords.get(ord,ord)
	try:
		inds = kwargs['%s_%s_inds'%(field,str(ord))]
		inds = inds[inds<y.shape[axis]]
	except:
		inds = slice(None)
	try:
		scale = kwargs['%s_%s_scale'%(field,str(ord))][inds]
	except:
		scale = 1
	try:
		arr = ((y_pred-y)[inds])*scale
	except:
		inds = slice(None)
		arr = ((y_pred-y)[inds])*scale
	try:
		C = (arr.shape[axis])**(-1/ord)
	except:
		C = 1
	return C*norm_norm(arr,axis=axis,ord=ord)

def loss_l2(y_pred,y,*args,**kwargs):
	ord = kwargs.pop('ord','l2')
	axis = kwargs.pop('axis',0)
	return loss_norm(y_pred,y,axis,ord,*args,**kwargs)

def loss_l1(y_pred,y,*args,**kwargs):
	ord = kwargs.pop('ord','l1')
	axis = kwargs.pop('axis',0)
	return loss_norm(y_pred,y,axis,ord,*args,**kwargs)

def loss_linf(y_pred,y,*args,**kwargs):
	ord = kwargs.pop('ord','linf')
	axis = kwargs.pop('axis',0)
	return loss_norm(y_pred,y,axis,ord,*args,**kwargs)

def loss_uniform(y_pred,y,*args,**kwargs):
	ord = kwargs.pop('ord','uniform')
	axis = kwargs.pop('axis',0)

	return loss_norm(y_pred,y,axis,ord,*args,**kwargs)

def loss_None(y_pred,y,*args,**kwargs):
	ord = kwargs.pop('ord','uniform')
	axis = kwargs.pop('axis',0)
	return loss_norm(y_pred,y,axis,ord,*args,**kwargs)


def loss_rmse(y_pred,y,*args,**kwargs):
	return loss_l2(y_pred,y,*args,**kwargs)

def loss_weighted(y_pred,y,*args,**kwargs):
	try:
		weights = kwargs['loss_weights']
	except:
		weights = {'l2':1}
	axis = kwargs.pop('axis',0)
	return sum([weights[ord]*loss_norm(y_pred,y,axis,ord,*args,**kwargs)
				for ord in weights])

def score(y_pred,y,score_func,*args,**kwargs):
	field = 'score'
	default = 'rmse'
	score_func = str(score_func)
	globs = globals()
	func = globs.get('_'.join([field,score_func]),globs['_'.join([field,default])])
	return func(y_pred,y,*args,**kwargs)

def set_score(score_func,*args,**kwargs):
	field = 'score'
	default = 'rmse'
	score_func = str(score_func)
	globs = globals()
	func = globs.get('_'.join([field,score_func]),globs['_'.join([field,default])])
	return wrapper(func,*args,**kwargs)

def score_norm(y_pred,y,axis,ord,*args,**kwargs):
	field = 'score'
	ords = {'l1':1,'l2':2,'linf':np.inf,'uniform':np.inf}
	ord = ords.get(ord,ord)
	try:
		inds = kwargs['%s_%s_inds'%(field,str(ord))]
	except:
		inds = slice(None)
	try:
		scale = kwargs['%s_%s_scale'%(field,str(ord))][inds]
	except:
		scale = 1	
	try:
		arr = ((y_pred-y)[inds])*scale
	except:
		inds = slice(None)
		arr = ((y_pred-y)[inds])*scale
	try:
		C = (arr.shape[axis])**(-1/ord)
	except:
		C = 1
	return -C*norm_norm(arr,axis=axis,ord=ord)

def score_l2(y_pred,y,*args,**kwargs):
	ord = kwargs.pop('ord','l2')
	axis = kwargs.pop('axis',0)
	return score_norm(y_pred,y,axis,ord,*args,**kwargs)

def score_l1(y_pred,y,*args,**kwargs):
	ord = kwargs.pop('ord','l1')
	axis = kwargs.pop('axis',0)
	return score_norm(y_pred,y,axis,ord,*args,**kwargs)

def score_linf(y_pred,y,*args,**kwargs):
	ord = kwargs.pop('ord','linf')
	axis = kwargs.pop('axis',0)
	return score_norm(y_pred,y,axis,ord,*args,**kwargs)

def score_uniform(y_pred,y,*args,**kwargs):
	ord = kwargs.pop('ord','uniform')
	axis = kwargs.pop('axis',0)
	return score_norm(y_pred,y,axis,ord,*args,**kwargs)

def score_rmse(y_pred,y,*args,**kwargs):
	return score_l2(y_pred,y,*args,**kwargs)

def score_weighted(y_pred,y,*args,**kwargs):
	try:
		weights = kwargs['score_weights']
	except:
		weights = {'l2':1}
	axis = kwargs.pop('axis',0)
	return sum([weights[ord]*score_norm(y_pred,y,axis,ord,*args,**kwargs)
				for ord in weights])


def score_r2(y_pred,y,*args,**kwargs):
	n = max(1,y.shape[0])
	axis = kwargs.pop('axis',0)
	ord = 'l2'
	return (norm_norm(y_pred-y,axis,ord,*args,**kwargs)*
			invert(norm_norm(y-y.mean(axis=axis),axis,ord,*args,**kwargs),constant=1.0)*
			((n-1)/(n-0)))



def criteria(loss,complexity_,losses,complexities_,criteria_func,*args,**kwargs):
	field = 'criteria'
	default = 'F_test'
	score_func = str(criteria_func)
	globs = globals()
	func = globs.get('_'.join([field,criteria_func]),globs['_'.join([field,default])])
	return func(loss,complexity_,losses,complexities_,*args,**kwargs)


def set_criteria(criteria_func,*args,**kwargs):
	field = 'criteria'
	default = 'F_test'
	criteria_func = str(criteria_func)
	globs = globals()
	func = globs.get('_'.join([field,criteria_func]),globs['_'.join([field,default])])
	return wrapper(func,*args,**kwargs)

def criteria_F_test(loss,complexity_,losses,complexities_,*args,**kwargs):
	try:
		return ((loss-losses[-1])*(complexities_[0]-complexities_[-1]))*(
				invert(((losses[-1])*(complexities_[-1]-complexity_)),constant=0.0))
	except Exception as e:
		return zeros(np.shape(loss))







# Check equality along axis
def equal_axis(a,axis,axis_axis=False):
	assert a.ndim == 2, "Multidimensional array: ndim > 2"
	if axis == 0:
		return np.all(a==take(a,0,axis),axis=axis if axis_axis else None)
	elif axis == 1:
		return np.all((a.T==take(a,0,axis)).T,axis=axis if axis_axis else None)
	return False


# Conjunction of logical conditions
def conjunction(*conditions):
	if conditions not in [[]]:
		return functools.reduce(np.logical_and, conditions)
	else:
		return None

# Disjunction of logical conditions
def disjunction(*conditions):
	if conditions not in [[]]:
		return functools.reduce(np.logical_or, conditions)
	return None




# Safely convert to constants
def convert(x,constant=1):
	isx = (np.isnan(x) | np.isinf(x) | (x==0))
	if isinstance(x,np.ndarray):
		x[isx] = constant
	else:
		x = constant if isx else x
	return x


# Convert between iterable types
def toiter(a,iterable):
	try:
		return iterable((toiter(i,iterable) for i in a))
	except:
		return a

# Make set of same iterable type
def toset(a):
	return toiter(set(toiter(a,tuple)),type(a))


# Check if iterable
def isiterable(obj):
	field = '__iter__'
	try:
		return hasattr(obj,field)
	except:
		return False


# Convert Series of lists to Array
def series_array(series,**kwargs):
	try:
		arr = np.stack(series.values,**kwargs)
	except ValueError:
		arr = [x for x in series.values]
	return arr

# Convert Array to Series of lists
def array_series(arr,**kwargs):
	series = pd.Series(arr.tolist(),**kwargs)
	return series


# Index array with list of lists indices
def masked_index(arr,indices,fill_value=-1):
	mask = np.array([len(index) for index in indices])
	mask = mask[:,None] <= arange(mask.max())    
	inds = np.full(mask.shape,fill_value,dtype=int)
	inds[~mask] = concatenate(indices)    
	out = np.ma.masked_array(arr[arange(arr.shape[0])[:,None],inds],mask,fill_value=0)
	return out


# Alias of keywords
def alias(kwds,aliases,**kwargs):
	kwds.update({aliases.get(k,k): (kwds[aliases.get(k,k)](kwargs[k]) if callable(kwds[aliases.get(k,k)]) else kwargs[k])
					for k in kwargs if ((aliases.get(k,k) in kwds) and 
					((k == aliases.get(k,k)) or (aliases.get(k,k) not in kwargs)))})
	kwds.update({k: kwds[k]() for k in kwds if callable(kwds[k])})
	return


# Check if callable
def iscallable(obj,*args,**kwargs):
	if callable(obj):
		return obj(*args,**kwargs)
	else:
		return obj

# Factors of integers
def factors(n):
	return sorted(set((f for i in range(1, int(n**0.5)+1) 
					if n % i == 0 
					for f in [i, n//i])))

# Convert between units
def units(value,bases):
	values = [value]
	for base in bases:
		values.extend(divmod(values[-1],base))



def isnumber(s):
	'''
	Check if object is a float or integer number
	
	Args:
		s(object): Object to be checked as number
	Returns:
		Boolean of whether object s is a number
	'''
	try:
		s = float(s)
		return True
	except:
		try:
			s = int(s)
			return True
		except:
			return False




def scinotation(number,decimals=2,base=10,order=2,zero=True,scilimits=[-1,1],usetex=True):
	'''
	Put number into scientific notation string
	
	Args:
		number (str,int,float): Number to be processed
		decimals (int): Number of decimals in base part of number
		base (int): Base of scientific notation
		order (int): Max power of number allowed for rounding
		zero (bool): Make numbers that equal 0 be the int representation
		scilimits (list): Limits on where not to represent with scientific notation
		usetex (bool): Render string with Latex
	Returns:
		String with scientific notation format for number
	'''
	if not isnumber(number):
		return str(number)
	try:
		number = int(number) if int(number) == float(number) else float(number)
	except:
		string = number
		return string

	maxnumber = base**order
	if number > maxnumber:
		number = number/maxnumber
		if int(number) == number:
			number = int(number)
		string = str(number)
	
	if zero and number == 0:
		string = '%d'%(number)
	
	elif isinstance(number,(int,np.integer)):
		string = str(number)
		# if usetex:
		# 	string = r'\textrm{%s}'%(string)
	
	elif isinstance(number,(float,np.float64)):		
		string = '%0.*e'%(decimals,number)
		string = string.split('e')
		basechange = np.log(10)/np.log(base)
		basechange = int(basechange) if int(basechange) == basechange else basechange
		flt = string[0]
		exp = str(int(string[1])*basechange)
		if int(exp) in range(*scilimits):
			flt = '%0.*f'%(decimals,float(flt)/(base**(-int(exp))))
			string = r'%s'%(flt)
		else:
			string = r'%s%s'%(flt,r'\cdot %d^{%s}'%(base,exp) if exp!= '0' else '')
	if usetex:
		string = r'%s'%(string.replace('$',''))
	else:
		string = string.replace('$','')
	return string

# Generator wrapper to restart stop number of times
def generator(stop=None):
	def wrap(func):
		def set(*args,**kwargs):
			return func(*args,*kwargs)
		@functools.wraps(func)
		def wrapper(*args,stop=stop,**kwargs):
			generator = set(*args,**kwargs)
			while stop:
				try:
					yield next(generator)
				except StopIteration:
					stop -= 1
					generator = set(*args,**kwargs)
					yield next(generator)
		return wrapper
	return wrap


# List from generator
def list_from_generator(generator,field=None):
	item = next(generator)
	if field is not None:
		item = item[field]    
	items = [item]
	for item in generator:
		if field is not None:
			item = item[field]
		if item == items[0]:
			break
		items.append(item)

	# Reset iterator state:
	for item in generator:
		if item == items[-1]:
			break
	return items


# Sliced indices
def slicing(arr,n,base,dim,offset=0):
	arr = arr.reshape([n]*dim)
	for d in range(dim):
		arr = take(arr,arange(offset,n,base),axis=d)
	arr = arr.reshape((-1))
	return arr

# Mesh refinement indices
def refine(n,dim,base,power):
	indices = arange(n**dim)
	slices = []
	refine = base**power
	for j in range(base):
		slices.append(slicing(indices,n,refine,dim,int(j*refine/base)))
	return slices


# Mesh refinement
def refinement(n,dim,base,powers=None,string='slice',boundary=False):
	# Given powers exponents, will refine dim-dimensional mesh 
	# by base^powers
	# powers is either None, list [power_min,power_max,power_step] or
	#	   numpy array of powers
	# i.e) L = nh, N = int(log_base(n))
	#	   n -> n/base^p for p in powers = [2,3,...,N]
	# 	   h -> h*base^p for p in powers = [2,3,...,N]
	n += (boundary*(1-n%base))

	N = int(np.log(n)/np.log(base))

	if powers is None or isinstance(powers,list):
		powers = [None,None,None] if powers is None else powers
		powers = [(powers[i] if powers[i]>=0 else N+powers[i]) if i<len(powers) and powers[i] is not None else {0:2,1:N,2:1}[i] 
					for i in range(3)]
		powers = arange(*powers)
	slices = {} 

	for power in powers:
		key = '%s_%d'%(string,power)
		slices[key] = {'%s_%d_%d'%(string,power,j):s for j,s in enumerate(refine(n,dim,base,power))}
	return slices




# Find indices of extrema of array
def extrema(x,**kwargs):

	# Find extremal points of array
	def _extrema(x,method,passes,**kwargs):
		defaults = {'distance':1}
		scales = {'min':-1,'max':1,'default':1}
		scale = scales.get(method,scales['default'])
		kwargs.update({k: kwargs.get(k,defaults[k]) for k in defaults})
		n = x.shape[0]
		inds = arange(n)
		for passer in range(passes):
			_inds = sp.signal.find_peaks(scale*x[inds],**kwargs)[0]
			inds = np.array([inds[(scale*x[inds]).argmax(axis=0)]]) if (len(_inds)==0) else _inds
		return inds


	# Weave maxima and minima of array
	def weaver(a,b,x):
		a = copy.deepcopy(a).tolist()
		b = copy.deepcopy(b).tolist()
		c = None
		d = None
		o = []
		while(len(a)>0 and len(b)>0):
			ab = a[0]<b[0]
			if a[0]==b[0]:
				a0 = a.pop(0)
				b0 = b.pop(0)
				if ab:
					o.append(a0)
				else:
					o.append(b0)
				continue
			elif ab:
				c = a
				d = b
			elif  not ab:
				c = b
				d = a
			extrema = [(i,c[i]) for i in range(len(c)) if c[i]<d[0]]
			extrema = [i for (i,x) in sorted(extrema,key=lambda i: x[i[1]],reverse=(not ab))]
			c0 = c[extrema[0]]
			for i in sorted(extrema,reverse=True):
				c.pop(i);
			if len(c) == 0:
				o.extend([c0,d[0]])
				break
			extrema = [(i,d[i]) for i in range(len(d)) if d[i]<=c[0]]
			extrema = [i for (i,x) in sorted(extrema,key=lambda i: x[i[1]],reverse=ab)]
			d0 = d[extrema[0]]
			for i in sorted(extrema,reverse=True):
				d.pop(i);

			o.extend([c0,d0])

		return o
	defaults = {'passes':1,'distance':1}
	kwargs.update({k: kwargs.get(k,defaults[k]) for k in defaults})
	n = len(x)
	inds_range = arange(n)
	inds_min = _extrema(x,'min',**kwargs)
	inds_max = _extrema(x,'max',**kwargs)
	inds = np.array(weaver(inds_min,inds_max,x))
	inds = endpoints(inds,inds_range)
	return inds


# Append endpoints of x at x_new
def endpoints(x_new,x,y_new=None,y=None):
	isx = len(x)>0
	isy = (y_new is not None) and (y is not None)

	if isx and (x[0] not in x_new):
		x_new = np.array([x[0],*x_new])
		if isy:
			y_new = np.array([y[0],*y_new])
	if isx and (x[-1] not in x_new):
		x_new = np.array([*x_new,x[-1]])
		if isy:
			y_new = np.array([*y_new,y[-1]])
	if isy:
		return x_new,y_new
	else:
		return x_new	




# Filter array by interpolating over average of envelope
def filtering(y,x=None,inds=None,**kwargs):


	# Interpolate y(x) at x_new
	def interpolate(x,y,x_new,**kwargs):
		min_points = 4
		x_min = np.min(x,axis=0)
		x_max = np.max(x,axis=0)
		x_new = x_new[(x_new>=x_min) & (x_new<=x_max)]
		if len(x)<min_points:
			return x,y
		else:
			return x_new,sp.interpolate.interp1d(x, y,**kwargs)(x_new)


	defaults = {'extrema':{'distance':1,'passes':1},'interpolate':{'kind':'cubic'}}
	kwargs.update({k:kwargs.get(k,defaults[k]) for k in defaults})

	inds = extrema(y,**kwargs['extrema']) if inds is None else inds

	n = len(y)
	m = len(inds)
	x = arange(n) if x is None else x

	x_filtered = np.split(x[inds],m//2,axis=-1).mean(axis=-1)
	y_filtered = np.split(y[inds],m//2,axis=-1).mean(axis=-1)
	x_filtered,y_filtered = endpoints(x_filtered,x,y_filtered,y)
	x_filtered,y_filtered = interpolate(x_filtered,y_filtered,x,**kwargs.get('interpolate',{}))
	return y_filtered,x_filtered,inds



# Compute size of basis
def basis_size(inputs,outputs,order,basis,constants,samples,intercept_):
	Nbasis = {
		None: lambda Ninputs,Noutputs,Nconstants,order,intercept_: (Ninputs - Nconstants + intercept_),
		'taylorseries': lambda Ninputs,Noutputs,Nconstants,order,intercept_: ((((Ninputs-Nconstants)**(order+1)-1)/((Ninputs-Nconstants)-1)) if (Ninputs-Nconstants)>1 else 1+order),
		'linear': lambda Ninputs,Noutputs,Nconstants,order,intercept_: (Ninputs - Nconstants + intercept_),
		'polynomial': lambda Ninputs,Noutputs,Nconstants,order,intercept_: ((((order+1)**Ninputs)-(1-intercept_)) if Ninputs>1 else order + intercept_) - Nconstants,
		**{k: (lambda Ninputs,Noutputs,Nconstants,order,intercept_: (Ninputs*(order+1)) - Nconstants + intercept_) 
			for k in ['monomial','chebyshev','legendre','hermite']}
		}
	Nconstant = {
		None: lambda inputs,outputs,constants: len(set([x for y in constants for x in constants[y]])) if (constants is not None and len(constants)>0) else 0,
		'taylorseries': lambda inputs,outputs,constants: max([len([x for x in constants[y] if x in inputs]) for y in constants]) if (constants is not None and len(constants)>0) else 0,
		'polynomial': lambda inputs,outputs,constants: len(set([x for y in constants for x in constants[y]])) if (constants is not None and len(constants)>0) else 0,
		**{k:(lambda inputs,outputs,constants: len(set([x for y in constants for x in constants[y]])) if (constants is not None and len(constants)>0) else 0)
			for k in ['linear','monomial','chebyshev','legendre','hermite']}
		}

	basis = None if basis not in Nconstant or basis not in Nbasis else basis
	order = max(order) if isinstance(order,list) else order


	Ninputs = len(inputs)
	Noutputs = len(outputs)
		
	Nconstants = Nconstant[basis](inputs,outputs,constants)
	Nbasises = int(Nbasis[basis](Ninputs,Noutputs,Nconstants,order,intercept_))

	return Nbasises


def position(site,n,d,dtype=np.int32):
	# Return position coordinates in d-dimensional n^{d} lattice 
	# from given linear site position in 1d N^{d} length array
	# i.e) [int(site/(self.n**(i))) % self.n for i in range(self.{d})]
	n_i = np.power(n,arange(d,dtype=dtype))

	isint = isinstance(site,(int,np.integer))

	if isint:
		site = np.array([site])
	position = np.mod(((site[:,None]/n_i)).astype(dtype),n)
	if isint:
		return position[0]
	else:
		return position

def site(position,n,d,dtype=np.int32):
	# Return linear site position in 1d n^{d} length array 
	# from given position coordinates in {d}-dimensional n^{d} lattice
	# i.e) sum(position[i]*self.n**i for i in range(self.{d}))
	
	n_i = np.power(n,arange(d,dtype=dtype))

	is1d = isinstance(position,(int,np.integer,list,tuple)) or position.ndim < 2

	if is1d:
		position = np.atleast_2d(position)
	
	site = position.dot(n_i).astype(dtype)

	if is1d:
		return site[0]
	else:
		return site






import sys
from types import ModuleType, FunctionType
from gc import get_referents

def getsizeof(obj,units='B'):
    """sum size of object & members."""

	# Custom objects know their class.
	# Function objects seem to know way too much, including modules.
	# Exclude modules as well.
    known = (type,ModuleType,FunctionType)

    if isinstance(obj, known):
        raise TypeError('getsize() does not take argument of type: '+ str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, known) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)

    conversions = {'B':1,'KB':2**10,'MB':2**20,'GB':2**30}
    conversion = conversions.get(units,1)
    size /= conversion
    return size





class lattice(object):
	
	# Define a (Square) Lattice class for lattice sites configurations with:
	# Lattice Length L, Lattice Dimension d
	
	def __init__(self,n=10,d=3):
		# Define parameters of system        
		self.n = n
		self.d = d
		self.N = n**d
		self.z = 2*d
		
		if self.N > 2**32-1:
			self.dtype = np.int64
		else:
			self.dtype=np.int32

		# Prepare arrays for Lattice functions

		# Define array of sites
		self.sites = arange(self.N)
		
		# n^i for i = 1:d array
		self.n_i = np.power(self.n,arange(self.d,dtype=self.dtype))
		
		# Arrays for finding coordinate and linear position in d dimensions
		self.I = np.identity(self.d)
		self.R = arange(1,np.ceil(self.n/2),dtype=self.dtype)

		
		# Calculate array of arrays of r-distance neighbour sites,
		# for each site, for r = 1 : n/2 
		# i.e) self.neighbour_sites = np.array([[self.neighboursites(i,r) 
		#                                 for i in range(self.N)]
		#                                 for r in range(1,
		#                                             int(np.ceil(self.n/2)))])
		self.neighbours = self.neighbour_sites()


		
	def position(self,site):
		# Return position coordinates in d-dimensional n^{d} lattice 
		# from given linear site position in 1d N^{d} length array
		# i.e) [int(site/(self.n**(i))) % self.n for i in range(self.{d})]
		isint = isinstance(site,(int,np.integer))

		if isint:
			site = np.array([site])
		position = np.mod(((site[:,None]/self.n_i)).
						astype(self.dtype),self.n)
		if isint:
			return position[0]
		else:
			return position
	
	def site(self,position):
		# Return linear site position in 1d N^{d} length array 
		# from given position coordinates in {d}-dimensional n^{d} lattice
		# i.e) sum(position[i]*self.n**i for i in range(self.{d}))
		is1d = isinstance(position,(list,tuple)) or position.ndim < 2

		if is1d:
			position = np.atleast_2d(position)
		
		site = position.dot(self.n_i).astype(self.dtype)

		if is1d:
			return site[0]
		else:
			return site
	
	def neighbour_sites(self,r=None,sites=None):
		# Return array of neighbour spin sites 
		# for a given site and r-distance neighbours
		# i.e) np.array([self.site(np.put(self.position(site),i,
		#                 lambda x: np.mod(x + p*r,self.n))) 
		#                 for i in range(self.d)for p in [1,-1]]) 
		#                 ( previous method Time-intensive for large n)
		
		if sites is None:
			sites = self.sites
		
		sitepos = self.position(sites)[:,None]
		
		if r is None:
			Rrange = self.R
		elif isinstance(r,list):
			Rrange = r
		else:
			Rrange = [r]
		return np.array([np.concatenate(
							(self.site(np.mod(sitepos+R*self.I,self.n)),
							 self.site(np.mod(sitepos-R*self.I,self.n))),1)
								for R in Rrange])                     

		
	def neighbour_states(self,r=1):
		# Return spins of r-distance neighbours for all spin sites
		return np.array([np.index(self.sites,self.neighbour_sites[r-1][i]) 
									for i in range(len(self.sites))])



