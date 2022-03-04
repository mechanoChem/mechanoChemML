 #!/usr/bin/env python

# Import python modules
import os,sys,copy,warnings,itertools

warnings.simplefilter("ignore", (UserWarning,DeprecationWarning,FutureWarning))


def _copier(key,value,_copy):
	'''
	Copy value based on associated key 

	Args:
		key (string): key associated with value to be copied
		value (python object): data to be copied
		_copy (bool,dict,None): boolean or None whether to copy value, or dictionary with keys on whether to copy value
	Returns:
		Copy of value
	'''

	# Check if _copy is a dictionary and key is in _copy and is True to copy value
	if ((not _copy) or (isinstance(_copy,dict) and (not _copy.get(key)))):
		return value
	else:
		return copy.deepcopy(value)



def _clone(iterable,twin,_copy=False):
	'''
	Shallow in-place copy of iterable to twin

	Args:
		iterable (dict): dictionary to be copied
		twin (dict): dictionary to be modified in-place with copy of iterable
		_copy (bool,dict,None): boolean or None whether to copy value, or dictionary with keys on whether to copy value
	'''	

	# Iterate through iterable and copy values in-place to twin dictionary
	for key in iterable:
		if isinstance(iterable[key],dict):
			if twin.get(key) is None:
				twin[key] = {}
			_clone(iterable[key],twin[key],_copy)
		else:
			twin[key] = _copier(key,iterable[key],_copy)
	return




def _set(iterable,elements,value,_split=False,_copy=False,_reset=True):
	'''
	Set nested value in iterable with nested elements keys

	Args:
		iterable (dict): dictionary to be set in-place with value
		elements (str,list): DELIMITER separated string or list to nested keys of location to set value
		value (python object): data to be set in iterable
		_split (bool,str,None): boolean or None or delimiter on whether to split string elements into list of nested keys
		_copy (bool,dict,None): boolean or None whether to copy value, or dictionary with keys on whether to copy value
		_reset (bool): boolean on whether to replace value at key with value, or update the nested dictionary
	'''

	# Get copy of value in elements
	i = iterable
	e = 0
	value = _copier(elements,value,_copy)
	
	assert isinstance(iterable,dict), "Error - iterable is not dictionary"

	# Convert string instance of elements to list, splitting string based on _split delimiter
	if isinstance(elements,str) and _split:
		elements = elements.split(_split)

	# Boolean whether elements is a list, otherwise is python object that is explicit key in dictionary
	islist = isinstance(elements,list)

	# Update iterable with elements 
	if not islist:
		# elements is python object and iterable is to be updated at first level of nesting
		isdict = not _reset and isinstance(i.get(elements),dict) and isinstance(value,dict)
		if isdict:
			i[elements].update(value)
		else:
			i[elements] = value
	else:
		# elements is list of nested keys and the nested values are to be extracted from iterable and set with value
		try:
			while e<len(elements)-1:
				if i.get(elements[e]) is None:
					i[elements[e]] = {}
				i = i[elements[e]]
				e+=1
			isdict = not _reset and isinstance(i.get(elements[e]),dict) and isinstance(value,dict)
			if isdict:
				i[elements[e]].update(value)
			else:
				i[elements[e]] = value
		except:
			pass

	return

def _get(iterable,elements,default=None,_split=False,_copy=False):
	'''
	Get nested value in iterable with nested elements keys

	Args:
		iterable (dict): dictionary of values
		elements (str,list): DELIMITER separated string or list to nested keys of location to get value
		default (python object): default data to return if elements not in nested iterable
		_split (bool,str,None): boolean or None or delimiter on whether to split string elements into list of nested keys
		_copy (bool,dict,None): boolean or None whether to copy value, or dictionary with keys on whether to copy value

	Returns:
		Value at nested keys elements of iterable
	'''	


	i = iterable
	e = 0
	
	# Convert string instance of elements to list, splitting string based on _split delimiter
	if isinstance(elements,str) and _split:
		elements = elements.split(_split)

	# Get nested element if iterable, based on elements
	if not isinstance(elements,list):
		# elements is python object and value is to be got from iterable at first level of nesting
		try:
			return i[elements]
		except:
			return default
	else:
		# elements is list of nested keys and the nested values are to be extracted from iterable
		try:
			while e<len(elements):
				i = i[elements[e]]
				e+=1			
		except:
			return default

	return _copier(elements[e-1],i,_copy)

def _pop(iterable,elements,default=None,_split=False,_copy=False):
	'''
	Pop nested value in iterable with nested elements keys

	Args:
		iterable (dict): dictionary to be popped in-place
		elements (str,list): DELIMITER separated string or list to nested keys of location to pop value
		default (python object): default data to return if elements not in nested iterable
		_split (bool,str,None): boolean or None or delimiter on whether to split string elements into list of nested keys
		_copy (bool,dict,None): boolean or None whether to copy value, or dictionary with keys on whether to copy value

	Returns:
		Value at nested keys elements of iterable
	'''		
	
	i = iterable
	e = 0

	# Convert string instance of elements to list, splitting string based on _split delimiter	
	if isinstance(elements,str) and _split:
		elements = elements.split(_split)

	if not isinstance(elements,list):
		# elements is python object and value is to be got from iterable at first level of nesting		
		try:
			return i.pop(elements)
		except:
			return default
	else:
		# elements is list of nested keys and the nested values are to be extracted from iterable		
		try:
			while e<(len(elements)-1):
				i = i[elements[e]]
				e+=1			
		except:
			return default

	return _copier(e,i.pop(elements[e],default),_copy)

def _has(iterable,elements,_split=False):
	'''
	Check if nested iterable has nested elements keys

	Args:
		iterable (dict): dictionary to be searched
		elements (str,list): DELIMITER separated string or list to nested keys of location to set value
		_split (bool,str,None): boolean or None or delimiter on whether to split string elements into list of nested keys

	Returns:
		Boolean value if nested keys elements are in iterable
	'''		

	i = iterable
	e = 0

	# Convert string instance of elements to list, splitting string based on _split delimiter	
	if isinstance(elements,str) and _split:
		elements = elements.split(_split)
	try:
		if not isinstance(elements,list):
			# elements is python object and value is to be got from iterable at first level of nesting				
			i = i[element]
		else:
			# elements is list of nested keys and the nested values are to be extracted from iterable		
			while e<len(elements):
				i = i[elements[e]]
				e+=1			
		return True
	except:
		return False

def _update(iterable,elements,_copy=False,_clear=True,_func=None):
	'''
	Update nested iterable with elements

	Args:
		iterable (dict): dictionary to be updated in-place
		elements (dict): dictionary of nested values to update iterable
		_copy (bool,dict,None): boolean or None whether to copy value, or dictionary with keys on whether to copy value
		_clear (bool): boolean of whether to clear iterable when the element's value is an empty dictionary
		_func(callable,None): Callable function that accepts key,iterable,elements arguments to modify value to be updated based on the given dictionaries
	'''		

	# Setup _func as callable
	if not callable(_func):
		_func = lambda key,iterable,elements: elements[key]

	# Clear iterable if _clear and elements is empty dictionary
	if _clear and elements == {}:
		iterable.clear()

	if not isinstance(elements,(dict)):
		# elements is python object and iterable is directly set as elements
		iterable = elements
		return

	# Recursively update iterable with elements
	for e in elements:
		if isinstance(iterable.get(e),dict):
			if e not in iterable:
				iterable.update({e: _copier(e,_func(e,iterable,elements),_copy)})
			else:
				_update(iterable[e],elements[e],_copy=_copy,_clear=_clear,_func=_func)
		else:
			iterable.update({e:_copier(e,_func(e,iterable,elements),_copy)})
	return

def _permute(dictionary,_copy=False,_groups=None,_ordered=True):
	'''
	Get all combinations of values of dictionary of lists

	Args:
		dictionary (dict): dictionary of keys with lists of values to be combined in all combinations across lists
		_copy (bool,dict,None): boolean or None whether to copy value, or dictionary with keys on whether to copy value
		_groups (list,None): List of lists of groups of keys that should not have their values permuted in all combinations, but should be combined in sequence element wise. For example groups = [[key0,key1]], where dictionary[key0] = [value_00,value_01,value_02],dictionary[key1] = [value_10,value_11,value_12], then the permuted dictionary will have key0 and key1 keys with only pairwise values of [{key0:value_00,key1:value_10},{key0:value_01,key1:value_11},{key0:value_02,key1:value_12}].
		_ordered (bool): Boolean on whether to return dictionaries with same ordering of keys as dictionary

	Returns:
		List of dictionaries with all combinations of lists of values in dictionary
	'''		
	def indexer(keys,values,_groups):
		'''
		Get lists of values for each group of keys in _groups
		'''
		_groups = copy.deepcopy(_groups)
		if _groups is not None:
			inds = [[keys.index(k) for k in g if k in keys] for g in _groups]
		else:
			inds = []
			_groups = []
		N = len(_groups)
		_groups.extend([[k] for k in keys if all([k not in g for g in _groups])])
		inds.extend([[keys.index(k) for k in g if k in keys] for g in _groups[N:]])
		values = [[values[j] for j in i ] for i in inds]
		return _groups,values

	def zipper(keys,values,_copy): 
		'''
		Get list of dictionaries with keys, based on list of lists in values, retaining ordering in case of grouped values
		'''
		return [{k:_copier(k,u,_copy) for k,u in zip(keys,v)} for v in zip(*values)]

	def unzipper(dictionary):
		'''
		Zip keys of dictionary of list, and values of dictionary as list
		'''
		keys, values = zip(*dictionary.items())	
		return keys,values

	def permuter(dictionaries): 
		'''
		Get all list of dictionaries of all permutations of sub-dictionaries
		'''
		return [{k:d[k] for d in dicts for k in d} for dicts in itertools.product(*dictionaries)]

	def nester(keys,values):
		'''
		Get values of permuted nested dictionaries in values.
		Recurse permute until values are lists and not dictionaries.
		'''
		keys,values = list(keys),list(values)
		for i,(key,value) in enumerate(zip(keys,values)):
			if isinstance(value,dict):
				if isinstance(_groups,dict):
					_group = _groups.get(key,_group)
				else:
					_group = _groups
				values[i] = _permute(value,_copy=_copy,_groups=_group)    
		return keys,values


	if dictionary in [None,{}]:
		return [{}]

	# Get list of all keys from dictionary, and list of lists of values for each key
	keys,values = unzipper(dictionary)


	# Get values of permuted nested dictionaries in values
	keys,values = nester(keys,values)

	# Retain ordering of keys in dictionary
	keys_ordered = keys
	
	# Get groups of keys based on _groups and get lists of values for each group
	keys,values = indexer(keys,values,_groups)

	# Zip keys with lists of lists in values into list of dictionaries
	dictionaries = [zipper(k,v,_copy) for k,v in zip(keys,values)]


	# Get all permutations of list of dictionaries into one list of dictionaries with all keys
	dictionaries = permuter(dictionaries)


	# Retain original ordering of keys if _ordered is True
	if _ordered:
		for i,d in enumerate(dictionaries):
			dictionaries[i] = {k: dictionaries[i][k] for k in keys_ordered}    
	return dictionaries



def _find(iterable,key):
	'''
	Find and yield key in nested iterable

	Args:
		iterable (dict): dictionary to search
		key (python object): key to find in iterable dictionary

	Yields:
		Found values with key in iterable
	'''	

	# Recursively find and yield value associated with key in iterable		
	try:
		if not isinstance(iterable,dict):
			raise
		for k in iterable:
			# print(k)
			if k == key:
				yield iterable[k]
			for v in _find(iterable[k],key):
				yield v
	except:
		pass
	return
				
def _replace(iterable,key,replacement,_append=False,_copy=True,_values=False):
	'''
	Find and replace key in-place in iterable with replacement key

	Args:
		iterable (dict): dictionary to be searched
		key (python object): key to be replaced with replacement key
		replacement (python object): dictionary key to replace key
		_append (bool): boolean on  whether to append replacement key to dictionary with value associated with key
		_copy (bool,dict,None): boolean or None whether to copy value, or dictionary with keys on whether to copy value
		_values (bool): boolean of whether to replace any values that equal key with replacement in the iterable 
	'''	

	# Recursively find where nested iterable keys exist, and replace or append in-place with replacement key
	try:
		keys = list(iterable)
		for k in keys:
			if k == key:
				if _append:
					iterable[replacement] = _copier(replacement,iterable.get(key),_copy)
					k = replacement
				else:
					iterable[replacement] = _copier(replacement,iterable.pop(key),_copy)
					k = replacement   
			if _values and iterable[k] == key:
				iterable[k] = _copier(k,replacement,_copy)
			_replace(iterable[k],key,replacement,_append=_append,_copy=_copy,_values=_values)
	except Exception as e:
		pass
	return



def _formatstring(key,iterable,elements,*args,**kwargs):

	'''
	Format values in iterable based on key and elements

	Args:
		key (python object): key to index iterable for formatting
		iterable (dict): dictionary with values to be formatted
		elements (dict): dictionary of elements to format iterable values

	Returns:
		Formatted value based on key,iterable, and elements
	'''	


	# Get value associated with key for iterable and elements dictionaries
	try:
		i = iterable[key]
	except:
		i = None
	e = elements[key]
	n = 0
	m = 0


	# Return elements[key] if kwargs[key] not passed to function, or elements[key] is not a type to be formatted
	if key not in kwargs or not isinstance(e,(str,tuple,list)):
		return e

	# Check for different cases of types of iterable[key] and elements[key] to be formatted

	# If iterable[key] is not a string, or iterable tuple or list, return value based on elements[key]
	if not isinstance(i,(str,tuple,list)):

		# If elements[key] is a string, string format elements[key] with args and kwargs and return the formatted value
		if isinstance(e,str):
			m = e.count('%')
			if m == 0:
				return e
			else:
				return e%(tuple((*args,*kwargs[key]))[:m])

		# If elements[key] is an iterable tuple or list, string format each element of elements[key] with args and kwargs and return the formatted value as a tuple
		elif isinstance(e,(tuple,list)):
			m = 0
			e = [x for x in e]
			c = [j for j,x in enumerate(e) if isinstance(x,str) and x.count('%')>0]
			for j,x in enumerate(e):
				if not isinstance(x,str):
					continue
				m = x.count('%')
				if m > 0:
					_j = c.index(j)
					e[j] = x%(tuple((*args,*kwargs[key]))[_j:m+_j])
			e = tuple(x for x in e)
			return e

		# If elements[key] is other python object, return elements[key]
		else:
			return e

	# If iterable[key] is a string, format iterable[key] based on elements[key]
	elif isinstance(i,str):

		# Get number of formatting elements in iterable[key] string to be formatted
		n = i.count('%')
		if n == 0:
			# If iterable[key] has no formatting elements, return based on elements[key]

			# If elements[key] is a string, string format elements[key] with args and kwargs and return the formatted value
			if isinstance(e,str):
				m = e.count('%')
				if m == 0:
					return e
				else:
					return e%(tuple((i,*args,*kwargs[key]))[:m])

			# If elements[key] is an iterable tuple or list, string format each element of elements[key] with args and kwargs and return the formatted value as a tuple
			elif isinstance(e,(tuple,list)):
				m = 0
				e = [x for x in e]
				c = [j for j,x in enumerate(e) if isinstance(x,str) and x.count('%')>0]	
				for j,x in enumerate(e):
					if not isinstance(x,str):
						continue
					m = x.count('%')
					if m > 0:
						_j = c.index(j)
						if isinstance(i,str):
							e[j] = x%(tuple((i,*args,*kwargs[key]))[_j:m+_j])
						else:
							e[j] = x%(tuple((*i,*args,*kwargs[key]))[_j:m+_j])										
				e = tuple(x for x in e)
				return e

			# If elements[key] is other python object, return elements[key]
			else:
				return e
		# If iterable[key] string has non-zero formatting elements, format iterable[key] string with elements[key], args, and kwargs
		else:
			if isinstance(e,str):
				return i%(tuple((e,*args,*kwargs[key]))[:n])
			elif isinstance(e,(tuple,list)):
				return i%(tuple((*e,*args,*kwargs[key]))[:n])
			else:
				return e

	# If iterable[key] is an iterable tuple or list, string format each element of iterable[key] with elements[key],args and kwargs and return the formatted value as a tuple
	elif isinstance(i,(tuple,list)):
		i = [str(x) for x in i]
		n = 0
		c = [j for j,x in enumerate(i) if isinstance(x,str) and x.count('%')>0]	
		for j,x in enumerate(i):
			n = x.count('%')
			if n > 0:
				_j = c.index(j)				
				if isinstance(e,str):
					i[j] = x%(tuple((e,*args,*kwargs[key]))[_j:n+_j])
				else:
					i[j] = x%(tuple((*e,*args,*kwargs[key]))[_j:n+_j])										

		if n == 0:
			if isinstance(e,str):
				m = e.count('%')
				if m == 0:
					return e
				else:
					return e%(tuple((i,*args,*kwargs[key]))[:m])
			elif isinstance(e,(tuple,list)):
				m = 0
				e = [x for x in e]
				c = [j for j,x in enumerate(e) if isinstance(x,str) and x.count('%')>0]					
				for j,x in enumerate(e):
					if not isinstance(x,str):
						continue
					m = x.count('%')
					if m > 0:
						_j = c.index(j)				
						if isinstance(i,str):
							e[j] = x%(tuple((i,*args,*kwargs[key]))[_j:m+_j])
						else:
							e[j] = x%(tuple((*i,*args,*kwargs[key]))[_j:m+_j])										
				e = tuple(x for x in e)
				return e
			else:
				return e			
			return e
		else:
			i = tuple(x for x in i)
			return i
	else:
		return e