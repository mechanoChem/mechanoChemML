# Import python modules
import os,sys,copy,itertools,re
import numpy as np
DELIMITER = '__'



def groupsort(groups,args):
	'''
	Get unique args from groups and sort in same order as in args
	'''

	sort = [v for v in args]
	sets = {u:[v for v in groups if groups[v] == u] for u in set([groups[v] for v in groups])}
	groups = list(sorted(sets,key=lambda u: min([sort.index(v) if v in sort else len(sort)+list(sets).index(u) 
												 for v in sets[u]])))



	return groups



def icombinations(iterable,n,unique=False):
	''' 
	Get all combinations of p number of non-negative integers that sum up to at most n
	Args:
		iterable (iterable): Number of integers or iterable of length p
		n (int,iterable): Maximum number of elements, or allowed number of elements
		unique (bool): Return unique combinations of integers and q = choose(p+n,n) else q = (p^(n+1)-1)/(p-1)
	Returns:
		combinations (list): All combinations of iterable with q list of lists of length up to n, or lengths in n
	'''
	iterable = list(iterable)
	p = len(iterable)
	n = range(n+1) if isinstance(n,(int,np.integer)) else n
	combinations = []
	for i in n:
		combos = list((tuple(sorted(j,key=lambda i:iterable.index(i))) for j in itertools.product(iterable,repeat=i)))
		if unique:
			combos = sorted(set(combos),key=lambda i:combos.index(i))
		combinations.extend(combos)
	return combinations



def findstring(string,strings,types,prefixes,labels,replacements,default=None,regex=False,usetex=True):
	'''
	Check if string in strings dictionary
	Args:
		string(str): String to be checked if in strings dictionary
		strings(dict): Dictionary of string keys with replacement strings values
		types(dict): Dictionary of variable types that strings may start with, with template values in order to determine how to render strings 
		prefixes(dict): Dictionary of prefix types that strings may start with, with template values in order to determine how to render strings 
		labels(dict): Dictionary of label prefixes for string templates to render strings
		default (str,None): default return value if string not in strings
		regex (bool): perform regex processing to find string pattern in strings
	Returns:
		Returned string based on logic of whether string is in strings
	'''
	if default is None:
		default = string

	latex = None
	func = lambda string,latex,iloc,labels,strings:latex
	iloc = None

	funcs = []
	ilocs = []
	prefixs = []
	_string = string
	isprefix = any([string.startswith('%s%s'%(prefix,DELIMITER)) for prefix in prefixes])


	# print('finding',string,isprefix)
	while(isprefix):
		for prefix in prefixes:			
			if string.startswith(prefix):
				func = prefixes[prefix]
				# print('startswith',prefix,func)
				if DELIMITER in string:
					iloc = string.split(DELIMITER)[1]
					string = DELIMITER.join(string.split(DELIMITER)[2:])
				else:
					iloc = ''
				#print('modified',string,iloc)
				prefixs.append(prefix)
				funcs.append(func)
				ilocs.append(iloc)
		isprefix = any([string.startswith('%s%s'%(prefix,DELIMITER)) for prefix in prefixes])

	
	prefixs = prefixs[::-1]
	funcs = funcs[::-1]
	ilocs = ilocs[::-1]

	# print(ilocs)

	for typed in types:
		for label in types[typed]:
			# print('Trying',string,typed,label,string in types[typed][label])
			if string in types[typed][label] and (any([p==typed or '%ss'%(p)==typed for p in prefixs]) or (len(prefixs)==0) or all([(p not in types) and ('%ss'%(p) not in types) for p in prefixs])):
				latex = types[typed][label][string]	
				# print('found latex',typed,label,latex)

			if latex is not None:
				break
		if latex is not None:
				break

	if latex is None:
		latex = strings.get(string,string)
		typed = None
		label = None

	#print('final latex',latex)


	for func,iloc in zip(funcs,ilocs):
		latex = func(string,latex,iloc,labels,strings)

	_latex = latex
	for t in replacements:
		latex = latex.replace(t,replacements[t])

	#print('returning',latex)

	if not regex:
		return latex

	for string in strings:
		restring = re.compile(r'%s'%(string))
		restring = restring.search(string)
		try:
			restring = restring.group(0)			
		except:
			continue
		try: 
			parser = strings.get(string)				
			if not callable(parser):
				replacement = parser
				parser = lambda s:s
			else:
				replacement = string
			return parser(re.sub(string,result,string))
		except:
			return string
	return default


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
	
	elif isinstance(number,(float,np.float)):		
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

# Texify strings
class Texify(object):
	'''
	Render strings as Latex strings

	Args:
		texstrings(dict): Dictionary of text strings with Latex string values
		texargs(dict): Dictionary of arguments and settings used in Latex rendering
		texlabels(dict): Dictionary of modifier strings for Latex rendering
		labels(dict): Dictionary of label prefixes for string templates for Latex rendering
		usetex(bool): Render as Latex strings, or render as text strings
	'''
	def __init__(self,texstrings={},texargs={},texlabels={},usetex=True):


		args_default = {
			'bases':{'monomial':1,'polynomial':0,'taylorseries':1,'derivative':1,'expansion':1},
			'order':2,
			'basis':3,
			'selection':tuple(range(1+1)),
			'iloc':[0,None],
			'unique':True,
			'operators':['partial','delta','Delta','d'],
			'weights':['stencil'],			
			'inputs': {'%s%d'%(x,i):r'{%s_{%d}}'%(x,i) for x in ['x'] for i in range(5)},
			'outputs': {'%s%d'%(x,i):r'{%s_{%d}}'%(x,i) for x in ['y'] for i in range(3)},
			'terms': {'%s%d'%(x,i):r'{%s_{%d}}'%(x,i) for x in ['x'] for i in range(3)},
			# 'groups':{**{'%s%d'%(x,i):r'{%s_{%d}}'%(x,i) for x in ['x'] for i in range(5)}},
			'constants':{},
			'texreplacements':{r'\\\\':'\\\\'},
			'replacements':{DELIMITER:'',r'\\\\':'','\\':'','textrm':'','_':r'\_','^{}':'','^':r'\^~',
							r'frac':'',r'}{{delta':r'}/{{delta',r'}{{partial':r'}/{{partial','abs':''},
		}

		args_special = {'inputs':{},'outputs':{},'constants':{},'order':1,'basis':1}
		args_keyed = ['iloc']
		args_dependent = {'groups': lambda args:groupsort(args.get('groups',{x: args['inputs'].get(x) for x in args['inputs']}),args['inputs']),
						  'selection': lambda args: tuple(range(args['order'] if not isinstance(args.get('selection'),(int,np.integer)) else args.get('selection')+1)) if not isinstance(args.get('selection'),(np.ndarray,list)) else args['selection'] }
		


		args = {}
		args.update(texargs)
		args.update({k:v for k,v in args_default.items() if k not in args})
		args.update({k:v for k,v in args_special.items() if args.get(k) is None})
		args.update({k: list(set([i for u in (args.get(k,{}) if isinstance(args.get(k),dict) else ([i for i in args.get(k,[])] if isinstance(args.get(k),list) else [args.get(k)]))
									for i in ((args.get(k,{}).get(u) if (
										isinstance(args.get(k,{}).get(u),(list,tuple,np.ndarray))) else (
										[args.get(k,{}).get(u)])) if isinstance(args.get(k),dict) else ([i for i in args.get(k,[])] if isinstance(args.get(k),list) else [args.get(k)]))]))
					for k in args_keyed})


		for k in args_dependent:
			args[k] = args_dependent[k](args)

		# Modifying
		labels = {
				'abs': r'{\abs{%s}}',	
				'bar': r'{\bar{%s}}',
				'tilde': r'{\tilde{%s}}',
				'hat': r'{\hat{%s}}',
				'brackets':r'{(%s)}',
				'leftrightbrackets':r'{\left({%s}\right)}',
				'squarebrackets':r'{[{%s}]}',
				'leftrightsquarebrackets':r'{\left[{%s}\right]}',
				'curlybrackets':r'{\{{%s}\}}',
				'leftrightcurlybrackets':r'{\left\{{%s}\right\}}',				
				'superscript':r'{%s}^{%s}',			
				'subscript':r'{%s}_{%s}',			
				'supersubscript':r'{%s}_{%s}^{%s}',			
				'partial':r'\partial',
				'func':r'%s(%s)',
				'd': r'd',
				'delta':r'\delta',
				'Delta':r'\Delta',
				'gamma':r'\gamma',				
				'derivative':r'{\frac{{%s}^{%s}%s}{%s {%s}^{%s}}}',
				'nderivative':r'{\frac{{%s}^{%s}%s}{%s}}',
				'coefficient': r'{%s}^{%s}',
				'factorial': r'{%s!}'
		}
		labels.update(texlabels)


		prefixes = {
			'variable': lambda string,latex,iloc,labels,strings: latex,
			'constant': lambda string,latex,iloc,labels,strings: labels['func']%(latex,'%s'%(','.join(['{%s}_{%s}'%(v,str(iloc) if iloc not in [None,'None',str(None)] else '') for v in args['groups']]))),
			'subscript': lambda string,latex,iloc,labels,strings: r'{%s}_{%s}'%(latex,str(iloc)),
			'superscript': lambda string,latex,iloc,labels,strings: r'{%s^{%s}'%(latex,str(iloc)),
			'monomial': lambda string,latex,iloc,labels,strings: r'{%s}'%(latex),
			'polynomial': lambda string,latex,iloc,labels,strings: r'{%s}'%(latex),
			'chebyshev': lambda string,latex,iloc,labels,strings: r'{%s}'%(latex),
			'legendre': lambda string,latex,iloc,labels,strings: r'{%s}'%(latex),
			'hermite': lambda string,latex,iloc,labels,strings: r'{%s}'%(latex),
			'expansion': lambda string,latex,iloc,labels,strings: r'{%s}'%(latex.replace('True',str(iloc) if iloc not in [None,'None',str(None)] else '')),
			'taylorseries': lambda string,latex,iloc,labels,strings: r'{%s}'%(latex.replace('True',str(iloc) if iloc not in [None,'None',str(None)] else '')),
			'derivative': lambda string,latex,iloc,labels,strings: r'{%s}'%(latex),
			'difference': lambda string,latex,iloc,labels,strings: r'{%s}'%(latex),
			'deltavar': lambda string,latex,iloc,labels,strings: r'%s{%s}'%(labels['delta'],latex),
			'Deltavar': lambda string,latex,iloc,labels,strings: r'%s{%s}'%(labels['Delta'],latex),
			'coefficient': lambda string,latex,iloc,labels,strings: labels['coefficient']%(labels['gamma'],latex),
			'factorial': lambda string,latex,iloc,labels,strings: labels['factorial']%(str(iloc)),
			'iteration': lambda string,latex,iloc,labels,strings: r'{%s}^{(%s)}'%(latex,iloc),
			}

		# Types
		def variables(variable,strings,args,labels,usetex):
			'''
			Variable string patterns
			Args:
				variable(str,list): Type of variable for variables (inputs,outputs,terms)			
				strings(dict): String patterns with text string keys and Latex string values
				args(dict): Dictionary of arguments and settings used in Latex rendering
				labels(dict): Dictionary of label prefixes for string templates for Latex rendering
			Returns:
				Dictionary of rendered string patterns
			'''					


			if isinstance(variable,str):
				variables = {x: strings.get(x,args[variable][x]) for x in args[variable]}
			else:
				variables = {x: strings.get(x,args[v][x]) for v in variable for x in args[v]}

			func = labels['superscript']

			values = variables

			return values


		def constants(variable,strings,args,labels,usetex):
			'''
			Variable string patterns
			Args:
				variable(str,list): Type of variable for constants (inputs,outputs,terms)			
				strings(dict): String patterns with text string keys and Latex string values
				args(dict): Dictionary of arguments and settings used in Latex rendering
				labels(dict): Dictionary of modifier strings for Latex rendering
			Returns:
				Dictionary of rendered string patterns
			For example:
				strings have notation:
					prefix__iloc__variable
				with prefixes
					[constant]					
				which becomes
					variable
			'''					

			if isinstance(variable,str):
				variables = {x: strings.get(x,args[variable][x]) for x in args[variable]}
			else:
				variables = {x: strings.get(x,args[v][x]) for v in variable for x in args[v]}

			func = labels['superscript']


			values = variables
			return values


		def derivative(symbol,strings,args,labels,usetex):
			'''
			Derivative string patterns
			Args:
				symbol(str): String key for labels on which derivative symbol to use (partial,delta,Delta)
				strings(dict): String patterns with text string keys and Latex string values
				args(dict): Dictionary of arguments and settings used in Latex rendering
				labels(dict): Dictionary of modifier strings for Latex rendering
			Returns:
				Dictionary of rendered string patterns
			For example:
				strings have notation:
					operation_0__operation_1__...__operation_order-1__order__function__variable_0__variable_1__...__variable_order-1__weight_0__weight_1__...__weight_order-1
				which becomes
					\frac{operation function}{operation_0 variable_0 ... operation_order-1 variable_order-1} operation order_0 variable_0 ... operation_order-1 variable_order-1
			'''					
			inputs = {x: strings.get(x,args['inputs'][x]) for x in args['inputs']}
			outputs = {y: strings.get(y,args['outputs'][y]) for y in args['outputs']}

			delimiter = DELIMITER
			func = labels['nderivative']
			Symbol = labels[symbol]


			values = {}
			if args['bases'].get('derivative'):
				values = {
					delimiter.join([
					delimiter.join([o]*j),
					str(j),
					y,
					delimiter.join([u for u in x]),
					delimiter.join([w]*j)
					]):func%(Symbol,str(j) if j>1 else '','%s'%(outputs[y]),
							' '.join([r'%s %s'%(Symbol,inputs[v]) for v in x]) if len(set([v for v in x]))>1 else r'{%s %s}^{%s}'%(Symbol,inputs[[v for v in x][0]],str(j) if j>1 else ''),
					)if usetex else (
						'd^%s%s/%s'%(str(j) if j>1 else '',y,''.join(['d%s'%(v) for v in x]))
						)				
					for o in [o for o in args['operators'] if o in [symbol]]
					for j in range(1,args['order']+1)
					for y in outputs
					for x in icombinations(inputs,[j],unique=args['unique'])
					for w in args['weights']
					}

			return values	

		def expansion(iloc,symbol,Symbol,strings,args,labels,usetex):
			'''
			Expansion string patterns
			Args:
				iloc(int,None): location of Expansion
				symbol(str): String key for labels on which derivative symbol to use (partial,delta,Delta)
				symbol(str): String key for labels on which delta symbol to use (partial,delta,Delta)
				strings(dict): String patterns with text string keys and Latex string values
				args(dict): Dictionary of arguments and settings used in Latex rendering
				labels(dict): Dictionary of modifier strings for Latex rendering
			Returns:
				Dictionary of rendered string patterns
			For example:
				strings have notation:
					prefix__iloc__operation_0__operation_1__...__operation_order-1__order__function__variable_0__variable_1__...__variable_order-1__weight_0__weight_1__...__weight_order-1
				with prefixes
					[expansion]
				which becomes
					\frac{operation function}{operation_0 variable_0 ... operation_order-1 variable_order-1} operation order_0 variable_0 ... operation_order-1 variable_order-1
			'''					
			inputs = {x: strings.get(x,args['inputs'][x]) for x in args['inputs']}
			outputs = {y: strings.get(y,args['outputs'][y]) for y in args['outputs']}

			delimiter = DELIMITER
			func = labels['nderivative']
			symbol = labels[symbol]
			Symbol = labels[Symbol]




			values = {}
			if args['bases'].get('expansion'):

				values.update({
					delimiter.join([
					delimiter.join([o]*j),
					str(j),
					y,
					delimiter.join([u for u in x]),
					delimiter.join([w]*j)
					]):('%s'%(						
						func%(
							symbol,
							str(j) if j>1 else '',
							'%s%s'%(outputs[y],'(%s)'%(','.join(['{%s}_{%s}'%(v,str(iloc) if iloc is not None else '') for v in args['groups']]))),
							' '.join([r'{%s %s}^{%s}'%(symbol,inputs[v],str(i) if i>1 else '') for i,v in sorted([(list(x).count(v),v) for v in set([v for v in x])],key=lambda v:list(x).index(v[1]))]))		
					) if usetex else (
					'd^%s%s/%s'%(						
						str(j) if j>1 else ''),
						y,
						''.join(['d%s'%(v) for v in x]), 
					))
					for o in args['operators']
					for j in range(1,args['order']+1)
					for y in outputs
					for x in icombinations(inputs,[j],unique=args['unique'])
					for w in args['weights']
				})
				
			return values	

		def taylorseries(iloc,symbol,Symbol,strings,args,labels,usetex):
			'''
			Taylorseries string patterns
			Args:
				iloc(int,None): location of Taylorseries
				symbol(str): String key for labels on which derivative symbol to use (partial,delta,Delta)
				symbol(str): String key for labels on which delta symbol to use (partial,delta,Delta)
				strings(dict): String patterns with text string keys and Latex string values
				args(dict): Dictionary of arguments and settings used in Latex rendering
				labels(dict): Dictionary of modifier strings for Latex rendering
			Returns:
				Dictionary of rendered string patterns
			For example:
				strings have notation:
					prefix__iloc__operation_0__operation_1__...__operation_order-1__order__function__variable_0__variable_1__...__variable_order-1__weight_0__weight_1__...__weight_order-1
				with prefixes
					[taylorseries]					
				which becomes
					\frac{1}{order !} \frac{operation function}{operation_0 variable_0 ... operation_order-1 variable_order-1} operation order_0 variable_0 ... operation_order-1 variable_order-1
			'''					
			inputs = {x: strings.get(x,args['inputs'][x]) for x in args['inputs']}
			outputs = {y: strings.get(y,args['outputs'][y]) for y in args['outputs']}

			delimiter = DELIMITER
			func = labels['nderivative']
			symbol = labels[symbol]
			Symbol = labels[Symbol]

			values = {}
			
			if args['bases'].get('taylorseries'):
				values.update({
					delimiter.join([
					delimiter.join([o]*j),
					str(j),
					y,
					delimiter.join([u for u in x]),
					delimiter.join([w]*j)
					]):('%s%s%s'%(
						(r'%s%s'%(
							(labels['coefficient']%(labels['gamma'],
								''.join([r'{{%s}^{%s}}'%(inputs[v],str(i) if i>1 else '') 
									for i,v in sorted([(list(x).count(v),v) for v in set([v for v in x])],key=lambda v:list(x).index(v[1]))]))), 
							(r'\frac{1}{%s}'%(''.join([labels['factorial']%(i) if i > 1 else '' for i,v in sorted([(list(x).count(v),v) for v in set([v for v in x])],key=lambda v:list(x).index(v[1]))])
								if args['unique'] else labels['factorial']%(j))
								if (j>1 and ((not args['unique']) or any([i!=1 for i,v in sorted([(list(x).count(v),v) for v in set([v for v in x])],key=lambda v:list(x).index(v[1]))]))) else '')
							)) if j>0 else '',							
						func%(
							symbol,
							str(j) if j>1 else '',
							'%s%s'%(outputs[y],'(%s)'%(','.join(['{%s}_{%s}'%(v,str(iloc) if iloc is not None else '') for v in args['groups']]))),
							' '.join([r'{%s %s}^{%s}'%(symbol,inputs[v],str(i) if i>1 else '') for i,v in sorted([(list(x).count(v),v) for v in set([v for v in x])],key=lambda v:list(x).index(v[1]))])),		
						' '.join([r'{%s %s}^{%s}'%(Symbol,inputs[v],str(i) if i>1 else '') for i,v in sorted([(list(x).count(v),v) for v in set([v for v in x])],key=lambda v:list(x).index(v[1]))
							])
					) if usetex else (
					'%s d^%s%s/%s %s'%(
						('%s**{%s} %s'%(
								'g',
								''.join([r'{{%s}^{%s}}'%(inputs[v],str(i) if i>1 else '') 
									for i,v in sorted([(list(x).count(v),v) for v in set([v for v in x])],key=lambda v:list(x).index(v[1]))]), 
							(r'1/%s'%(''.join([labels['factorial']%(i) if i > 1 else '' for i,v in sorted([(list(x).count(v),v) for v in set([v for v in x])],key=lambda v:list(x).index(v[1]))])
								if args['unique'] else labels['factorial']%(j)) if (j>1 and ((not args['unique']) or any([i!=1 for i,v in sorted([(list(x).count(v),v) for v in set([v for v in x])],key=lambda v:list(x).index(v[1]))]))) else '') if j>0 else ''
							) if j>0 else ''),						
						str(j) if j>1 else '',
						y,
						''.join(['d%s'%(v) for v in x]), 
						' '.join(['d%s'%(v) for v in x])
					)))
					for o in args['operators']
					for j in range(1,args['order']+1)
					for y in outputs
					for x in icombinations(inputs,[j],unique=args['unique'])
					for w in args['weights']
				})


				# values.update({
				# 	delimiter.join([
				# 	delimiter.join([o]*j),
				# 	str(j),
				# 	y,
				# 	delimiter.join([u for u in x]),
				# 	delimiter.join([w]*j)
				# 	]):'%s%s%s'%(
				# 		r'%s\frac{1}{%s}'%(labels['coefficient']%(labels['gamma'],''.join([r'%s'%(inputs[v]) for v in x]) if len(set([v for v in x]))>1 else r'{%s}^{%s}'%(inputs[[v for v in x][0]],str(j) if j>1 else '')
				# 			),''.join([labels['factorial']%(i) if i > 1 else '' for i in [list(x).count(v) for v in set([v for v in x])]]) if args['unique'] else labels['factorial']%(j)) if (j>1 and ((not args['unique']) or any([i!=1 for i in [list(x).count(v) for v in set([v for v in x])]]))) else (
				# 			labels['coefficient']%(labels['gamma'],''.join([r'%s'%(inputs[v]) for v in x]) if len(set([v for v in x]))>1 else r'{%s}^{%s}'%(inputs[[v for v in x][0]],str(j) if j>1 else ''))),
				# 		func%(symbol,str(j) if j>1 else '','%s%s'%(outputs[y],'(%s)'%(','.join(['{%s}_{%s}'%(v,str(iloc) if iloc is not None else '') for v in args['groups']]))),
				# 			' '.join([r'%s %s'%(symbol,inputs[v]) for v in x]) if len(set([v for v in x]))>1 else r'{%s %s}^{%s}'%(symbol,inputs[[v for v in x][0]],str(j) if j>1 else '')),
				# 		' '.join([r'%s %s'%(Symbol,inputs[v]) for v in x]) if len(set([v for v in x]))>1 else r'{%s %s}^{%s}'%(Symbol,inputs[[v for v in x][0]],str(j) if j>1 else '')
				# 	) if usetex else (
				# 		'%s d^%s%s/%s %s'%('1/%s!'%(str(j)) if j>1 else '',str(j) if j>1 else '',y,''.join(['d%s'%(v) for v in x]), ' '.join(['d%s'%(v) for v in x]))
				# 	)				
				# 	for o in args['operators']
				# 	for j in range(1,args['order']+1)
				# 	for y in outputs
				# 	for x in icombinations(inputs,[j],unique=args['unique'])
				# 	for w in args['weights']
				# })




			return values				



		def monomials(variable,replacements,strings,args,labels,usetex):
			'''
			Monomial string patterns
			Args:
				variable(str,list): Type of variable for monomials (inputs,outputs,terms)			
				replacements(dict): Replacements for variable strings				
				strings(dict): String patterns with text string keys and Latex string values
				args(dict): Dictionary of arguments and settings used in Latex rendering
				labels(dict): Dictionary of modifier strings for Latex rendering
			Returns:
				Dictionary of rendered string patterns
			For example:
				strings have notation:
					prefix__iloc__variable_power
				with prefixes
					[monomial]					
				which becomes
					{variable}^{power}
			'''

			if isinstance(variable,str):
				variables = copy.deepcopy(list(args[variable]))
				for i,v in enumerate(variables):
					for r in replacements:
						variables[i] = variables[i].replace(r,replacements[r])
				variables = {x: strings.get(x,args[variable].get(x)) for x in variables}
			else:
				variables = copy.deepcopy({v:[x for x in args[v]] for v in variable})

				for i,v in enumerate(variables):
					for j,x in enumerate(variables[v]):
						for r in replacements:
							variables[v][j] = variables[v][j].replace(r,replacements[r])
				variables = {x: strings.get(x,args[v].get(x)) for v in variables for x in variables[v]}


			if usetex:
				func = labels['superscript']
			else:
				func = '%s^%s'

			delimiter = '_'
			constant = r''

			values = {}

			if args['bases'].get('monomial'):

				values.update({delimiter.join([x,str(j)]):func%(variables[x],str(j) if j>1 else '') if j>0 else constant
							for j in range(args['order']+1)
							for x in variables
						})


			return values

		def polynomial(variable,replacements,strings,args,labels,usetex):
			'''
			Polynomial string patterns
			Args:
				variable(str,list): Type of variable for polynomial (inputs,outputs,terms)
				replacements(dict): Replacements for variable strings
				strings(dict): String patterns with text string keys and Latex string values
				args(dict): Dictionary of arguments and settings used in Latex rendering
				labels(dict): Dictionary of modifier strings for Latex rendering
			Returns:
				Dictionary of rendered string patterns
			For example:
				strings have notation:
					prefix__iloc__variable_0_power_0-variable_1_power_1-...-variable_order-1_power_order-1
				with prefixes
					[polynomial]					
				which becomes
					{variable_0}^{power_0} {variable_1}^{power_1} ... {variable_order-1}^{power_order-1} 				
			'''			

			

			if isinstance(variable,str):
				variables = copy.deepcopy(list(args[variable]))
				for i,v in enumerate(variables):
					for r in replacements:
						variables[i] = variables[i].replace(r,replacements[r])
				variables = {x: strings.get(x,args[variable].get(x)) for x in variables}
			else:
				variables = copy.deepcopy({v:[x for x in args[v]] for v in variable})

				for i,v in enumerate(variables):
					for j,x in enumerate(variables[v]):
						for r in replacements:
							variables[v][j] = variables[v][j].replace(r,replacements[r])
				variables = {x: strings.get(x,args[v].get(x)) for v in variables for x in variables[v]}

			if usetex:
				func = labels['superscript']
			else:
				func = '%s^%s'

			if isinstance(args['selection'],tuple):
				I = itertools.product(args['selection'],repeat=len(variables))
			else:
				I = args['selection']

			if not isinstance(I,list):
				I = list(I)
			splitter = '-'
			delimiter = '_'
			constant = r'1'
			if usetex:
				separator = ''
			else:
				separator = ' '

			values = {}

			if args['bases'].get('polynomial'):
				values.update({
					splitter.join([delimiter.join([x,str(j)]) for x,j in zip(variables,i)]):
					 separator.join([func%(variables[x],str(j) if j>1 else '') if j>0 else '' for x,j in zip(variables,i)]) if sum(i)>0 else constant
					for i in I
					})


			return values				


		def chebyshev(variable,replacements,strings,args,labels,usetex):
			'''
			Chebyshev string patterns
			Args:
				variable(str,list): Type of variable for Chebyshev polynomial (inputs,outputs,terms)			
				replacements(dict): Replacements for variable strings				
				strings(dict): String patterns with text string keys and Latex string values
				args(dict): Dictionary of arguments and settings used in Latex rendering
				labels(dict): Dictionary of modifier strings for Latex rendering
			Returns:
				Dictionary of rendered string patterns
			For example:
				strings have notation:
					prefix__iloc__variable_order
				with prefixes
					[chebyshev]					
				which becomes
					T_{order}(variable)
			'''


			if isinstance(variable,str):
				variables = copy.deepcopy(list(args[variable]))
				for i,v in enumerate(variables):
					for r in replacements:
						variables[i] = variables[i].replace(r,replacements[r])
				variables = {x: strings.get(x,args[variable].get(x)) for x in variables}
			else:
				variables = copy.deepcopy({v:[x for x in args[v]] for v in variable})

				for i,v in enumerate(variables):
					for j,x in enumerate(variables[v]):
						for r in replacements:
							variables[v][j] = variables[v][j].replace(r,replacements[r])
				variables = {x: strings.get(x,args[v].get(x)) for v in variables for x in variables[v]}

			if usetex:
				func = labels['subscript']
			else:
				func = '%s_%s'
			symbol = 'T'

			delimiter = '_'
			constant = r'1'

			values = {}

			if args['bases'].get('chebyshev'):			
				values.update({delimiter.join([x,str(j)]):r'%s(%s)'%(func%(symbol,str(j)),variables[x]) if j>0 else constant
							for j in range(args['order']+1)
							for x in variables
						})

			return values

		def hermite(variable,replacements,strings,args,labels,usetex):
			'''
			Hermite string patterns
			Args:
				variable(str,list): Type of variable for Hermite polynomial (inputs,outputs,terms)			
				replacements(dict): Replacements for variable strings				
				strings(dict): String patterns with text string keys and Latex string values
				args(dict): Dictionary of arguments and settings used in Latex rendering
				labels(dict): Dictionary of modifier strings for Latex rendering
			Returns:
				Dictionary of rendered string patterns
			For example:
				strings have notation:
					prefix__iloc__variable_order
				with prefixes
					[hermite]					
				which becomes
					H_{order}(variable)				
			'''

			if isinstance(variable,str):
				variables = copy.deepcopy(list(args[variable]))
				for i,v in enumerate(variables):
					for r in replacements:
						variables[i] = variables[i].replace(r,replacements[r])
				variables = {x: strings.get(x,args[variable].get(x)) for x in variables}
			else:
				variables = copy.deepcopy({v:[x for x in args[v]] for v in variable})

				for i,v in enumerate(variables):
					for j,x in enumerate(variables[v]):
						for r in replacements:
							variables[v][j] = variables[v][j].replace(r,replacements[r])
				variables = {x: strings.get(x,args[v].get(x)) for v in variables for x in variables[v]}

			if usetex:
				func = labels['subscript']
			else:
				func = '%s_%s'
			symbol = 'H'

			delimiter = '_'
			constant = r'1'
			
			values = {}

			if args['bases'].get('hermite'):			
				values.update({delimiter.join([x,str(j)]):r'%s(%s)'%(func%(symbol,str(j)),variables[x]) if j>0 else constant
							for j in range(args['order']+1)
							for x in variables
						})

			return values		

		def legendre(variable,replacements,strings,args,labels,usetex):
			'''
			Legendre string patterns
			Args:
				variable(str,list): Type of variable for Legendre polynomial (inputs,outputs,terms)			
				replacements(dict): Replacements for variable strings				
				strings(dict): String patterns with text string keys and Latex string values
				args(dict): Dictionary of arguments and settings used in Latex rendering
				labels(dict): Dictionary of modifier strings for Latex rendering
			Returns:
				Dictionary of rendered string patterns
			For example:
				strings have notation:
					prefix__iloc__variable_order
				with prefixes
					[legendre]					
				which becomes
					L_{order}(variable)				
			'''

			if isinstance(variable,str):
				variables = copy.deepcopy(list(args[variable]))
				for i,v in enumerate(variables):
					for r in replacements:
						variables[i] = variables[i].replace(r,replacements[r])
				variables = {x: strings.get(x,args[variable].get(x)) for x in variables}
			else:
				variables = copy.deepcopy({v:[x for x in args[v]] for v in variable})

				for i,v in enumerate(variables):
					for j,x in enumerate(variables[v]):
						for r in replacements:
							variables[v][j] = variables[v][j].replace(r,replacements[r])
				variables = {x: strings.get(x,args[v].get(x)) for v in variables for x in variables[v]}

			if usetex:
				func = labels['subscript']
			else:
				func = '%s_%s'
			symbol = 'L'

			delimiter = '_'
			constant = r'1'
			
			values = {}

			if args['bases'].get('legendre'):			
				values.update({delimiter.join([x,str(j)]):r'%s(%s)'%(func%(symbol,str(j)),variables[x]) if j>0 else constant
							for j in range(args['order']+1)
							for x in variables
						})

			return values			


		texstrings.update({s.replace('partial',r):texstrings[s].replace(r'\delta',r'\%s'%r if r not in ['d'] else r) for s in texstrings for r in args['operators']})

		funcs = { 
			'misc': lambda strings,args,labels,usetex:{
				'default':{
					'complexity_':r'N_{\textrm{terms}}',
					'coef_':labels['abs']%(labels['gamma']),
					'data':r'\mathcal{D}_{\textrm{data}}',
					'intercept_':r'\beta',
					'term_':r'\frac{\partial f}{\partial x}',
					'loss':r'\textrm{Loss}', 
					'State': r'\textrm{State}',	
					'radius':r'r'},
				'strings':texstrings,	
				},			
			'variables': lambda strings,args,labels,usetex: {'%s_%s'%(k,'_'.join(list(set(list(r.values()))))): monomials(k,r,strings,args,labels,usetex) for k in ['inputs','outputs','terms',['inputs','terms']]  for r in [{'partial':x} for x in args['operators'] if x not in ['partial']]},
			'constants': lambda strings,args,labels,usetex: {'%s_%s'%(k,'_'.join(list(set(list(r.values()))))): monomials(k,r,strings,args,labels,usetex) for k in ['inputs','outputs','terms',['inputs','terms']]  for r in [{'partial':x} for x in args['operators'] if x not in ['partial']]},
			'derivative': lambda strings,args,labels,usetex: {'%s'%(k): derivative(k,strings,args,labels,usetex) for k in args['operators']},
			'expansion': lambda strings,args,labels,usetex: {'%s_%s_%s'%(str(i),k,K): expansion(i,k,K,strings,args,labels,usetex) for i in args['iloc'] for k,K in [('partial','Delta'),('delta','delta'),('Delta','Delta'),('d','Delta')]},
			'taylorseries': lambda strings,args,labels,usetex: {'%s_%s_%s'%(str(i),k,K): taylorseries(i,k,K,strings,args,labels,usetex) for i in args['iloc'] for k,K in [('partial','Delta'),('delta','delta'),('Delta','Delta'),('d','Delta')]},
			'monomials': lambda strings,args,labels,usetex: {'%s_%s'%(k,'_'.join(list(set(list(r.values()))))): monomials(k,r,strings,args,labels,usetex) for k in ['inputs','outputs','terms',['inputs','terms']] for r in [{'partial':x} for x in args['operators'] if x not in ['partial']]},
			'polynomial': lambda strings,args,labels,usetex: {'%s_%s'%(k,'_'.join(list(set(list(r.values()))))): polynomial(k,r,strings,args,labels,usetex) for k in ['inputs','outputs','terms',['inputs','terms']] for r in [{'partial':x} for x in args['operators'] if x not in ['partial']]},
			'chebyshev': lambda strings,args,labels,usetex: {'%s_%s'%(k,'_'.join(list(set(list(r.values()))))): chebyshev(k,r,strings,args,labels,usetex) for k in ['inputs','outputs','terms',['inputs','terms']] for r in [{'partial':x} for x in args['operators'] if x not in ['partial']]},
			'hermite': lambda strings,args,labels,usetex: {'%s_%s'%(k,'_'.join(list(set(list(r.values()))))): hermite(k,r,strings,args,labels,usetex) for k in ['inputs','outputs','terms',['inputs','terms']] for r in [{'partial':x} for x in args['operators'] if x not in ['partial']]},
			'legendre': lambda strings,args,labels,usetex: {'%s_%s'%(k,'_'.join(list(set(list(r.values()))))): legendre(k,r,strings,args,labels,usetex) for k in ['inputs','outputs','terms',['inputs','terms']] for r in [{'partial':x} for x in args['operators'] if x not in ['partial']]},
			}



		strings = {}
		types = {}

		strings.update(texstrings)


		for func in funcs:
			values = funcs[func](strings,args,labels,usetex)
			types[func] = values
			strings.update({s: values[k][s] for k in values for s in values[k] if s not in strings})
		


		self.args = args
		self.strings  = strings
		self.labels = labels
		self.types = types
		self.prefixes = prefixes
		self.usetex = usetex


		return
	 

	def texify(self,string,texstrings={},texargs={},texlabels={},usetex=True):
		'''
		Render string as Latex string
		Args:
			string(str): String to be rendered
			texstrings(dict): Dictionary of text strings with Latex string values
			texargs(dict): Dictionary of arguments and settings used in Latex rendering
			texlabels(dict): Dictionary of modifier strings for Latex rendering
			usetex(bool): Render as Latex strings, or render as text strings
		Returns:
			Latex rendered string
		'''			

		_string = string

		usetex = usetex if self.usetex is None else self.usetex

		String = ""
		if not isinstance(string,tuple):
			string = (string,)

		string = list(((scinotation(s,decimals=3,zero=True,usetex=usetex) for s in string)))


		for texstring in texstrings:
			self.strings['misc']['strings'].update({texstring:texstrings[texstring]})
		self.args.update(texargs)
		self.labels.update(texlabels)


		replacements = self.args['texreplacements'] if usetex else self.args['replacements']

		for s in string:

			s = scinotation(s,decimals=3,zero=True,usetex=usetex)

			if (0) and (s not in self.strings):
				#print('texify',s,s in self.strings,findstring(s,self.strings,self.types,self.prefixes,self.labels,s))
				pass
			s = findstring(s,self.strings,self.types,self.prefixes,self.labels,replacements,s,usetex=usetex)
			String = ' - '.join([String,s]) if len(String)>0 else s

		if usetex:
			String = '\n'.join([r'$%s$'%(s.replace('$','')) if len(s)>0 else s for s in String.split('\n')])
		else:
			String = '\n'.join([r'%s'%(s.replace('$','')) if len(s)>0 else s for s in String.split('\n')])


		return String    
