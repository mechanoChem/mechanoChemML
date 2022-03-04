#!/usr/bin/env python

import os,sys,json,copy

import dolfin
from dolfin import MPI,XDMFFile,HDF5File
from dolfin import IntervalMesh,CompiledSubDomain, MeshFunction, CellVolume
from dolfin import VectorFunctionSpace, FunctionSpace,Function,TestFunction,Expression
from dolfin import project,inner,assemble,vertex_to_dof_map,dx, Measure
from dolfin import DirichletBC
from dolfin import NonlinearVariationalProblem,NonlinearVariationalSolver
from dolfin import dot, Constant, diff, div, grad,derivative

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt



# Logging
import logging,logging.handlers
log = 'warning'


rootlogger = logging.getLogger()
rootlogger.setLevel(getattr(logging,log.upper()))
stdlogger = logging.StreamHandler(sys.stdout)
stdlogger.setLevel(getattr(logging,log.upper()))
rootlogger.addHandler(stdlogger)	


logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging,log.upper()))



class dns:
	def __init__(self):
		self.defaults = {
			'path':'data',
                        'file':{'xdmf':'data','hdf5':'data','data':'data','log':'log','plot':'plot','observables':'data'},
                        'ext':{'xdmf':'xdmf','hdf5':'h5','data':'csv','log':'log','plot':'pdf','observables':'csv'},
			'fields':{
				'c':{
					'type':'scalar','dimension':1,'space':'V',
					'initial':'(1.0/5.0)*pow(x[0],5) - (1.0/3.0)*pow(x[0],3) + 1/(5.0*3.0)','bcs':'neumann',
					'plot':{'x':['Time'],'y':['Phi_1p'],'texify':{'Time':'t','x':'x','Phi_1p':'\\varphi'},'mplstyle':'plot.mplstyle','animate':1},
					}
				},
			'spaces': {
				'V':{'type':'FunctionSpace','family':'CG','degree':2},
				'W':{'type':'VectorFunctionSpace','family':'CG','degree':1},
				'Z':{'type':'FunctionSpace','family':'DG','degree':0},
				},
			'potential':{
				'expression':'0.25 - 0.5 * pow(c,2) + 0.25 * pow(c,4)',
				'degree':4,
				'derivative':'-c + pow(c,3)',
				'kwargs':{'c':None}
				},
			'num_elem':10,
			'L_0':0,
			'L_1':1,
			'D':1,
			'alpha':1,
			'dt':0.1,
			'N':10,
			'tol':1e-5,
			'plot':True
		}

		return


	def set_parameters(self,parameters={},reset=False):
		field = '_parameters'
		default = {}
		if not hasattr(self,field) or reset:
			setattr(self,field,default)
		if isinstance(parameters,str):
			with open(parameters,'r') as file:
				getattr(self,field).update(json.load(file))
		elif isinstance(parameters,dict):
			getattr(self,field).update(parameters)

		getattr(self,field).update({parameter: self.defaults[parameter] 
						   for parameter in self.defaults if parameter not in getattr(self,field)})

		self.set_paths()
		return

	def get_parameters(self):
		field = '_parameters'		
		return getattr(self,field)

	def set_data(self,data={},reset=False):
		field = '_data'
		default = {key: [] for key in data}

		paths = self.get_paths()		

		if not hasattr(self,field) or reset:
			setattr(self,field,default)
		for key in data:
			try:
				getattr(self,field)[key].append(data[key])
			except:
				getattr(self,field)[key] = [data[key]] if data[key] not in [[]] else data[key]


		for key in getattr(self,field):
			path = paths['data'].replace('.','_%s.'%(str(key)))
			self.dump({key:np.array(getattr(self,field)[key])},path)
		return

	def get_data(self):
		field = '_data'
		default = {}
		paths = self.get_paths()		
		parameters = self.get_parameters()
		if not hasattr(self,field):
			try:
				values = {}
				for key in parameters['fields']:
					path = paths['data'].replace('.','_%s.'%(str(key)))
					values.update({key: self.load(path)})
			except:
				values = default
			setattr(self,field,values)
		else:
			values = getattr(self,field)
		return values


	def set_observables(self,observables={},reset=False):
		header = {
			'Time':'Time',
			'total_energy':'TE','chi':'Chi',
			'total_energy_p':'TE_P','chi_p':'Chi_P','total_energy_m':'TE_M','chi_m':'Chi_M',
			'Phi_0p':'Phi_0P','Phi_1p':'Phi_1P','Phi_2p':'Phi_2P','Phi_3p':'Phi_3P','Phi_4p':'Phi_4P','Phi_5p':'Phi_5P',
			'Phi_0m':'Phi_0M','Phi_1m':'Phi_1M','Phi_2m':'Phi_2M','Phi_3m':'Phi_3M','Phi_4m':'Phi_4M','Phi_5m':'Phi_5M',					
			'gradient_energy':'GradE','landau_energy':'LanE','diffusion_energy':'LapC','spinodal_energy':'dLan',
			'gradient_energy_p':'GradE_P','landau_energy_p':'LanE_P','diffusion_energy_p':'LapC_P','spinodal_energy_p':'dLan_P',					
			'gradient_energy_m':'GradE_M','landau_energy_m':'LanE_M','diffusion_energy_m':'LapC_M','spinodal_energy_m':'dLan_M',										
		}	

		field = '_observables'
		default = {}

		paths = self.get_paths()		
		if not hasattr(self,field) or reset:
			setattr(self,field,default)

		for key in observables:
			for observable in observables[key]:
				try:
					getattr(self,field)[key][observable].append(observables[key][observable])
				except:
					try:
						getattr(self,field)[key][observable] = [observables[key][observable]]
					except:
						getattr(self,field)[key] = {}
						getattr(self,field)[key][observable] = [observables[key][observable]]



		for key in getattr(self,field):
			path = paths['observables'].replace('.','_%s.'%(str(key)))
			self.dump({key:{header.get(k,k): getattr(self,field)[key][k] for k in getattr(self,field)[key]}},path)
		return


	def get_observables(self):
		field = '_observables'
		default = {}
		paths = self.get_paths()		
		parameters = self.get_parameters()
		if not hasattr(self,field):
			try:
				values = {}
				for key in parameters['fields']:
					path = paths['observables'].replace('.','_%s.'%(str(key)))
					values.update({key: self.load(path)})
			except:
				values = default
			setattr(self,field,values)
		else:
			values = getattr(self,field)
		return values


	def set_paths(self):
		field = '_paths'
		parameters = self.get_parameters()
		paths = {file:os.path.abspath(os.path.join(parameters['path'],'%s.%s'%(parameters['file'][file],parameters['ext'][file])))
					for file in parameters['file']}



		for path in paths:
			directory = os.path.dirname(paths[path])
			if not os.path.exists(directory):
				os.makedirs(directory)


		setattr(self,field,paths)

		return

	def get_paths(self):
		field = '_paths'
		return getattr(self,field)


	def set_logger(self,logger):

		field = '_logger'

		if logger is None:
			logger = logging.getLogger(__name__)
			logger.setLevel(getattr(logging,log.upper()))

		filelogger = logging.handlers.RotatingFileHandler(self.get_paths()['log'])
		fileloggerformatter = logging.Formatter(
			fmt='%(asctime)s: %(message)s',
			datefmt='%Y-%m-%d %H:%M:%S')
		filelogger.setFormatter(fileloggerformatter)
		filelogger.setLevel(getattr(logging,log.upper()))
		if len(rootlogger.handlers) == 2:
			rootlogger.removeHandler(rootlogger.handlers[-1])
		rootlogger.addHandler(filelogger)

		setattr(self,field,getattr(logger,log))

		return


	def get_logger(self):
		field = '_logger'
		return getattr(self,field)

	def load(self,path,wr='r'):
		ext = path.split('.')[-1]
		if ext in ['csv']:
			data = pd.read_csv(path)
		elif ext in ['npy']:
			data = np.load(path)
		return data

	def dump(self,data,path,wr='w'):
		for key in data:
			if isinstance(data[key],dict):

				values = data[key]
				header = list(values)
				values = np.atleast_2d([values[observable] for observable in values]).T
				
				# kwargs = {'fname':path,'X':values,'header':header,'fmt':'%0.8f','delimiter':',','comments':''}
				# np.savetxt(**kwargs)

				values = pd.DataFrame(values,columns=header)
				kwargs = {'index':False}
				values.to_csv(path,**kwargs)

			elif isinstance(data[key],np.ndarray):
				np.save(path,data[key])

		return

	def simulate(self,parameters=None,logger=None):
		self.model(parameters,logger)
		

	def model(self,parameters=None,logger=None):


		def format(arr):
			return np.array(arr)

		# Setup parameters, logger
		self.set_parameters(parameters)
		self.set_logger(logger)

		# Get parameters
		parameters = self.get_parameters()


		#SS#if not parameters.get('simulate'):
		#SS#	return


		# Setup data
		self.set_data(reset=True)
		self.set_observables(reset=True)



		# Show simulation settings
		self.get_logger()('\n\t'.join(['Simulating:',*['%s : %r'%(param,parameters[param] if not isinstance(parameters[param],dict) else list(parameters[param])) 
														for param in ['fields','num_elem','N','dt','D','alpha','tol']]]))


		# Data mesh
		mesh = IntervalMesh(MPI.comm_world,parameters['num_elem'],parameters['L_0'],parameters['L_1'])

		# Fields
		V = {}
		W = {}
		fields_n = {}
		fields = {}		
		w = {}
		fields_0 = {}
		potential = {}
		potential_derivative = {}
		bcs = {}
		R = {}
		J = {}
		# v2d_vector = {}
		observables = {}
		for field in parameters['fields']:

			# Define functions
			V[field] = {}
			w[field] = {}
			for space in parameters['spaces']:
				V[field][space] = getattr(dolfin,parameters['spaces'][space]['type'])(mesh,parameters['spaces'][space]['family'],parameters['spaces'][space]['degree'])
				w[field][space] = TestFunction(V[field][space])

			
			space = V[field][parameters['fields'][field]['space']]
			test = w[field][parameters['fields'][field]['space']]

			fields_n[field] = Function(space,name='%sn'%(field))
			fields[field] = Function(space,name=field)


			# Inital condition
			fields_0[field] = Expression(parameters['fields'][field]['initial'],element = space.ufl_element())
			fields_n[field] = project(fields_0[field],space)
			fields[field].assign(fields_n[field])


			# Define potential
			if parameters['potential'].get('kwargs') is None:
				parameters['potential']['kwargs'] = {}
			for k in parameters['potential']['kwargs']:
				if parameters['potential']['kwargs'][k] is None:
					parameters['potential']['kwargs'][k] = fields[k]

			potential[field] = Expression(parameters['potential']['expression'], degree=parameters['potential']['degree'],**parameters['potential']['kwargs'])
			potential_derivative[field] = Expression(parameters['potential']['derivative'], degree=parameters['potential']['degree']-1,**parameters['potential']['kwargs'])

			#Subdomain for defining Positive grain
			sub_domains = MeshFunction('size_t', mesh, mesh.topology().dim(), 0)

			#BC condition
			bcs[field] = []
			if parameters['fields'][field]['bcs'] == 'dirichlet':
				BC_l =  CompiledSubDomain('near(x[0], side) && on_boundary', side = parameters['L_0'])
				BC_r=  CompiledSubDomain('near(x[0], side) && on_boundary', side = parameters['L_1'])
				bcl = DirichletBC(V, fields_n[field], BC_l)
				bcr = DirichletBC(V, fields_n[field], BC_r)
				bcs[field].extend([bcl,bcr])
			elif parameters['fields'][field]['bcs'] == 'neumann':
				bcs[field].extend([])


			# Residual and Jacobian
			R[field] = (((fields[field]-fields_n[field])/parameters['dt']*test*dx) + 
						(inner(parameters['D']*grad(test),grad(fields[field]))*dx) + 
						(parameters['alpha']*potential_derivative[field]*test*dx))
			
			J[field] = derivative(R[field], fields[field])



			# Observables
			observables[field] = {}


		files = {
			'xdmf':XDMFFile(MPI.comm_world,self.get_paths()['xdmf']),
			'hdf5':HDF5File(MPI.comm_world,self.get_paths()['hdf5'],'w'),
			}			

		files['hdf5'].write(mesh,'/mesh')
		eps = lambda n,key,field,observables,tol: (n==0) or (abs(observables[key]['total_energy'][n]-observables[key]['total_energy'][n-1])/(observables[key]['total_energy'][0]) > tol)
		flag = {field: True for field in parameters['fields']}		
		tol = {field: parameters['tol'][field] if isinstance(parameters['tol'],dict) else parameters['tol'] for field in parameters['fields']}		
		phases = {'p':1,'m':2}
		n=0
		problem = {}
		solver = {}
		while(n<parameters['N'] and any([flag[field] for field in parameters['fields']])):

			self.get_logger()('Time: %d'%(n))
			
			for field in parameters['fields']:

				if not flag[field]:
					continue

				# Solve
				problem[field] = NonlinearVariationalProblem(R[field],fields[field],bcs[field],J[field])
				solver[field] = NonlinearVariationalSolver(problem[field])
				solver[field].solve()


				# Get field array
				array = assemble((1/CellVolume(mesh))*inner(fields[field], w[field]['Z'])*dx).get_local()


 
				# Observables
				observables[field]['Time'] = parameters['dt']*n

				# observables[field]['energy_density'] = format(0.5*dot(grad(fields[field]),grad(fields[field]))*dx)
				observables[field]['gradient_energy'] = format(assemble(0.5*dot(grad(fields[field]),grad(fields[field]))*dx(domain=mesh)))
				observables[field]['landau_energy'] = format(assemble(potential[field]*dx(domain=mesh)))
				observables[field]['diffusion_energy'] = format(assemble(project(div(project(grad(fields[field]),V[field]['W'])),V[field]['Z'])*dx(domain=mesh))) #Diffusion part of chemical potential
				observables[field]['spinodal_energy'] = format(assemble(potential_derivative[field]*dx(domain=mesh))) #Spinodal part of chemical potential

				observables[field]['total_energy'] = parameters['D'] * observables[field]['gradient_energy'] + parameters['alpha'] *observables[field]['landau_energy']
				observables[field]['chi'] =  parameters['alpha']*observables[field]['diffusion_energy'] - parameters['D']*observables[field]['spinodal_energy']





				# Phase observables
				sub_domains.set_all(0)
				sub_domains.array()[:] = np.where(array > 0.0, phases['p'],phases['m'])
				
				phases_dxp = Measure('dx', domain = mesh, subdomain_data = sub_domains)
				for phase in phases:
					dxp = phases_dxp(phases[phase])

					observables[field]['Phi_0%s'%(phase)] = format(assemble(1*dxp))
					observables[field]['Phi_1%s'%(phase)] = format(assemble(fields[field]*dxp))
					observables[field]['Phi_2%s'%(phase)] = format(assemble(fields[field]*fields[field]*dxp))
					observables[field]['Phi_3%s'%(phase)] = format(assemble(fields[field]*fields[field]*fields[field]*dxp))
					observables[field]['Phi_4%s'%(phase)] = format(assemble(fields[field]*fields[field]*fields[field]*fields[field]*dxp))
					observables[field]['Phi_5%s'%(phase)] = format(assemble(fields[field]*fields[field]*fields[field]*fields[field]*fields[field]*dxp))

					observables[field]['gradient_energy_%s'%(phase)] = format(assemble(0.5*dot(grad(fields[field]),grad(fields[field]))*dxp))	
					observables[field]['landau_energy_%s'%(phase)] = format(assemble(potential[field]*dxp))
					observables[field]['total_energy_%s'%(phase)] = parameters['D'] * observables[field]['gradient_energy_%s'%(phase)] + parameters['alpha'] *observables[field]['landau_energy_%s'%(phase)]			

					observables[field]['diffusion_energy_%s'%(phase)] = format(assemble(project(div(project(grad(fields[field]),V[field]['W'])),V[field]['Z'])*dxp)) #Diffusion part of chemical potential
					observables[field]['spinodal_energy_%s'%(phase)] = format(assemble(potential_derivative[field]*dxp)) #Spinodal part of chemical potential

					observables[field]['chi_%s'%(phase)] =  parameters['alpha']*observables[field]['spinodal_energy_%s'%(phase)] - parameters['D']*observables[field]['diffusion_energy_%s'%(phase)]


				files['hdf5'].write(fields[field],'/%s'%(field),n)
				files['xdmf'].write(fields[field],n)

				fields_n[field].assign(fields[field])
		
				self.set_data({field: array})
				self.set_observables({field: observables[field]})


				flag[field] = eps(n,field,fields[field],self.get_observables(),tol[field])


			n+=1



		for file in files:
			files[file].close()

		return

	def plot(self):

		parameters = self.get_parameters()
		data = self.get_data()
		observables = self.get_observables()
		paths = self.get_paths()



		plot = parameters.get('plot')

		if plot in [{},None,False]:
			return
		if not isinstance(plot,dict):
			plot = {}


		path = paths['plot']


		for key in parameters['fields']:

			plotting = parameters['fields'][key].get('plot')

			if plotting in [{},None,False]:
				continue

			plotting.update({k:plot[k] for k in plot if k not in plotting})
	
			self.get_logger()('\n'.join(['Plotting %s : %r'%(key,np.shape(data[key]))]))

			N = len(data[key])

			animate = plotting.get('animate')

			if animate:
				path = os.path.join(os.path.dirname(path),'plots_%s'%(key),os.path.basename(path))
				directory = os.path.dirname(path)
				if not os.path.exists(directory):
					os.makedirs(directory)



			with matplotlib.style.context(plotting.get('mplstyle',matplotlib.matplotlib_fname())):

				# Data plot
				fig,ax = plt.subplots()

				if not animate:
					Nrange = range(0,N,20) 
				else:
					Nrange = range(N)


				#Nrange = [150]


				x = 'x'
				v = ['x','t']
				y = key


				
				
				for i in Nrange:

					self.get_logger()('Plotting %s %s'%(key,str(i)))

					size = data[key][i].size
					
					plotprops = {
						'label':r'$t = {%s}_{}$'%(str(i)) if N>0 else None,
						'linewidth':3,			
						'color':'dimgrey'			
						}

					fillprops = [
						{'color':'tab:blue','alpha':0.8,'hatch':r''},
						{'color':'tab:blue','alpha':0.6,'hatch':r''},
					]

					legprops = {
						# 'title':r'\textrm{Time}',
						'ncol':1,'loc':(0.85,0.75),
						'framealpha':0,
						'handletextpad':-2.0, 'handlelength':0,
						}


					X = np.linspace(parameters['L_0'],parameters['L_1'],size)
					Y = data[key][i]


					ax.plot(X,Y,**plotprops)
					plt.fill_between(X[Y>0],0*X[Y>0],Y[Y>0],**fillprops[0])
					plt.fill_between(X[Y<0],0*X[Y<0],Y[Y<0],**fillprops[1])


					ax.set_xlabel(r'$%s_{}$'%(plotting['texify'].get(x,x)))
					ax.set_ylabel(r'${%s}_{}%s$'%(plotting['texify'].get(y,y),'(%s)'%(','.join([plotting['texify'].get(u,u) for u in v])) if len(v)>0 else ''))
					ax.set_xlim(-0.05,1.05)
					ax.set_ylim(-1.2,1.2)
					ax.set_xticks([0,0.25,0.5,0.75,1.0])
					ax.set_yticks([-1,-0.5,0,0.5,1])
					ax.grid(True,alpha=0.7)
					ax.annotate(r'$\large{\varphi}$',xy=(0.11,0.26))
					ax.annotate(r'$\large{\bar{\varphi}}$',xy=(0.6,-0.30))
					if N>0:
						ax.legend(**legprops)						
					if animate:
						file = path.replace('.','_data_%s_%s.'%(str(key),str(i)))
						fig.savefig(file,bbox_inches='tight')
						ax.clear()


				# if N>0 and not animate:
				

				if not animate:
					path = paths['plot'].replace('.','_data_%s.'%(str(key)))
					fig.savefig(path,bbox_inches='tight')

				# Observables plot
				for x,y in zip(plotting['x'],plotting['y']):
					fig,ax = plt.subplots()
					X = observables[key][x]
					Y = observables[key][y]
					ax.plot(X,Y)
					ax.set_xlabel(r'${%s}_{}$'%(plotting['texify'].get(x,x)))
					ax.set_ylabel(r'${%s}_{}$'%(plotting['texify'].get(y,y)))
					path = paths['plot'].replace('.','_observables_%s_%s_%s.'%(str(key),str(x),str(y)))
					fig.savefig(path,bbox_inches='tight')

		return


if __name__ == '__main__':

	if len(sys.argv) > 1: 
		parameters = sys.argv[1]
	else:
		print('usage: dns.py settings.prm')
		exit()

	model=dns()
	model.set_parameters(parameters)
	parameters = copy.deepcopy(model.get_parameters())
	
	#For random initial conditions
	#Set the number of Random Initial Conditions
	N = [i for i in range(0,100)]
	settings = {
		'a0':{'func':np.random.uniform,'args':[0.0,1.0],'kwargs':{},'value':None,'placement':[0,2]},
		'a1':{'func':np.random.uniform,'args':[-0.25,0.25],'kwargs':{},'value':None,'placement':[1]},
	}
	##For fixed initial conditions
	##Set the sample id
	#N = [1102] #Sample ID
	#settings = {
	#	'a0':{'func':lambda *args,**kwargs: 2.0,'args':[0.0,1.0],'kwargs':{},'value':None,'placement':[0,2]},
	#	'a1':{'func':lambda *args,**kwargs: 0.2,'args':[-0.25,0.25],'kwargs':{},'value':None,'placement':[1]},        	
	#}

	funcs = {'fields':{'c':{'initial':'%f /(1+exp( (x[0]-0.5 + %f)/ 0.2 )) - 0.5*%f'} }}

	for i in N:

		params = copy.deepcopy(parameters)

		updates = {}
		for setting in settings:
			settings[setting]['value'] = settings[setting]['func'](*settings[setting]['args'],**settings[setting]['kwargs'])


		values = {k:settings[setting]['value'] for setting in settings for k in settings[setting]['placement']}
		values = tuple([values[k] for k in range(len(values))])
		for func in funcs:
			if isinstance(funcs[func],dict):
				updates[func] = {}
				for k in funcs[func]:
					if isinstance(funcs[func][k],dict):
						updates[func][k] = {}
						for v in funcs[func][k]:
							if '%' in funcs[func][k][v]:	
								updates[func][k][v] = funcs[func][k][v]%(values)							

					else:
						if '%' in funcs[func][k]:	
							updates[func][k] = funcs[func][k]%(values)
			else:
				if '%' in funcs[func]:	
					updates[func] = funcs[func]%(values)

		updates.update({'path':os.path.join(params['path'],'Sample%d'%i)})
		for update in updates:
			if update not in params:
				params[update] = updates[update]
			elif isinstance(params[update],dict):
				for param in updates[update]:
					if isinstance(params[update].get(param),dict):
						params[update][param].update(updates[update][param])
					else:
						params[update].update({param: updates[update][param]})
			else:
				params.update({update:updates[update]})


		model.set_parameters(params)
		model.simulate()
		model.plot()	


