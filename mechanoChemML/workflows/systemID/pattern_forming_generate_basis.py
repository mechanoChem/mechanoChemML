from ufl import *
from dolfin import *
import numpy as np
import h5py as h5
import os


def generate_basis(data_list,results_dir='results'):
  mesh=Mesh()
  hdf5=HDF5File(MPI.comm_world, results_dir+'/data.h5','r')
  hdf5.read(mesh,'/mesh',False)
  P1 = FiniteElement('P', triangle, 1)
  element = MixedElement([P1, P1])
  V = FunctionSpace(mesh, element)
  #
  # Define functions
  C_all = Function(V)
  C_all_n = Function(V)
  C1, C2 = split(C_all)
  C1_n, C2_n = split(C_all_n)

  ############
  #residual
  ############

  bcs=[]
  w1, w2 = TestFunctions(V)
  grad_w1=grad(w1)
  grad_w2=grad(w2)
  grad_C1=grad(C1)
  grad_C2=grad(C2)
  dt=Constant(0.25)
  def assemble_R(basis_id):
    if basis_id==0:
      R = -inner(grad_w1,grad_C1)*dx
    elif basis_id==1:
      R = -inner(grad_w1,grad_C2)*dx
    elif basis_id==2:
      R = 1*w1*dx
    elif basis_id==3:
      R = C1*w1*dx
    elif basis_id==4:  
      R = C2*w1*dx
    elif basis_id==5:
      R = C1*C1*C2*w1*dx
    elif basis_id==6:
      R = -inner(grad_w2,grad_C1)*dx
    elif basis_id==7:
      R = -inner(grad_w2,grad_C2)*dx
    elif basis_id==8:
      R = 1*w2*dx
    elif basis_id==9:
      R = C1*w2*dx
    elif basis_id==10:  
      R = C2*w2*dx
    elif basis_id==11:
      R = C1*C1*C2*w2*dx
    elif basis_id==12:
      R= (C1-C1_n)/dt*w1*dx
    elif basis_id==13:
      R= (C2-C2_n)/dt*w2*dx
    
      
    R_=assemble(R)
    for bc in bcs:
      bc.apply(R_)
    R_value=R_.get_local()

    return R_value

  sigma=0
  basis_path='basis'
  if not os.path.isdir(basis_path):
    os.mkdir(basis_path)
  print('save operators to '+basis_path)
  for step in data_list:
    if step==data_list[-1]:
      break
    hdf5.read(C_all,'C_all/vector_'+str(step+1))
    hdf5.read(C_all_n,'C_all/vector_'+str(step))


    #add noise
    C_all.vector()[:]=C_all.vector()[:]+np.random.normal(0,sigma,C_all.vector()[:].size)
    C_all_n.vector()[:]=C_all_n.vector()[:]+np.random.normal(0,sigma,C_all_n.vector()[:].size)
    basis=np.column_stack([assemble_R(basis_id) for basis_id in range(14)])
    #
    
    print('saving operators at time step ',step+1)
    np.savetxt('basis/basis_sigma_'+str(sigma)+'_step_'+str(step+1)+'.dat',basis)





