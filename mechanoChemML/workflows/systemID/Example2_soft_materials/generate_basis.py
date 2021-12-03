from ufl import *
from dolfin import *
import numpy as np
import h5py as h5
import os


def generate_basis(data_list,results_dir='results/'):
  print('generating basis...')
  q_degree = 10
  zeros = Constant((0.0, 0.0, 0.0))
  mesh=Mesh()
  hdf5=HDF5File(MPI.comm_world, results_dir+data_list[0]+'.h5','r')
  hdf5.read(mesh,'/mesh',False)

  V = VectorFunctionSpace(mesh, "Lagrange", 1)
  u = Function(V,name='u') 
  #rectangular
  x_0=0
  x_1= 10
  y_0=0
  y_1=2

  BC1 =  CompiledSubDomain("near(x[0], side) && on_boundary", side = x_0 )
  BC2 =  CompiledSubDomain("near(x[0], side) && on_boundary", side = x_1 )
  BC3 =  CompiledSubDomain("near(x[1], side) && on_boundary", side = y_0 )
  BC4 =  CompiledSubDomain("near(x[1], side) && on_boundary", side = y_1 )

  boundary_subdomains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
  boundary_subdomains.set_all(0)
  BC2.mark(boundary_subdomains,1)
  BC3.mark(boundary_subdomains,2)
  BC4.mark(boundary_subdomains,3)

  dss = ds(subdomain_data=boundary_subdomains)

  v = TestFunction(V) 
  # Kinematics
  d=len(u)
  I = Identity(d)             # Identity tensor
  Fe = I + grad(u)          # Deformation gradient    
  
  J=det(Fe)             
  C = Fe.T*Fe                # Right Cauchy-Green tensor
  invC=inv(C)              
  I1 = tr(C)
  I2=0.5*(I1*I1-tr(C*C) )
  I3  = det(C)

  # fiber direction
  a=Constant((1.0,0.0,0.0))
  A=outer(a, a)
  I4=dot(a,C*a)
  I5=dot(a,C*C*a)
  
  barI1=J**(-2/3)*I1
  barI2=J**(-4/3)*I2
  barI4=J**(-2/3)*I4
  barI5=J**(-4/3)*I5

  bcs=[]
  def assemble_R(basis_id):
    if basis_id==0:
      R = -dot(T,v)*dss(surface)
    elif basis_id==1:
      S=(J**(-2./3.)*I-1./3.*barI1*invC)
      P=Fe*2*S
      R=inner(P,grad(v))*dx(metadata={'quadrature_degree': q_degree})
    elif basis_id==2:
      S=2*(barI1-3)*(J**(-2./3.)*I-1./3.*barI1*invC)
      P=Fe*2*S
      R=inner(P,grad(v))*dx(metadata={'quadrature_degree': q_degree})
    elif basis_id==3:
      S=(J**(-2./3.)*barI1*I- J**(-4./3.)*C-2./3.*barI2*invC )
      P=Fe*2*S
      R=inner(P,grad(v))*dx(metadata={'quadrature_degree': q_degree})
    elif basis_id==4:
      S=2*(barI2-3)*(J**(-2./3.)*barI1*I- J**(-4./3.)*C-2./3.*barI2*invC )
      P=Fe*2*S
      R=inner(P,grad(v))*dx(metadata={'quadrature_degree': q_degree})
    elif basis_id==5:
      S=(J**(-2./3.)*A-1./3.*barI4*invC)
      P=Fe*2*S
      R=inner(P,grad(v))*dx(metadata={'quadrature_degree': q_degree})
    elif basis_id==6:
      S=2*(barI4-1)*(J**(-2./3.)*A-1./3.*barI4*invC)
      P=Fe*2*S
      R=inner(P,grad(v))*dx(metadata={'quadrature_degree': q_degree})
    elif basis_id==7:
      S=(J-1)*J*invC
      P=Fe*2*S
      R=inner(P,grad(v))*dx(metadata={'quadrature_degree': q_degree})
    elif basis_id==8:
      S=2*(J-1)*(J-1)*(J-1)*J*invC
      P=Fe*2*S
      R=inner(P,grad(v))*dx(metadata={'quadrature_degree': q_degree})
   
    R_=assemble(R)
    for bc in bcs:
      bc.apply(R_)
    R_value=R_.get_local()
    return R_value
  
  #shape_list=['extension','extension_2','bending','torsion']
  bcl_1 = DirichletBC(V, zeros, BC1)
  bcl_1_y = DirichletBC(V, zeros, BC3)
  for shape in data_list:
    bcs=[bcl_1]
    surface=1
    if shape=='extension_2':
      force=80
      T = Constant((0, force, 0))
      surface=3
      bcs=[bcl_1_y]
    if shape=='extension':
      force=40 
      T = Constant((force, 0, 0))
    elif shape=='bending':
      force=0.5
      T = Constant((0, 0, force) )
    elif shape=='torsion':
      force=5
      T = Expression(("0"," f*sqrt( (x[1]-1)*(x[1]-1)+(x[2]-1)*(x[2]-1) )*sin(atan2(x[2]-1,x[1]-1))","-f*sqrt((x[1]-1)*(x[1]-1)+(x[2]-1)*(x[2]-1) )*cos(atan2(x[2]-1,x[1]-1)) "),degree =1,f=force)
      
    
    file=results_dir+shape+'.h5'
    print('save to',file)
    hdf5=HDF5File(MPI.comm_world, file,'r')
    hdf5.read(u, 'u')
    # sigma=0
    # u.vector()[:]=u.vector()[:]+np.random.normal(0,sigma,u.vector()[:].size)
    basis=np.column_stack([assemble_R(basis_id) for basis_id in range(9)])

    np.savetxt('basis/'+shape+'.dat',basis)