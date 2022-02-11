from ufl import *
from dolfin import *
import numpy as np
import os

def threeField_neo_Hookean(results_dir='results'):
  print('generating data... ')
  set_log_active(False)
  zeros = Constant((0.0, 0.0, 0.0))
  point0=Point(0,0,0)
  point1=Point(10,2,2)
  mesh = BoxMesh(MPI.comm_world,point0,point1,25,5,5)
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

  barI1=J**(-2/3)*I1

  omega1=Constant(0)
  omega2=Constant(0)

  P=2*Fe*(omega1*(J**(-2/3)*I-1./3.*barI1*invC))+omega2*2*(J-1)*J*inv(Fe.T)


  x = SpatialCoordinate(mesh)
  #force=40,0.5,3
  #force
  def loss_eval(m):
    error=0
    omega1.assign(m[0])
    omega2.assign(m[1])
    shape_list=['extension','extension_2','bending','torsion']
    bcl_1 = DirichletBC(V, zeros, BC1)
    bcl_1_y = DirichletBC(V, zeros, BC3)
  
    for shape in shape_list:
      u.vector()[:]=0
      print('running deformation modes: ', shape)
      load_scale_list=[0.01,0.1,0.4,0.8,1]
      hdf5=HDF5File(MPI.comm_world, 'results/'+shape+'.h5','w')
      hdf5.write(mesh,'/mesh')
      for load_scale in load_scale_list:
        bcs=[bcl_1]  
        surface=1
        if shape=='extension_2':
          force=80
          T = Constant((0, force*load_scale, 0))
          surface=3
          bcs=[bcl_1_y]
        if shape=='extension':
          force=40 
          T = Constant((force*load_scale, 0, 0))
        elif shape=='bending':
          force=0.5
          T = Constant((0, 0, force*load_scale) )
        elif shape=='torsion':
          force=5
          T = Expression(("0"," f*sqrt( (x[1]-1)*(x[1]-1)+(x[2]-1)*(x[2]-1) )*sin(atan2(x[2]-1,x[1]-1))","-f*sqrt((x[1]-1)*(x[1]-1)+(x[2]-1)*(x[2]-1) )*cos(atan2(x[2]-1,x[1]-1)) "),degree =1,f=force*load_scale)
      
        R=inner(P,grad(v))*dx-dot(T,v)*dss(surface)
        Jobian=derivative(R, u)
        problem = NonlinearVariationalProblem(R, u, bcs, Jobian)
        solver = NonlinearVariationalSolver(problem)

        prm = solver.parameters      
        solver.solve()
    
      hdf5.write(u,'u')
      file = XDMFFile(MPI.comm_world,'results/visualization/'+shape+'.xdmf')
      file.write(u)


  m=[40.0,400.0]
  loss_eval(m)















