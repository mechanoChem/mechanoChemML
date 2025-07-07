from ufl import *
from dolfin import *
import numpy as np
from dolfin_adjoint import *
import os
import random
import sys
from functools import reduce
from stepwise_minimization import *
from numpy import linalg as LA
from boundaryConditions import *
from itertools import product
import time

parameters["mesh_partitioner"] = "ParMETIS"
parameters["dof_ordering_library"] = "Boost"

def forward_loss(theta, tendonName, condition, disp, meshName, 
                 lossFactor1, lossFactor2, listFactor):

 #################################### Define mesh ############################
  tendonDate = tendonName[6:]
  
  mesh=Mesh()
  with XDMFFile(tendonDate + "/convergenceAnalysis_V2/mesh/Tendon" + tendonDate+ 
                "_" + condition + "_" + meshName + ".xdmf") as infile:
    infile.read(mesh)
  
  V = VectorFunctionSpace(mesh, "Lagrange", 2)
  W = FunctionSpace(mesh, "DG", 0)

  x=SpatialCoordinate(mesh)
  dof_coordinates = V.tabulate_dof_coordinates()                    
  dof_coordinates.resize((V.dim(), mesh.geometry().dim()))                   
  
  ######################### Define regions to apply boundary conditions  ##########################
  dim = mesh.topology().dim()
  facetfct = MeshFunction('size_t', mesh, dim - 1)
  facetfct.set_all(0)
  mesh.init(dim - 1, dim) # relates facets to cells

  equationDict = globals()[tendonName]["equations"][condition].items()
  keyList = list(globals()[tendonName]["equations"][condition].keys())
  
  for f in facets(mesh):
    for key, equation in equationDict:
      if equation(f):
        facetfct[f.index()] = keyList.index(key) + 1

  n = FacetNormal(mesh)

  # verify that the bottom side is encastered and not rolling - verified
  q_degree = 5

  # Define functions
  u = Function(V,name='u')
  u_real = Function(V,name='u')
  boneDisp = Function(V, name='boneDisp')
  
  file_U = HDF5File(MPI.comm_world, tendonDate + 
                    "/convergenceAnalysis_V2/displacementHighRes/Tendon" + tendonDate +
                    "_" + meshName + "_U_" + condition + "_" + str(disp) + "_CG2.h5", 'r')
  file_U.read(u_real, 'U_' + condition + str(disp))
  file_U.read(boneDisp, 'U_' + condition + str(disp))

  # This is to calculate the predictions
  v = TestFunction(V) 
  # Kinematics
  d = len(u)
  I = Identity(d)          # Identity tensor
  F = I + grad(u)          # Deformation gradient    

  J=det(F)             
  C = F.T*F               # Right Cauchy-Green tensor

  # Invariants
  invC=inv(C)              
  I1 = tr(C)
  I2=0.5*(I1*I1-tr(C*C) )
  I3  = det(C)

  # Fiber direction
  file_UVW = HDF5File(MPI.comm_world, tendonDate + 
                      "/convergenceAnalysis_V2/UVWHighRes/Tendon" + tendonDate +
                      "_" + meshName + "_UVW_" + condition + "_CG2.h5", 'r')

  a = Function(V, name='UVW')
  file_UVW.read(a, '/UVW_' + condition)

  I4=dot(a,C*a)

  barI1=J**(-2/3)*I1
  barI2=J**(-4/3)*I2

  dx_ = dx(metadata={'quadrature_degree': q_degree})
  
  I14 = theta[4]*I1+(1.-3.*theta[4])*I4-1.
  
  force_used = np.loadtxt(tendonDate + "/forceData/" + tendonDate + "_MedianForceData_" + \
                          condition + "_" + str(disp) + "mm.txt")

  P=2*F*(theta[0]*(J-1)*J*invC + \
       theta[1]*(J**(-2./3.)*I-1./3.*barI1*invC) + \
        theta[2]*I14*exp(theta[3]*I14*I14)*(theta[4]*I + \
          (1.-3.*theta[4])*outer(a,a)))

  dss = Measure("ds", domain=mesh, subdomain_data=facetfct,
                subdomain_id=keyList.index("tendon")+1, metadata={'quadrature_degree': q_degree})

  if MPI.comm_world.rank == 0:
    print("Condition: ", condition)
    print("Load: ", disp)
  
  #Define traction on top surface
  normal_to_surf = n
  for i,factor in enumerate(listFactor):
    if MPI.comm_world.rank == 0:
      print("Step: ", factor)

    valueBC = Function(V)
    valueBC.assign(project(factor*boneDisp, V))
    # valueBC.vector()[:] = factor*boneDisp.vector()

    bcs = []

    for key in keyList:
      bc_key = DirichletBC(V, valueBC, facetfct, keyList.index(key) + 1)
      bcs.append(bc_key)
    
    traction = dot(P, normal_to_surf)

    R = inner(P,grad(v))*dx_ - dot(traction,v)*dss #- dot(u,normal_to_surf)*dssHead
    Jac=derivative(R, u)
    
    if MPI.comm_world.rank == 0:
      print('===============PRE load finished===============')
    
    problem = NonlinearVariationalProblem(R, u, bcs, Jac)
    solver = NonlinearVariationalSolver(problem)

    prm = solver.parameters
    prm["newton_solver"]["absolute_tolerance"] = 1E-8
    prm["newton_solver"]["relative_tolerance"] = 1E-9
    prm["newton_solver"]["maximum_iterations"] = 10
    # if i == 0:
    #   prm['newton_solver']['relaxation_parameter'] = 0.5
    #   prm["newton_solver"]["maximum_iterations"] = 100
    # else:
    #   prm['newton_solver']['relaxation_parameter'] = 1.
    #   prm["newton_solver"]["maximum_iterations"] = 10
    prm["newton_solver"]["linear_solver"] = 'mumps'
    prm["newton_solver"]["error_on_nonconvergence"] = False
    solver.solve()
  
  load_dir = Constant((-1.0, 0.0, 0.0))
  loadOnFace = dot(traction, load_dir)*dss # This is what needs to change with the other loading modes
  P_ext = force_used
  
  ######################## Calculate losses ##########################
  
  # Calculate maximum displacement to make loss dimensionless
  dim_ureal = u_real.function_space().num_sub_spaces()
  umax_real = u_real.vector()[:]
  umax_real = umax_real.reshape((-1, dim_ureal))
  umax_real = np.linalg.norm(umax_real, axis=1)
  umax_real = np.max(umax_real)

  comm = u_real.function_space().mesh().mpi_comm()
  max_global = MPI.max(comm, umax_real)

  volume = assemble(Constant((1))*dx(domain=mesh))

  alphaMatrix = Constant(((1., 0., 0.),(0., 1., 0.),(0., 0., 1.)))
  loss1 = assemble((inner(u - u_real, dot(alphaMatrix,u - u_real))/(max_global**2))*dx)*lossFactor1/volume
  loss2 = pow(assemble(loadOnFace) - P_ext,2)*lossFactor2/P_ext**2

  #penaltySum = sum(float(theta[i].values()[0])**2 for i in range(4))

  loss = loss1 + loss2 #+ 0.5*lam*penaltySum
  if MPI.comm_world.rank == 0:
    print("Loss1 = ", loss1)
    print("Loss2 = ", loss2)
    print("Loss = ", loss)
  
  return loss

def addLosses(theta, tendonStamp, meshName, lossFactor1, lossFactor2, 
              combinationList, listFactor):
  loss = 0.
  for k in range(len(combinationList)):
    loss += forward_loss(theta, "Tendon" + tendonStamp, combinationList[k][0],
                        combinationList[k][1], meshName, lossFactor1,
                        lossFactor2, listFactor)
  return loss

if __name__ == "__main__":
    
  ###################################### Adjoint running #########################
  method = "L-BFGS-B"
  #method = "SLSQP"
  #method = 'BFGS'
  
  tendonStampList = ["20231012", "20231017", "20231107", "20231114", 
                     "20231201", "20231206", "20231212", "20240503", 
                     "20240517", "20241127_1", "20241127_2", "20241219"]
  indexList = [0,1,2,3,4,5,6,7,8,9,10,11]
  indexList = [0]
  conditionList = ["Intact", "Torn"]
  dispList = ["1", "2"]
  combinationList = np.array(list(product(conditionList, dispList)))
  combinationList = combinationList[1:,] # this leaves the intact set, 1 mm load out

  lossFactor1 = 1.
  lossFactor2 = 1.e-2

  listFactor = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]

  lam = 0.
  meshName = "Coarse"
  identifier = 'LF1E_6_NoBounds_'+ meshName + 'Mesh' 
  
  paramsFile = 'AllCoeffs_AllTendons_' + identifier + '_HGOHighOrderI1I2_SquaredNorm_PureHGO.dat'
  data = np.loadtxt(paramsFile)
  for index in indexList:
    adjointTrialsFile = tendonStampList[index] + '/results/HGOHighOrderI1I2_SquaredNorm/' + \
                tendonStampList[index] + "adjoint_log_"+ identifier + "_LF21E_2.dat"
    derivativeFile = tendonStampList[index] + '/results/HGOHighOrderI1I2_SquaredNorm/' + \
                tendonStampList[index] + "derivative_log_"+ identifier + "_LF21E_2.dat"
    def eval_cb_save(m):
      if MPI.comm_world.rank == 0:
        print ("m = ", m)
        with open(adjointTrialsFile, 'a') as solver_log:
          np.savetxt(solver_log, m[None, :])

    def derivative_cb(j, dj, m):
      if MPI.comm_world.rank == 0:
        print("j = %f, dj = %s, m = %s." %(j, [float(k) for k in dj], [float(k) for k in m]))
        print(f"grad norm = {np.linalg.norm([float(k) for k in dj]):.2e}")
        combined = np.array([j] + [float(k) for k in dj] + [float(k) for k in m] + [np.linalg.norm([float(k) for k in dj])])
        with open(derivativeFile, 'a') as derivativeLog:
          np.savetxt(derivativeLog, combined[None, :])
      return dj

    if MPI.comm_world.rank == 0:
      print("Solving tendon: ", tendonStampList[index])

    parameters_value = data[index,:]
    
    theta=[0]*len(parameters_value)
    for j in range(len(parameters_value)):
        theta[j]=Constant(parameters_value[j])
    
    if MPI.comm_world.rank == 0:
      print("parameters_value=",parameters_value)
    
    control_index = [0,1,2,3,4]
    bounds=np.zeros((2,len(control_index)))
    bounds[0,:]= 0.	#All terms positive
    bounds[1,:]= np.inf
    bounds[0,0] = parameters_value[0]
    bounds[1,0] = parameters_value[0]*5.
    bounds[0,1] = parameters_value[1]*0.1
    bounds[1,1] = parameters_value[1]*10.
    bounds[0,2] = parameters_value[2]
    bounds[1,2] = parameters_value[2]*20.
    bounds[0,3] = parameters_value[3]
    bounds[1,3] = parameters_value[3]*50.
    bounds[1,4] = 1./3.

    control_parameter=[Control(theta[i]) for i in control_index]
    reduced_functional = ReducedFunctional(addLosses(theta, tendonStampList[index],
                     meshName, lossFactor1, lossFactor2, combinationList,
                     listFactor), control_parameter, derivative_cb_post=derivative_cb)

    ###Starting adjoint
    if MPI.comm_world.rank == 0:
      print("Starting adjoint for tendon: ", tendonStampList[index])
    
    results_opt = minimize(reduced_functional, method = method, bounds=bounds,
                    tol = 1e-8, options = {'ftol':1.0e-11, 'gtol':1.0e-11, 'maxiter':200, 
                               "disp": True},callback = eval_cb_save)
    if MPI.comm_world.rank == 0:
      print('==========Adjoint Finished==========')
      
    results_opt=np.array(results_opt)
    if MPI.comm_world.rank == 0:
      print('Optimized results = ',results_opt)
    
    alphaMatIdentifier = "1"
    
    save_to_file = str(tendonStampList[index]) + '/results/HGOHighOrderI1I2_SquaredNorm/' \
            + str(tendonStampList[index]) + '_adjoint_lam_'\
                + str(lam) + '_' + identifier + '_alpha_' + \
                    alphaMatIdentifier + '_LF21E_2.dat'
    np.savetxt(save_to_file,results_opt)