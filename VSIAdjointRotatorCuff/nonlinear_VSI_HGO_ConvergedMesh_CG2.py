from ufl import *
from dolfin import *
import numpy as np
import random
import sys
from functools import reduce
from stepwise_minimization_HGOConverged import *
from numpy import linalg as LA
from boundaryConditions import *
from itertools import product
import os

num_theta=12
num_basis=9
theta=[]
for i in range(num_theta):
  theta.append(Constant(0.0))

def nonlinear_VSI_HGO(target_array, tendonName, condition, disp, loss_factor2, meshName, current_activate_index,
                      target_index):
  #################################### Define mesh ############################
  tendonDate = tendonName[6:]
  
  mesh=Mesh()
  with XDMFFile(tendonDate + "/convergenceAnalysis_V2/mesh/Tendon" + tendonDate+ 
                "_" + condition + "_" + meshName + ".xdmf") as infile:
    infile.read(mesh)
    
  V = VectorFunctionSpace(mesh, "Lagrange", 2)

  x=SpatialCoordinate(mesh)
  dof_coordinates = V.tabulate_dof_coordinates()                    
  dof_coordinates.resize((V.dim(), mesh.geometry().dim()))                          
  
  ######################### Define regions to apply boundary conditions  ##########################
  dim = mesh.topology().dim()
  # print("dim = ", V.dim())
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

  q_degree = 5

  ########################### Define functions, invariants, fiber direction ######################################
  u = Function(V,name='u') 

  v = TestFunction(V) 
  # Kinematics
  d=len(u)
  I = Identity(d)          # Identity tensor
  F = I + grad(u)          # Deformation gradient    
  
  J=det(F)
  C = F.T*F               # Right Cauchy-Green tensor
  B=F*F.T
  
  # Invariants
  invC=inv(C)              
  I1 = tr(C) # volumetric
  I2=0.5*(I1*I1-tr(C*C) ) # biaxial state
  I3  = det(C) # triaxial

  # Fiber direction
  file_UVW = HDF5File(MPI.comm_world, tendonDate + 
                      "/convergenceAnalysis_V2/UVWHighRes/Tendon" + tendonDate +
                      "_" + meshName + "_UVW_" + condition + "_CG2.h5", 'r')

  a = Function(V, name='UVW')
  file_UVW.read(a, '/UVW_' + condition)

  I4=dot(a,C*a)

  barI1=J**(-2./3.)*I1
  barI2=J**(-4./3.)*I2
  logJ = ln(J)

  dss = Measure("ds", domain=mesh, subdomain_data=facetfct,
                subdomain_id=keyList.index("tendon")+1, metadata={'quadrature_degree': q_degree})

  I14 = theta[4]*I1+(1.-3.*theta[4])*I4-1.

  ################## Candidate functions for strain energy density ###########################
  hS0 = (J-1)*J*invC
  # hS0 = 0.5*logJ*invC
  hS1 = (J**(-2./3.)*I-1./3.*barI1*invC)
  # hS1 = 0.5*I - 0.5*invC - (1./3.)*logJ*invC

  # Compressible anisotropic part
  hS2 = theta[3]*I14*exp(theta[3]*I14*I14)*(theta[4]*I + (1.-3.*theta[4])*outer(a,a))

  # I1 terms
  hS3 = 2*(barI1-3)*(J**(-2./3.)*I-1./3.*barI1*invC) # quadratic
  hS4 = 3*(barI1-3)**2*(J**(-2./3.)*I-1./3.*barI1*invC) # cubic
  hS5 = 4*(barI1-3)**3*(J**(-2./3.)*I-1./3.*barI1*invC) # fourth order

  # I2 terms
  hS6 = 2*(barI2-3)*(J**(-2./3.)*barI1*I- J**(-4./3.)*C-2./3.*barI2*invC) # quadratic
  hS7 = 3*(barI2-3)**2*(J**(-2./3.)*barI1*I- J**(-4./3.)*C-2./3.*barI2*invC) # cubic
  hS8 = 4*(barI2-3)**3*(J**(-2./3.)*barI1*I- J**(-4./3.)*C-2./3.*barI2*invC) # fourth order

  # Here we build the basis function
  basis_pool=[0]*num_basis
  basis_pool[0]=theta[0]*inner(2*F*hS0, grad(v) )*dx(metadata=
                                                     {'quadrature_degree': q_degree})
  basis_pool[1]=theta[1]*inner(2*F*hS1, grad(v) )*dx(metadata=
                                                     {'quadrature_degree': q_degree})
  basis_pool[2]=theta[2]*inner(2*F*hS2, grad(v) )*dx(metadata=
                                                     {'quadrature_degree': q_degree})
  basis_pool[3]=theta[5]*inner(2*F*hS3, grad(v) )*dx(metadata=
                                                     {'quadrature_degree': q_degree})
  basis_pool[4]=theta[6]*inner(2*F*hS4, grad(v) )*dx(metadata=
                                                     {'quadrature_degree': q_degree})
  basis_pool[5]=theta[7]*inner(2*F*hS5, grad(v) )*dx(metadata=
                                                     {'quadrature_degree': q_degree})
  basis_pool[6]=theta[8]*inner(2*F*hS6, grad(v) )*dx(metadata=
                                                     {'quadrature_degree': q_degree})
  basis_pool[7]=theta[9]*inner(2*F*hS7, grad(v) )*dx(metadata=
                                                     {'quadrature_degree': q_degree})
  basis_pool[8]=theta[10]*inner(2*F*hS8, grad(v) )*dx(metadata=
                                                     {'quadrature_degree': q_degree})

  # Define residual
  R=0
  for i in range(len(basis_pool)):
    R+=basis_pool[i]

  ########################### Define displacement and force for each tendon ##################

  file_U = HDF5File(MPI.comm_world, tendonDate + 
                    "/convergenceAnalysis_V2/displacementHighRes/Tendon" + tendonDate +
                    "_" + meshName + "_U_" + condition + "_" + str(disp) + "_CG2.h5", 'r')
  force_used = np.loadtxt(tendonDate + "/forceData/" + tendonDate + 
                          "_MedianForceData_" + condition + "_" + str(disp) + "mm.txt")

  P=2*F*(theta[0]*hS0 + theta[1]*hS1 + theta[2]*hS2 + theta[5]*hS3 + theta[6]*hS4 
         + theta[7]*hS5 + theta[8]*hS6 + theta[9]*hS7 + theta[10]*hS8)

  boneDisp = Function(V, name='boneDisp')

  ##################################### VSI implementation ###################################
  
  for i in range(len(theta)):
    theta[i].assign(0.0)
  theta[target_index].assign(1.0)
   
  target_array_index=0
  for i in current_activate_index:
    theta[i].assign(target_array[target_array_index])
    target_array_index+=1
  
  # Assign displacement 
  file_U.read(u, 'U_' + condition + str(disp))
  file_U.read(boneDisp, 'U_' + condition + str(disp))
  normal_to_surf = n
  
  # Calculate traction, external force, and residual
  load_dir = Constant((-1.0, 0.0, 0.0))
  traction = dot(P, normal_to_surf)
  loadOnFace = dot(traction, load_dir)*dss
  P_ext = force_used
  tem=assemble(R-dot(traction,v)*dss) 

  # Apply boundary conditions

  for key in keyList:
    bc_key = DirichletBC(V, boneDisp, facetfct, keyList.index(key) + 1)
    bc_key.apply(tem)
  
  tem=tem[:]
  loss1 = np.inner(tem,tem)/(np.size(tem)*P_ext**2) 
  loss2 = pow(assemble(loadOnFace) - P_ext,2)/(P_ext**2) 
  loss_factor1 = 1.
#   print("loss1 = ", loss_factor1*loss1)
#   print("loss2 = ", loss_factor2*loss2)
  lam=0. # penalty term
  a = [0]*len(theta)
  penaltySum = 0
  for i in range(len(theta)):
    a[i]= theta[i].values()
    penaltySum += a[i]**2

  loss = loss_factor1*loss1 + loss_factor2*loss2 + 0.5*lam*penaltySum
  return loss 

def add_up_losses(target_array, tendonStamp, meshName, loss_factor2, combination, current_activate_index,
                      target_index):
  losses = 0.

  for i in range(len(combination)):
      losses += nonlinear_VSI_HGO(target_array, tendonStamp, str(combination[i][0]), 
                                  str(combination[i][1]), loss_factor2, meshName, current_activate_index,
                                  target_index)
  return losses
  

###################### Here's where the code execution starts ####################
if __name__ == "__main__":
  target_index=11
  activate_basis_index=[0,1,2,3,4,5,6,7,8,9,10]
  coeffs0=np.zeros(len(activate_basis_index))
  coeffs0 = np.array([1., 1., 1., 1., 1./3., 1., 1., 1., 1., 1., 1.]) 
  print("target_index=",target_index)
  
  meshName = "Coarse"
  conditionList = ["Intact", "Torn"]
  dispList = ["1", "2"]
  combinationList = np.array(list(product(conditionList, dispList)))
  combinationList = combinationList[1:,]
  loss_factor2 = 1e-6
  tendonStampList = ["20231012", "20231017", "20231107", "20231114", "20231201", 
                     "20231206", "20231212", "20240503", "20240517", "20241127_1",
                     "20241127_2", "20241219"]
  indexList = [0,1,2,3,4,5,6,7,8,9,10,11]
  indexList = [0]
  
  identifier = 'LF1E_6_NoBounds_CoarseMesh'
  for index in indexList:
    print("Tendon ", tendonStampList[index])
    loss0 = add_up_losses(coeffs0, "Tendon" + tendonStampList[index], meshName, loss_factor2, 
                          combinationList, activate_basis_index, target_index)
    print("Loss0 = ", loss0)
    
    bounds=np.zeros((coeffs0.size,2))
    bounds[:,0]= 0.#1e-5	#All terms positive
    bounds[:,1] = np.inf
    bounds[0,0] = 0.1
    bounds[1,0] = 0.04
    bounds[3,0] = 1.e-4
    bounds[4,1] = 1./3.
    num_activate_index=len(activate_basis_index)
    gamma_matrix=np.zeros((num_activate_index,num_activate_index))
    args={'num_theta':num_theta,'activate_basis_index':activate_basis_index,
            'target_vector_index':target_index,
            'frozen_index':[0,1,2,3,4], 'max_eliminate_step':6,
            'method':'SLSQP','bounds':bounds, 
            'tendonStamp': "Tendon" + tendonStampList[index], 
            'meshName': meshName, 'combination': combinationList, 
            'lossFactor': loss_factor2}
    
    folder_name = tendonStampList[index] + '/results/HGOHighOrderI1I2_SquaredNorm'

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    save_to_file= folder_name + '/NoStag_' + identifier + '_CG2.dat'

    print(save_to_file)
    method_options={'disp':False,'ftol':1.0e-15,'eps': 1e-10,'maxiter': 1000 }

    gamma_matrix, loss=stepwise_minimization_HGOConverged(add_up_losses, coeffs0, 
                                                args_dict=args, grad_f=[],
                                                method_options=method_options,
                                                save_to_file=save_to_file)
    
    save_gamma= folder_name + '/gamma_NoStag_' + identifier + '_CG2.dat'
    np.savetxt(save_gamma,gamma_matrix)