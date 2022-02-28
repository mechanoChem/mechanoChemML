import numpy as np

import tensorflow as tf

import mechanoChemML.src.pde_layers as pde_layers
from mechanoChemML.workflows.pde_solver.pde_workflow_steady_state import PDEWorkflowSteadyState

class LayerLinearElasticityBulkResidual(pde_layers.LayerBulkResidual):
    """
    Linear elasticity bulk residual
    """
    # data: [batch, in_height, in_width, in_channels]
    # filter: [filter_height, filter_width, in_channels, out_channels]
    # dh is needed.

    def __init__(self, dh, E0=2.5, nu0=0.3, normalization_factor=2.0, name='R_bulk_elasticity'):
        super(LayerLinearElasticityBulkResidual, self).__init__(name=name)

        self.dh = dh
        self.dof = 2
        self.normalization_factor = normalization_factor
        self.lambda0, self.mu0 = self.E_nu_to_lambda_mu(E=E0, nu=nu0)
        self.initialize_arrays()

    def call(self, input):
        """ 
        apply the int (B^T P) dV for element wise u value with 8 nodal value
        - input data: [batch, in_height, in_width, 8] (2x2x2 nodal values for u)
        - output: [batch, in_height, in_width, 8] (nodal value residual)
        """
        # scaled_data = non_scaled/normalization_factor + 0.5 for range  [-x, +x]
        # non_scaled = scaled_data * normalization_factor - 0.5 * normalization_factor

        data = self.GetElementInfo(input)
        data = data * self.normalization_factor - 0.5 * self.normalization_factor

        shape = data.get_shape()[0:].as_list()    
        domain_shape = shape[1:3]
        gradu1, gradu2, gradu3, gradu4 = self.ComputeGraduAtGPs(data)

        I4, I2x2 = self.Get2ndOrderIdentityTensor(gradu1, domain_shape)
        epsilon = self.GetEpsilon(gradu1, gradu2, gradu3, gradu4, domain_shape)

        sigma1, sigma2, sigma3, sigma4 = self.ConstitutiveRelation(epsilon, I2x2)
        R = self.ComputeIntTranBxP(sigma1, sigma2, sigma3, sigma4, domain_shape)
        return R

    def ConstitutiveRelation(self, epsilon, I2x2):
        """
        Linear elasticity constitutive relationship
        """
        epsilon_trace = tf.linalg.trace(epsilon)
        epsilon_trace = tf.expand_dims(epsilon_trace, 4)
        epsilon_trace = tf.expand_dims(epsilon_trace, 5)

        # get sigma for linear elasticity
        sigma = self.lambda0 * tf.math.multiply(epsilon_trace, I2x2) + 2.0 * self.mu0 * epsilon

        sigma1 = sigma[:,:,:,0,:,:]
        sigma2 = sigma[:,:,:,1,:,:]
        sigma3 = sigma[:,:,:,2,:,:]
        sigma4 = sigma[:,:,:,3,:,:]

        sigma1 = tf.reshape(sigma1, [-1, 4])
        sigma2 = tf.reshape(sigma2, [-1, 4])
        sigma3 = tf.reshape(sigma3, [-1, 4])
        sigma4 = tf.reshape(sigma4, [-1, 4])
        return sigma1, sigma2, sigma3, sigma4


class WeakPDELinearElasticity(PDEWorkflowSteadyState):
    def __init__(self):
        super().__init__()
        self.dof = 2
        self.dof_name = ['Ux', 'Uy']
        self.problem_name = 'linear-elasticity'
        self.E0 = 25
        self.nu0 = 0.3
        self.UseTwoNeumannChannel = False

    def _bulk_residual(self, y_pred):
        """
        bulk residual for linear elasticity
        """
        elem_bulk_residual=LayerLinearElasticityBulkResidual(dh=self.dh, E0=self.E0, nu0=self.nu0)(y_pred)
        return elem_bulk_residual


if __name__ == '__main__':
    """ Weak PDE constrained NN for linear elasticity """
    problem = WeakPDELinearElasticity()
    problem.run()
    # problem.test(test_folder='DNS')
    # problem.test(test_folder='Test_inter')
    # problem.test(test_folder='Test_extra')
    # problem.debug_problem(use_label=False)
    # problem.debug_problem(use_label=True)
    # problem.test_residual_gaussian(noise_std=1e-4, sample_num=1000)
