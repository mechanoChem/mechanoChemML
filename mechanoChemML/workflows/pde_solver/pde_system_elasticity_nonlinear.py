import numpy as np

import tensorflow as tf

import mechanoChemML.src.pde_layers as pde_layers
from mechanoChemML.workflows.pde_solver.pde_workflow_steady_state import PDEWorkflowSteadyState

class LayerNonLinearElasticityBulkResidual(pde_layers.LayerBulkResidual):
    """
    Non-linear elasticity bulk residual
    """
    # data: [batch, in_height, in_width, in_channels]
    # filter: [filter_height, filter_width, in_channels, out_channels]
    # dh is needed.

    def __init__(self, dh, E0=2.5, nu0=0.3, normalization_factor=2.0, name='R_bulk_elasticity'):
        super(LayerNonLinearElasticityBulkResidual, self).__init__(name=name)

        self.dh = dh
        self.dof = 2
        self.normalization_factor = normalization_factor
        self.lambda0, self.mu0 = self.E_nu_to_lambda_mu(E=E0, nu=nu0)
        self.initialize_arrays()

    def call(self, input):
        """ 
        apply the int (B^T P) dV for element wise u value with 8 nodal value
        - input data: [batch, in_height, in_width, 4*dof] (2x2x2 nodal values for u)
        - output: [batch, in_height, in_width, 4*dof] (nodal value residual)
        """

        data = self.GetElementInfo(input)
        data = data * self.normalization_factor - 0.5 * self.normalization_factor
        shape = data.get_shape()[0:].as_list()    
        domain_shape = shape[1:3]
        gradu1, gradu2, gradu3, gradu4 = self.ComputeGraduAtGPs(data)

        I4, I2x2 = self.Get2ndOrderIdentityTensor(gradu1, domain_shape)
        F2x2 = self.GetF(gradu1, gradu2, gradu3, gradu4, I4, domain_shape)
        P1, P2, P3, P4 = self.ConstitutiveRelation(F2x2, I2x2)
        R = self.ComputeIntTranBxP(P1, P2, P3, P4, domain_shape)
        return R

    def ConstitutiveRelation(self, F2x2, I2x2):
        """
        Non-linear elasticity constitutive relationship
        """
        detF = tf.expand_dims(tf.linalg.det(F2x2), 4)
        detF = tf.expand_dims(detF, 5)

        #---------------------------------------------------------------------------------
        # get detF mask to make sure inv(F2x2) works.
        # It is very possible that F2x2 is not invertible. The following will set
        # the region where detF < 0.5 or detF > 3.0 to identity tensor to make sure
        # F2x2 is invertible, as a numerical solution.

        detF_mask_finite = tf.where(tf.math.is_finite(detF), tf.fill(tf.shape(detF), 1.0), tf.fill(tf.shape(detF), 0.0))
        detF_mask_negative = tf.where(detF < 0.1, tf.fill(tf.shape(detF), 0.0), tf.fill(tf.shape(detF), 1.0))
        detF_mask_large = tf.where(detF > 5.0, tf.fill(tf.shape(detF), 0.0), tf.fill(tf.shape(detF), 1.0))

        detF_mask = tf.multiply(detF_mask_negative, detF_mask_large)
        detF_mask = tf.multiply(detF_mask, detF_mask_finite)
        detF_mask_reverse = tf.where( detF_mask == 0, tf.fill(tf.shape(detF_mask), 1.0), tf.fill(tf.shape(detF_mask), 0.0))

        F2x2_modified = tf.multiply(F2x2, detF_mask) + tf.multiply(I2x2, detF_mask_reverse)

        detF = tf.expand_dims(tf.linalg.det(F2x2_modified), 4)
        detF = tf.expand_dims(detF, 5)
        #---------------------------------------------------------------------------------

        # get other values
        try:
            InvF = tf.linalg.inv(F2x2_modified)
        except tensorflow.python.framework.errors_impl.InvalidArgumentError:
            raise ValueError ('F2x2 not invertable', F2x2_modified)

        TransInvF = tf.transpose(InvF, perm=[0,1,2,3,5,4])

        # get P
        P = self.lambda0 * (tf.math.multiply(detF,detF) - detF) * TransInvF + self.mu0 * ( F2x2_modified - TransInvF)
        P = tf.multiply(P, detF_mask)

        P1 = P[:,:,:,0,:,:]
        P2 = P[:,:,:,1,:,:]
        P3 = P[:,:,:,2,:,:]
        P4 = P[:,:,:,3,:,:]

        P1 = tf.reshape(P1, [-1, 4])
        P2 = tf.reshape(P2, [-1, 4])
        P3 = tf.reshape(P3, [-1, 4])
        P4 = tf.reshape(P4, [-1, 4])
        return P1, P2, P3, P4


class WeakPDENonLinearElasticity(PDEWorkflowSteadyState):
    def __init__(self):
        super().__init__()
        self.dof = 2
        self.dof_name = ['Ux', 'Uy']
        self.problem_name = 'nonlinear-elasticity'
        self.E0 = 25
        self.nu0 = 0.3
        self.UseTwoNeumannChannel = False

    def _bulk_residual(self, y_pred):
        """
        bulk residual for nonlinear elasticity
        """
        elem_bulk_residual=LayerNonLinearElasticityBulkResidual(dh=self.dh, E0=self.E0, nu0=self.nu0)(y_pred)
        return elem_bulk_residual


if __name__ == '__main__':
    """ Weak PDE constrained NN for nonlinear elasticity """
    problem = WeakPDENonLinearElasticity()
    problem.run()
    # problem.test(test_folder='DNS')
    # problem.test(test_folder='Test_inter')
    # problem.test(test_folder='Test_extra')
    # problem.debug_problem(use_label=False)
    # problem.debug_problem(use_label=True)
    # problem.test_residual_gaussian(noise_std=1e-4, sample_num=1000)
