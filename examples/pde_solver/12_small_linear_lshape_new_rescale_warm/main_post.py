import numpy as np
import tensorflow as tf
from mechanoChemML.workflows.pde_solver.pde_workflow_steady_state import PDEWorkflowSteadyState
from mechanoChemML.workflows.pde_solver.pde_system_elasticity_linear import LayerLinearElasticityBulkResidual
from mechanoChemML.workflows.pde_solver.pde_system_elasticity_linear import WeakPDELinearElasticity as thisPDESystem


class thisPDESystem(PDEWorkflowSteadyState):
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
        elem_bulk_residual=LayerLinearElasticityBulkResidual(dh=self.dh, E0=self.E0, nu0=self.nu0, normalization_factor=0.2)(y_pred)
        return elem_bulk_residual
    
    def _build_optimizer(self):
        """ 
        Build the optimizer for weak-PDE constrained NN
        """
        # not using the decay learning rate function.
        # global_step = tf.Variable(0, name='global_step', trainable=False)
        # LearningRate = tf.compat.v1.train.exponential_decay(
            # learning_rate=self.LR0,
            # global_step=global_step,
            # decay_steps=1000,
            # decay_rate=0.8,
            # staircase=True)
        # print("new learning rate")
        initial_learning_rate = self.LR0
        LearningRate = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=500,
            decay_rate=0.97,
            staircase=True)

        if self.NNOptimizer.lower() == 'Adam'.lower() :
            optimizer = tf.keras.optimizers.Adam(learning_rate=LearningRate)
        elif self.NNOptimizer.lower() == 'Nadam'.lower() :
            optimizer = tf.keras.optimizers.Nadam(learning_rate=self.LR0)
        elif self.NNOptimizer.lower() == 'SGD'.lower() : # not very well
            optimizer = tf.keras.optimizers.SGD(learning_rate=LearningRate) 
        else:
            raise ValueError('Unknown optimizer option:', self.NNOptimizer)
        return optimizer


if __name__ == '__main__':
    problem = thisPDESystem()
    problem.run()
    inputs = []
    labels = []
    mean = []
    var = []
    std = []
    # problem.test(test_folder='DNS', output_reaction_force=False)
    _inputs, _labels, _mean, _var, _std = problem.test(test_folder='DNS', plot_png=False)
    print(np.shape(_inputs), '_inputs')
    print(np.shape(_labels), '_labels')
    print(np.shape(_mean),   '_mean  ')
    print(np.shape(_var),    '_var   ')
    print(np.shape(_std),    '_std   ')
    np.save('inputs.npy',_inputs)
    np.save('labels.npy',_labels)
    np.save('mean.npy',_mean)
    np.save('var.npy',_var)
    np.save('std.npy',_std)

    # inputs.append(_inputs)
    # labels.append(_labels)
    # mean.append(_mean)
    # var.append(_var)
    # std.append(_std)
