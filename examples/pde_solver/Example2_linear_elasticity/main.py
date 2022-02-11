from mechanoChemML.workflows.physics_constrained_learning.src.pde_system_elasticity_linear import WeakPDELinearElasticity as thisPDESystem

if __name__ == '__main__':
    problem = thisPDESystem()
    problem.run()
    problem.test(test_folder='DNS')
    problem.test(test_folder='Inter')
    problem.test(test_folder='Extra')
    # problem.debug_problem(use_label=False)
    # problem.debug_problem(use_label=True)
    # problem.test_residual_gaussian(noise_std=1e-4, sample_num=1000)
