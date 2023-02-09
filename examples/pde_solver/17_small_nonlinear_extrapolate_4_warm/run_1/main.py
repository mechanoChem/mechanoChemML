from mechanoChemML.workflows.pde_solver.pde_system_elasticity_nonlinear import WeakPDENonLinearElasticity as thisPDESystem

if __name__ == '__main__':
    problem = thisPDESystem()
    problem.run()
    problem.test(test_folder='Test')
    problem.test(test_folder='DNS', output_reaction_force=False)
    problem.test(test_folder='Inter', output_reaction_force=False)
    problem.test(test_folder='Extra', output_reaction_force=False)
    # problem.debug_problem(use_label=False)
    # problem.debug_problem(use_label=True)
    # problem.test_residual_gaussian(noise_std=1e-4, sample_num=1000)
