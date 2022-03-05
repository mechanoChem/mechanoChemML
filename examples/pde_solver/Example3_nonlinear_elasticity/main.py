from mechanoChemML.workflows.pde_solver.pde_system_elasticity_nonlinear import WeakPDENonLinearElasticity as thisPDESystem

if __name__ == '__main__':
    problem = thisPDESystem()
    problem.run()
