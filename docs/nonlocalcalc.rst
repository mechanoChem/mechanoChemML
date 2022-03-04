:orphan:

*******************
Non-Local Calculus
*******************

Description
===========

Graph-based representation methods have shown great success in reduced-order modeling for a range of physical phenomena, that often lend themselves to a PDE form. Non-local calculus on graphs provide a mathematical framework for defining and evaluating these PDE models. This library allows the user to estimate partial derivatives in unstrutured data by systematically constructing graphs with controlled accuracy. The analysis of these methods are provided in [1]. Following section discusses the implementation of examples provided in [2].
 
Examples
========

Example 1: Derivative Calculation
---------------------------------
This example shows the basic I/O format for the datapoints and the resulting derivatives. 

Data preparation
^^^^^^^^^^^^^^^^

Synthetic data can be generated using the following commands

.. code-block:: bash

    cd mechanoChemML/examples/non_local_calculus/Example1_Derivative_Calculation
    python generate_data.py

The generated data is saved in the `data/` folder. 

Estimation of derivatives using Non-Local Calculus
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following commands will estimate the derivatives

.. code-block:: bash

    cd mechanoChemML/examples/non_local_calculus/Example1_Derivative_Calculation
    python generate_data.py

The result is saved in the `result/` folder. 

Example 2: Reduced order modeling for Allen Cahn dynamics
---------------------------------------------------------
In this example a reduced order model is trained for 1D Reaction-Diffusion system. The basis for operators is generated using the "Non-Local Calculus" module. The regression in carried out using the "SystemID" workflow.

Data preparation
^^^^^^^^^^^^^^^^

Synthetic data generation requires solving the Reaction-Diffusion system. Finite Element Method is used for numerically solving the PDE using the Fenics library. The global observables are evaluated and saved in this step. Following commands are used for generating the data. 

.. code-block:: bash

    cd mechanoChemML/examples/non_local_calculus/Example2_Allen_Cahn/dns
    python dns.py settings.prm

This generates 100 samples corresponding to different initial conditions and save the individual trajectories in the `dns/data/Sample<i>/` folder. 

Reduced-order modeling
^^^^^^^^^^^^^^^^^^^^^^
First, the derivative terms are generated using the following code: 

.. code-block:: bash

    cd mechanoChemML/examples/non_local_calculus/Example2_Allen_Cahn/
    python estimate_derivatives.py

Th generated basis is saved in the `dns/data/Sample<i>/ProcessDump/` folder for each trajectory. These basis terms from all the trajectories are then combined and regression is carried out using the SystemID workflow. Following commands are used to estimate the reduced order model:

.. code-block:: bash
	
    cd mechanoChemML/examples/non_local_calculus/Example2_Allen_Cahn/
    python train_model.py config_allen_cahn

	

References
==========

[1]. Duschenes, M. and Garikipati, K., 2021. Reduced order models from computed states of physical systems using non-local calculus on finite weighted graphs. arXiv preprint arXiv:2105.01740.

[2]. Zhang, X., Teichert, G.H., Wang, Z., Duschenes, M., Srivastava, S., Sunderarajan, A., Livingston, E. and Garikipati, K., 2021. mechanoChemML: A software library for machine learning in computational materials physics. arXiv preprint arXiv:2112.04960.