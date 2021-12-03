**************
Code structure
**************


Following is the folder structure of the mechanoChemML library:

.. code-block:: text

   | mechanoChemML-master
   | ├── docs
   | │   ├── _build
   | │   ├── figures
   | │   ├── conf.py
   | │   ├── *.rst
   | │   └── Makefile
   | ├── LICENSE
   | ├── README.md
   | ├── requirements.txt
   | ├── mechanoChemML
   | │   ├── src
   | │   ├── testing
   | │   ├── third_party
   | │   └── workflows
   | │       ├── active_learning
   | │       |   ├── Example1_NiAl
   | │       |   └── Example2_LCO
   | │       ├── multi_resolution_learning
   | │       |   └── Example1_elasticity_microstructures
   | │       ├── pde_solver
   | │       |   ├── Example1_diffusion_steady_state
   | │       |   ├── Example2_linear_elasticity
   | │       |   └── Example3_nonlinear_elasticity
   | │       │── systemID
   | │       |   ├── Example1_pattern_forming
   | │       |   ├── Example2_soft_materials
   | │       |   └── Example3_polynomial


In the folder structure above:

- ``mechanoChemML-master`` is the folder we get when we issue a ``git pull/clone`` command
- ``mechanoChemML-master/docs`` is the directory where our Sphinx documentation will reside
- ``mechanoChemML-master/docs/_build`` being the Sphinx build directory. This folder is auto-generated for us by Sphinx.
- ``mechanoChemML-master/mechanoChemML`` is the actual Python package directory, where our Python source files reside.
- ``mechanoChemML-master/mechanoChemML/src`` is the directory, where the machine learning classes and functions reside.
- ``mechanoChemML-master/mechanoChemML/testing`` is the directory, where the various testing functions reside.
- ``mechanoChemML-master/mechanoChemML/third_party`` is the directory, where scripts to use third party libraries reside.
- ``mechanoChemML-master/mechanoChemML/workflows`` is the directory, where the workflows reside.
- ``mechanoChemML-master/mechanoChemML/workflows/active_learning`` is the directory, where the active learning workflows reside.
- ``mechanoChemML-master/mechanoChemML/workflows/multi_resolution_learning`` is the directory, where the multi-resolution learning workflows reside.
- ``mechanoChemML-master/mechanoChemML/workflows/pde_solver`` is the directory, where the NN-based PDE solver workflows reside.
- ``mechanoChemML-master/mechanoChemML/workflows/systemID`` is the directory, where the system identification workflows reside.

