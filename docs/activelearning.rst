:orphan:

***************
Active learning
***************

.. figure:: figures/activeLearning.png
   :scale: 50 %
   :align: center

   Schematic of the active learning workflow for learning free energy density function.

Description
===========
The active learning workflow combines global and local sampling to create a well-sampled dataset with a corresponding surrogate model. 

Examples
========

Example1: Learning free energy density for scale bringing: Ni-Al system
-----------------------------------------------------------------------

This example manages the creation of a free energy density function for the Ni-Al system, where the data are found using Monte Carlo simulations based on statistical mechanics, using in-house versions of the CASM software [github.com/prisms-center/CASMcode]. The surrogate model is an integrable deep neural network (IDNN).

The following python class defines the active learning workflow for the Ni-Al free energy project. First, the class constructor, which sets up necessary directories and defines an initial IDNN (other IDNNs will also be created later on, at each iteration of the workflow).

.. literalinclude:: ../mechanoChemML/workflows/active_learning/Example1_NiAl/active_learning.py
   :lines: 33-56

Next, we read in the parameters defined in the configuration file, such as the number of workflow iterations, the number of CPUs to use for the direct numerical simulations, the number of hyperparameter sets to compare, etc.

.. literalinclude:: ../mechanoChemML/workflows/active_learning/Example1_NiAl/active_learning.py
   :lines: 60-73

The following function defines a test set, or a sample set. Here, points are first sampled from the sublattice composition space, since the bounds are well-defined: [0,1]. Sampling is done with a Sobol sequence because if its space-filling and noncollapsing design. These values are then converted from sublattice compositions to order parameters through a linear operation. Finally, only points with average composition less than or equal to 0.25 are taken, since the crystal structure changes from FCC to BCC at that point.
	   
.. literalinclude:: ../mechanoChemML/workflows/active_learning/Example1_NiAl/active_learning.py
   :lines: 77-95

At each iteration of the workflow, an approximation of the free energy and the chemical potential are needed. For the first iteration before any data have been computed, we use the free energy of an ideal solution. For subsequent iterations, the current IDNN is used.
	   
.. literalinclude:: ../mechanoChemML/workflows/active_learning/Example1_NiAl/active_learning.py
   :lines: 99-112

The active learning workflow consists of alternating between global and local sampling of the composition/order parameter space. Global sampling uses the create_test_set function defined previously to sample across the full domain, then submitting these points to the CASM code to compute the chemical potentials and order parameters.
	   
.. literalinclude:: ../mechanoChemML/workflows/active_learning/Example1_NiAl/active_learning.py
   :lines: 116-139

Local sampling is based on two characteristics: areas with large pointwise error and areas with local energy wells. The pointwise error is calculated for all data points from the most recent global sampling and is defined at each point as the square of the difference between the predicted and the actual chemical potential values, summed across all four chemical potentials. All these points are sorted according to the error. For each of the N (e.g. 200) points with the highest error, n (e.g. 4) random perturbations of the point are selected to be computed with CASM. This is repeated, with a lower value of n, for the next N points in the sorted list. By sampling in regions with high IDNN error, we can more rapidly improve the overall error in the IDNN.

Local energy wells define areas of phase stability. For this reason, local wells are identified and additional sampling takes place there. To do this, a large set of points are sampled across the full domain, using a Sobol sequence. The current IDNN is used to determine if each point is in a region of convexity (a potential well). For each point, the norm of the gradient also computed. The points are sorted by the norm of the gradient, since the points nearest to the minium of the well with have the lowest gradient norm. The M points in a region of convexity that have the lowest gradient norm are used to defined M*n additional sampling points, through random perturbation.

.. literalinclude:: ../mechanoChemML/workflows/active_learning/Example1_NiAl/active_learning.py
   :lines: 143-167

A hyperparameter search is performed to improve training.

.. literalinclude:: ../mechanoChemML/workflows/active_learning/Example1_NiAl/active_learning.py
   :lines: 171-173

The free energy of this system is known, due to crystal symmetries, to be invariant to certain transformations of the order parameters. Those transformations are defined here, and the IDNN then becomes of a function of these transformations. In this way, the known symmetries are built in exactly.

.. literalinclude:: ../mechanoChemML/workflows/active_learning/Example1_NiAl/active_learning.py
   :lines: 177-189

This function defines the training of the IDNN. Note that we receive DNS data for the four chemical potentials, which are the partial derivatives of the free energy with respect to the four order parameters. Since we only have derivative data, we will have an arbitrary constant of integration. While this is not a huge problem, it is inconvenient. To remedy this, we will also create free energy data that (somewhat arbitratily) defines the free energy to be zero when all order parameters are zero. (Because of the way the neural network code is defined, the number of free energy points has to be the same as the number of chemical potential points, but they do not have to be defined at the same order parameter values.) A snapshot of the IDNN with its current weights is saved at each workflow iteration.
	   
.. literalinclude:: ../mechanoChemML/workflows/active_learning/Example1_NiAl/active_learning.py
   :lines: 193-224

The main_workflow function is fairly simple, but it defines the overall workflow. Again, it begines with the global sampling. After these points are calculated through CASM, and IDNN is trained to all calculated data, including data from previous iterations. (On the second iteration of the workflow, a hyperparameter search is performed first to determine the size of the IDNN and a good value for the learning rate.) Once trained, the current IDNN is used to perform local sampling, which completes the workflow iteration. Currently, the number of iterations is given directly, but a stopping condition could be defined based on the overall change in the IDNN from one iteration to the next.
	   
.. literalinclude:: ../mechanoChemML/workflows/active_learning/Example1_NiAl/active_learning.py
   :lines: 228-250

Data preparation
^^^^^^^^^^^^^^^^

Configuration file
^^^^^^^^^^^^^^^^^^

Standard parameters in the workflow can be defined in the .ini configuration file.

The deep numerical simlation (DNS) and high performance computing (HPC) settings can be defined here, including the DNS model and any addition parameters required by the model:

.. literalinclude:: ../mechanoChemML/workflows/active_learning/Example1_NiAl/NiAl_free_energy.ini
   :lines: 1-6

Next, define parameters related to the neural network surrogate model:
	   
.. literalinclude:: ../mechanoChemML/workflows/active_learning/Example1_NiAl/NiAl_free_energy.ini
   :lines: 8-18

Finally, there are additional parameters specific to the active learning workflow itself:

.. literalinclude:: ../mechanoChemML/workflows/active_learning/Example1_NiAl/NiAl_free_energy.ini
   :lines: 19-21


References
==========

G.H. Teichert, A.R. Natarajan, A. Van der Ven, K. Garikipati. "Machine learning materials physics: Integrable deep neural networks enable scale bridging by learning free energy functions," Computer Methods in Applied Mechanics and Engineering. Vol 353, 201-216, 2019, doi:10.1016/j.cma.2019.05.019

G.H. Teichert, A.R. Natarajan, A. Van der Ven, K. Garikipati. "Active learning workflows and integrable deep neural networks for representing the free energy functions of alloys," Computer Methods in Applied Mechanics and Engineering. Vol 371, 113281, 2020, doi:10.1016/j.cma.2020.113281

G.H. Teichert, S. Das, M. Aykol, C. Gopal, V. Gavini, K. Garikipati. "LixCoO2 phase stability studied by machine learning-enabled scale bridging between electronic structure, statistical mechanics and phase field theories," arXiv preprint: arxiv.org/abs/2104.08318
