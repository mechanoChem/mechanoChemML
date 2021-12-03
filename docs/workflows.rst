****************
Workflow summary
****************

Following are provided workflows with specific scientific examples:

Active learning
===============
The active learning workflow combines global and local sampling to create a well-sampled dataset with a corresponding surrogate model. 

See details and examples of this workflow at :doc:`Active learning <activelearning>`

Multi-resolution learning
=========================
The system identification workflow uses a class of inverse modeling techniques that allows physics discovery from data.

See details and examples of this workflow at :doc:`Multi-resolution learning <multiresolutionlearning>`

NN-based PDE solver
===================
The high-throughput solution of PDEs for inverse modelling, design and optimization leads to requirements of very fast solutions that are largely beyond the capability of traditional PDE solver libraries.The NN-based PDE solver workflow is developed for such purpose, which could predict the full field solutions orders faster than the traditional PDE solvers. Such solver works for both small dataset, which only contains a few of BVPs, and large dataset, which could contain hundreds of thousands BVPs.

See details and examples of this workflow at :doc:`NN-based PDE solver <nnbasedpdesolver>`

System identification
=====================
In materials physics problems, it is not uncommon to encounter data that possesses a hierarchical structure. Multi-resolution learning can be used to capture the details in the data, which were not well-delineated by the pre-trained model for the dominant characteristics of data.

See details and examples of this workflow at :doc:`System identification <systemid>`

