:orphan:

*************************
Multi-resolution learning
*************************


.. figure:: figures/multi-resolution-learning.png
   :scale: 100 %
   :align: center

   Illustration of the NN architecture used for the multi-resolution learning workflow.

Description
===========
In materials physics problems, it is not uncommon to encounter data that possesses a hierarchical structure. Multi-resolution learning can be used to capture the details in the data, which were not well-delineated by the pre-trained model for the dominant characteristics of data.

Examples
========

Example 1: Learn the free energy and nonlinear elastic response of evolving microstructures
-------------------------------------------------------------------------------------------

When studying the homogenized stress-strain response of a family of multi-component crystalline microstructures, the free energy of each microstructure has a multi-resolution structure with a dominant trajectory from phase transformations that drive evolution of the microstructure, and small-scale fluctuations from strains that explore the effective elastic response of a given microstructure. The dominant trajectory strongly depends on the microstructural information, such as the volume fraction, the location and orientation of each crystalline phase, and the interfaces, whereas the small-scale fluctuations are related to the applied loading. 

Data preparation
^^^^^^^^^^^^^^^^
The required data is stored at

.. code-block:: text

    examples/multi_resolution_learning/Example1_elasticity_microstructures/mrnn-1-microstructure-dnn/data

Step 1: Hyper-parameter search for DNNs to learn the main feature of data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    cd examples/multi_resolution_learning/Example1_elasticity_microstructures/mrnn-1-microstructure-dnn/step1_hyper_parameter_search_main_feature
    python hyper_parameter_search.py

Step 2: Train the optimal DNN to learn the main feature of data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    cd examples/multi_resolution_learning/Example1_elasticity_microstructures/mrnn-1-microstructure-dnn/step2_final_dnn_main_feature
    python dnn_1dns_final.py

Step 3: Hyper-parameter search for MRNNs to learn the detail feature of data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    cd examples/multi_resolution_learning/Example1_elasticity_microstructures/mrnn-1-microstructure-dnn/step3_hyper_parameter_search_mrnn_detail_feature
    python hyper_parameter_search.py

Step 4: Train the optimal MRNN to learn the detail feature of data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    cd examples/multi_resolution_learning/Example1_elasticity_microstructures/mrnn-1-microstructure-dnn/step4_final_mrnn_no_penalize_P
    python kbnn_1_frame_dnn.py

References
==========

X Zhang, K Garikipati. "Machine learning materials physics: Multi-resolution neural networks learn the free energy and nonlinear elastic response of evolving microstructures", Computer Methods in Applied Mechanics and Engineering, 372, 113362, 2020, `doi:10.1016/j.cma.2020.113362 <https://doi.org/10.1016/j.cma.2020.113362>`_, preprint at `arXiv:2001.01575 <https://arxiv.org/abs/2001.01575>`_.