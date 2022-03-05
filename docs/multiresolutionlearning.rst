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

    examples/mr_learning/Example1_single_microstructure_dnn/data

Step 1: Hyper-parameter search for DNNs to learn the main feature of data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    cd examples/mr_learning/Example1_single_microstructure_dnn/step1_hp_search_main
    python hyper_parameter_search.py

During the hyper-parameter search, different NNs are ranked based on their averaged final loss.  

Step 2: Train the optimal DNN to learn the main feature of data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One needs to modify the file `dnn-free-energy-1dns-final.ini`

.. code-block:: bash

    cd examples/mr_learning/Example1_single_microstructure_dnn/step2_final_dnn_main
    vi dnn-free-energy-1dns-final.ini

and provide the best NN architecture obtained from step 1 to the following variable in the input file

.. code-block:: bash

    [MODEL]
    NodesList = 76
    Activation = softplus

Here, :code:`76` indicates that a DNN contains one densely connected layer with 76 neuron on it and a :code:`softplus` activation function.

Once the input file is modified, one can run the following to train the NN.

.. code-block:: bash

    cd examples/mr_learning/Example1_single_microstructure_dnn/step2_final_dnn_main
    python dnn_1dns_final.py

At the end of the training, the final weights of the NN will be saved as 

.. code-block:: bash

    saved_weight/cp-10000.ckpt 

Step 3: Hyper-parameter search for MRNNs to learn the detail feature of data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One needs to modify the file `kbnn-load-dnn-1-frame-hyperparameter-search.ini` by providing the information of the NN that learned the main feature of the data by

.. code-block:: bash

    cd examples/mr_learning/Example1_single_microstructure_dnn/step3_hp_search_mrnn_detail
    vi kbnn-load-dnn-1-frame-hyperparameter-search.ini

The information at the section :code:`[KBNN]` should be updated, such as

.. code-block:: bash

    [KBNN]
    LabelShiftingModels = ../step2_final_dnn_main/dnn-free-energy-1dns-final.ini
    OldShiftFeatures = vol_rectangle_p, vol_rectangle_m,len_c,len_s_r_p,len_s_r_m
    OldShiftMean = 0.195798, 0.195384, 0.074566, 0.051366,0.055325
    OldShiftStd = 0.014416, 0.017324, 0.033848, 0.014693, 0.016482
    OldShiftLabelScale = 100
    OldShiftDataNormOption = 3

One the input file is prepared, one can run the following to perform the hyper-parameter search for the multi-resolution NN to learn the detailed feature.

.. code-block:: bash

    cd examples/mr_learning/Example1_single_microstructure_dnn/step3_hp_search_mrnn_detail
    python hyper_parameter_search.py

Again, different NNs are ranked based on their averaged final loss.  

Step 4: Train the optimal MRNN to learn the detail feature of data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One needs to modify the file `kbnn-load-dnn-1-frame.ini`

.. code-block:: bash

    cd examples/mr_learning/Example1_single_microstructure_dnn/step4_final_mrnn_no_penalize_P
    vi kbnn-load-dnn-1-frame.ini

and provide the best NN architecture obtained from step 3 to the following variable in the input file

.. code-block:: bash

    [MODEL]
    NodesList = 26, 26
    Activation = softplus, softplus

Here, :code:`26,26` indicates that a DNN contains two densely connected layers with 76 neuron on each and the :code:`softplus` activation function for each layer.

Once the input file is modified, one can run the following to train the NN.

.. code-block:: bash

    cd examples/mr_learning/Example1_single_microstructure_dnn/step4_final_mrnn_no_penalize_P
    python kbnn_1_frame_dnn.py

References
==========

X Zhang, K Garikipati. "Machine learning materials physics: Multi-resolution neural networks learn the free energy and nonlinear elastic response of evolving microstructures", Computer Methods in Applied Mechanics and Engineering, 372, 113362, 2020, `doi:10.1016/j.cma.2020.113362 <https://doi.org/10.1016/j.cma.2020.113362>`_, preprint at `arXiv:2001.01575 <https://arxiv.org/abs/2001.01575>`_.
