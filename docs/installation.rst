************
Installation
************


Create a virtual environment
============================

A virtual Python package installation environment is isolated from the Python environment from the operating system.

Option 1 with Anaconda (recommended)
------------------------------------

One can follow the `installation instruction <https://docs.anaconda.com/anaconda/install/>`_ to install `Anaconda <https://www.anaconda.com/>`_ or `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_. After conda is being activated, one can create a new environment.

.. code-block:: bash

  $ (base) conda create --name mechanochemml python==3.7

  $ (base) conda activate mechanochemml


Option 2 with pip virtualenv
----------------------------

.. code-block:: bash

  $ sudo pip3 install -U virtualenv 

  $ python3 -m venv --system-site-packages ./mechanochemml

  $ source mechanochemml/bin/activate


Install mechanoChemML
=====================

The following command will install the `mechanoChemML` library and the required libraries, except `TensorFlow`, whose version depends on the CUDA version on their machine.

.. code-block:: bash

  $ (mechanochemml) pip install mechanoChemML


Download examples
=================

One can either download the whole `mechanoChemML` library

.. code-block:: bash

  $ (mechanochemml) git clone https://github.com/mechanoChem/mechanoChemML.git mechanoChemML-master

Or just download the examples provided by the `mechanoChemML` library

.. code-block:: bash

  $ (mechanochemml) svn export https://github.com/mechanoChem/mechanoChemML/trunk/examples ./examples


Install TensorFlow
==================
One needs to run the following command to install the proper `TensorFlow` version that is compatible with their CUDA version

.. code-block:: bash

  $ (mechanochemml) python3 examples/install_tensorflow.py


Compile local documentation
===========================

.. code-block:: bash

  $ (mechanochemml) cd mechanoChemML-master/docs

  $ (mechanochemml) make html

Local code development
======================

For developers, one can use the following command to re-compile the `mechanoChemML` library and install it locally to reflect the latest GitHub changes that are not available on `PyPi <https://pypi.org/project/mechanoChemML/>`_. The newly compiled `mechanoChemML` library will overwrite the old installed version.  

.. code-block:: bash

  $ (mechanochemml) cd mechanoChemML-master/

  $ (mechanochemml) python3 setup.py bdist_wheel sdist

  $ (mechanochemml) pip3 install -e .

Run examples
============

Please refer to the documentation page of each workflow (and its examples) for instructions to run testing examples.
