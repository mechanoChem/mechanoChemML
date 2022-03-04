************
Installation
************


Create a virtual environment
============================

A virtual Python package installation environment is isolated from the Python environment from the operating system.

Option 1 with Anaconda (recommended)
------------------------------------


.. code-block:: bash

  $ conda create --name mechanochemml python==3.7

  $ conda activate mechanochemml


Option 2 with pip virtualenv
----------------------------

.. code-block:: bash

  $ sudo pip3 install -U virtualenv 

  $ python3 -m venv --system-site-packages ./mechanochemml

  $ source mechanochemml/bin/activate


Install mechanoChemML
=====================

.. code-block:: bash

  $ (mechanochemml) pip install mechanoChemML


Download examples
=================

One can either download the whole mechanoChemML library

.. code-block:: bash

  $ (mechanochemml) git clone https://github.com/mechanoChem/mechanoChemML.git mechanoChemML-master

  $ (mechanochemml) cd mechanoChemML-master/examples

Or just download the examples provided by the mechanoChemML library

.. code-block:: bash

  $ (mechanochemml) svn export https://github.com/mechanoChem/mechanoChemML/trunk/examples ./examples

  $ (mechanochemml) cd examples

One needs to run the following command to install the proper TensorFlow version, which is compatible with their CUDA version

.. code-block:: bash

  $ (mechanochemml) python3 install_tensorflow.py


Compile local documentation
===========================

.. code-block:: bash

  $ (mechanochemml) cd mechanoChemML-master/doc

  $ (mechanochemml) make html

Local code development
======================

.. code-block:: bash

  $ (mechanochemml) cd mechanoChemML-master/

  $ (mechanochemml) python3 setup.py bdist_wheel sdist

  $ (mechanochemml) pip3 install -e . --user

Run examples
============

Please refer to the documentation page of each workflow (and its examples) for instructions to run testing examples.
