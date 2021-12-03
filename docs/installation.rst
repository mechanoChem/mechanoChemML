************
Installation
************


Create a virtual environment
============================

A virtual Python package installation environment is isolated from the Python environment from the operating system.

Option 1 with Anaconda (recommended)
------------------------------------


.. code-block:: bash

  $ conda create --name mechanochemml python==3.6.9

  $ conda activate mechanochemml


Option 2 with pip virtualenv
----------------------------

.. code-block:: bash

  $ sudo pip3 install -U virtualenv 

  $ python3 -m venv --system-site-packages ./mechanochemml

  $ source mechanochemml/bin/activate


Install required packages
=========================

.. code-block:: bash

  $ (mechanochemml) git clone https://github.com/mechanoChem/mechanoChemML.git mechanoChemML-master

  $ (mechanochemml) cd mechanoChemML-master

  $ (mechanochemml) pip3 install -r requirements.txt

Compile local documentation
===========================

.. code-block:: bash

  $ (mechanochemml) cd mechanoChemML-master/doc

  $ (mechanochemml) make html

Run examples
============

Please refer to the documentation page of each workflow (and its examples) for instructions to run testing examples.
