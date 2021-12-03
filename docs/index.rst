*********************
mechanoChemML library
*********************

mechanoChemML is a machine learning software library for computational materials physics. It is designed to function as an interface between platforms that are widely used for scientific machine learning on one hand, and others for solution of partial differential equations-based models of physics. Of special interest here, and the focus of mechanoChemML, are applications to computational materials physics. These typically feature the coupled solution of material transport, reaction, phase transformation, mechanics, heat transport and electrochemistry. mechanoChemML is developed and maintained by the `Computational Physics Group <http://websites.umich.edu/~compphys/index.html>`_ at the University of Michigan, Ann Arbor. Following is the list of the main contribtors:

* Arjun Sundararajan
* Elizabeth Livingston
* Greg Teichert
* Matt Duschenes
* Sid Srivastava
* Xiaoxuan Zhang
* Zhenlin Wang
* Krishna Garikipati


Using this library
==================
:doc:`installation`
  How to install this library.

:doc:`codestructure`
  Details of the code structure of this library.
  
:doc:`workflows`
  Various workflows shipped with this library.

:doc:`autoapi/index`  
  Auto generated API information.


Development
===========

:doc:`contribute`  
  How to contribute to this library.
  
:doc:`changelog`
  The library development changelog.

Cite mechanoChemML
==================
If you find this code useful in your research, please consider citing:

  Author List (2021), mechanoChemML, arXiv preprint arXiv:xxxxx.xxxxx.

.. toctree::
   :caption: mechanoChemML
   :maxdepth: 2
   :hidden:

   installation
   codestructure
   workflows
   autoapi/index
   contribute
   changelog

.. toctree::
   :caption: List of workflows
   :maxdepth: 2
   :hidden:

   activelearning
   multiresolutionlearning
   nnbasedpdesolver
   systemid
