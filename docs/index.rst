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


Development
===========

:doc:`contribute`  
  How to contribute to this library.
  
:doc:`changelog`
  The library development changelog.

List of examples
================

:doc:`activelearning`  

:doc:`multiresolutionlearning`  

:doc:`nnbasedpdesolver`  

:doc:`systemid`  

:doc:`nonlocalcalc`  

API reference
=============

:doc:`autoapi/mechanoChemML/src/index`  

:doc:`autoapi/mechanoChemML/workflows/index`  

Cite mechanoChemML
==================
If you find this code useful in your research, please consider citing:

  X. Zhang, G.H. Teichert, Z. Wang, M. Duschenes, S. Srivastava, A. Sunderarajan, E. Livingston, K. Garikipati (2021), mechanoChemML: A software library for machine learning in computational materials physics, arXiv preprint `arXiv:2112.04960 <https://arxiv.org/abs/2112.04960>`_.

.. toctree::
   :caption: mechanoChemML
   :maxdepth: 2
   :hidden:

   installation
   codestructure
   workflows
   contribute
   changelog
   PyPI project page <https://pypi.org/project/mechanoChemML/>

.. toctree::
   :caption: List of examples
   :maxdepth: 2
   :hidden:

   activelearning
   multiresolutionlearning
   nnbasedpdesolver
   systemid
   nonlocalcalc

.. toctree::
   :caption: API reference
   :maxdepth: 2
   :hidden:

   autoapi/mechanoChemML/src/index
   autoapi/mechanoChemML/workflows/index
