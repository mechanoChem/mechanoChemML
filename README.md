# mechanoChemML library

Developed by the [Computational Physics Group](http://www.umich.edu/~compphys/index.html) at the University of Michigan.

List of contributors (alphabetical order):
* Arjun Sundararajan
* Elizabeth Livingston
* Greg Teichert
* Matt Duschenes
* Sid Srivastava
* Xiaoxuan Zhang
* Zhenlin Wang
* Krishna Garikipati

# Overview

mechanoChemML is a machine learning software library for computational materials physics. It is designed to function as an interface between platforms that are widely used for scientific machine learning on one hand, and others for solution of partial differential equations-based models of physics. Of special interest here, and the focus of mechanoChemML, are applications to computational materials physics. These typically feature the coupled solution of material transport, reaction, phase transformation, mechanics, heat transport and electrochemistry. 

# Version information

This is version 0.1.0.

# License

GNU Lesser General Public License (LGPL) v3.0. Please see the file LICENSE for details. 

# Installation

```
  $ conda create --name mechanochemml python==3.6.9

  $ conda activate mechanochemml

  $ (mechanochemml) git clone https://github.com/mechanoChem/mechanoChemML.git mechanoChemML-master

  $ (mechanochemml) cd mechanoChemML-master

  $ (mechanochemml) pip3 install -r requirements.txt
```

# Documentation and usage 

The documentation of this library is available at https://mechanochemml.readthedocs.io/en/latest/index.html, where one can find instructions of using the provided classes, functions, and workflows provided by the mechanoChemML library.

To create a local copy of the documentation, one can use

```
  $ (mechanochemml) cd mechanoChemML-master/doc

  $ (mechanochemml) make html
```

# Acknowledgements

This code has been developed under the support of the following:

- Toyota Research Institute, Award #849910 "Computational framework for data-driven, predictive, multi-scale and multi-physics modeling of battery materials"


# Referencing this code

If you write a paper using results obtained with the help of this code, please consider citing

- X. Zhang, G.H. Teichert, Z. Wang, M. Duschenes, S. Srivastava, A. Sunderarajan, E. Livingston, K. Garikipati (2021), mechanoChemML: A software library for machine learning in computational materials physics, arXiv preprint [arXiv:2112.04960](https://arxiv.org/abs/2112.04960).
