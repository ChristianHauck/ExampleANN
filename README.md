# ExampleANN
Python 2.7 code for a Multi-Layer-Perceptron (mlp)

The code is not designed to be efficient, sophisticated or whatever - but just to be easy to read and
understand. The implementation was done according to the pseudo-code in "Russell,Norvig, Artificial Intelligence"
and "Hertz,Krogh,Palmer, Introduction to the Theory of Neural Computation". Look there (or in any other textbook
chapter about ANNs) for a detailed explanation.


# ANACONDA
The anaconda toolkit provides jit-compilation to C and CUDA support. To test this interesting technology in the
context of python neural networks, folllow these steps:

 1. Download and install anaconda from [continuum](https://www.continuum.io/downloads)
 2. conda -v     - shows, if conda is installed correctly
Now jit should work.
 3. To install the CUDA stuff, open a shell and run: conda install cudatoolkit

Adapt the mlp. E.g. according to the following 
[example](http://nbviewer.jupyter.org/gist/harrism/f5707335f40af9463c43)

A comparison of acceleration technologies for python is put together by 
[IBM](https://www.ibm.com/developerworks/community/blogs/jfp/entry/How_To_Compute_Mandelbrodt_Set_Quickly?lang=en)


# License

The project and all its files are provided under the GNU GENERAL PUBLIC LICENSE Version 3, see the file LICENSE.

(c) 2016-2017 Christian Hauck, all rights reserved.

