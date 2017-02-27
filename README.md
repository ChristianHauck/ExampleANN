-# ExampleANN

Python 2.7 code for a Single-Layer-Perceptron (slp) and a flexible Multi-Layer-Perceptron (mlp)

The code is not designed to be efficient, sophisticated or whatever - but just to be easy to read and understand. The implementation was done according to the pseudo-code in "Russell,Norvig, Artificial Intelligence" and "Hertz,Krogh,Palmer, Introduction to the Theory of Neural Computation". Look there (or in any other textbook chapter about ANNs) for a detailed explanation.

**Update:** Much more efficient version added: MLP2 / Layer2. This version is up to 1600 times faster than MLP / Layer - simply because I missed a single numpy operation. Implicitely realized as a python loop resulted in this terrible slow down. To start a runtime comparison simply start program.py.

## License

The project and all its files are provided under the GNU GENERAL PUBLIC LICENSE Version 3, see the file LICENSE.

(c) 2016-2017 Christian Hauck, all rights reserved.

