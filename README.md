# kerpy
python code framework for kernel methods in hypothesis testing. 
some code on kernel computation was adapted from https://github.com/karlnapf/kameleon-mcmc 


### independence_testing

Code for HSIC-based large-scale independence tests. The methods are described in:

Q. Zhang, S. Filippi, A. Gretton, and D. Sejdinovic, __Large-Scale Kernel Methods for Independence Testing__, _Statistics and Computing_, to appear, 2017. [url](http://link.springer.com/article/10.1007%2Fs11222-016-9721-7)

For an example use of the code, demonstrating how to run an HSIC-based large-scale independence test on either simulated data or data loaded from a file, see ExampleHSIC.py. 

To reproduce results from the paper, see ExperimentsHSICPermutation.py, ExperimentsHSICSpectral.py, ExperimentsHSICBlock.py. 
