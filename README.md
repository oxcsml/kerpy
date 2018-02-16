# kerpy
python code framework for kernel methods in hypothesis testing. 
some code on kernel computation was adapted from https://github.com/karlnapf/kameleon-mcmc 

To set up as a package run in terminal:

``python setup.py develop``


### independence_testing

Code for HSIC-based large-scale independence tests. The methods are described in:

Q. Zhang, S. Filippi, A. Gretton, and D. Sejdinovic, __Large-Scale Kernel Methods for Independence Testing__, _Statistics and Computing_, to appear, 2017. [url](http://link.springer.com/article/10.1007%2Fs11222-016-9721-7)

For an example use of the code, demonstrating how to run an HSIC-based large-scale independence test on either simulated data or data loaded from a file, see ExampleHSIC.py. 

To reproduce results from the paper, see ExperimentsHSICPermutation.py, ExperimentsHSICSpectral.py, ExperimentsHSICBlock.py. 



### weak_conditional_independence_testing

Code for feature-to-feature regression for a two-step conditional independence tests (i.e. testing for weak conditional independence). The methods are described in:

Q. Zhang, S. Filippi, S. Flaxman, and D. Sejdinovic, __Feature-to-Feature Regression for a Two-Step Conditional Independence Test__, UAI, 2017.


To reproduce results from the paper, see WHO_KRESITvsRESIT.py, SyntheticDim_KRESIT.py, PCalg_twostep_flags.py (the nodes names correspond to the varibales in Synthetic_DAGexample.csv file, to run it for Boston Housing Data (BH_prewhiten.csv) or Ozone Data (Ozone_prewhiten.csv) simply change the label names in the script.)
