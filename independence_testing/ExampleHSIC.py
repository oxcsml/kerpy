'''
Example script for running large-scale independence tests with HSIC
https://github.com/oxmlcs/kerpy
'''


#adding relevant folder to your pythonpath
import os, sys
BASE_DIR = os.path.join( os.path.dirname( __file__ ), '..' )
sys.path.append(BASE_DIR)


from kerpy.GaussianKernel import GaussianKernel
from SimDataGen import SimDataGen
from HSICTestObject import HSICTestObject
from numpy import shape,savetxt,loadtxt,transpose,shape,reshape,concatenate
from independence_testing.HSICSpectralTestObject import HSICSpectralTestObject
from independence_testing.HSICBlockTestObject import HSICBlockTestObject

'''
Given a data set data_x and data_y of paired observations, 
we wish to test the hypothesis of independence between the two. 
If dealing with vectorial data, data_x and data_y should be 2d-numpy arrays of shape (n,dim), 
where n is the number of observations and dim is the dimension of these observations 
--- note: one-dimensional observations should also be in a 2d-numpy array format (n,1)
'''



#here we simulate a dataset of size 'num_samples' in the correct format
num_samples = 10000
data_x, data_y = SimDataGen.LargeScale(num_samples, dimension=20)
#SimDataGen.py contains more examples of data generating functions


'''
# Alternatively, we can load a dataset from a file as follows: 
#-- here file is assumed to be a num_samples by (dimx+dimy) table
data = loadtxt('ExampleData.txt')
num_samples,D = shape(data)
#assume that x corresponds to all but the last column in the file
data_x = data[:,:(D-1)]
#and that y is just the last column
data_y = data[:,D-1]
#need to ensure data_y is a 2d array
data_y=reshape(data_y,(num_samples,1))
'''


print "shape of data_x:", shape(data_x)
print "shape of data_y:", shape(data_y)

'''
First, we need to specify the kernels for X and Y. We will use Gaussian kernels -- default value of the width parameter is 1.0
the widths can be either kept fixed or set to a median heuristic based on the data when running a test
'''
kernelX=GaussianKernel()
kernelY=GaussianKernel()




'''
HSICSpectralTestObject/HSICPermutationTestObject:
=================================================
num_samples:                Integer values -- the number of data samples 
data_generator:             If we use simulated data, which function to use to generate data for repeated tests to investigate power;
                            Examples are given in SimDataGen.py, e.g. data_generator = SimDataGen.LargeScale;
                            Default value is None (if only a single test will be run).
                            
kernelX, kernelY:           The kernel functions to use for X and Y respectively. (Examples are included in kerpy folder) 
                            E.g. kernelX = GaussianKernel(); alternatively, for a kernel with fixed width: kernelY = GaussianKernel(float(1.5))
kernelX_use_median,
kernelY_use_median:         "True" or "False" -- if median heuristic should be used to select the kernel bandwidth. 

rff:                        "True" or "False" -- if random Fourier Features should be used.
num_rfx, num_rfy:           Even integer values -- the number of random features for X and Y respectively.

induce_set:                 "True" or "False" -- if Nystrom method should be used.
num_inducex, num_inducey:    Integer values -- the number of inducing variables for X and Y respectively.

num_nullsims:                An integer value -- the number of simulations from the null distribution for spectral approach.
num_shuffles:                An integer value -- the number of shuffles for permutation approach.
unbiased:                    "True" or "False" -- if unbiased HSIC test statistics is preferred.


HSICBlockTestObject: 
====================
blocksize:                  Integer value -- the size of each block. 
nullvarmethod:              "permutation", "direct" or "across" -- the method of estimating the null variance. 
                            Refer to the paper for more details of each.
'''


#example usage of HSIC spectral test with random Fourier feature approximation
myspectralobject = HSICSpectralTestObject(num_samples, kernelX=kernelX, kernelY=kernelY, 
                                          kernelX_use_median=True, kernelY_use_median=True,
                                          rff=True, num_rfx=20, num_rfy=20, num_nullsims=1000)
pvalue = myspectralobject.compute_pvalue(data_x, data_y)

print "Spectral test p-value:", pvalue

#example usage of HSIC block test:
myblockobject = HSICBlockTestObject(num_samples, kernelX=kernelX, kernelY=kernelY, 
                                    kernelX_use_median=True, kernelY_use_median=True,
                                    blocksize=50, nullvarmethod='permutation')
pvalue = myblockobject.compute_pvalue(data_x, data_y)

print "Block test p-value:", pvalue