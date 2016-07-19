'''
adding relevant folder to your pythonpath
'''
import os, sys
BASE_DIR = os.path.join( os.path.dirname( __file__ ), '..' )
sys.path.append(BASE_DIR)


from kerpy.GaussianKernel import GaussianKernel
from SimDataGen import SimDataGen
from HSICTestObject import HSICTestObject
from numpy import shape,savetxt,loadtxt,transpose
from independence_testing.HSICSpectralTestObject import HSICSpectralTestObject
from independence_testing.HSICBlockTestObject import HSICBlockTestObject

'''
Given a data set data_x and data_y, we wish to test the independence between the two. 
Both data_x and data_y should have the number of observations being the number of rows.  
As an example, we simulate some data here. (Note, either data are simulated through the data_generator, 
or direct values for X and Y should be given.)

'''

num_samples = 10000
dim = 20
data_x, data_y = SimDataGen.LargeScale(num_samples, dim)

# Or, if we import the data from some txt files. For convenience, we illustrate by just 
# saving the above simulated data and loaded again.
savetxt('ExampleDataX.txt', data_x) 
savetxt('ExampleDataY.txt',data_y)
data_x = loadtxt('ExampleDataX.txt')
data_y = loadtxt('ExampleDataY.txt')
# As data_y is 1-D, so we put it into our desired data structure.
data_y = transpose([data_y])


'''
First, we need to specify the kernels for X and Y. We will use Gaussian kernels -- default value of the width parameter is 1.0
the widths can be either kept fixed or set to a median heuristic based on the data when running a test
'''
kernelX=GaussianKernel()
kernelY=GaussianKernel()

data_generator = None


'''
HSICSpectralTestObject: (or HSICPermutationTestObject)
=====================================================
num_samples:                Integer values -- the number of data samples 
data_generator:             if we use simulated data, which function to use (examples are in SimDataGen.py). 
                            E.g. data_generator = SimDataGen.LargeScale
                            
kernelX, kernelY =          the kernel functions to use for X and Y respectively. (Examples are included in kerpy folder) 
                            E.g. kernelX = GaussianKernel()
kernelX_use_median:         Takes "True" or "False" -- if median heuristic should be used. 

rff:                        Takes "True" or "False" -- if random Fourier Features should be used.
num_rfx, num_rfy:           Takes even integers, gives the number of random features for X and Y respectively.

induce_set:                 "True" or "False" -- if Nystrom method should be used.
num_inducex, num_inducey:    Takes integers, gives the number of inducing variables for X and Y respectively.

num_nullsims:                An integer value -- the number of simulations from the null distribution for Spectral approach.
num_shuffles:                An integer value -- the number of shuffles for permutation approach.
unbiased:                    "True" or "False" -- if unbiased HSIC test statistics should be used.


HSICBlockTestObject: 
====================
blocksize:                  Integer value -- the size of each block. 
nullvarmethod:              "permutation", "direct" or "across" -- the method of estimating the null variance. 
                            Refer to the paper for more details of each.
'''


#HSIC Spectral Test using random Fourier features, as specified by rff = True.
myspectralobject = HSICSpectralTestObject(num_samples, data_generator, kernelX, kernelY, 
                kernelX_use_median=True,kernelY_use_median=True,
                rff=True, num_rfx=20, num_rfy=20, num_nullsims=1000)
pvalue,_ = myspectralobject.compute_pvalue(data_x, data_y)

print "Spectral test p-value:", pvalue

# Or, if we would like to use HSIC Block Test:
myblockobject = HSICBlockTestObject(num_samples, data_generator, kernelX, kernelY,
                 kernelX_use_median=True, kernelY_use_median=True,
                 blocksize=50, nullvarmethod='permutation')
pvalue,_ = myblockobject.compute_pvalue(data_x, data_y)

print "Block test p-value:", pvalue