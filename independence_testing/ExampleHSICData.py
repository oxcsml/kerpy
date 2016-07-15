from kerpy.GaussianKernel import GaussianKernel
from SimDataGen import SimDataGen
from HSICTestObject import HSICTestObject
from numpy import shape
from independence_testing.HSICSpectralTestObject import HSICSpectralTestObject
from independence_testing.HSICBlockTestObject import HSICBlockTestObject

'''
Given a data set data_x and data_y, we wish to test the independence between the two. 
Both data_x and data_y should have the number of observations being the number of rows.  
As an example, we simulate some data here.
First, we need to specify the kernels for X and Y. Here, we use Gaussian Kernel with Median heuristic for both. But, 
we will use a temporary setup for the bandwidth.  
'''

data_x, data_y = SimDataGen.SimpleLn(200, 4)
num_samples = shape(data_x)[0]
kernelX=GaussianKernel(1.)
kernelY=GaussianKernel(1.)

data_generator = None

# We use the exact HSIC Spectral Test:
myspectralobject = HSICSpectralTestObject(num_samples, data_generator, kernelX, kernelY, kernel_width_x=True,kernel_width_y=True,
                 rff=False,num_rfx=None,num_rfy=None,induce_set=False, num_inducex = None, num_inducey = None,
                 num_nullsims=1000, unbiased=False)
pvalue,_ = myspectralobject.compute_pvalue(data_x, data_y)

# Or, if we would like to use HSIC Block Test:
myblockobject = HSICBlockTestObject(num_samples, data_generator, kernelX, kernelY,
                 kernel_width_x=True,kernel_width_y=True,
                 blocksize=20, streaming=False, nullvarmethod='permutation', freeze_data=False)

pvalue,_ = myblockobject.compute_pvalue(data_x, data_y)