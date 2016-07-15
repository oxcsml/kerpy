'''
adding relevant folder to your pythonpath
'''
import os, sys
BASE_DIR = os.path.join( os.path.dirname( __file__ ), '..' )
sys.path.append(BASE_DIR)


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

'''

num_samples = 10000
dim = 20
data_x, data_y = SimDataGen.LargeScale(num_samples, dim)

'''
First, we need to specify the kernels for X and Y. We will use Gaussian kernels -- default value of the width parameter is 1.0
the widths can be either kept fixed or set to a median heuristic based on the data when running a test
'''
kernelX=GaussianKernel()
kernelY=GaussianKernel()

data_generator = None

#HSIC Spectral Test using random Fourier features, as specified by rff = True
myspectralobject = HSICSpectralTestObject(num_samples, data_generator, kernelX, kernelY, 
                kernel_width_x=True,kernel_width_y=True,
                rff=True, num_rfx=20, num_rfy=20, num_nullsims=1000)
pvalue,_ = myspectralobject.compute_pvalue(data_x, data_y)

print "Spectral test p-value:", pvalue

# Or, if we would like to use HSIC Block Test:
myblockobject = HSICBlockTestObject(num_samples, data_generator, kernelX, kernelY,
                 kernel_width_x=True, kernel_width_y=True,
                 blocksize=50, nullvarmethod='permutation')
pvalue,_ = myblockobject.compute_pvalue(data_x, data_y)

print "Block test p-value:", pvalue