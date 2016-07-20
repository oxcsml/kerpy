from HSICTestObject import HSICTestObject
from numpy import zeros
import time
from numpy.random import permutation

class SubHSICTestObject(HSICTestObject):

    def __init__(self, num_samples, data_generator=None, kernelX=None, kernelY=None, kernelX_use_median=False,
                 kernelY_use_median=False, num_rfx=None, num_rfy=None, rff=False, num_shuffles=1000, unbiased=True):
        HSICTestObject.__init__(self, num_samples, data_generator=data_generator, kernelX=kernelX, kernelY=kernelY, 
                                kernelX_use_median=kernelX_use_median,kernelY_use_median=kernelY_use_median,
                                num_rfx=num_rfx, num_rfy=num_rfy, rff=rff)
        self.num_samples = num_samples
        self.num_shuffles = num_shuffles
        self.unbiased = unbiased
        
     
    def compute_pvalue_with_time_tracking(self,data_x=None,data_y=None):
        if data_x is None and data_y is None:
            if not self.streaming and not self.freeze_data:
                start = time.clock()
                self.generate_data()
                data_generating_time = time.clock()-start
                data_x = self.data_x
                data_y = self.data_y
            else:
                data_generating_time = 0.
        else:
            data_generating_time = 0.
        #print 'data generating time passed: ', data_generating_time
        SubHSIC_statistic = self.SubHSIC_statistic(unbiased=self.unbiased,data_x=data_x, data_y = data_y)
        null_samples=zeros(self.num_shuffles)
        for jj in range(self.num_shuffles):
            pp = permutation(self.num_samples)
            yy = self.data_y[pp,:]
            null_samples[jj]=self.SubHSIC_statistic(data_x = data_x, data_y = yy, unbiased = self.unbiased)
        pvalue = ( sum( null_samples > SubHSIC_statistic ) ) / float( self.num_shuffles )
        return pvalue, data_generating_time