from TestObject import TestObject
from numpy import shape, zeros
from scipy.stats import pearsonr
import time
from numpy.random import permutation


class CorrTestObject(TestObject):
    def __init__(self, num_samples, data_generator, streaming=False, freeze_data=False,num_shuffles=1000):
        TestObject.__init__(self,self.__class__.__name__,streaming=streaming, freeze_data=freeze_data)
        self.num_samples = num_samples #We have same number of samples from X and Y in independence testing
        self.data_generator = data_generator
        self.num_shuffles = num_shuffles
    
    
    def generate_data(self):
        self.data_x, self.data_y = self.data_generator(self.num_samples)
        return self.data_x, self.data_y
    
    
    def SubCorr_statistic(self,data_x=None,data_y=None):
        if data_x is None:
            data_x=self.data_x
        if data_y is None:
            data_y=self.data_y
        dx = shape(data_x)[1]
        stats_value = zeros(dx)
        for dd in range(dx):
            stats_value[dd] = pearsonr(data_x[:,[dd]],data_y)[0]**2
        SubCorr = sum(stats_value)/float(dx)
        return SubCorr
    
    
    def compute_pvalue_with_time_tracking(self,data_x = None, data_y = None):
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
        print 'data generating time passed: ', data_generating_time
        SubCorr_statistic = self.SubCorr_statistic(data_x=data_x,data_y=data_y)
        null_samples=zeros(self.num_shuffles)
        for jj in range(self.num_shuffles):
            pp = permutation(self.num_samples)
            yy = self.data_y[pp,:]
            null_samples[jj]=self.SubCorr_statistic(data_x = data_x, data_y = yy)
        pvalue = ( sum( null_samples > SubCorr_statistic ) ) / float( self.num_shuffles )
        return pvalue, data_generating_time
    
    
    
    