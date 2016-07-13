from TestObject import TestObject
from HSICTestObject import HSICTestObject
from numpy import mean, sum, zeros, var, sqrt
from scipy.stats import norm
import time

class HSICBlockTestObject(HSICTestObject):
    def __init__(self,num_samples, data_generator, kernelX, kernelY,
                 kernel_width_x=False,kernel_width_y=False,
                  rff=False, num_rfx=None, num_rfy=None,
                 blocksize=50, streaming=False, nullvarmethod='permutation', freeze_data=False):
        HSICTestObject.__init__(self, num_samples, data_generator, kernelX, kernelY, 
                                kernel_width_x=kernel_width_x,kernel_width_y=kernel_width_y,
                                 rff=rff, streaming=streaming, num_rfx=num_rfx, num_rfy=num_rfy,
                                freeze_data=freeze_data)
        self.blocksize = blocksize
        #self.blocksizeY = blocksizeY
        self.nullvarmethod = nullvarmethod
        
    def compute_pvalue(self):
        if not self.streaming and not self.freeze_data:
            start = time.clock()
            self.generate_data()
            data_generating_time = time.clock()-start
        else: 
            data_generating_time = 0.
        #print 'Total block data generating time passed: ', data_generating_time
        if self.kernel_width_x:
            sigmax = self.kernelX.get_sigma_median_heuristic(self.data_x)
            self.kernelX.set_width(float(sigmax))
        if self.kernel_width_y:
            sigmay = self.kernelY.get_sigma_median_heuristic(self.data_y)
            self.kernelY.set_width(float(sigmay))
        num_blocks = ( self.num_samples ) / self.blocksize
        block_statistics = zeros(num_blocks)
        null_samples = zeros(num_blocks)
        null_varx = zeros(num_blocks)
        null_vary = zeros(num_blocks)
        for bb in range(num_blocks):
            if self.streaming:
                data_x, data_y = self.data_generator(self.blocksize, self.blocksize)
            else:
                data_x = self.data_x[(bb*self.blocksize):((bb+1)*self.blocksize)]
                data_y = self.data_y[(bb*self.blocksize):((bb+1)*self.blocksize)]
            if self.nullvarmethod == 'permutation':
                block_statistics[bb], null_samples[bb], _, _, _, _, _ = \
                    self.HSICmethod(data_x=data_x, data_y=data_y, unbiased=True, num_shuffles=1, estimate_nullvar=False,isBlockHSIC=True)
            elif self.nullvarmethod == 'direct':
                block_statistics[bb], _, null_varx[bb], null_vary[bb], _, _, _ = \
                    self.HSICmethod(data_x=data_x, data_y=data_y, unbiased=True, num_shuffles=0, estimate_nullvar=True,isBlockHSIC=True)
            elif self.nullvarmethod == 'across':
                block_statistics[bb], _, _, _, _, _, _ = \
                    self.HSICmethod(data_x=data_x, data_y=data_y, unbiased=True, num_shuffles=0, estimate_nullvar=False,isBlockHSIC=True)
            else:
                raise NotImplementedError()
        BTest_Statistic = sum(block_statistics) / float(num_blocks)
        #print BTest_Statistic
        if self.nullvarmethod == 'permutation':
            BTest_NullVar = self.blocksize**2*var(null_samples)
        elif self.nullvarmethod == 'direct':
            overall_varx = mean(null_varx)
            overall_vary = mean(null_vary)
            BTest_NullVar = 2.*overall_varx*overall_vary
        elif self.nullvarmethod == 'across':
            BTest_NullVar = var(block_statistics)
        #print BTest_NullVar
        Z_score = sqrt(self.num_samples*self.blocksize)*BTest_Statistic / sqrt(BTest_NullVar) 
        #print Z_score
        pvalue = 2*min(norm.cdf(Z_score),norm.cdf(-Z_score))
        return pvalue, data_generating_time
    
    
