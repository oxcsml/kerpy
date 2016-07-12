from kerpy.BagKernel import BagKernel
import numpy as np
from tools.GenericTests import GenericTests
from kerpy.GaussianKernel import GaussianKernel
from abc import abstractmethod

class LinearBagKernel(BagKernel):
    def __init__(self,data_kernel):
        BagKernel.__init__(self,data_kernel)
        
    def __str__(self):
        s=self.__class__.__name__+ "["
        s += "" + BagKernel.__str__(self)
        s += "]"
        return s
    
    def rff_generate(self,mdata=20,dim=1):
        '''
        mdata:: number of random features for data kernel
        dim:: data dimensionality
        '''
        self.data_kernel.rff_generate(mdata,dim=dim)
        self.rff_num=mdata
    
    def rff_expand(self,bagX):
        nx=len(bagX)
        featuremeans=np.zeros((nx,self.data_kernel.rff_num))
        for ii in range(nx):
            featuremeans[ii]=np.mean(self.data_kernel.rff_expand(bagX[ii]),axis=0)
        return featuremeans
    
    def compute_BagKernel_value(self,bag1,bag2):
        innerK=self.data_kernel.kernel(bag1,bag2)
        return np.mean(innerK[:])
    
    
if __name__ == '__main__':
    from tools.UnitTests import UnitTests
    UnitTests.UnitTestBagKernel(LinearBagKernel)