from kerpy.Kernel import Kernel
import numpy as np
from tools.GenericTests import GenericTests
from abc import abstractmethod

class BagKernel(Kernel):
    def __init__(self,data_kernel):
        Kernel.__init__(self)
        self.data_kernel=data_kernel
        
    def __str__(self):
        s="BagKernel["
        s += "data_kernel=" + self.data_kernel.__str__()
        s += "]"
        return s
    
    def kernel(self, bagX, bagY=None):
        #GenericTests.check_type(bagX,'bagX',list)
        nx=len(bagX)
        if bagY is None:
            K=np.zeros((nx,nx))
            for ii in range(nx):
                zi = bagX[ii]
                for jj in range(ii,nx):
                    zj = bagX[jj]
                    K[ii,jj]=self.compute_BagKernel_value(zi,zj)
            K=self.symmetrize(K)
        else:
            #GenericTests.check_type(bagY,'bagY',list)
            ny=len(bagY)
            K=np.zeros((nx,ny))
            for ii in range(nx):
                zi = bagX[ii]
                for jj in range(ny):
                    zj = bagY[jj]
                    K[ii,jj]=self.compute_BagKernel_value(zi,zj)
        return K
    
    @abstractmethod
    def compute_BagKernel_value(self,bag1,bag2):
        raise NotImplementedError()
    
    @staticmethod
    def symmetrize(a):
        return a + a.T - np.diag(a.diagonal())