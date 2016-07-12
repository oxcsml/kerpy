from kerpy.Kernel import Kernel
import numpy as np


class SumKernel(Kernel):
    def __init__(self, list_of_kernels):
        Kernel.__init__(self)
        self.list_of_kernels = list_of_kernels
        
    def __str__(self):
        s=self.__class__.__name__+ "=["
        s += ", " + Kernel.__str__(self)
        s += "]"
        return s
    
    def kernel(self, X, Y=None):
        return np.sum([individual_kernel.kernel(X,Y) for individual_kernel in self.list_of_kernels],0)