from matplotlib.pyplot import show, imshow
from numpy import shape, reshape
import numpy as np
from scipy.spatial.distance import squareform, pdist, cdist

from Kernel import Kernel
from tools.GenericTests import GenericTests


class BrownianKernel(Kernel):
    def __init__(self, alpha=1.0):
        Kernel.__init__(self)
        GenericTests.check_type(alpha,'alpha',float)
        self.alpha = alpha
    
    def __str__(self):
        s=self.__class__.__name__+ "=["
        s += "alpha="+ str(self.alpha)
        s += ", " + Kernel.__str__(self)
        s += "]"
        return s
    
    def set_kerpar(self, kerpar):
        if (kerpar<=0 or kerpar>=2):
            raise ValueError("incorrect parameter value")
        self.alpha=kerpar
    
    
    def kernel(self, X, Y=None):
        
        GenericTests.check_type(X,'X',np.ndarray,2)
        # if X=Y, use more efficient pdist call which exploits symmetry
        normX=reshape(np.linalg.norm(X,axis=1),(len(X),1))
        if Y is None:
            dists = squareform(pdist(X, 'euclidean'))
            normY=normX.T
        else:
            GenericTests.check_type(Y,'Y',np.ndarray,2)
            assert(shape(X)[1]==shape(Y)[1])
            normY=reshape(np.linalg.norm(Y,axis=1),(1,len(Y)))
            dists = cdist(X, Y, 'euclidean')
        K=0.5*(normX**self.alpha+normY**self.alpha-dists**self.alpha)
        return K
    
    def gradient(self, x, Y):
        raise NotImplementedError()
    
if __name__ == '__main__':
    from tools.UnitTests import UnitTests
    UnitTests.UnitTestDefaultKernel(BrownianKernel)
