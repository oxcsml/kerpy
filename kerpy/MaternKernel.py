from matplotlib.pyplot import show, imshow
from numpy import exp, shape, sqrt, reshape
import numpy as np
from scipy.spatial.distance import squareform, pdist, cdist


from kerpy.Kernel import Kernel
from tools.GenericTests import GenericTests


class MaternKernel(Kernel):
    def __init__(self, width, nu=1.5, sigma=1.0):
        Kernel.__init__(self)
        #GenericTests.check_type(width,'width',float)
        GenericTests.check_type(nu,'nu',float)
        GenericTests.check_type(sigma,'sigma',float)
        
        self.width = width
        self.nu = nu
        self.sigma = sigma
    
    def __str__(self):
        s=self.__class__.__name__+ "["
        s += "width="+ str(self.width)
        s += ", nu="+ str(self.nu)
        s += ", sigma="+ str(self.sigma)
        s += "]"
        return s
    
    def kernel(self, X, Y=None):
        
        GenericTests.check_type(X,'X',np.ndarray,2)
        # if X=Y, use more efficient pdist call which exploits symmetry
        if Y is None:
            dists = squareform(pdist(X, 'euclidean'))
        else:
            GenericTests.check_type(Y,'Y',np.ndarray,2)
            assert(shape(X)[1]==shape(Y)[1])
            dists = cdist(X, Y, 'euclidean')
        if self.nu==0.5:
            #for nu=1/2, Matern class corresponds to Ornstein-Uhlenbeck Process
            K = (self.sigma**2.) * exp( -dists / self.width )                 
        elif self.nu==1.5:
            K = (self.sigma**2.) * (1+ sqrt(3.)*dists / self.width) * exp( -sqrt(3.)*dists / self.width )
        elif self.nu==2.5:
            K = (self.sigma**2.) * (1+ sqrt(5.)*dists / self.width + 5.0*(dists**2.) / (3.0*self.width**2.) ) * exp( -sqrt(5.)*dists / self.width )
        else:
            raise NotImplementedError()
        return K
    
    def gradient(self, x, Y):
        assert(len(shape(x))==1)
        assert(len(shape(Y))==2)
        assert(len(x)==shape(Y)[1])
        
        if self.nu==1.5 or self.nu==2.5:
            x_2d=reshape(x, (1, len(x)))
            lower_order_width = self.width * sqrt(2*(self.nu-1)) / sqrt(2*self.nu)
            lower_order_kernel = MaternKernel(lower_order_width,self.nu-1,self.sigma)
            k = lower_order_kernel.kernel(x_2d, Y)
            differences = Y - x
            G = ( 1.0 / lower_order_width ** 2 ) * (k.T * differences)
            return G
        else:
            raise NotImplementedError()
    
if __name__ == '__main__':
    Z = np.random.randn(50,2)
    Z2 = np.random.randn(50,2)
    kernel = MaternKernel(1.0)
    K = kernel.kernel(Z, Z2)
    imshow(K, interpolation="nearest")
    show()

