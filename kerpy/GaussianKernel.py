from kerpy.Kernel import Kernel
from numpy import exp, shape, reshape, sqrt, median
from numpy.random import permutation,randn
from scipy.spatial.distance import squareform, pdist, cdist
import warnings
from tools.GenericTests import GenericTests
import numpy as np

class GaussianKernel(Kernel):
    def __init__(self, sigma=1.0, is_sparse = False):
        Kernel.__init__(self)
        self.width = sigma
        self.is_sparse = is_sparse
    
    def __str__(self):
        s=self.__class__.__name__+ "["
        s += "width="+ str(self.width)
        s += "]"
        return s
    
    def kernel(self, X, Y=None):
        """
        Computes the standard Gaussian kernel k(x,y)=exp(-0.5* ||x-y||**2 / sigma**2)
        
        X - 2d numpy.ndarray, first set of samples:
            number of rows: number of samples
            number of columns: dimensionality
        Y - 2d numpy.ndarray, second set of samples, can be None in which case its replaced by X
        """
        if self.is_sparse:
            X = X.todense()
            Y = Y.todense()
        GenericTests.check_type(X, 'X',np.ndarray)
        assert(len(shape(X))==2)
        
        # if X=Y, use more efficient pdist call which exploits symmetry
        if Y is None:
            sq_dists = squareform(pdist(X, 'sqeuclidean'))
        else:
            GenericTests.check_type(Y, 'Y',np.ndarray)
            assert(len(shape(Y))==2)
            assert(shape(X)[1]==shape(Y)[1])
            sq_dists = cdist(X, Y, 'sqeuclidean')
        
        K = exp(-0.5 * (sq_dists) / self.width ** 2)
        return K
    
    
    def gradient(self, x, Y):
        """
        Computes the gradient of the Gaussian kernel wrt. to the left argument, i.e.
        k(x,y)=exp(-0.5* ||x-y||**2 / sigma**2), which is
        \nabla_x k(x,y)=1.0/sigma**2 k(x,y)(y-x)
        Given a set of row vectors Y, this computes the
        gradient for every pair (x,y) for y in Y.
        """
        if self.is_sparse:
            x = x.todense()
            Y = Y.todense()
        assert(len(shape(x))==1)
        assert(len(shape(Y))==2)
        assert(len(x)==shape(Y)[1])
        
        x_2d=reshape(x, (1, len(x)))
        k = self.kernel(x_2d, Y)
        differences = Y - x
        G = (1.0 / self.width ** 2) * (k.T * differences)
        return G
    
    
    def rff_generate(self,m,dim=1):
        self.rff_num=m
        self.unit_rff_freq=randn(int(m/2),dim)
        self.rff_freq=self.unit_rff_freq/self.width
    
    @staticmethod
    def get_sigma_median_heuristic(X, is_sparse = False):
        if is_sparse:
            X = X.todense()
        n=shape(X)[0]
        if n>1000:
            X=X[permutation(n)[:1000],:]
        dists=squareform(pdist(X, 'euclidean'))
        median_dist=median(dists[dists>0])
        sigma=median_dist/sqrt(2.)
        return sigma
