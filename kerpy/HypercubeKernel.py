from numpy import tanh
import numpy
from scipy.spatial.distance import squareform, pdist, cdist

from kerpy.Kernel import Kernel


class HypercubeKernel(Kernel):
    def __init__(self, gamma):
        Kernel.__init__(self)
        
        if type(gamma) is not float:
            raise TypeError("Gamma must be float")
        
        self.gamma = gamma
    
    def __str__(self):
        s = self.__class__.__name__ + "=["
        s += "gamma=" + str(self.gamma)
        s += ", " + Kernel.__str__(self)
        s += "]"
        return s
    
    def kernel(self, X, Y=None):
        """
        Computes the hypercube kerpy k(x,y)=tanh(gamma)^d(x,y), where d is the
        Hamming distance between x and y
        
        X - 2d numpy.bool8 array, samples on right left side
        Y - 2d numpy.bool8 array, samples on left hand side.
            Can be None in which case its replaced by X
        """
        
        if not type(X) is numpy.ndarray:
            raise TypeError("X must be numpy array")
        
        if not len(X.shape) == 2:
            raise ValueError("X must be 2D numpy array")
        
        if not X.dtype == numpy.bool8:
            raise ValueError("X must be boolean numpy array")
        
        if not Y is None:
            if not type(Y) is numpy.ndarray:
                raise TypeError("Y must be None or numpy array")
            
            if not len(Y.shape) == 2:
                raise ValueError("Y must be None or 2D numpy array")
            
            if not Y.dtype == numpy.bool8:
                raise ValueError("Y must be boolean numpy array")
        
            if not X.shape[1] == Y.shape[1]:
                raise ValueError("X and Y must have same dimension if Y is not None")
        
        # un-normalise normalised hamming distance in both cases
        if Y is None:
            K = tanh(self.gamma) ** squareform(pdist(X, 'hamming') * X.shape[1])
        else:
            K = tanh(self.gamma) ** (cdist(X, Y, 'hamming') * X.shape[1])
            
        return K
    
    def gradient(self, x, Y):
        """
        Computes the gradient of the hypercube kerpy wrt. to the left argument
        
        x - single sample on right hand side (1D vector)
        Y - samples on left hand side (2D matrix)
        """
        pass

