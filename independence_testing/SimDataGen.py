from numpy.random import uniform, permutation, multivariate_normal,normal
from numpy import pi, prod, empty, sin, cos, asscalar, shape,zeros, identity,arange,sign,sum,sqrt,transpose, tanh,sinh
import numpy as np 



class SimDataGen(object):
    def __init__(self):
        pass
    
    
    @staticmethod
    def LargeScale(num_samples, dimension=4):
        ''' dimension takes large even numbers, e.g. 50, 100 '''
        Xmean = zeros(dimension)
        Xcov = identity(dimension)
        data_x = multivariate_normal(Xmean, Xcov, num_samples)
        dd = dimension/2
        Zmean = zeros(dd+1)
        Zcov = identity(dd+1)
        Z = multivariate_normal(Zmean, Zcov, num_samples)
        first_term = sqrt(2./dimension)*sum(sign(data_x[:,arange(0,dimension,2)]* data_x[:,arange(1,dimension,2)])*abs(Z[:,range(dd)]),axis=1,keepdims=True)
        second_term = Z[:,[dd]] #take the last dimension of Z 
        data_y = first_term + second_term
        return data_x, data_y
    
    
    @staticmethod
    def VaryDimension(num_samples, dimension = 5):
        Xmean = zeros(dimension)
        Xcov = identity(dimension)
        data_x = multivariate_normal(Xmean, Xcov, num_samples)
        data_z = transpose([normal(0,1,num_samples)])
        data_y = 20*sin(4*pi*(data_x[:,[0]]**2 + data_x[:,[1]]**2)) + data_z
        return data_x,data_y
    
    
    @staticmethod
    def SimpleLn(num_samples, dimension = 5):
        Xmean = zeros(dimension)
        Xcov = identity(dimension)
        data_x = multivariate_normal(Xmean, Xcov, num_samples)
        data_z = transpose([normal(0,1,num_samples)])
        data_y = data_x[:,[0]] + data_z
        return data_x, data_y
    
    
    @staticmethod
    def turn_into_null(fn):
        def null_fn(*args, **kwargs):
            dataX,dataY=fn(*args, **kwargs)
            num_samples=shape(dataX)[0]
            pp = permutation(num_samples)
            return dataX,dataY[pp]
        return null_fn
