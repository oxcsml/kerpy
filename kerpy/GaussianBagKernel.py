from kerpy.BagKernel import BagKernel
from abc import abstractmethod
from numpy import exp, zeros, dot, cos, sin, concatenate, sqrt, mean, median
from numpy.random.mtrand import randn
import numpy as np

class GaussianBagKernel(BagKernel):
    def __init__(self,data_kernel,sigma=0.5):
        BagKernel.__init__(self,data_kernel)
        self.width=sigma
        
    def __str__(self):
        s=self.__class__.__name__+ "["
        s += "width="+ str(self.width)
        s += ", " + BagKernel.__str__(self)
        s += "]"
        return s
    
    def rff_generate(self,mbags,mdata=100,dim=1):
        '''
        mbags:: number of random features for bag kernel
        mdata:: number of random features for data kernel
        dim:: data dimensionality
        '''
        self.data_kernel.rff_generate(mdata,dim=dim)
        self.rff_num=mbags
        self.unit_rff_freq=randn(mbags/2,mdata)
        self.rff_freq=self.unit_rff_freq/self.width
    
    def rff_expand(self,bagX):
        if self.rff_freq is None:
            raise ValueError("rff_freq has not been set. use rff_generate first")
        nx=len(bagX)
        featuremeans=zeros((nx,self.data_kernel.rff_num))
        for ii in range(nx):
            featuremeans[ii]=mean(self.data_kernel.rff_expand(bagX[ii]),axis=0)
        xdotw=dot(featuremeans,(self.rff_freq).T)
        return sqrt(2./self.rff_num)*concatenate( ( cos(xdotw),sin(xdotw) ) , axis=1 )
    
    
    def compute_BagKernel_value(self,bag1,bag2):
        return exp(-0.5 * self.data_kernel.estimateMMD(bag1,bag2) / self.width ** 2)
    
    def get_sigma_median_heuristic(self,X):
        nx=np.shape(X)[0]
        if nx>200:
            X=X[np.random.permutation(nx)[:200]]
        n=min(nx,200)
        D=zeros((n,n))
        for ii in range(n):
            zi = X[ii]
            for jj in range(ii+1,n):
                zj = X[jj]
                D[ii,jj]=sqrt(self.data_kernel.estimateMMD(zi,zj))
        D=self.symmetrize(D)
        median_dist=median(D[D>0])
        sigma=median_dist/sqrt(2.)
        return sigma
    
if __name__ == '__main__':
    from tools.UnitTests import UnitTests
    UnitTests.UnitTestBagKernel(GaussianBagKernel)