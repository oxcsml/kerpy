from numpy import shape, fill_diagonal, zeros, mean, sqrt,identity,dot,diag
from numpy.random import permutation, randn
from independence_testing.TestObject import TestObject
import numpy as np
from abc import abstractmethod
from kerpy.Kernel import Kernel
import time
from scipy.linalg import sqrtm,inv
from numpy.linalg import eigh,svd



class HSICTestObject(TestObject):
    def __init__(self, num_samples, data_generator=None, kernelX=None, kernelY=None, kernelZ = None,
                 kernelX_use_median=False,kernelY_use_median=False,kernelZ_use_median=False,
                  rff=False, num_rfx=None, num_rfy=None, induce_set=False, 
                  num_inducex = None, num_inducey = None,
                  streaming=False, freeze_data=False):
        TestObject.__init__(self,self.__class__.__name__,streaming=streaming, freeze_data=freeze_data)
        self.num_samples = num_samples #We have same number of samples from X and Y in independence testing
        self.data_generator = data_generator
        self.kernelX = kernelX
        self.kernelY = kernelY
        self.kernelZ = kernelZ
        self.kernelX_use_median = kernelX_use_median #indicate if median heuristic for Gaussian Kernel should be used
        self.kernelY_use_median = kernelY_use_median
        self.kernelZ_use_median = kernelZ_use_median
        self.rff = rff
        self.num_rfx = num_rfx
        self.num_rfy = num_rfy
        self.induce_set = induce_set
        self.num_inducex = num_inducex
        self.num_inducey = num_inducey
        if self.rff|self.induce_set: 
            self.HSICmethod = self.HSIC_with_shuffles_rff
        else:
            self.HSICmethod = self.HSIC_with_shuffles
    
    def generate_data(self,isConditionalTesting = False):
        if not isConditionalTesting:
            self.data_x, self.data_y = self.data_generator(self.num_samples)
            return self.data_x, self.data_y
        else: 
            self.data_x, self.data_y, self.data_z = self.data_generator(self.num_samples)
            return self.data_x, self.data_y, self.data_z
        ''' for our SimDataGen examples, one argument suffice'''
    
    
    @staticmethod
    def HSIC_U_statistic(Kx,Ky):
        m = shape(Kx)[0]
        fill_diagonal(Kx,0.)
        fill_diagonal(Ky,0.)
        K = np.dot(Kx,Ky)
        first_term = np.trace(K)/float(m*(m-3.))
        second_term = np.sum(Kx)*np.sum(Ky)/float(m*(m-3.)*(m-1.)*(m-2.))
        third_term = 2.*np.sum(K)/float(m*(m-3.)*(m-2.))
        return first_term+second_term-third_term
    
    
    @staticmethod
    def HSIC_V_statistic(Kx,Ky):
        Kxc=Kernel.center_kernel_matrix(Kx)
        Kyc=Kernel.center_kernel_matrix(Ky)
        return np.sum(Kxc*Kyc)
    
    
    @staticmethod
    def HSIC_V_statistic_rff(phix,phiy):
        m=shape(phix)[0]
        phix_c=phix-mean(phix,axis=0)
        phiy_c=phiy-mean(phiy,axis=0)
        featCov=(phix_c.T).dot(phiy_c)/float(m)
        return np.linalg.norm(featCov)**2
    
    
    # generalise distance correlation ---- a kernel interpretation
    @staticmethod
    def dCor_HSIC_statistic(Kx,Ky,unbiased=False):
        if unbiased:
            first_term = HSICTestObject.HSIC_U_statistic(Kx,Ky)
            second_term = HSICTestObject.HSIC_U_statistic(Kx,Kx)*HSICTestObject.HSIC_U_statistic(Ky,Ky)
            dCor = first_term/float(sqrt(second_term))
        else:
            first_term = HSICTestObject.HSIC_V_statistic(Kx,Ky)
            second_term = HSICTestObject.HSIC_V_statistic(Kx,Kx)*HSICTestObject.HSIC_V_statistic(Ky,Ky)
            dCor = first_term/float(sqrt(second_term))
        return dCor
    
    
    # approximated dCor using rff/Nystrom 
    @staticmethod
    def dCor_HSIC_statistic_rff(phix,phiy):
        first_term = HSICTestObject.HSIC_V_statistic_rff(phix,phiy)
        second_term = HSICTestObject.HSIC_V_statistic_rff(phix,phix)*HSICTestObject.HSIC_V_statistic_rff(phiy,phiy)
        approx_dCor = first_term/float(sqrt(second_term))
        return approx_dCor
    
    
    def SubdCor_HSIC_statistic(self,data_x=None,data_y=None,unbiased=True):
        if data_x is None:
            data_x=self.data_x
        if data_y is None:
            data_y=self.data_y
        dx = shape(data_x)[1]
        stats_value = zeros(dx)
        for dd in range(dx):
            Kx, Ky = self.compute_kernel_matrix_on_data(data_x[:,[dd]], data_y)
            stats_value[dd] = HSICTestObject.dCor_HSIC_statistic(Kx, Ky, unbiased)
        SubdCor = sum(stats_value)/float(dx)
        return SubdCor
    
    
    def SubHSIC_statistic(self,data_x=None,data_y=None,unbiased=True):
        if data_x is None:
            data_x=self.data_x
        if data_y is None:
            data_y=self.data_y
        dx = shape(data_x)[1]
        stats_value = zeros(dx)
        for dd in range(dx):
            Kx, Ky = self.compute_kernel_matrix_on_data(data_x[:,[dd]], data_y)
            if unbiased: 
                stats_value[dd] = HSICTestObject.HSIC_U_statistic(Kx, Ky) 
            else:
                stats_value[dd] = HSICTestObject.HSIC_V_statistic(Kx, Ky)
        SubHSIC = sum(stats_value)/float(dx)
        return SubHSIC
    
    
    def HSIC_with_shuffles(self,data_x=None,data_y=None,unbiased=True,num_shuffles=0,
                           estimate_nullvar=False,isBlockHSIC=False):
        start = time.clock()
        if data_x is None:
            data_x=self.data_x
        if data_y is None:
            data_y=self.data_y
        time_passed = time.clock()-start
        if isBlockHSIC:
            Kx, Ky = self.compute_kernel_matrix_on_dataB(data_x,data_y)
        else:
            Kx, Ky = self.compute_kernel_matrix_on_data(data_x,data_y)
        ny=shape(data_y)[0]
        if unbiased:
            test_statistic = HSICTestObject.HSIC_U_statistic(Kx,Ky)
        else:
            test_statistic = HSICTestObject.HSIC_V_statistic(Kx,Ky)
        null_samples=zeros(num_shuffles)
        for jj in range(num_shuffles):
            pp = permutation(ny)
            Kpp = Ky[pp,:][:,pp]
            if unbiased:
                null_samples[jj]=HSICTestObject.HSIC_U_statistic(Kx,Kpp)
            else:
                null_samples[jj]=HSICTestObject.HSIC_V_statistic(Kx,Kpp)
        if estimate_nullvar:
            nullvarx, nullvary = self.unbiased_HSnorm_estimate_of_centred_operator(Kx,Ky)
            nullvarx = 2.* nullvarx
            nullvary = 2.* nullvary
        else:
            nullvarx, nullvary = None, None
        return test_statistic,null_samples,nullvarx,nullvary,Kx, Ky, time_passed
    
    
    
    def HSIC_with_shuffles_rff(self,data_x=None,data_y=None,
                               unbiased=True,num_shuffles=0,estimate_nullvar=False):
        start = time.clock()
        if data_x is None:
            data_x=self.data_x
        if data_y is None:
            data_y=self.data_y
        time_passed = time.clock()-start
        if self.rff:
            phix, phiy = self.compute_rff_on_data(data_x,data_y)
        else:
            phix, phiy = self.compute_induced_kernel_matrix_on_data(data_x,data_y)
        ny=shape(data_y)[0]
        if unbiased:
            test_statistic = HSICTestObject.HSIC_U_statistic_rff(phix,phiy)
        else:
            test_statistic = HSICTestObject.HSIC_V_statistic_rff(phix,phiy)
        null_samples=zeros(num_shuffles)
        for jj in range(num_shuffles):
            pp = permutation(ny)
            if unbiased:
                null_samples[jj]=HSICTestObject.HSIC_U_statistic_rff(phix,phiy[pp])
            else:
                null_samples[jj]=HSICTestObject.HSIC_V_statistic_rff(phix,phiy[pp])
        if estimate_nullvar:
            raise NotImplementedError()
        else:
            nullvarx, nullvary = None, None
        return test_statistic, null_samples, nullvarx, nullvary,phix, phiy, time_passed
    
    
    def get_spectrum_on_data(self, Mx, My):
        '''Mx and My are Kx Ky when rff =False
           Mx and My are phix, phiy when rff =True'''
        if self.rff|self.induce_set:
            Cx = np.cov(Mx.T)
            Cy = np.cov(My.T)
            lambdax=np.linalg.eigvalsh(Cx)
            lambday=np.linalg.eigvalsh(Cy)
        else:
            Kxc = Kernel.center_kernel_matrix(Mx)
            Kyc = Kernel.center_kernel_matrix(My)
            lambdax=np.linalg.eigvalsh(Kxc)
            lambday=np.linalg.eigvalsh(Kyc)
        return lambdax,lambday
    
    
    @abstractmethod
    def compute_kernel_matrix_on_data(self,data_x,data_y):
        if self.kernelX_use_median:
            sigmax = self.kernelX.get_sigma_median_heuristic(data_x)
            self.kernelX.set_width(float(sigmax))
        if self.kernelY_use_median:
            sigmay = self.kernelY.get_sigma_median_heuristic(data_y)
            self.kernelY.set_width(float(sigmay))
        Kx=self.kernelX.kernel(data_x)
        Ky=self.kernelY.kernel(data_y)
        return Kx, Ky
    
    
    @abstractmethod
    def compute_kernel_matrix_on_dataB(self,data_x,data_y):
        Kx=self.kernelX.kernel(data_x)
        Ky=self.kernelY.kernel(data_y)
        return Kx, Ky
    
    
    
    @abstractmethod
    def compute_kernel_matrix_on_data_CI(self,data_x,data_y,data_z):
        if self.kernelX_use_median:
            sigmax = self.kernelX.get_sigma_median_heuristic(data_x)
            self.kernelX.set_width(float(sigmax))
        if self.kernelY_use_median:
            sigmay = self.kernelY.get_sigma_median_heuristic(data_y)
            self.kernelY.set_width(float(sigmay))
        if self.kernelZ_use_median:
            sigmaz = self.kernelZ.get_sigma_median_heuristic(data_z)
            self.kernelZ.set_width(float(sigmaz))
        Kx=self.kernelX.kernel(data_x)
        Ky=self.kernelY.kernel(data_y)
        Kz=self.kernelZ.kernel(data_z)
        return Kx, Ky,Kz
    
    
    
    
    def unbiased_HSnorm_estimate_of_centred_operator(self,Kx,Ky):
        '''returns an unbiased estimate of 2*Sum_p Sum_q lambda^2_p theta^2_q
        where lambda and theta are the eigenvalues of the centered matrices for X and Y respectively'''
        varx = HSICTestObject.HSIC_U_statistic(Kx,Kx)
        vary = HSICTestObject.HSIC_U_statistic(Ky,Ky)
        return varx,vary
    
    
    @abstractmethod
    def compute_rff_on_data(self,data_x,data_y):
        self.kernelX.rff_generate(self.num_rfx,dim=shape(data_x)[1])
        self.kernelY.rff_generate(self.num_rfy,dim=shape(data_y)[1])
        if self.kernelX_use_median:
            sigmax = self.kernelX.get_sigma_median_heuristic(data_x)
            self.kernelX.set_width(float(sigmax))
        if self.kernelY_use_median:
            sigmay = self.kernelY.get_sigma_median_heuristic(data_y)
            self.kernelY.set_width(float(sigmay))
        phix = self.kernelX.rff_expand(data_x)
        phiy = self.kernelY.rff_expand(data_y)
        return phix, phiy
    
    
    @abstractmethod
    def compute_induced_kernel_matrix_on_data(self,data_x,data_y):
        '''Z follows the same distribution as X; W follows that of Y.
        The current data generating methods we use 
        generate X and Y at the same time. '''
        size_induced_set = max(self.num_inducex,self.num_inducey)
        #print "size_induce_set", size_induced_set
        if self.data_generator is None:
            subsample_idx = np.random.randint(self.num_samples, size=size_induced_set)
            self.data_z = data_x[subsample_idx,:]
            self.data_w = data_y[subsample_idx,:]
        else:
            self.data_z, self.data_w = self.data_generator(size_induced_set)
            self.data_z[[range(self.num_inducex)],:]
            self.data_w[[range(self.num_inducey)],:]
        #print 'Induce Set'
        if self.kernelX_use_median:
            sigmax = self.kernelX.get_sigma_median_heuristic(data_x)
            self.kernelX.set_width(float(sigmax))
        if self.kernelY_use_median:
            sigmay = self.kernelY.get_sigma_median_heuristic(data_y)
            self.kernelY.set_width(float(sigmay))
        Kxz = self.kernelX.kernel(data_x,self.data_z)
        Kzz = self.kernelX.kernel(self.data_z)
        #R = inv(sqrtm(Kzz))
        R = inv(sqrtm(Kzz + np.eye(np.shape(Kzz)[0])*10**(-6)))
        phix = Kxz.dot(R)
        Kyw = self.kernelY.kernel(data_y,self.data_w)
        Kww = self.kernelY.kernel(self.data_w)
        #S = inv(sqrtm(Kww))
        S = inv(sqrtm(Kww + np.eye(np.shape(Kww)[0])*10**(-6)))
        phiy = Kyw.dot(S)
        return phix, phiy
    
    
    def compute_pvalue(self,data_x=None,data_y=None):
        pvalue,_=self.compute_pvalue_with_time_tracking(data_x,data_y)
        return pvalue
    
    
    
    
    
    
    
    
