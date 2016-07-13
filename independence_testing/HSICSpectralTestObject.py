'''
Created on 15 Nov 2015

@author: qinyi
'''
from HSICTestObject import HSICTestObject
import numpy as np
import time

class HSICSpectralTestObject(HSICTestObject):

    def __init__(self, num_samples, data_generator, kernelX, kernelY, kernel_width_x=False,kernel_width_y=False,
                 rff=False,num_rfx=None,num_rfy=None,induce_set=False, num_inducex = None, num_inducey = None,
                 num_nullsims=1000, unbiased=True):
        HSICTestObject.__init__(self, num_samples, data_generator, kernelX, kernelY, kernel_width_x=kernel_width_x,
                                kernel_width_y=kernel_width_y, num_rfx=num_rfx, num_rfy=num_rfy, rff=rff,
                                induce_set=induce_set, num_inducex = num_inducex, num_inducey = num_inducey)
        self.num_nullsims = num_nullsims
        self.unbiased = unbiased
    
    
    def get_null_samples_with_spectral_approach(self,Mx,My):
        lambdax, lambday = self.get_spectrum_on_data(Mx,My)
        Dx=len(lambdax)
        Dy=len(lambday)
        null_samples=np.zeros(self.num_nullsims)
        for jj in range(self.num_nullsims):
            zz=np.random.randn(Dx,Dy)**2
            if self.unbiased:
                zz = zz - 1
            null_samples[jj]=np.dot(lambdax.T,np.dot(zz,lambday))
        return null_samples
    
    def compute_pvalue(self):
        start = time.clock()
        if not self.streaming and not self.freeze_data:
            self.generate_data()
        data_generating_time = time.clock()-start
        print 'data generating time passed: ', data_generating_time
        hsic_statistic, _, _, _, Mx, My, _ = self.HSICmethod(unbiased=self.unbiased)
        null_samples = self.get_null_samples_with_spectral_approach(Mx, My)
        pvalue = ( sum( null_samples > self.num_samples*hsic_statistic ) ) / float( self.num_nullsims )
        return pvalue, data_generating_time
