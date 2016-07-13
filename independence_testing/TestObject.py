"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Dino Sejdinovic
"""

from abc import abstractmethod
from scipy.stats import norm as normaldist



class TestObject(object):
    def __init__(self, test_type, streaming=False, freeze_data=False):
        self.test_type=test_type
        self.streaming=streaming
        self.freeze_data=freeze_data
        if self.freeze_data:
            self.generate_data()
            assert not self.streaming
    
    @abstractmethod
    def compute_Zscore(self):
        raise NotImplementedError
    
    @abstractmethod
    def generate_data(self):
        raise NotImplementedError
    
    def compute_pvalue(self):
        Z_score = self.compute_Zscore()
        pvalue = normaldist.sf(Z_score)
        return pvalue
    
    def perform_test(self, alpha):
        pvalue=self.compute_pvalue()
        return pvalue<alpha


