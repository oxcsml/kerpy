"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Dino Sejdinovic
"""
from numpy import arange,zeros,mean
import os
from pickle import load, dump
import time

class TestExperiment(object):
    def __init__(self, name, param, test_object):
        self.name=name
        self.param=param
        self.test_object=test_object
        self.folder="results/res_"+self.name
    
    def compute_pvalue(self):
        return self.test_object.compute_pvalue()
    
    def compute_pvalue_with_time_tracking(self):
        return self.test_object.compute_pvalue_with_time_tracking()
    
    def perform_test(self, alpha):
        return self.test_object.perform_test(alpha)
    
    def run_test_trials(self, numTrials, alpha=0.05):
        completedTrials = 0
        counter_init = 0
        save_filename = self.folder+"/results.bin"
        average_time_init=0.0
        pvalues_init=list()
        if not os.path.exists(self.folder):
            os.mkdir(self.folder)
        elif os.path.exists(save_filename):
            load_f = open(save_filename,"r")
            [counter_init, completedTrials, _, average_time_init,pvalues_init] = load(load_f)
            load_f.close()
            print "Found %d completed trials" % completedTrials
            if completedTrials >= numTrials:
                print "Exiting"
                return 0
            else:
                print "Continuing"
        counter = counter_init
        pvalues = pvalues_init
        times_passed=zeros(numTrials-completedTrials)
        for trial in arange(completedTrials,numTrials):
            start=time.clock()
            print "Trial %d" % trial
            pvalue, data_generating_time = self.compute_pvalue_with_time_tracking()
            counter += pvalue<alpha
            pvalues.append(pvalue)
            time_passed = time.clock()-start-data_generating_time
            times_passed[trial-completedTrials]=time_passed
            print 'p-value: ',pvalue
            print 'testing time passed: ', time_passed
            #save results every 50 trials
            if not (trial+1)%5 or trial+1==numTrials:
                    save_f = open(save_filename,"w")
                    average_time=mean(times_passed[:trial+1-completedTrials])*float(trial+1-completedTrials)/(trial+1)+\
                                average_time_init*float(completedTrials)/(trial+1)
                    dump([counter, trial+1, self.param, average_time, pvalues], save_f)
                    save_f.close()
                    print "...Dumped into file"
                    print "Rejection rate: %d / %d" % (counter, trial+1)
                    print "Average time: ", average_time
        return 0
        
'''We are directly calculating the power if under alternative;
calculating the Type I error (alpha) if under null'''