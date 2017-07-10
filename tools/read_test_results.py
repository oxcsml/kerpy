import os,sys 
BASE_DIR = os.path.join( os.path.dirname( __file__ ), '..' )
sys.path.append(BASE_DIR)

from numpy import sqrt
import os
from pickle import load
import sys


os.chdir(sys.argv[1])
load_filename = "results.bin"
load_f = open(load_filename,"r")
[counter, numTrials, param, average_time, pvalues] = load(load_f)
load_f.close()

rate = counter/float(numTrials)
stder = 1.96*sqrt( rate*(1-rate) / float(numTrials) )
'''this stder is symmetrical in terms of rate'''

print "Parameters:"
for keys,values in param.items():
    print(keys)
    print(values)
print "Rejection rate: %.3f +- %.3f (%d / %d)" % (rate, stder, counter, numTrials)
print "Average test time: %.5f sec" % average_time 
os.chdir('..')
#Minor: need to change the above for Gaussian Kernel Median Heuristic

