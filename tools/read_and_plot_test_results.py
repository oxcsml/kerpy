from matplotlib.pyplot import figure, gca, errorbar, grid, show, legend, xlabel, \
    ylabel, ylim
from numpy import sqrt, argsort, asarray
import os
from pickle import load
import sys

import matplotlib as mpl


mpl.rcParams['text.usetex']=True
mpl.rcParams['text.latex.unicode']=True


load_filename = "results.bin"
#print sys.argv[1:]
folders_greps = [x.replace('results/','') for x in sys.argv[1:]]
#print folders_greps
#if len(sys.argv)==1:
#	sys.argv[1:] = raw_input('Read null or alter: ').split()
which_case = 'alter'

lsdirs = os.listdir('results/')
print lsdirs
figure()
ax=gca()
legend_str=list()
for folders_grep in folders_greps:
    counter=list()
    numTrials=list()
    rate=list()
    stder=list()
    num_samples=list()
    ii=0
    for lsdir in lsdirs:
        if os.path.isdir('results/'+lsdir) and lsdir.startswith(folders_grep):
            os.chdir('results/'+lsdir)
            load_f = open(load_filename,"r")
            [counter_current, numTrials_current, param,_,_] = load(load_f)
            print 'reading ' + lsdir + ' -found ' + str(numTrials_current) + ' trials'
            num_samples.append( param['num_samplesX'] )
            numTrials.append(numTrials_current)
            counter.append(counter_current)
            rate.append( counter[ii]/float(numTrials[ii]) )
            stder.append( 1.96*sqrt( rate[ii]*(1-rate[ii]) / float(numTrials[ii]) ) )
            print "Rejection rate: %.3f +- %.3f (%d / %d)" % (rate[ii], stder[ii], counter[ii], numTrials[ii])
            os.chdir('../..')
            ii+=1
    #stat_test sizes may not be ordered
    legend_str.append(param['name'].split('_')[0])
    #legend_str.append(param['name'])
    order = argsort(num_samples)
    
    
    errorbar(asarray(num_samples)[order],\
    		asarray(rate)[order],\
    		yerr=asarray(stder)[order])
    
legend(legend_str,loc=4)
xlabel("number of samples",fontsize=12)
if which_case=='null':
    ylim([0,0.12])
    ax.set_yticks([0.01, 0.03, 0.05, 0.07, 0.09, 0.11])
    ylabel("rejection rate (Type I error)",fontsize=12)
elif which_case=='alter':
    #ylabel("rejection rate (1-Type II error)",fontsize=12)
    ylabel("rejection rate",fontsize=12)
ax.set_xscale("log",basex=2)
grid()
show()
