'''
    adding relevant folder to your pythonpath
'''
import os, sys
BASE_DIR = os.path.join( os.path.dirname( __file__ ), '..' )
#print BASE_DIR
sys.path.append(BASE_DIR)
#print sys.path.append(BASE_DIR)

from kerpy.GaussianKernel import GaussianKernel
from kerpy.BrownianKernel import BrownianKernel
from HSICTestObject import HSICTestObject
from HSICSpectralTestObject import HSICSpectralTestObject
from TestExperiment import TestExperiment
from SimDataGen import SimDataGen
from tools.ProcessingObject import ProcessingObject

#example usage: python ExperimentsHSICSpectral.py 500 --dimX 4 --hypothesis null --rff --num_rfx 50 --num_rfy 50 
# the above says that 500 samples; null hypothesis; rff True; 50 random Fourier features for X and Y.


data_generating_function = SimDataGen.VaryDimension
data_generating_function_null = SimDataGen.turn_into_null(SimDataGen.VaryDimension)
#data_generating_function = SimDataGen.LargeScale
#data_generating_function_null = SimDataGen.turn_into_null(SimDataGen.LargeScale)
args = ProcessingObject.parse_arguments()

'''unpack the arguments needed:''' 
num_samples=args.num_samples
hypothesis=args.hypothesis
dimX = args.dimX
kernelX_use_median = args.kernelX_use_median
kernelY_use_median = args.kernelY_use_median
rff=args.rff
num_rfx = args.num_rfx
num_rfy = args.num_rfy
induce_set = args.induce_set
num_inducex = args.num_inducex
num_inducey = args.num_inducey


# This will only be a temporary set up for the kernels when we use median heuristics. 
#kernelX = GaussianKernel(1.)
#kernelY = GaussianKernel(1.)

# Brownian kernel with H = 0.5 equivalently alpha = 1.0
kernelX = BrownianKernel(1.)
kernelY = BrownianKernel(1.) 

if hypothesis=="alter":
    data_generator=lambda num_samples: data_generating_function(num_samples,dimension=dimX)
elif hypothesis=="null":
    data_generator=lambda num_samples: data_generating_function_null(num_samples,dimension=dimX)
else:
    raise NotImplementedError()


test_object=HSICSpectralTestObject(num_samples, data_generator, kernelX, kernelY, 
                                   kernelX_use_median=kernelX_use_median,kernelY_use_median=kernelY_use_median, 
                                   rff = rff, num_rfx=num_rfx,num_rfy=num_rfy, unbiased=False,
                                   induce_set = induce_set, num_inducex = num_inducex, num_inducey = num_inducey)


# file name of the results
if rff:
    name = os.path.basename(__file__).rstrip('.py')+'Bsine'+'_'+hypothesis+'_rff_'+str(num_rfx)+\
    str(num_rfy)+'_d_'+str(dimX)+'_n_'+str(num_samples)
elif induce_set:
    name = os.path.basename(__file__).rstrip('.py')+'Bsine'+'_'+hypothesis+'_induce_'+str(num_inducex)+\
    str(num_inducey)+'_d_'+str(dimX)+'_n_'+str(num_samples)
else:
    name = os.path.basename(__file__).rstrip('.py')+'Bsine'+'_'+hypothesis+'_d_'+str(dimX)+'_n_'+str(num_samples)

param={'name': name,\
       'dim': dimX,\
       'kernelX': kernelX,\
       'kernelY': kernelY,\
       'num_rfx': num_rfx,\
       'num_rfy': num_rfy,\
       'num_inducex': num_inducex,\
       'num_inducey': num_inducey,\
       'data_generator': data_generator.__name__,\
       'hypothesis':hypothesis,\
       'num_samples': num_samples}

experiment=TestExperiment(name, param, test_object)

numTrials = 100
alpha=0.05
experiment.run_test_trials(numTrials, alpha=alpha)

