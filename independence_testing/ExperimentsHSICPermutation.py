'''
    adding relevant folder to your pythonpath
    '''
import os, sys
BASE_DIR = os.path.join( os.path.dirname( __file__ ), '..' )
sys.path.append(BASE_DIR)

from kerpy.GaussianKernel import GaussianKernel
from HSICTestObject import HSICTestObject
from TestExperiment import TestExperiment
from SimDataGen import SimDataGen
from tools.ProcessingObject import ProcessingObject
from HSICPermutationTestObject import HSICPermutationTestObject

#example use: python ExperimentsHSICPermutation.py 500 --dimX 3 --hypothesis null --rff --num_rfx 50 --num_rfy 50 

data_generating_function = SimDataGen.VaryDimension
data_generating_function_null = SimDataGen.turn_into_null(SimDataGen.VaryDimension)
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
num_shuffles = args.num_shuffles


# A temporary set up for the kernels: 
kernelX = GaussianKernel(1.)
kernelY = GaussianKernel(1.)


if hypothesis=="alter":
    data_generator=lambda num_samples: data_generating_function(num_samples,dimension=dimX)
elif hypothesis=="null":
    data_generator=lambda num_samples: data_generating_function_null(num_samples,dimension=dimX)
else:
    raise NotImplementedError()




test_object=HSICPermutationTestObject(num_samples, data_generator, kernelX, kernelY, 
                                   kernelX_use_median=kernelX_use_median,kernelY_use_median=kernelY_use_median, 
                                   num_rfx=num_rfx,num_rfy=num_rfy, unbiased=False, rff=rff,
                                   induce_set=induce_set,num_inducex=num_inducex,num_inducey=num_inducey,
                                   num_shuffles=num_shuffles)


if rff:
    name = os.path.basename(__file__).rstrip('.py')+'_VD_'+hypothesis+'_d_'+str(dimX)+\
    '_shuffles_'+str(num_shuffles)+'_rff_'+str(num_rfx)+str(num_rfy)+'_n_'+str(num_samples)
elif induce_set:
    name = os.path.basename(__file__).rstrip('.py')+'_VD_'+hypothesis+'_d_'+str(dimX)+\
    '_shuffles_'+str(num_shuffles)+'_induce_'+str(num_inducex)+str(num_inducey)+'_n_'+str(num_samples)
else:
    name = os.path.basename(__file__).rstrip('.py')+'_VD_'+hypothesis+'_d_'+str(dimX)+\
    '_shuffles_'+str(num_shuffles)+'_n_'+str(num_samples)



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