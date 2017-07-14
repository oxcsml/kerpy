'''
Example run in terminal:
1) KRESIT:
$ python SyntheticDim_KRESIT.py 40 --dimZ 1 
--kernelX_use_median --kernelY_use_median 

(Simulate 100 sets of 40 samples with 1 dimensional conditioning set from the null_model
and run KRESIT with Gaussian kernel median Heuristic on the variables X and Y. The kernel on Z
is set by default to be Gaussian median Heuristic. The regularisation parameters is 
set by default to use grid search. )

2) RESIT:
$ python SyntheticDim_KRESIT.py 40 --dimZ 1 
--kernelX --kernelY 
--kernelRxz --kernelRyz
--kernelRxz_use_median --kernelRyz_use_median 
--RESIT_type

(Simulate 100 sets of 40 samples with 1 dimensional conditioning set from the null_model
and run RESIT. The kernels on X and Y are set to be linear. The kernels on the residuals Rxz and 
Ryz are Gaussian with median Heuristic bandwidth.The regularisation parameters is 
set by default to use grid search. )

'''

import os, sys
BASE_DIR = os.path.join( os.path.dirname( __file__ ), '..' )
sys.path.append(BASE_DIR)

from kerpy.GaussianKernel import GaussianKernel
from kerpy.LinearKernel import LinearKernel
from TwoStepCondTestObject import TwoStepCondTestObject
from independence_testing.TestExperiment import TestExperiment
from SimDataGen import SimDataGen
import numpy as np
from tools.ProcessingObject import ProcessingObject



data_generating_function = SimDataGen.null_model
#data_generating_function = SimDataGen.alternative_model
args = ProcessingObject.parse_arguments()

'''unpack the arguments needed:''' 
num_samples = args.num_samples 
dimZ = args.dimZ # Integer dimension of the conditioning set.
kernelX = args.kernelX #Default: GaussianKernel(1.)
kernelY = args.kernelY #Default: GaussianKernel(1.)
kernelX_use_median = args.kernelX_use_median #Default: False
kernelY_use_median = args.kernelY_use_median #Default: False
kernelRxz = args.kernelRxz #Default: LinearKernel 
kernelRyz = args.kernelRyz #Default: LinearKernel
kernelRxz_use_median = args.kernelRxz_use_median #Default: False 
kernelRyz_use_median = args.kernelRyz_use_median #Default: False
RESIT_type = args.RESIT_type #Default: False
optimise_lambda_only = args.optimise_lambda_only #Default: True
grid_search = args.grid_search #Default: True
GD_optimise = args.GD_optimise #Default: False 


data_generator=lambda num_samples: data_generating_function(num_samples,dimension=dimZ)


num_lambdaval = 30
lambda_val = 10**np.linspace(-6,-1, num=num_lambdaval)
#num_bandwidth = 20
#z_bandwidth = 10**np.linspace(-5,1,num = num_bandwidth)
z_bandwidth = None
kernelZ = GaussianKernel(1.)


test_object = TwoStepCondTestObject(num_samples, data_generator, 
                 kernelX, kernelY, kernelZ, 
                 kernelX_use_median=kernelX_use_median,
                 kernelY_use_median=kernelY_use_median, 
                 kernelZ_use_median=True, 
                 kernelRxz = kernelRxz, kernelRyz = kernelRyz,
                 kernelRxz_use_median = kernelRxz_use_median, 
                 kernelRyz_use_median = kernelRyz_use_median,
                 RESIT_type = RESIT_type,
                 num_shuffles=800,
                 lambda_val=lambda_val,lambda_X = None, lambda_Y = None,
                 optimise_lambda_only = optimise_lambda_only, 
                 sigmasq_vals = z_bandwidth ,sigmasq_xz = 1., sigmasq_yz = 1.,
                 K_folds=5, grid_search = grid_search,
                 GD_optimise=GD_optimise, learning_rate=0.1,max_iter=300,
                 initial_lambda_x=0.5,initial_lambda_y=0.5, initial_sigmasq = 1)


# file name of the results
name = os.path.basename(__file__).rstrip('.py')+'_d_'+str(dimZ)+'_n_'+str(num_samples)

param={'name': name,\
       'dim_conditioning_set': dimZ,\
       'kernelX': kernelX,\
       'kernelY': kernelY,\
       'kernelZ': kernelZ,\
       'RESIT_type': RESIT_type,\
       'optimise_lambda_only': optimise_lambda_only,\
       'grid_search': grid_search, \
       'data_generator': str(data_generating_function),\
       'num_samples': num_samples}

experiment=TestExperiment(name, param, test_object)

numTrials = 100
alpha=0.05
experiment.run_test_trials(numTrials, alpha=alpha)


