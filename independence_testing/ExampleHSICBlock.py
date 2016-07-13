import os 

from kerpy.GaussianKernel import GaussianKernel
from HSICTestObject import HSICTestObject
from HSICBlockTestObject import HSICBlockTestObject
from TestExperiment import TestExperiment
from SimDataGen import SimDataGen
from tools.ProcessingObject import ProcessingObject

#example use: python ExampleHSICBlock.py 500 --dimX 3 --kernel_width_x --kernel_width_y --blocksize 10

data_generating_function = SimDataGen.LargeScale
data_generating_function_null = SimDataGen.turn_into_null(SimDataGen.LargeScale)
args = ProcessingObject.parse_arguments()

'''unpack the arguments needed:''' 
num_samples=args.num_samples
hypothesis=args.hypothesis
dimX = args.dimX
kernel_width_x = args.kernel_width_x
kernel_width_y = args.kernel_width_y
blocksize = args.blocksize 
#currently, we are using the same blocksize for both X and Y

# A temporary set up for the kernels: 
kernelX = GaussianKernel(1.)
kernelY = GaussianKernel(1.)

if hypothesis=="alter":
    data_generator=lambda num_samples: data_generating_function(num_samples,dimension=dimX)
elif hypothesis=="null":
    data_generator=lambda num_samples: data_generating_function_null(num_samples,dimension=dimX)
else:
    raise NotImplementedError()


test_object=HSICBlockTestObject(num_samples, data_generator, kernelX, kernelY, 
                                   kernel_width_x=kernel_width_x,kernel_width_y=kernel_width_y,
                                   nullvarmethod='permutation',
                                   blocksize=blocksize)

name = os.path.basename(__file__).rstrip('.py')+'_LS_'+hypothesis+'_d_'+str(dimX)+'_B_'+str(blocksize)+'_n_'+str(num_samples)

param={'name': name,\
       'kernelX': kernelX,\
       'kernelY': kernelY,\
       'blocksize':blocksize,\
       'data_generator': data_generator.__name__,\
       'hypothesis':hypothesis,\
       'num_samples': num_samples}


experiment=TestExperiment(name, param, test_object)

numTrials = 100
alpha=0.05
experiment.run_test_trials(numTrials, alpha=alpha)
