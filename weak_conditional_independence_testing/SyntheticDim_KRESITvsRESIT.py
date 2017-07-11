'''
The results of KRESIT and RESIT are compared with 
KCI-test (Zhang et al. 2011)
CI-perm (Fukumizu et al. 2008)
KCIPT (Doran et al. 2014)
Using the Matlab implementation of Zhang et al. 2011

Data are simulated from the Null and Alternative models 
and saved in .mat files in the folders. 

Here, the script reads in the .mat files one by one 
Run KRESIT/RESIT and save the pvalues in a .csv file 

Example run in terminal:
1) KRESIT:
$ python SyntheticDim_KRESITvsRESIT.py 40 --dimZ 1 
--kernelX_use_median --kernelY_use_median 

(It takes .mat files from the Null_num_40_dimZ_1 folder in the Null_model folder
and run KRESIT with Gaussian kernel median Heuristic on the variables X and Y. The kernel on Z
is set by default to be Gaussian median Heuristic. The regularisation parameters is 
set by default to use grid search. )

2) RESIT:
$ python SyntheticDim_KRESITvsRESIT.py 40 --dimZ 1 
--kernelX --kernelY 
--kernelRxz --kernelRyz
--kernelRxz_use_median --kernelRyz_use_median 
--RESIT_type

(It takes .mat files from the Null_num_40_dimZ_1 folder in the Null_model folder
and run RESIT. The kernels on X and Y are set to be linear. The kernels on the residuals Rxz and 
Ryz are Gaussian with median Heuristic bandwidth.The regularisation parameters is 
set by default to use grid search. The resulting CPDAG is saved "test_graph_RESIT.pdf".)


'''
import os, sys
BASE_DIR = os.path.join( os.path.dirname( __file__ ), '..' )
sys.path.append(BASE_DIR)

import scipy.io as sio
import numpy as np
from kerpy.GaussianKernel import GaussianKernel
from kerpy.LinearKernel import LinearKernel
from TwoStepCondTestObject import TwoStepCondTestObject
from tools.ProcessingObject import ProcessingObject

wd = os.getcwd()


args = ProcessingObject.parse_arguments()
# Choices for number of samples: 40, 80, 120, 160, 200
num_samples = args.num_samples 
# Choices for dimensions of Z: 1,2,3,4,5,6,7
dimZ = args.dimZ
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

lsdirs = os.listdir(wd + "/Null_model/Null_num_" + str(num_samples) + "_dimZ_" + str(dimZ) + "/")



data_generator = None 
num_lambdaval = 30
lambda_val = 10**np.linspace(-6,-1, num=num_lambdaval)
#num_bandwidth = 20
#z_bandwidth = 10**np.linspace(-5,1,num = num_bandwidth)
z_bandwidth = None

kernelZ = GaussianKernel(1.)

pval = np.zeros(100)
count = 0 

for lsdir in lsdirs:
    print count 
    mat_contents = sio.loadmat(wd+"/Null_model/Null_num_" + str(num_samples) + "_dimZ_" + str(dimZ) + "/"+lsdir)
    data_x = mat_contents['data_x']
    data_y = mat_contents['data_y']
    data_z = mat_contents['data_z']
    
    
    mytestobj = TwoStepCondTestObject(num_samples, data_generator, 
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
    
    
    pval[count], _ = mytestobj.compute_pvalue(data_x, data_y, data_z)
    count = count + 1

rejection_rate = sum(pval < 0.05)/float(100)
print "rejection rate", rejection_rate 

if RESIT_type:
    my_save_file_name = "RESIT_Null_pval_num_" + str(num_samples) + "_dimZ_" + str(dimZ) + ".csv"
else:
    my_save_file_name = "Null_pval_num_" + str(num_samples) + "_dimZ_" + str(dimZ) + ".csv"

np.savetxt(my_save_file_name, pval, delimiter=",")
