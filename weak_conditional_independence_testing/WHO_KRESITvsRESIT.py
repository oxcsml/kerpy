'''
Comparing RESIT and KRESIT on WHO data.
To run the script in terminal:
$ python WHO_KRESITvsRESIT.py 
The p-values will be printed and a data plot will be shown.

'''

import os, sys
BASE_DIR = os.path.join( os.path.dirname( __file__ ), '..' )
sys.path.append(BASE_DIR)

import os 
import pandas as pd
import numpy as np 
from numpy.random import normal
from scipy.stats import t
from scipy import stats, linalg
import matplotlib.pyplot as plt
from pylab import rcParams 

from kerpy.GaussianKernel import GaussianKernel
from kerpy.LinearKernel import LinearKernel
from TwoStepCondTestObject import TwoStepCondTestObject
# Import data 
data = pd.DataFrame.from_csv("WHO_dataset.csv") #WHO_dataset.csv is the same as WHO1 original.csv
Gross_national_income = np.reshape(data.iloc[:,1],(202,1))
Expenditure_on_health = np.reshape(data.iloc[:,5],(202,1))

# Remove missing values 
data_yz = np.concatenate((Gross_national_income,Expenditure_on_health),axis=1)
data_yz = data_yz[~np.isnan(data_yz).any(axis=1)]
data_z = data_yz[:,[0]] #(178,1)
data_y = data_yz[:,[1]]

# log transform data z to make the concentrated data more spread out
data_z = np.log(data_z)


# range of values for grid search 
num_lambdaval = 30
lambda_val = 10**np.linspace(-6,-1, num=num_lambdaval)
z_bandwidth = None
#num_bandwidth = 20
#z_bandwidth = 10**np.linspace(-5,1,num = num_bandwidth)

# some parameter settings
num_samples = np.shape(data_z)[0]
data_generator=None
num_trials = 1
pvals_KRESIT = np.reshape(np.zeros(num_trials),(num_trials,1))
pvals_RESIT = np.reshape(np.zeros(num_trials),(num_trials,1))



# computing Type I error (Null model is true)
for jj in xrange(num_trials):
        #print "number of trial:", jj
        
        data_x = np.reshape(np.zeros(num_samples),(num_samples,1))
        noise_x = np.reshape(normal(0,1,np.shape(data_z)[0]),(np.shape(data_z)[0],1))
        coin_flip_x = np.random.choice([0,1],replace=True,size=num_samples)
        data_x[coin_flip_x == 0] = (data_z[coin_flip_x == 0]-10)**2 
        data_x[coin_flip_x == 1] = -(data_z[coin_flip_x == 1]-10)**2+35
        data_x = data_x + noise_x
        
        
        # KRESIT:
        kernelX = GaussianKernel(1.)
        kernelY = GaussianKernel(1.)
        kernelZ = GaussianKernel(1.)
        mytestobject = TwoStepCondTestObject(num_samples, None, 
                                         kernelX, kernelY, kernelZ, 
                                         kernelX_use_median=True,
                                         kernelY_use_median=True, 
                                         kernelZ_use_median=True, 
                                         kernelRxz = LinearKernel(), kernelRyz = LinearKernel(),
                                         kernelRxz_use_median = False, 
                                         kernelRyz_use_median = False,
                                         RESIT_type = False,
                                         num_shuffles=800,
                                         lambda_val=lambda_val,lambda_X = None, lambda_Y = None,
                                         optimise_lambda_only = True, 
                                         sigmasq_vals = z_bandwidth ,sigmasq_xz = 1., sigmasq_yz = 1.,
                                         K_folds=5, grid_search = True,
                                         GD_optimise=False)
        
        pvals_KRESIT[jj,], _ = mytestobject.compute_pvalue(data_x,data_y,data_z)
        
        
        # RESIT:
        kernelX = LinearKernel()
        kernelY = LinearKernel()
        mytestobject_RESIT = TwoStepCondTestObject(num_samples, None, 
                                         kernelX, kernelY, kernelZ, 
                                         kernelX_use_median=False,
                                         kernelY_use_median=False, 
                                         kernelZ_use_median=True, 
                                         kernelRxz = GaussianKernel(1.), kernelRyz = GaussianKernel(1.),
                                         kernelRxz_use_median = True, 
                                         kernelRyz_use_median = True,
                                         RESIT_type = True,
                                         num_shuffles=800,
                                         lambda_val=lambda_val,lambda_X = None, lambda_Y = None,
                                         optimise_lambda_only = True, 
                                         sigmasq_vals = z_bandwidth ,sigmasq_xz = 1., sigmasq_yz = 1.,
                                         K_folds=5, grid_search = True,
                                         GD_optimise=False)
        
        pvals_RESIT[jj,], _ = mytestobject_RESIT.compute_pvalue(data_x,data_y,data_z)



#np.savetxt("WHO_KRESIT_rejection_rate.csv", pvals_KRESIT, delimiter=",")
#np.savetxt("WHO_RESIT_rejection_rate.csv", pvals_RESIT, delimiter=",")


if num_trials > 1:
    print "Type I error (KRESIT):", np.shape(filter(lambda x: x<0.05 ,pvals_KRESIT))[0]/float(num_trials)
    print "Type I error (RESIT):", np.shape(filter(lambda x: x<0.05 ,pvals_RESIT))[0]/float(num_trials)
elif num_trials == 1:
    print "pval our approach:", pvals_KRESIT
    print "pval RESIT:", pvals_RESIT


# Plot of the data
rcParams['figure.figsize'] = 9, 4.7
plt.figure(1)
plt.subplot(121)
plt.plot(data_z, data_y,'.')
plt.ylabel('Y = Expenditure on health per cap')
plt.xlabel('Z = log(Gross national income per cap)')

plt.subplot(122)
plt.plot(data_z,data_x,'.')
plt.ylabel('X ')
plt.xlabel('Z')
plt.show()

#plot_name = "WHO_" + "logZ_"+ "quadraticX" + ".pdf"
#plt.savefig(plot_name, format='pdf')
#plt.show()



