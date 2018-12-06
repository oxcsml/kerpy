'''
Created on 21 Jun 2017


Combine HSICConditionalTestObject.py (KRESIT)
and RESITCondTestObject.py (RESIT & LRESIT)
in one framework with the following parameter optimisation options:
- Fix step gradient descent on lambda 
- Fix step gradient descent on sigmasq and lambda 
- grid search on lambda 
- grid search on sigmasq and lambda

Such parameter optimisation is only for the regression part.

'''

import os, sys
BASE_DIR = os.path.join( os.path.dirname( __file__ ), '..' )
sys.path.append(BASE_DIR)


from independence_testing.HSICTestObject import HSICTestObject
from kerpy.Kernel import Kernel
from kerpy.LinearKernel import LinearKernel
from kerpy.GaussianKernel import GaussianKernel
from scipy.spatial.distance import squareform, pdist, cdist
from sklearn.model_selection import KFold
from scipy.linalg import solve


import numpy as np
from numpy.random import permutation 
from numpy import trace,eye, sqrt, median,exp,log
from scipy.linalg import cholesky, cho_solve
import time


class TwoStepCondTestObject(HSICTestObject):

    def __init__(self, num_samples, data_generator, 
                 kernelX, kernelY, kernelZ, kernelX_use_median=False,
                 kernelY_use_median=False, kernelZ_use_median=False, 
                 kernelRxz = LinearKernel(), kernelRyz = LinearKernel(),
                 kernelRxz_use_median=False, kernelRyz_use_median=False,
                 RESIT_type = False,
                 num_shuffles=800,
                 lambda_val=[0.5,1,5,10],lambda_X = None, lambda_Y = None,
                  optimise_lambda_only = False, 
                 sigmasq_vals = [1,2,3] ,sigmasq_xz = 1., sigmasq_yz = 1.,
                 K_folds=5, grid_search = False,
                 GD_optimise=True, learning_rate=0.001,max_iter=3000,
                 initial_lambda_x=0.5,initial_lambda_y=0.5, initial_sigmasq = 1):
        HSICTestObject.__init__(self, num_samples, data_generator, kernelX, kernelY, kernelZ,
                                kernelX_use_median=kernelX_use_median, kernelY_use_median=kernelY_use_median,
                                kernelZ_use_median=kernelZ_use_median)
        
        self.kernelRxz = kernelRxz
        self.kernelRyz = kernelRyz
        self.kernelRxz_use_median = kernelRxz_use_median
        self.kernelRyz_use_median = kernelRyz_use_median
        self.RESIT_type = RESIT_type
        self.num_shuffles = num_shuffles
        self.lambda_val = lambda_val
        self.lambda_X = lambda_X
        self.lambda_Y = lambda_Y
        self.optimise_lambda_only = optimise_lambda_only
        self.sigmasq_vals = sigmasq_vals
        self.sigmasq_xz = sigmasq_xz
        self.sigmasq_yz = sigmasq_yz
        self.K_folds = K_folds
        self.GD_optimise = GD_optimise
        self.learning_rate = learning_rate
        self.grid_search = grid_search
        self.initial_lambda_x = initial_lambda_x
        self.initial_lambda_y = initial_lambda_y
        self.initial_sigmasq = initial_sigmasq
        self.max_iter = max_iter
    
    
    
    
    # Pre-compute the kernel matrices needed for the total cv error and its gradient
    def compute_matrices_for_gradient_totalcverr(self, train_x, train_y, train_z):
        if self.kernelX_use_median:
            sigmax = self.kernelX.get_sigma_median_heuristic(train_x)
            self.kernelX.set_width(float(sigmax))
        if self.kernelY_use_median:
            sigmay = self.kernelY.get_sigma_median_heuristic(train_y)
            self.kernelY.set_width(float(sigmay))
        kf = KFold( n_splits=self.K_folds)
        matrix_results = [[[None] for _ in range(self.K_folds)]for _ in range(8)] 
        # xx=[[None]*10]*6 will give the same id to xx[0][0] and xx[1][0] etc. as 
        # this command simply copied [None] many times. But the above gives different ids.
        count = 0
        for train_index, test_index in kf.split(np.ones((self.num_samples,1))):
            X_tr, X_tst = train_x[train_index], train_x[test_index]
            Y_tr, Y_tst = train_y[train_index], train_y[test_index]
            Z_tr, Z_tst = train_z[train_index], train_z[test_index]
            matrix_results[0][count] = self.kernelX.kernel(X_tst, X_tr) #Kx_tst_tr
            matrix_results[1][count] = self.kernelX.kernel(X_tr, X_tr) #Kx_tr_tr
            matrix_results[2][count] = self.kernelX.kernel(X_tst, X_tst) #Kx_tst_tst
            matrix_results[3][count] = self.kernelY.kernel(Y_tst, Y_tr) #Ky_tst_tr
            matrix_results[4][count] = self.kernelY.kernel(Y_tr, Y_tr) #Ky_tr_tr
            matrix_results[5][count] = self.kernelY.kernel(Y_tst,Y_tst) #Ky_tst_tst
            matrix_results[6][count] = cdist(Z_tst, Z_tr, 'sqeuclidean') #D_tst_tr: square distance matrix
            matrix_results[7][count] = cdist(Z_tr, Z_tr, 'sqeuclidean') #D_tr_tr: square distance matrix
            count = count + 1
        return matrix_results
    
    
    
    
    
    # compute the gradient of the total cverror with respect to lambda
    def compute_gradient_totalcverr_wrt_lambda(self,matrix_results,lambda_val,sigmasq_z):
        # 0: K_tst_tr; 1: K_tr_tr; 2: D_tst_tr; 3: D_tr_tr
        num_sample_cv = self.num_samples
        ttl_num_folds = np.shape(matrix_results)[1]
        gradient_cverr_per_fold = np.zeros(ttl_num_folds)
        for jj in range(ttl_num_folds):
            uu = np.shape(matrix_results[3][jj])[0] # number of training samples
            M_tst_tr = exp(matrix_results[2][jj]*float(-1/2)*sigmasq_z**(-1))
            M_tr_tr = exp(matrix_results[3][jj]*float(-1/2)*sigmasq_z**(-1))
            lower_ZZ = cholesky(M_tr_tr+ lambda_val*eye(uu), lower=True)
            ZZ = cho_solve((lower_ZZ,True),eye(uu))
            first_term = matrix_results[0][jj].dot(ZZ.dot(ZZ.dot(M_tst_tr.T)))
            second_term = M_tst_tr.dot(ZZ.dot(ZZ.dot(
                            matrix_results[1][jj].dot(ZZ.dot(M_tst_tr.T)))))
            gradient_cverr_per_fold[jj] = trace(first_term-second_term)
        return 2*sum(gradient_cverr_per_fold)/float(num_sample_cv)
    
    
    # lambda = exp(eta)
    def compute_gradient_totalcverr_wrt_eta(self,matrix_results,lambda_val,sigmasq_z):
        # 0: K_tst_tr; 1: K_tr_tr; 2: D_tst_tr; 3: D_tr_tr
        #eta = log(lambda_val)
        #gamma = log(sigmasq_z)
        num_sample_cv = self.num_samples
        ttl_num_folds = np.shape(matrix_results)[1]
        gradient_cverr_per_fold = np.zeros(ttl_num_folds)
        for jj in range(ttl_num_folds):
            uu = np.shape(matrix_results[3][jj])[0] # number of training samples
            M_tst_tr = exp(matrix_results[2][jj]*float(-1/2)*sigmasq_z**(-1))
            M_tr_tr = exp(matrix_results[3][jj]*float(-1/2)*sigmasq_z**(-1))
            lower_ZZ = cholesky(M_tr_tr+ lambda_val*eye(uu), lower=True)
            ZZ = cho_solve((lower_ZZ,True),eye(uu))
            EE = lambda_val*eye(uu)
            first_term = matrix_results[0][jj].dot(ZZ.dot(EE.dot(ZZ.dot(M_tst_tr.T))))
            second_term = first_term.T
            third_term = -M_tst_tr.dot(ZZ.dot(EE.dot(ZZ.dot(
                            matrix_results[1][jj].dot(ZZ.dot(M_tst_tr.T))))))
            forth_term = -M_tst_tr.dot(ZZ.dot(
                            matrix_results[1][jj].dot(ZZ.dot(EE.dot(ZZ.dot(M_tst_tr.T))))))
            gradient_cverr_per_fold[jj] = trace(first_term + second_term + third_term + forth_term)
        return sum(gradient_cverr_per_fold)/float(num_sample_cv)
    
     
    
    
    
    # compute the gradient of the total cverror with respect to sigma_z squared 
    def compute_gradient_totalcverr_wrt_sqsigma(self,matrix_results,lambda_val,sigmasq_z):
        # 0: K_tst_tr; 1: K_tr_tr; 2: D_tst_tr; 3: D_tr_tr
        num_sample_cv = self.num_samples
        ttl_num_folds = np.shape(matrix_results)[1]
        gradient_cverr_per_fold = np.zeros(ttl_num_folds)
        for jj in range(ttl_num_folds):
            uu = np.shape(matrix_results[3][jj])[0]
            log_M_tr_tst = matrix_results[2][jj].T*float(-1/2)*sigmasq_z**(-1)
            M_tr_tst = exp(log_M_tr_tst)
            log_M_tr_tr = matrix_results[3][jj]*float(-1/2)*sigmasq_z**(-1)
            M_tr_tr = exp(log_M_tr_tr)
            lower_ZZ = cholesky(M_tr_tr+ lambda_val*eye(uu), lower=True)
            ZZ = cho_solve((lower_ZZ,True),eye(uu))
            term_1 = matrix_results[0][jj].dot(ZZ.dot((M_tr_tr*sigmasq_z**(-1)*(-log_M_tr_tr)).dot(ZZ.dot(M_tr_tst))))
            term_2 = -matrix_results[0][jj].dot(ZZ.dot(M_tr_tst*(-log_M_tr_tst*sigmasq_z**(-1))))
            term_3 = (sigmasq_z**(-1)*(M_tr_tst.T)*(-log_M_tr_tst.T)).dot(ZZ.dot(matrix_results[1][jj].dot(ZZ.dot(M_tr_tst))))
            term_4 = -(M_tr_tst.T).dot(ZZ.dot((M_tr_tr*sigmasq_z**(-1)*(-log_M_tr_tr)).dot(ZZ.dot(matrix_results[1][jj].dot(
                                                                                    ZZ.dot(M_tr_tst))))))
            term_5 = -(M_tr_tst.T).dot(ZZ.dot(matrix_results[1][jj].dot(ZZ.dot((M_tr_tr*sigmasq_z**(-1)*(-log_M_tr_tr)).dot(
                                                                                    ZZ.dot(M_tr_tst))))))
            term_6 = (M_tr_tst.T).dot(ZZ.dot(matrix_results[1][jj].dot(ZZ.dot(M_tr_tst*sigmasq_z**(-1)*(-log_M_tr_tst)))))
            gradient_cverr_per_fold[jj] = trace(2*term_1 + 2*term_2 + term_3 + term_4 + term_5 + term_6)
        return sum(gradient_cverr_per_fold)/float(num_sample_cv)
    
    
    
    
    def compute_gradient_totalcverr_wrt_gamma(self,matrix_results,lambda_val,sigmasq_z):
        # 0: K_tst_tr; 1: K_tr_tr; 2: D_tst_tr; 3: D_tr_tr
        #eta = log(lambda_val)
        #gamma = log(sigmasq_z)
        num_sample_cv = self.num_samples
        ttl_num_folds = np.shape(matrix_results)[1]
        gradient_cverr_per_fold = np.zeros(ttl_num_folds)
        for jj in range(ttl_num_folds):
            uu = np.shape(matrix_results[3][jj])[0]
            log_M_tr_tst = matrix_results[2][jj].T*float(-1/2)*sigmasq_z**(-1)
            M_tr_tst = exp(log_M_tr_tst)
            log_M_tr_tr = matrix_results[3][jj]*float(-1/2)*sigmasq_z**(-1)
            M_tr_tr = exp(log_M_tr_tr)
            lower_ZZ = cholesky(M_tr_tr+ lambda_val*eye(uu), lower=True)
            ZZ = cho_solve((lower_ZZ,True),eye(uu))
            term_1 = matrix_results[0][jj].dot(ZZ.dot((M_tr_tr*(-log_M_tr_tr)).dot(ZZ.dot(M_tr_tst))))
            term_2 = -matrix_results[0][jj].dot(ZZ.dot(M_tr_tst*(-log_M_tr_tst)))
            term_3 = (M_tr_tst.T*(-log_M_tr_tst).T).dot(ZZ.dot(matrix_results[1][jj].dot(ZZ.dot(M_tr_tst))))
            term_4 = -(M_tr_tst.T).dot(ZZ.dot((M_tr_tr*(-log_M_tr_tr)).dot(ZZ.dot(matrix_results[1][jj].dot(
                                                                                    ZZ.dot(M_tr_tst))))))
            term_5 = -(M_tr_tst.T).dot(ZZ.dot(matrix_results[1][jj].dot(ZZ.dot((M_tr_tr*(-log_M_tr_tr)).dot(
                                                                                    ZZ.dot(M_tr_tst))))))
            term_6 = (M_tr_tst.T).dot(ZZ.dot(matrix_results[1][jj].dot(ZZ.dot(M_tr_tst*(-log_M_tr_tst)))))
            gradient_cverr_per_fold[jj] = trace(2*term_1 + 2*term_2 + term_3 + term_4 + term_5 + term_6)
        return sum(gradient_cverr_per_fold)/float(num_sample_cv)
    
    
    
    # compute the total cverror
    def compute_totalcverr(self,matrix_results,lambda_val,sigmasq_z):
        # 0: K_tst_tr; 1: K_tr_tr; 2: K_tst_tst; 3: D_tst_tr; 4: D_tr_tr 
        num_sample_cv = self.num_samples
        ttl_num_folds = np.shape(matrix_results)[1]
        cverr_per_fold = np.zeros(ttl_num_folds)
        for jj in range(ttl_num_folds):
            uu = np.shape(matrix_results[4][jj])[0] # number of training samples 
            M_tst_tr = exp(matrix_results[3][jj]*float(-1/2)*sigmasq_z**(-1))
            M_tr_tr = exp(matrix_results[4][jj]*float(-1/2)*sigmasq_z**(-1))
            lower_ZZ = cholesky(M_tr_tr+ lambda_val*eye(uu), lower=True)
            ZZ = cho_solve((lower_ZZ,True),eye(uu))
            first_term = matrix_results[2][jj]
            second_term = - matrix_results[0][jj].dot(ZZ.dot(M_tst_tr.T))
            third_term = np.transpose(second_term)
            fourth_term = M_tst_tr.dot(ZZ.dot(
                            matrix_results[1][jj].dot(ZZ.dot(M_tst_tr.T))))
            cverr_per_fold[jj] = trace(first_term + second_term + third_term + fourth_term)
        return sum(cverr_per_fold)/float(num_sample_cv)
    
    
    
    
    def compute_GD_lambda_sigmasq_for_TotalCVerr_with_fix_step_logspace(self, matrix_results,initial_lambda, initial_sigmasq):
        EE = log(initial_lambda) # initialisation of the lambda value
        GG = log(initial_sigmasq) # initialisation of the sigma square value for z
        count = 0
        log_lambda_path = [EE]
        log_sigma_square_path = [GG]
        Error_path = [self.compute_totalcverr(matrix_results,lambda_val = exp(EE),sigmasq_z=exp(GG))]
        d_part_matrix_results = [matrix_results[ii] for ii in [0,1,3,4]]
        Grad_EE = self.compute_gradient_totalcverr_wrt_eta(d_part_matrix_results, exp(EE), exp(GG))
        Grad_GG = self.compute_gradient_totalcverr_wrt_gamma(d_part_matrix_results, exp(EE), exp(GG))
        while (sum(np.array([abs(Grad_EE),abs(Grad_GG)]) >= 0.00001) == 2 and count < self.max_iter):
            Grad_EE_old = self.compute_gradient_totalcverr_wrt_eta(d_part_matrix_results, exp(EE), exp(GG))
            EE = EE - self.learning_rate*Grad_EE_old
            Grad_EE = self.compute_gradient_totalcverr_wrt_eta(d_part_matrix_results, exp(EE), exp(GG))
            log_lambda_path = np.concatenate((log_lambda_path,[EE]))
            Error_path = np.concatenate((Error_path,[self.compute_totalcverr(matrix_results,lambda_val = exp(EE), sigmasq_z=exp(GG))]))
            
            if sum(np.array([abs(Grad_EE),abs(Grad_GG)]) >= 0.00001) == 2 and count < self.max_iter:
                Grad_GG_old = self.compute_gradient_totalcverr_wrt_gamma(d_part_matrix_results, exp(EE), exp(GG))
                GG = GG - self.learning_rate*Grad_GG_old
                Grad_GG = self.compute_gradient_totalcverr_wrt_gamma(d_part_matrix_results, exp(EE), exp(GG))
                log_sigma_square_path = np.concatenate((log_sigma_square_path,[GG]))
                Error_path = np.concatenate((Error_path,[self.compute_totalcverr(matrix_results,lambda_val = exp(EE), sigmasq_z=exp(GG))]))
                
            else:
                break
            count = count+1
        return log_lambda_path[count], log_lambda_path, log_sigma_square_path[count], log_sigma_square_path,Error_path
    
    
    
    def compute_GD_lambda_for_TotalCVerr_with_fix_step_logspace(self, matrix_results,initial_lambda, sigmasq_z):
        EE = log(initial_lambda) # initialisation of the lambda value
        count = 0
        log_lambda_path = [EE]
        Error_path = [self.compute_totalcverr(matrix_results,lambda_val = exp(EE),sigmasq_z=sigmasq_z)]
        d_part_matrix_results = [matrix_results[ii] for ii in [0,1,3,4]]
        Grad_EE = self.compute_gradient_totalcverr_wrt_eta(d_part_matrix_results, exp(EE), sigmasq_z)
        while (abs(Grad_EE) >= 0.00001 and count < self.max_iter):
            Grad_EE_old = Grad_EE
            EE = EE - self.learning_rate*Grad_EE_old
            Grad_EE = self.compute_gradient_totalcverr_wrt_eta(d_part_matrix_results, exp(EE), sigmasq_z)
            log_lambda_path = np.concatenate((log_lambda_path,[EE]))
            Error_path = np.concatenate((Error_path,[self.compute_totalcverr(matrix_results,lambda_val = exp(EE), sigmasq_z=sigmasq_z)]))
            count = count+1
        
        return log_lambda_path[count], log_lambda_path, Error_path
    
    
    
    def compute_lambda_sigmasq_through_grid_search(self, matrix_results_x, matrix_results_y, lambda_vals, sigmasq_vals):
        # 0: K_tst_tr; 1: K_tr_tr; 2: K_tst_tst; 3: D_tst_tr; 4: D_tr_tr 
        #print "parameter opt via grid search"
        num_of_lambdas = np.shape(lambda_vals)[0]
        num_of_sigmasq = np.shape(sigmasq_vals)[0]
        total_cverr_matrix_x = np.reshape(np.zeros(num_of_sigmasq*num_of_lambdas), (num_of_sigmasq,num_of_lambdas))
        total_cverr_matrix_y = np.reshape(np.zeros(num_of_sigmasq*num_of_lambdas), (num_of_sigmasq,num_of_lambdas))
        for ss in range(num_of_sigmasq):
            for ll in range(num_of_lambdas):
                #print "Bandwidth numb; Lambdaval numb:", (ss,ll)
                total_cverr_matrix_x[ss,ll] = self.compute_totalcverr(matrix_results_x, lambda_vals[ll], sigmasq_vals[ss])
                total_cverr_matrix_y[ss,ll] = self.compute_totalcverr(matrix_results_y, lambda_vals[ll], sigmasq_vals[ss])
        x_sigmasq_idx, x_lambda_idx = np.where(total_cverr_matrix_x == np.min(total_cverr_matrix_x))
        y_sigmasq_idx, y_lambda_idx = np.where(total_cverr_matrix_y == np.min(total_cverr_matrix_y))
        if np.shape(x_sigmasq_idx)[0] > 1:
            x_sigmasq = sigmasq_vals[x_sigmasq_idx[0]]
            x_lambda = lambda_vals[x_lambda_idx[0]]
        else:
            x_sigmasq = sigmasq_vals[x_sigmasq_idx]
            x_lambda = lambda_vals[x_lambda_idx]
        if np.shape(y_sigmasq_idx[0]) > 1:
            y_sigmasq = sigmasq_vals[y_sigmasq_idx[0]]
            y_lambda = lambda_vals[y_lambda_idx[0]]
        else:
            y_sigmasq = sigmasq_vals[y_sigmasq_idx]
            y_lambda = lambda_vals[y_lambda_idx]
        return x_sigmasq, x_lambda, y_sigmasq, y_lambda, total_cverr_matrix_x,total_cverr_matrix_y
    
    
    
    
    
    def compute_lambda_through_grid_search(self, matrix_results_x, lambda_vals, sigmasq_xz):
        # 0: K_tst_tr; 1: K_tr_tr; 2: K_tst_tst; 3: D_tst_tr; 4: D_tr_tr 
        #print "lambda parameter opt via grid search"
        num_of_lambdas = np.shape(lambda_vals)[0]
        total_cverr_matrix_x = np.reshape(np.zeros(num_of_lambdas), (num_of_lambdas,1))
        for ll in range(num_of_lambdas):
            total_cverr_matrix_x[ll,0] = self.compute_totalcverr(matrix_results_x, lambda_vals[ll], sigmasq_xz)
        
        x_lambda_idx = np.where(total_cverr_matrix_x == np.min(total_cverr_matrix_x))
        if np.shape(x_lambda_idx)[0] > 1:
            x_lambda = lambda_vals[x_lambda_idx[0]]
        else:
            x_lambda = lambda_vals[x_lambda_idx]
        return x_lambda, total_cverr_matrix_x
    
    
    
    
    
    
    def compute_test_statistics_and_others(self, data_x, data_y, data_z):
        if self.grid_search or self.GD_optimise:
            matrix_results = self.compute_matrices_for_gradient_totalcverr(data_x,data_y,data_z)
            matrix_results_x = [matrix_results[ii] for ii in [0,1,2,6,7]]
            matrix_results_y = [matrix_results[ii] for ii in [3,4,5,6,7]]
            if self.GD_optimise: # Gradient descent with fixed learning rate
                if self.optimise_lambda_only:
                    if self.kernelZ_use_median:
                        sigmaz = self.kernelZ.get_sigma_median_heuristic(data_z)
                        self.kernelZ.set_width(float(sigmaz))
                        self.sigmasq_xz = self.sigmasq_yz = sigmaz**2
                    #print "Gradient Descent Optimisation in log space, fixed step for lambda X"
                    log_lambda_X, log_lambda_pathx, X_CVerror = self.compute_GD_lambda_for_TotalCVerr_with_fix_step_logspace(matrix_results_x,
                                                                                self.initial_lambda_x, self.sigmasq_xz)
                    self.lambda_X = exp(log_lambda_X)
                    #print X_CVerror
                    #print "Gradient Descent Optimisation in log space, fixed step for lambda Y"
                    log_lambda_Y, log_lambda_pathy, Y_CVerror = self.compute_GD_lambda_for_TotalCVerr_with_fix_step_logspace(matrix_results_y,
                                                                                self.initial_lambda_y, self.sigmasq_yz)
                    self.lambda_Y = exp(log_lambda_Y)
                    #print Y_CVerror
                else:
                    #print "Gradient Descent Optimisation in log space, fixed step for lambda X and sigma XZ"
                    log_lambda_X, _, log_sigmasq_xz, _, X_CVerror = self.compute_GD_lambda_sigmasq_for_TotalCVerr_with_fix_step_logspace(matrix_results_x,
                                                                initial_lambda=self.initial_lambda_x, initial_sigmasq=self.initial_sigmasq)
                    self.lambda_X = exp(log_lambda_X)
                    self.sigmasq_xz = exp(log_sigmasq_xz)
                    #print X_CVerror
                    #print "Gradient Descent Optimisation in log space, fixed step for lambda Y and sigma YZ"
                    log_lambda_Y, _, log_sigmasq_yz, _, Y_CVerror = self.compute_GD_lambda_sigmasq_for_TotalCVerr_with_fix_step_logspace(matrix_results_y,
                                                                initial_lambda=self.initial_lambda_y, initial_sigmasq=self.initial_sigmasq)
                    self.lambda_Y = exp(log_lambda_Y)
                    self.sigmasq_yz = exp(log_sigmasq_yz)
                    #print Y_CVerror
                
            elif self.grid_search:
                if self.optimise_lambda_only:
                    if self.kernelZ_use_median:
                        sigmaz = self.kernelZ.get_sigma_median_heuristic(data_z)
                        self.kernelZ.set_width(float(sigmaz))
                        self.sigmasq_xz = self.sigmasq_yz = sigmaz**2
                    #print "Grid Search Optimisation in log space for lambda X"
                    self.lambda_X, X_CVerror = self.compute_lambda_through_grid_search(matrix_results_x, self.lambda_val,self.sigmasq_xz)
                    #print X_CVerror
                    #print "Grid Search Optimisation in log space for lambda Y"
                    self.lambda_Y, Y_CVerror = self.compute_lambda_through_grid_search(matrix_results_y, self.lambda_val,self.sigmasq_yz)
                    #print Y_CVerror
                else:
                    self.sigmasq_xz, self.lambda_X, self.sigmasq_yz, self.lambda_Y, X_CVerror, Y_CVerror = \
                    self.compute_lambda_sigmasq_through_grid_search(matrix_results_x, matrix_results_y, self.lambda_val, self.sigmasq_vals)
                    #print X_CVerror
                    #print Y_CVerror
            else:
                raise NotImplementedError
                
        else:
            if self.lambda_X == None:
                self.lambda_X = self.lambda_val[0]
            if self.lambda_Y == None:
                self.lambda_Y = self.lambda_val[0]
            if self.sigmasq_xz == None:
                sigmaz = self.kernelZ.get_sigma_median_heuristic(data_z)
                self.kernelZ.set_width(float(sigmaz))
                self.sigmasq_xz = sigmaz**2
            if self.sigmasq_yz == None:
                sigmaz = self.kernelZ.get_sigma_median_heuristic(data_z)
                self.kernelZ.set_width(float(sigmaz))
                self.sigmasq_yz = sigmaz**2
            X_CVerror = 0
            Y_CVerror = 0
        
        #print "lambda value for (X,Y)", (self.lambda_X,self.lambda_Y)
        #print "sigma squared for (XZ, YZ)", (self.sigmasq_xz, self.sigmasq_yz)
        test_size = self.num_samples
        if not self.RESIT_type:
            test_Kx, test_Ky, _  = self.compute_kernel_matrix_on_data_CI(data_x, data_y, data_z)
            D_z = cdist(data_z, data_z, 'sqeuclidean')
            test_Kzx = exp(D_z*(-0.5)*self.sigmasq_xz**(-1))
            test_Kzy = exp(D_z*(-0.5)*self.sigmasq_yz**(-1))
            weight_xz = solve(test_Kzx/float(self.lambda_X)+np.identity(test_size),np.identity(test_size))
            weight_yz = solve(test_Kzy/float(self.lambda_Y)+np.identity(test_size),np.identity(test_size))
            K_epsilon_x = weight_xz.dot(test_Kx.dot(weight_xz))
            K_epsilon_y = weight_yz.dot(test_Ky.dot(weight_yz))
        else:
            #print "RESIT Computation"
            if self.kernelZ_use_median:
                sigmaz = self.kernelZ.get_sigma_median_heuristic(data_z)
                self.kernelZ.set_width(float(sigmaz))
            test_Kz = self.kernelZ.kernel(data_z)
            weight_xz = solve(test_Kz/float(self.lambda_X)+np.identity(test_size),np.identity(test_size))
            weight_yz = solve(test_Kz/float(self.lambda_Y)+np.identity(test_size),np.identity(test_size))
            residual_xz = weight_xz.dot(data_x)
            residual_yz = weight_yz.dot(data_y)
            if self.kernelRxz_use_median:
                sigmaRxz = self.kernelRxz.get_sigma_median_heuristic(residual_xz)
                self.kernelRxz.set_width(float(sigmaRxz))
            if self.kernelRyz_use_median:
                sigmaRyz = self.kernelRyz.get_sigma_median_heuristic(residual_yz)
                self.kernelRyz.set_width(float(sigmaRyz))
            K_epsilon_x = self.kernelRxz.kernel(residual_xz)
            K_epsilon_y = self.kernelRyz.kernel(residual_yz)
        
        hsic_statistic = self.HSIC_V_statistic(K_epsilon_x, K_epsilon_y)
        #print "HSIC Statistics", hsic_statistic
        return hsic_statistic, K_epsilon_x, K_epsilon_y, X_CVerror, Y_CVerror
    
    
    
    
    
    def compute_null_samples_and_pvalue(self,data_x=None,data_y=None,data_z=None):
        ''' data_x,data_y, data_z are the given data that we wish to test 
        the conditional independence given data_z. 
        > each data set has the number of samples = number of rows 
        > the bandwidth for training set and test set will be different (as we will calculate as soon as data comes in)
        '''
        if data_x is None and data_y is None and data_z is None: 
            if not self.streaming and not self.freeze_data:
                start = time.clock()
                self.generate_data(isConditionalTesting=True)
                data_generating_time = time.clock()-start
                data_x = self.data_x
                data_y = self.data_y
                data_z = self.data_z
                #print "dimension of data:", np.shape(data_x)
            else:
                data_generating_time = 0.
            
        else:
            data_generating_time = 0.
        #print 'Data generating time passed: ', data_generating_time
        hsic_statistic, K_epsilon_x, K_epsilon_y, X_CVerror, Y_CVerror = self.compute_test_statistics_and_others(data_x, data_y, data_z)
        if self.num_shuffles != 0:
            ny = np.shape(K_epsilon_y)[0]
            null_samples = np.zeros(self.num_shuffles)
            for jj in range(self.num_shuffles):
                pp = permutation(ny)
                Kp = K_epsilon_y[pp,:][:,pp]
                null_samples[jj] = self.HSIC_V_statistic(K_epsilon_x, Kp)
            pvalue = ( sum( null_samples > hsic_statistic ) + 1) / float( self.num_shuffles + 1)
            #print "P-value:", pvalue
        else:
            pvalue = None 
            null_samples = 0  
            #print "Not interested in P-value"
        return null_samples, hsic_statistic, pvalue, X_CVerror, Y_CVerror,data_generating_time
    
    
    
    def compute_pvalue_with_time_tracking(self, data_x = None, data_y = None, data_z = None):
        if self.lambda_X is not None and self.lambda_Y is not None:
            self.GD_optimise = False
            self.grid_search = False
            self.lambda_val = [1]
        _, _, pvalue, _, _, data_generating_time = self.compute_null_samples_and_pvalue(data_x = data_x,
                                                                            data_y = data_y, data_z = data_z)
        return pvalue, data_generating_time
    
    
    
    
