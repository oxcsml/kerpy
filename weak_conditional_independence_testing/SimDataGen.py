from numpy.random import uniform, permutation, multivariate_normal,normal
from numpy import pi, prod, empty, sin, cos, asscalar, shape,zeros, identity,arange,sign,sum,sqrt,transpose, tanh,sinh
import numpy as np 



class SimDataGen(object):
    def __init__(self):
        pass
    
    
    @staticmethod
    def null_model(num_samples, dimension = 1, rho=0):
        data_z = np.reshape(uniform(0,5,num_samples*dimension),(num_samples,dimension))
        coin_flip_x = np.random.choice([0,1],replace=True,size=num_samples)
        coin_flip_y = np.random.choice([0,1],replace=True,size=num_samples)
        mean_noise = [0,0]
        cov_noise = [[1,0],[0,1]]
        noise_x, noise_y = multivariate_normal(mean_noise, cov_noise, num_samples).T
        data_x = zeros(num_samples)
        data_x[coin_flip_x == 0,] = 1.7*data_z[coin_flip_x == 0,0] 
        data_x[coin_flip_x == 1,] = -1.7*data_z[coin_flip_x == 1,0]
        data_x = data_x + noise_x
        data_y = zeros(num_samples)
        data_y[coin_flip_y == 0,] = (data_z[coin_flip_y == 0,0]-2.7)**2
        data_y[coin_flip_y == 1,] = -(data_z[coin_flip_y == 1,0]-2.7)**2+13
        data_y = data_y + noise_y
        data_x = np.reshape(data_x, (num_samples,1))
        data_y = np.reshape(data_y, (num_samples,1))
        return data_x, data_y, data_z
    
    
    @staticmethod
    def alternative_model(num_samples,dimension = 1, rho=0.15):
        data_z = np.reshape(uniform(0,5,num_samples*dimension),(num_samples,dimension))
        rr = uniform(0,1, num_samples)
        idx_rr = np.where(rr < rho)
        coin_flip_x = np.random.choice([0,1],replace=True,size=num_samples)
        coin_flip_y = np.random.choice([0,1],replace=True,size=num_samples)
        coin_flip_y[idx_rr] = coin_flip_x[idx_rr]
        mean_noise = [0,0]
        cov_noise = [[1,0],[0,1]]
        noise_x, noise_y = multivariate_normal(mean_noise, cov_noise, num_samples).T
        data_x = zeros(num_samples)
        data_x[coin_flip_x == 0] = 1.7*data_z[coin_flip_x == 0,0] 
        data_x[coin_flip_x == 1] = -1.7*data_z[coin_flip_x == 1,0]
        data_x = data_x + noise_x
        data_y = zeros(num_samples)
        data_y[coin_flip_y == 0] = (data_z[coin_flip_y == 0,0]-2.7)**2
        data_y[coin_flip_y == 1] = -(data_z[coin_flip_y == 1,0]-2.7)**2+13
        data_y = data_y + noise_y
        data_x = np.reshape(data_x, (num_samples,1))
        data_y = np.reshape(data_y, (num_samples,1))
        return data_x, data_y, data_z
    
    
    
    @staticmethod
    def DAG_simulation_version1(num_samples):
        dimension = 1
        rho = 0
        data_z = np.reshape(uniform(0,5,num_samples*dimension),(num_samples,dimension))
        rr = uniform(0,1, num_samples)
        idx_rr = np.where(rr < rho)
        coin_flip_x = np.random.choice([0,1],replace=True,size=num_samples)
        coin_flip_y = np.random.choice([0,1],replace=True,size=num_samples)
        coin_flip_y[idx_rr] = coin_flip_x[idx_rr]
        mean_noise = [0,0]
        cov_noise = [[1,0],[0,1]]
        noise_x, noise_y = multivariate_normal(mean_noise, cov_noise, num_samples).T
        data_x = zeros(num_samples)
        data_x[coin_flip_x == 0] = 1.7*data_z[coin_flip_x == 0,0] 
        data_x[coin_flip_x == 1] = -1.7*data_z[coin_flip_x == 1,0]
        data_x = data_x + noise_x
        data_y = zeros(num_samples)
        data_y[coin_flip_y == 0] = (data_z[coin_flip_y == 0,0]-2.7)**2
        data_y[coin_flip_y == 1] = -(data_z[coin_flip_y == 1,0]-2.7)**2+13
        data_y = data_y + noise_y
        data_x = np.reshape(data_x, (num_samples,1))
        data_y = np.reshape(data_y, (num_samples,1))
        coin_x = np.reshape(coin_flip_x, (num_samples,1))
        coin_y = np.reshape(coin_flip_y, (num_samples,1))
        
        noise_A, noise_B = multivariate_normal(mean_noise, cov_noise, num_samples).T
        noise_A = np.reshape(noise_A, (num_samples,1))
        noise_B = np.reshape(noise_B, (num_samples,1))
        
        data_A = (data_y-5)**2/float(3) + 5 + noise_A
        #data_A = (data_y-5)**2/float(11)+ 5 +noise_A
        #data_B = 5*np.tanh(data_y) + noise_B # tanh version 
        #data_B = 5*np.sin(data_y) + noise_B # sine version
        data_B = 5.5*np.tanh(data_y) + noise_B 
        
        data_matrix = np.concatenate((data_x,data_y,data_z,data_A,data_B,coin_x, coin_y),axis=1)
        return data_matrix
        # data_x, data_y, data_z, data_A, data_B, coin_x, coin_y,
