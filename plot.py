# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 19:36:59 2022

@author: USER
""" 
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import gennorm, norm, laplace,energy_distance
import scipy
import pickle

#------------------------------------------------------------------------------
'Functions'
def W(p, u, v):
    assert len(u) == len(v)
    return np.mean(np.abs(np.sort(u)[1:u.size-1]-np.sort(v)[1:v.size-1])**p)**(1/p)

def calculate_alpha(beta, delta):
    return delta*np.sqrt(scipy.special.gamma(1/beta)/scipy.special.gamma(3/beta))

#------------------------------------------------------------------------------
'Modifiyable variables'    
epochs = 100
times = 10
nSample = 3000

#------------------------------------------------------------------------------
'Main Code'
distance = np.zeros((6, times, epochs)) # [different distribution, epoch]
para = np.zeros((4, times, epochs))

param_gen = []
param_norm = []
param_lap = []
for time in range(times):
    with open('gradient_17000nodes'+str(time)+'.pickle', 'rb') as f:
        gradient = pickle.load(f)
    for epoch in range(epochs):
        data = np.array(gradient[epoch])
        #-----------------------------------------------------------------------------#
        rv_x_new = np.random.choice(data, nSample)
        #----data normalize-----------------------------------------------------------#
        std_val = rv_x_new.std()
        data_n = rv_x_new/rv_x_new.std()
        # generate the fit distribution to calculate the wasserstein_distance
        
        #---Norm---------------------------------------------------------------
        [mean_fit,std_fit] = norm.fit(data_n, floc=0)  # estimate norm distribution
        param_norm.append(std_fit)
        
        # min_val = float("inf")
        # for i0 in param_norm:
        #     temp = W(2,norm.ppf(np.linspace(0,1,data_n.size),0, i0),data_n)
        #     min_val = min(temp, min_val)
        # distance[0, epoch] += min_val    
        
        # min_val = float("inf")
        # for i0 in param_norm:
        #     a = norm.ppf(np.linspace(0,1,data_n.size),0, i0)
        #     temp = energy_distance(a[1:a.size-1],data_n[1:data_n.size-1])
        #     min_val = min(temp, min_val)
        # distance[3, epoch] += min_val
        
        #W2 Distance
        distance[0, time, epoch] = min(W(2, norm.ppf(np.linspace(0,1,data_n.size),0, i0), data_n) for i0 in param_norm)
        
        #Energy Distance
        distance[3, time, epoch] = min(energy_distance(norm.ppf(np.linspace(0,1,data_n.size),0, i0)[1:data_n.size-1],data_n[1:data_n.size-1]) for i0 in param_norm)
        
        #---Laplace-------------------------------------------------------------
        [arg1,arg2] = laplace.fit(data_n, floc=0)  # estimate laplace distribution
        param_lap.append(arg2)

        #W2 Distance
        distance[1, time, epoch] = min(W(2, laplace.ppf(np.linspace(0,1,data_n.size),0, i0), data_n) for i0 in param_lap)
        
        #Energy Distance
        distance[4, time, epoch] = min(energy_distance(laplace.ppf(np.linspace(0,1,data_n.size),0, i0)[1:data_n.size-1],data_n[1:data_n.size-1]) for i0 in param_lap)

       #---GenNorm-------------------------------------------------------------
        [arg4, arg5, arg6]= gennorm.fit(data_n, floc=0)  # estimate gennorm distribution
        param_gen.append([arg4,arg6])

        #W2 Distance
        min_val, best_beta, best_alpha = min([[W(2,gennorm.ppf(np.linspace(0,1,data_n.size),
                                                               beta=i0, loc=0,scale=i1),data_n), i0, i1] for i0, i1 in param_gen], key = lambda x: x[0])
        distance[2, time, epoch] = min_val
        
        #Energy Distance
        min_val, _, _ = min([[energy_distance(gennorm.ppf(np.linspace(0,1,data_n.size),
                                                                           beta=i0, loc=0,scale=i1)[1:data_n.size-1],data_n[1:data_n.size-1]), i0, i1] for i0, i1 in param_gen], key = lambda x: x[0])
        distance[5, time, epoch] = min_val
        #----------------------------------------------------------------------
        para[0, time, epoch] = best_beta # best beta obtained through W2
        para[1, time, epoch] = best_alpha*std_val # best alpha obtained through W2
        para[2, time, epoch] = arg4 # after normalize the data 
        para[3, time, epoch] = arg6*std_val # true alpha value
    
    param_gen = []
    param_norm = []
    param_lap = []
    
    print('Simulation #', time, ' done')
    
distance_mean = distance.mean(axis=1)
para_mean = para.mean(axis=1)

#------------------------------------------------------------------------------
"Plotting"
plt.figure(1)
plt.plot(distance_mean[0],color='red',label='norm')
plt.plot(distance_mean[1],color='blue',label='laplace')
plt.plot(distance_mean[2],color='yellow',label='gennorm')
plt.xlabel('Epochs')
plt.ylabel('Wasserstein Distance')
plt.plot()
plt.legend()
plt.grid()

plt.figure(2)
plt.plot(distance_mean[3],color='red',label='norm')
plt.plot(distance_mean[4],color='blue',label='laplace')
plt.plot(distance_mean[5],color='yellow',label='gennorm')
plt.xlabel('Epochs')
plt.ylabel('Energy Distance')
plt.plot()
plt.legend()
plt.grid()

plt.figure(3)
plt.plot(para_mean[0], color = 'blue',label='Best $\\beta$')
plt.plot(para_mean[2], color = 'red',label='Epoch\'s $\\beta$')
plt.xlabel('Epoch')
plt.ylabel('$\\beta$')
plt.title('Average $\\beta$ across ' + str(times) +' simulations')
plt.plot()
plt.legend()
plt.grid()
plt.show()

plt.figure(4)
plt.plot(para_mean[1], color = 'blue',label='Best $\\alpha$')
plt.plot(para_mean[3], color = 'red',label='Epoch\'s $\\alpha$')
plt.xlabel('Epoch')
plt.ylabel('$\\alpha$ (scale parameter)')
plt.title('Average $\\alpha$ across ' + str(times) +' simulations (in log scale)')
plt.plot()
plt.yscale("log")
plt.legend()
plt.grid()