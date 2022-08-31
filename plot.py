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

def W(p, u, v):
    assert len(u) == len(v)
    return np.mean(np.abs(np.sort(u)[1:u.size-1]-np.sort(v)[1:v.size-1])**p)**(1/p)

def calculate_alpha(beta, delta):
    return delta*np.sqrt(scipy.special.gamma(1/beta)/scipy.special.gamma(3/beta))

distance = np.zeros((6,100)) # [different distribution, epoch]
para = np.zeros((2,100))
epochs = 100
times = 10
nSample = 3000
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
        [mean_fit,std_fit] = norm.fit(data_n, floc=0)  # estimate norm distribution
        param_norm.append(std_fit)
        min_val = float("inf")
        for i0 in param_norm:
            temp = W(2,norm.ppf(np.linspace(0,1,data_n.size),0, i0),data_n)
            min_val = min(temp, min_val)
        distance[0, epoch] += min_val
        #------#
        in_val = float("inf")
        for i0 in param_norm:
            a = norm.ppf(np.linspace(0,1,data_n.size),0, i0)
            temp = energy_distance(a[1:a.size-1],data_n[1:data_n.size-1])
            min_val = min(temp, min_val)
        distance[3, epoch] += min_val
        ##
        [arg1,arg2] = laplace.fit(data_n, floc=0)  # estimate laplace distribution
        param_lap.append(arg2)
        in_val = float("inf")
        for i0 in param_lap:
            a = laplace.ppf(np.linspace(0,1,data_n.size),0, i0)
            temp = energy_distance(a[1:a.size-1],data_n[1:data_n.size-1])
            min_val = min(temp, min_val)
        distance[4, epoch] += min_val
        
        min_val = float("inf")
        for i0 in param_lap:
            temp = W(2,laplace.ppf(np.linspace(0,1,data_n.size),0, i0),data_n)
            min_val = min(temp, min_val)
        distance[1, epoch] += min_val

       
        [arg4, arg5, arg6]= gennorm.fit(data_n, floc=0)  # estimate gennorm distribution
        param_gen.append([arg4,arg6])
        min_val = float("inf")
        for i0,i1 in param_gen:
            temp = W(2,gennorm.ppf(np.linspace(0,1,data_n.size), beta=i0, loc=0,scale=i1),data_n)
            min_val = min(temp, min_val)

        distance[2, epoch] += min_val
        min_val = float("inf")
        for i0,i1 in param_gen:
            c = gennorm.ppf(np.linspace(0,1,data_n.size), beta=i0, loc=0,scale=i1)
            temp = energy_distance(c[1:c.size-1],data_n[1:data_n.size-1])
            min_val = min(temp, min_val)
        distance[5, epoch] += min_val

        para[0,epoch] += arg4 # after normalize the data 
        para[1,epoch] += arg6*std_val # true alpha value
    param_gen = []
distance_mean = distance/times
para_mean = para/times


plt.figure(1)
plt.plot(distance_mean[0,:],color='red',label='norm')
plt.plot(distance_mean[1,:],color='blue',label='laplace')
plt.plot(distance_mean[2,:],color='yellow',label='gennorm')
plt.xlabel('training epochs')
plt.ylabel('wasserstein_distance')
plt.plot()
plt.legend()
plt.grid()

plt.figure(2)
plt.plot(para_mean[1,:], color = 'red',label='α')
plt.xlabel('training epochs')
plt.ylabel('value of alpha')
plt.plot()
plt.legend()
plt.grid()

plt.figure(3)
plt.plot(para_mean[0,:], color = 'red',label=' true β')
plt.xlabel('training epochs')
plt.ylabel('value of beta')
plt.title('average of β with 10 times experiment')
plt.plot()
plt.legend()
plt.grid()
plt.show()

plt.figure(4)
plt.plot(distance_mean[3,:],color='red',label='norm')
plt.plot(distance_mean[4,:],color='blue',label='laplace')
plt.plot(distance_mean[5,:],color='yellow',label='gennorm')
plt.xlabel('training epochs')
plt.ylabel('energy_distance')
plt.plot()
plt.legend()
plt.grid()

# for i in range(len(distance_mean[0,:])):
#     if distance_mean[2,i] >= distance_mean[1,i]:
#         print(i)
# k = np.array(gradient[29])
# std_k = k.std()
# k_n = k/k.std()
# gg = np.random.choice(k, nSample)
# [arg1,arg2] = laplace.fit(gg, floc=0)
# [arg4, arg5, arg6]= gennorm.fit(gg, floc=0)    
     