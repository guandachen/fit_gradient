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
for time in range(times):
    with open('gradient6-'+str(time)+'.pickle', 'rb') as f:
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
        distance[0, epoch] += W(2,norm.ppf(np.linspace(0,1,data_n.size),mean_fit, std_fit),data_n)
        a = norm.ppf(np.linspace(0,1,data_n.size),mean_fit, std_fit)
        distance[3, epoch] += energy_distance(a[1:a.size-1],data_n[1:data_n.size-1])
        # distance_norm.append(distance1)
       
        [arg1,arg2] = laplace.fit(data_n, floc=0)  # estimate laplace distribution
        distance[1, epoch] += W(2,laplace.ppf(np.linspace(0,1,data_n.size), arg1, arg2),data_n)
        b = laplace.ppf(np.linspace(0,1,data_n.size), arg1, arg2)
        distance[4, epoch] += energy_distance(b[1:b.size-1],data_n[1:data_n.size-1])
        # distance_laplace.append(distance2)
       
        [arg4, arg5, arg6]= gennorm.fit(data_n, floc=0)  # estimate gennorm distribution
        distance[2, epoch] += W(2,gennorm.ppf(np.linspace(0,1,data_n.size), beta=arg4, loc=arg5,scale= arg6),data_n)
        c = gennorm.ppf(np.linspace(0,1,data_n.size), beta=arg4, loc=arg5,scale= arg6)
        distance[5, epoch] += energy_distance(c[1:c.size-1],data_n[1:data_n.size-1])
        # alpha_est = calculate_alpha(arg4, 1)
        # print("Alpha:", arg6)
        # print('Estimated', alpha_est)
        # print('Error', arg6 - alpha_est)
       
        # distance_gennorm.append(distance3)
        para[0,epoch] += arg4
        para[1,epoch] += arg6*std_val
distance_mean = distance/times
para_mean = para/times

sum_beta = 0


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
     