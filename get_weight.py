# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 16:18:22 2022

@author: USER
"""
import os 
import numpy as np 
import pickle
import tensorflow as tf 
import matplotlib.pyplot as plt
import seaborn as sn
from numpy import linalg as LA
# times = 1      # for each cases 
# path = 'D:\simulation data\\'
path = 'C:\\Users\\USER\\Desktop\\Python project\\Python project\\different hidden layer\\cnn case\\'
# path = 'D:\MNIST Train\Dense'
# node1_B = [11000, 13000, 15000, 17000, 19000] # overparameter
# node1_B = [5000, 7000, 9000]         # underparameter
node1_B = 5000
times = 10
# model_name = 'ConvNet'
model_name = 'Dense'
# features = 100
# sample_Data = np.zeros((times,int(features*node1_B*0.1)))  # For DNN with Teacher and Student Network
sample_Data = np.zeros((times,int(5*5*20*10)))                # For CNN with Teacher and Student Network
# sample_Data = np.zeros((times,int(3*3*32*64*0.1)))         # For ConvNet with MNIST dataset
# sample_Data = np.zeros((times,int(512*512*0.1)))           # For MLP with MNIST dataset
# sample_Data = np.zeros((times,int(784*10*0.1)))              # For Dense with MNIST dataset
save_Data = np.zeros((times,times))
for time in range(times):
    # with open(path+str(node1_B)+'//'+'sim_'+str(time)+'sample_weight.pickle', 'rb') as f:
    with open (path+'sim_'+str(time)+'_sample_weight.pickle','rb') as f:
    # with open (path+'\\'+str(model_name)+str(time)+'weight.pickle','rb') as f:    
        sample_Data[time,:] = pickle.load(f)

for time in range(times):
    for x in range(times):
        save_Data[time,x] = (sample_Data[time,:].dot(sample_Data[x,:]))/(LA.norm(sample_Data[time,:])*LA.norm(sample_Data[x,:]))
        save_Data[time,x] = np.abs(np.round(save_Data[time,x], 5))
# with open(path+str(node1_B)+'//'+'sim_with_size' + str(node1_B)+'cosine similarity.pickle', 'wb') as file:
#     pickle.dump(save_Data, file)
    
# for i in range(times):
#     save_Data[i,i] = 0 
sn.heatmap(save_Data, annot=True, fmt='g',cmap='YlGnBu_r')
plt.title('size of different times simulation cosine similarity')
plt.show()