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
# path = 'C:\\Users\\USER\\Desktop\\Python project\\Python project\\different hidden layer\\cnn case\\'
path = 'D:\\student_teacher_save_weight\\'
# path = 'D:\MNIST Train\ConvNet'
# path = path = 'D:\\MNIST Train\\Dense'
# node1_B = [11000, 13000, 15000, 17000, 19000] # overparameter
# node1_B = [5000, 7000, 9000]         # underparameter
# node1_B = 5000
nodes = 5 #for teacher_student Network [100, 500, 1000, 5000,10000]
times = 10
# model_name = 'ConvNet'
# model_name = 'MLP'
# model_name = 'Dense'
features = 50
epochs = 2000
# sample_Data = np.zeros((times,int(features*node1_B*0.1)))  # For DNN with Teacher and Student Network(BIG)
# sample_Data = np.zeros((times,int(5*5*20*10)))             # For CNN with Teacher and Student Network
sample_Data = np.zeros((times,int(3*3*32*64*0.1)))         # For ConvNet with MNIST dataset
# sample_Data = np.zeros((times,int(784*512*0.1)))           # For MLP with MNIST dataset
# sample_Data = np.zeros((times,int(784*10*0.1)))            # For Dense with MNIST dataset
# sample_Data = np.zeros((times,int(features*nodes*0.1)))      # For DNN with Teacher and Student Network(FIND similarity)
# sample_Data = np.zeros((times, int(features*nodes*0.1)))
save_Data = np.zeros((times,times))
#----------------------------------------------------------------------------------------------#
# for time in range(times):
#     # with open(path+str(node1_B)+'//'+'sim_'+str(time)+'sample_weight.pickle', 'rb') as f:
#     # with open (path+'sim_'+str(time)+'_sample_weight.pickle','rb') as f:
#     with open (path+'\\'+str(model_name)+str(time)+'weight.pickle','rb') as f:
#     # with open(path+str(nodes)+'\\'+'sim_'+str(time)+'sample_weight.pickle','rb') as f:     
#         sample_Data[time,:] = pickle.load(f)


# for time in range(times):
#     for x in range(times):
#         save_Data[time,x] = (sample_Data[time,:].dot(sample_Data[x,:]))/(LA.norm(sample_Data[time,:])*LA.norm(sample_Data[x,:]))
#         save_Data[time,x] = np.abs(np.round(save_Data[time,x], 5))
#         # print(save_Data[time,x] )
# denom = np.ones_like(save_Data)
# norm = np.linalg.norm(save_Data)/np.linalg.norm(denom)



# sn.heatmap(save_Data, annot=True, fmt='g',cmap='YlGnBu_r')
# plt.title('across 10 times simulation cosine similarity' + '\nFrobenius norm of ' + str(norm))
# plt.title(' 10 times simulation cosine similarity' + '\nFrobenius norm of ' + str(norm))
# plt.show()

#----------------------------------------------------------------------------------------------#
# sample_Data = np.zeros((times,epochs, int(features*nodes)))
sample_Data = np.zeros((times,epochs,50))
save_Data = np.zeros((epochs,times,times))
# frob_norm = np.zeros((5,1200))
frob_norm = np.zeros((times,epochs))
for epoch in range(epochs):
    for time in range(times):
        # with open(path+str(node1_B)+'//'+'sim_'+str(time)+'sample_weight.pickle', 'rb') as f:
        # with open (path+'sim_'+str(time)+'_sample_weight.pickle','rb') as f:
        # with open (path+'\\'+str(model_name)+str(time)+'weight.pickle','rb') as f:
        with open(path+str(nodes)+'permute'+'\\'+'sim_'+str(time)+str(epoch)+'layer2_weight.pickle','rb') as f:     
            sample_Data[time,epoch,:] = pickle.load(f)
        
for epoch in range(epochs):  
    for time in range(times):
        for x in range(times):        
            save_Data[epoch,time,x] = (sample_Data[time,epoch,:].dot(sample_Data[x,epoch,:]))/(LA.norm(sample_Data[time,epoch,:])*LA.norm(sample_Data[x,epoch,:]))
            save_Data[epoch,time,x] = np.abs(np.round(save_Data[epoch,time,x], 5))
        denom = np.ones_like(save_Data[epoch,:,:])
        frob_norm[time,epoch] = np.linalg.norm(save_Data[epoch,:,:])/np.linalg.norm(denom)
avg = np.mean(frob_norm,axis=0)
plt.figure()
plt.xlabel('epochs')
plt.ylabel('frobenius norm')
plt.title('frobenius norm  for the hidden layer size 5')
plt.plot(np.arange(len(avg)), avg)
plt.show()
#--------------------------------------------------------------------------------------------------#
# i = 0
# for index in range(1200):
#     if frob_norm[index]>=0.55 :#and i !=1 :
#         print(index)
#         i+=1 