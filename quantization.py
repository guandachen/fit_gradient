# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 12:43:43 2022

@author: USER
"""
import os 
import numpy as np 
import pickle
import tensorflow as tf 
import matplotlib.pyplot as plt
from numpy import linalg as LA
import seaborn as sn
'parameter' 
path = 'D:\\student_teacher_save_weight\\'
nodes = 5
times = 1
bins = 5
neurons = 5

# with open(path+str(nodes)+'permute'+'\\'+'sim_'+'layer1_weight.pickle', 'wb') as file:
#     pickle.dump(sample_layer1, file)
# with open(path+str(nodes)+'permute'+'\\'+'sim_'+'layer0_weight.pickle', 'wb') as file:
#     pickle.dump(sample_layer0, file) 


'loading data'
# q_weight0 = np.zeros((times,50,5))
# q_weight1 = np.zeros((times,50,5))
for time in range(times):
    with open(path+str(nodes)+'permute'+'\\'+'sim_'+str(time)+'sample_weight.pickle', 'rb') as file:
         q_weight[time,:,:] = pickle.load(file)
weight_permute = np.zeros((times,50,5))

'processing'
quan_sort = np.zeros((times,5,10)) # times, num of neuron, bins level
quan_sort_check = np.zeros((times,5,10))
weight_max = np.max(q_weight)
q_weight = q_weight/weight_max

'statictics'
mean = np.mean(q_weight)
var = np.var(q_weight)

'quantization'
diff = (np.max(q_weight)-np.min(q_weight))/10
bins = np.array([np.min(q_weight), np.min(q_weight)+1*diff, np.min(q_weight)+2*diff, np.min(q_weight)+3*diff, np.min(q_weight)+4*diff, np.min(q_weight)+5*diff, 
                 np.min(q_weight)+6*diff, np.min(q_weight)+7*diff, np.min(q_weight)+8*diff, np.min(q_weight)+9*diff, np.max(q_weight)])

for time in range(times):
    for neuron in range(neurons):
        # quan_sort[time,neuron,:], _ = np.histogram(q_weight[time,:,neuron]+np.random.normal(0,0.01*var),bins)
        quan_sort[time,neuron,:], _ = np.histogram(0.99*(q_weight[time,:,neuron]),bins) 
        quan_sort_check[time,neuron,:], _ = np.histogram(q_weight[time,:,neuron],bins)

# distance = np.zeros((5,5))
# for x in range(neurons):
#     for y in range(neurons):
#         distance[x,y] = LA.norm(quan_sort[0, x, :] - quan_sort[1, y, :])
# sn.heatmap(distance)
# plt.show()

'sanity check'
distance_check = np.zeros((5,5))
for x in range(neurons):
    for y in range(neurons):
        k = quan_sort[0, x, :] - (quan_sort_check[0, y, :])
        distance_check[x,y] = LA.norm(quan_sort[0, x, :] - (quan_sort_check[0, y, :]))
        # print(k)
plt.figure(1)
plt.title('weight v.s weight+noise')
sn.heatmap(distance_check, annot=True)
plt.xlabel('neuron')
plt.ylabel('neuron')
plt.show()

#-------------------------------------------------------------------------------------------------#

'Sanity check of the quantization error to explain the similarity heatmap is not symmertric '
check_neuron2_Noise = 0.99*(q_weight[0,:,2])
check_neuron3_Noise = 0.99*(q_weight[0,:,3])
check_neuron3 = q_weight[0,:,3]
check_neuron2 = q_weight[0,:,2 
                          ]
for i in range(len(check_neuron2)):
    # print('here')    
    if check_neuron3[i] >= -0.011 and check_neuron3[i] <= 0.1911:
        print(check_neuron3[i],'is belong to label 5',i)
    if check_neuron3_Noise[i] >= -0.011 and check_neuron3_Noise[i] <= 0.1911:
        print(check_neuron3_Noise[i],'is belong to label 5(noise)',i)
    
    # if check_neuron2[i] >= 0.1911 and check_neuron2[i] <= 0.3933:
    #     print(check_neuron2[i],'is belong to label 6')
    # if check_neuron2_Noise[i] >= 0.1911 and check_neuron2_Noise[i] <= 0.3933:
    #     print(check_neuron2_Noise[i],'is belong to label 6(noise)')
    
    # if check_neuron2[i] >= 0.3933 and check_neuron2[i] <= 0.5955:
    #     print(check_neuron2[i],'is belong to label 7')
    # if check_neuron2_Noise[i] >= 0.3933 and check_neuron2_Noise[i] <= 0.5955:
    #     print(check_neuron2_Noise[i],'is belong to label 7(noise)',i)
    
    # if check_neuron2[i] >= 0.5955 and check_neuron2[i] <= 0.7977:
    #     print(check_neuron2[i],'is belong to label 8',i)
    # if check_neuron2_Noise[i] >= 0.5955 and check_neuron2_Noise[i] <= 0.7977:
    #     print(check_neuron2_Noise[i],'is belong to label 8(noise)',i)
#------------------------------------------------------------------------------------------------#



