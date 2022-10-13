# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 19:25:58 2022

@author: USER
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import os
#---------------------
'variables'
epochs = 500
samples = 100   # like one picture of the minst dataset
features = 100  # like the []
node1_all = [10000, 5000, 1000, 500, 100]
node2 = 100
# node1_B = [11000, 13000, 15000, 17000, 19000] # overparameter
# node1_B = [5000, 7000, 9000, 10000, 11000]         # underparameter
# cases = 5
times = 15      # for each cases 
learning_rate=0.001
name = "weight"
path = 'C:\\Users\\USER\\Desktop\\Python project\\Python project\\different hidden layer\\'
#-------------------------------------------------------------------------------
# construct network A for generating data (fixed weight:seed 100)
class Network_A(tf.keras.Model):
    def __init__(self, node1):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units = node1 ,
                           kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=1, seed = 100 ),
                           use_bias=False,
                           )
        self.dense2 = tf.keras.layers.Dense(units = node2,
                           kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=1, seed = 100 ),
                           use_bias=False,
                           )
    def call(self, input):
        x = self.dense1(input)
        output = self.dense2(x)
        return output

# construct network B for training

class Network_B(tf.keras.Model):
    def __init__(self, node1):
        super().__init__()
        self.dense1= tf.keras.layers.Dense(units = node1,
                          kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=1),
                          use_bias=False,
                          )
        self.dense2 = tf.keras.layers.Dense(units = node2,
                           kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=1),
                           use_bias=False,
                           )

    def call(self, input):
        x = self.dense1(input)
        output = self.dense2(x)
        return output

# change tf to numpy    
def change_type(grads):
    for idx in range(len(grads)):
           grads[idx] = grads[idx].numpy().flatten()
    return np.array(grads,dtype=object)

# losses = np.zeros((times,epochs,cases))
losses = np.zeros((times,epochs))
# case = 0
for nodes in node1_all:
    sample_index_nodeB = np.random.choice(np.arange(features*nodes), size=int(features*nodes*0.1))
    # isExists = os.path.exists(path+str(nodes))
    # if not isExists:
    #     os.makedirs(path+str(nodes))
    #     print("file with "+str(nodes)+'create successful')
    #     pass
    # else:
    #     print("file with "+str(nodes)+'create successful')
    #     pass
    for time in range(times):
        model = Network_B(nodes)
        optimizer = tf.keras.optimizers.SGD(learning_rate)
        A = Network_A(nodes)
        loss_fn = tf.keras.losses.MeanSquaredError()
        empty = [] # prepare for sotting gradient
        
        np.random.seed(0) #fit the input data state
    #-----------------------------------------------------------------#
        for epoch in range(epochs):
            # the data input should be the random variable
            X = np.random.normal(0, 1, size=(samples, features)).astype(np.float32) #  [batch sizes , feature ]# we can also change 
            y = A(X)
            # record the gradient and updated the weight by the MMSE
            with tf.GradientTape() as tape:
                y_pred = model(X)
                loss = loss_fn(y, y_pred) # use MMSE should start from construct the instance
                # losses[time,epoch,case] = loss
                losses[time,epoch] = loss
                print("epoch %d: loss %f" % (epoch+1, loss.numpy()))
            grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
            new = change_type(grads) # tf to numpy
            # d = new.flatten()
            empty.append(new[1]) # only sort for the hidden layer
        print(str(time) + ' simulations done')
        ## sort the 'gradient' data which is in the hidden layer 
        # with open('gradient_'+str(nodes)+'nodes'+str(time)+'.pickle', 'wb') as f:
        #     pickle.dump(empty, f)
        ## sort the 'weight' of the model  which is in the hidden layer 
        # weights = model.get_weights() # the weight of the model 
        # layer1 = weights[1].flatten()
        # sample_layer1 = layer1[sample_index_nodeB]
        # np.set_printoptions(threshold=np.inf)
        # with open(path+str(nodes)+'//'+'sim_'+str(time)+'sample_weight.pickle', 'wb') as file:
        #     pickle.dump(sample_layer1, file)
    # case+=1
# losses = losses.mean(axis=0)
# plt.figure()
# plt.plot(losses,color='red', label='size of 7000')
# # plt.plot(losses[:,1],color='blue', label='size of 13000')
# # plt.plot(losses[:,2],color='green', label='size of 15000')
# # plt.plot(losses[:,3],color='yellow', label='size of 17000')
# # plt.plot(losses[:,4],color='black', label='size of 19000')
# plt.xlabel('training epochs')
# plt.ylabel('training loss')
# plt.yscale("log")
# plt.title('loss with overparameter cases')
# plt.legend()
# plt.grid()
# plt.show()  
    


