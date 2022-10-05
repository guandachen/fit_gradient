# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 14:05:02 2022

@author: USER
"""

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
'variables'
epochs = 500
samples = 100   # like one picture of the minst dataset
features = 100  # like the pixel number in one picture
# node1_A = 10000
node2 = 100
times = 10     # for each cases 
learning_rate=0.01
name = "weight"

sample_index_nodeB = np.random.choice(np.arange(200*25), size=int(200*25*0.1))
# construct network A for generating data (fixed weight:seed 100)
class Network_A(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv_B = tf.keras.layers.Conv2D(
            filters=400,
            kernel_size=[5,5],
            padding ='valid',
            activation=None,
            use_bias=False)
        self.dense2 = tf.keras.layers.Dense(units = node2,
                           kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=1,seed=0),
                           use_bias=False,
                           )
    def call(self, input):
        x = self.conv_B(input)
        # x = self.pool_B(x)
        output = self.dense2(x)
        return output 

class CNN_B(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv_B = tf.keras.layers.Conv2D(
            filters=200,
            kernel_size=[5,5],
            padding ='valid',
            activation=None,
            use_bias=False)
        # self.pool_B = tf.keras.layers.MaxPool2D(pool_size=[2,2],strides=2)
        self.dense2 = tf.keras.layers.Dense(units = node2,
                           kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=1),
                           use_bias=False,
                           )
    def call(self, input):
        x = self.conv_B(input)
        # x = self.pool_B(x)
        output = self.dense2(x)
        return output 
    
# change tf to numpy    
def change_type(grads):
    for idx in range(len(grads)):
           grads[idx] = grads[idx].numpy().flatten()
    return np.array(grads,dtype=object)
for time in range(times): 
    # model setting
    optimizer = tf.keras.optimizers.SGD(learning_rate)
    loss_fn = tf.keras.losses.MeanSquaredError()
    # sort the model data about gradient or weight
    A = Network_A()
    model = CNN_B()
    empty = [] 
    losses = []
    #fit the input data state
    np.random.seed(0) 
    # train
    for epoch in range(epochs):
        X = np.random.normal(0, 1, size=(samples, 10, 10, 1)).astype(np.float32) #  [batch sizes , feature ]# we can also change 
        y = A(X)
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = loss_fn(y, y_pred) # use MMSE should start from construct the instance
            # losses[time,epoch,case] = loss
            losses.append(loss)
            print("epoch %d: loss %f" % (epoch+1, loss.numpy()))
        grads_cnn = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads_cnn, model.variables))
        # new_cnn = change_type(grads_cnn) # tf to numpy
        weights = model.get_weights() # the weight of the model 
        layer1 = weights[1].flatten()
        sample_layer1 = layer1[sample_index_nodeB]
        # np.set_printoptions(threshold=np.inf)
        with open('sim_'+str(time)+'_sample_weight.pickle', 'wb') as file:
            pickle.dump(sample_layer1, file)
print(A.summary())
print(model.summary())
plt.figure()
plt.plot(losses,color='red', label='size of 5000 ')
plt.xlabel('training epochs')
plt.ylabel('training loss')
# plt.yscale("log")
plt.title('loss for CNN cases')
plt.legend()
plt.grid()
plt.show() 
    