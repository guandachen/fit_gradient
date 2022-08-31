# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 19:25:58 2022

@author: USER
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy
# from scipy.interpolate import interp1d
# from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import gennorm, norm, laplace
import pickle
epochs = 100
node1_A = 10000
node2 = 100
node1_B = 1000
# construct network A for generating data (fixed weight:seed 100)

class Network_A(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units = node1_A ,
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
    def __init__(self):
        super().__init__()
        self.dense1= tf.keras.layers.Dense(units = node1_B,
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
times = 10


for time in range(times):
    model = Network_B()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
    A = Network_A()
    loss_fn = tf.keras.losses.MeanSquaredError()
    losses = []
    empty = []
#-----------------------------------------------------------------#
    for epoch in range(epochs):
        # the data input should be the random variable
        X = np.random.normal(0, 1, size=(100, 1)).astype(np.float32)
        y = A(X)
        # record the gradient and updated the weight by the MMSE
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = loss_fn(y, y_pred) # use MMSE should start from construct the instance
            losses.append(loss)
            print("epoch %d: loss %f" % (epoch+1, loss.numpy()))
        grads = tape.gradient(loss, model.variables)
        #------------
        #modify gradients to fp4 or fp8 here
        # grads = fp4(grads)
        #------------
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
        new = change_type(grads) # tf to numpy
        d = new.flatten()
        empty.append(d[1]) # only sort for the hidden layer
    with open('gradient6-'+str(time)+'.pickle', 'wb') as f:
        pickle.dump(empty, f)


plt.plot(losses,color='red', label='loss')
plt.xlabel('training epochs')
plt.ylabel('training loss')
plt.legend()
plt.grid()
plt.show()
