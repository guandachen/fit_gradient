# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 13:08:05 2023

@author: USER
"""
import time
import pickle
import os
import sys
import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.models import model_from_json

import tensorflow as tf
from keras.models import load_model
# nodes = 10
# times = 2
node1 = 5
samples = 100
features = 50
def elem(size,i,j):
    matrix = np.eye(size)
    matrix[i][i] = 0
    matrix[j][j] = 0
    matrix[i][j] = 1
    matrix[j][i] = 1
    return matrix
class Student(tf.keras.Model):
    def __init__(self, node1):
        super().__init__()
        self.dense1= tf.keras.layers.Dense(units = node1,
                          kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=1),
                          use_bias=False,
                          )
        self.dense2 = tf.keras.layers.Dense(units = 10,
                           kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=1),
                           use_bias=False,
                           activation = 'softmax'
                           )

    def call(self, input):
        x = self.dense1(input)
        output = self.dense2(x)
        return output

x = np.random.normal(0,1,(samples, features))
model = Student(node1)
model.build(input_shape=(samples, features))
model.load_weights('Student_weights.h5', by_name = True)
y = model(x)
print('without permutation output:', y[0])
rest = model.get_weights()
rest = np.asarray(rest)
print('neuron 2 :',rest[0][0][2])
print('neuron 3:',rest[0][0][3])

rest[0] = np.matmul(rest[0],elem(5,2,3))
rest[1] = np.matmul(np.linalg.inv(elem(5,2,3)),rest[1])
print('----after permutation----')
print('neuron 2:',rest[0][0][2])
print('neuron 3:',rest[0][0][3])

# rest = rest.tolist()
# k = model.layers[0].get_weights()
# k = rest[0]
model.layers[0] = rest[0]
model.layers[1] = rest[1]
# model.layers[0].set_weights(rest[0])
# model.layers[1].set_weights(rest[1])
# model_P = Student(node1)
# model_P.build(input_shape=(samples, features))
# model_P.load_weights('Student_weights.h5', by_name = True)
y_p = model(x)
print('permutation output:', y_p[0])