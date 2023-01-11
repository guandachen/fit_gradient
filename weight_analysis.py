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

model_P = Student(node1)
model_P.build(input_shape=(samples, features))
model_P.load_weights('Student_weights.h5', by_name = True)
y_p = model(x)