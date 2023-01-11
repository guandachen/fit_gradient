# -*- coding: utf-8 -*-

"""
## Setup
"""
import time
import sys
import pickle
import math
import numpy as np
import heapq

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import Progbar
from numpy.linalg import matrix_rank


def Dense(input_shape, num_classes):
    'Single Layer Dense'
    tf.random.set_seed(time.time())
    return keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Flatten(),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

def MLP(input_shape, num_classes):
    'Multi Layer Dense'
    tf.random.set_seed(time.time())
    return keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dense(512, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

def ConvNet(input_shape, num_classes):
    'Convolution + Dense Layers'
    tf.random.set_seed(time.time())
    return keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

architecture = {'Dense': Dense,
                'MLP': MLP,
                'ConvNet': ConvNet}

def step(x, y):
    # global opt, model
    # keep track of our gradients
    with tf.GradientTape() as tape:
        # make a prediction using the model and then calculate the loss
        logits = model(x, training=True)
        loss = keras.losses.categorical_crossentropy(y,logits)
        cross_acc = keras.metrics.CategoricalAccuracy()
        cross_acc.update_state(y,logits)
        
        # Compute the loss and the initial gradient
        grads = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

    for idx in range(len(grads)):
        grads[idx] = grads[idx].numpy().flatten()
    return np.array(grads,dtype=object), loss.numpy().mean(), cross_acc.result().numpy()


"""
## Prepare the data
"""
model_name = 'MLP'
learning_rate = 0.01
momentum = 0
batch_size = 128
epochs = 15
keras_fit = True
times = 10
assert model_name in list(architecture.keys()), 'Error! Model does not exist!'
# path = 'D:\\MNIST Train\\Dense'
path = 'D:\\MNIST Train\\'+str(model_name)
# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

"""
## Build the model
"""

model = architecture[model_name](input_shape, num_classes)

model.summary()

"""
## Train the model
"""
# sample_index_nodeB = np.random.choice(np.arange(3*3*32*64), size=int(3*3*32*64*0.1)) #For ConvNet
# sample_index_nodeB = np.random.choice(np.arange(512*512), size=int(512*512*0.1)) #For ConvNet
# sample_index_nodeB = np.random.choice(np.arange(784*10), size=int(784*10*0.1)) #For Dense       
for time in range(times):
    opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    
    if keras_fit:
        tf.random.set_seed(0)
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))
    else:
        total_size = len(x_train)
        batch_total = math.ceil(total_size/batch_size)
        train_acc = np.zeros(epochs)
        test_acc = np.zeros(epochs)
        train_loss = np.zeros(epochs)
        test_loss = np.zeros(epochs)
        
        elapsed = time.time()
        np.random.seed(0)
    
        for epoch in range(epochs):
            epoch_time = time.time()
            a0 = np.zeros(batch_total)
            l0 = np.zeros(batch_total)
            
            pb_i = Progbar(batch_total + 1, stateful_metrics=['loss', 'accuracy'])
            
            'Randomizing the input index'
            ind_perm = np.random.permutation(total_size)
            ind_perm = ind_perm[:-(total_size%batch_size)].reshape(-1, batch_size).tolist() + [ind_perm[-(total_size%batch_size):].tolist()]
            
            print('Epoch ' + str(epoch+1) + '/' + str(epochs))
            for batches, idx in enumerate(ind_perm):
                grads, l1, a1 = step(x_train[idx], y_train[idx])
        
                a0[batches] = a1
                l0[batches] = l1
                
                values=[('loss', l1), ('accuracy', a1)]
                pb_i.add(1, values=values)
        
            t_loss, t_acc = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
            
            train_acc[epoch] = a0.mean()
            test_acc[epoch] = t_acc
            train_loss[epoch] = l0.mean()
            test_loss[epoch] = t_loss
            epoch_time = time.time() - epoch_time
            
            values=[('accuracy', a0.mean()), ('val_acc', t_acc), ('epoch time (s)', epoch_time)]
            pb_i.update(batch_total + 1, values=values, finalize=True)
            
    """
    ## Evaluate the trained model
    """
    
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    
    ## sort the weight of the the model 
    
    weights = model.get_weights() # the weight of the model 
# for layer in range(len(weights)):
#     print(matrix_rank(weights[layer]))
    # # layer1 = weights[2].flatten() # for ConvNet & MLP
    layer1 = weights[0].flatten()   # for Desne
    layer1_k = np.abs(layer1)
    # # sample_layer1 = layer1
    top_k=784*512*0.1
    # if time == 0:
    bigval_index = layer1_k.argsort()[::-1][0:int(top_k)]
    sample_layer1 = layer1[bigval_index]
    
    with open(path+'\\'+str(model_name)+str(time)+'weight.pickle', 'wb') as file:
        pickle.dump(sample_layer1, file)
    