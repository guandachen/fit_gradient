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
from keras.models import load_model
from keras.models import model_from_json
#---------------------
'variables'
epochs = 60000  
samples = 100  # like one picture of the minst dataset
features = 50  # like the []
# node1_all = [10000, 5000, 1000, 500, 100]
node1_all = [5]
node2 = 10
# node1_B = [11000, 13000, 15000, 17000, 19000] # overparameter
# node1_B = [5000, 7000, 9000, 10000, 11000]    # underparameter
times = 1      # for each cases 
# learning_rate=[0.001, 0.001, 0.001, 0.01, 0.01]
learning_rate = [0.01]
path = 'D:\\student_teacher_save_weight\\'
#-------------------------------------------------------------------------------
# construct network A for generating data (fixed weight:seed 100)
class Network_A(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units = 10,
                           kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=1, seed = 100),
                           use_bias=False,
                           )
        self.dense2 = tf.keras.layers.Dense(units = node2,
                           kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=1, seed = 100),
                           use_bias=False,
                           activation = "softmax"
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
                           activation = 'softmax'
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
acc = np.zeros((times,epochs))
# shape = tf.TensorSpec(shape=(samples, features),dtype=tf.dtypes.float32)
# case = 0
for nodes, lr in zip(node1_all, learning_rate):
    # sample_index_nodeB = np.random.choice(np.arange(features*nodes), size=int(features*nodes*0.1))
    # isExists = os.path.exists(path+str(nodes))
    # if not isExists:
    #     os.makedirs(path+str(nodes))
    #     print("file with "+str(nodes)+'create successful')
    #     pass
    # else:
    #     print("file with "+str(nodes)+'create successful')
    #     pass
    isExists = os.path.exists(path+str(nodes)+'permute')
    if not isExists:
        os.makedirs(path+str(nodes)+'permute')
        print("file with "+str(nodes)+'_permute'+'create successful')
        pass
    else:
        print("file with "+str(nodes)+'_permute'+'create successful')
        pass
    for time in range(times):
        model = Network_B(nodes)
        # model._set_inputs(shape)
        optimizer = tf.keras.optimizers.SGD(lr)
        A = Network_A()
        # loss_fn = tf.keras.losses.MeanSquaredError()
        loss_fn = tf.keras.losses.CategoricalCrossentropy()
        np.random.seed(0) #fit the input data state
    #-----------------------------------------------------------------#
        for epoch in range(epochs):
            X = np.random.normal(0, 1, size=(samples, features)).astype(np.float32) #  [batch sizes , feature ]
            # the data input should be the random variable
            y = A(X)
            # record the gradient and updated the weight by the MMSE
            with tf.GradientTape() as tape:
                y_pred = model(X)
                loss = loss_fn(y, y_pred) #  start from construct the instance
                cross_acc = tf.keras.metrics.CategoricalAccuracy()
                losses[time,epoch] = loss
                cross_acc.update_state(y,y_pred)
                grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
            acc[time,epoch]=cross_acc.result().numpy()
            # print("epoch %d: loss %f" % (epoch+1, loss.numpy()))
            print(f"epoch {epoch+1}: acc {cross_acc.result().numpy()}% ")          
            # if epoch < 10000 :
            #     weights = model.get_weights() # the weight of the model  
            #     layer1 = weights[1].flatten()
            #     sample_layer1 = layer1
            #     with open(path+str(nodes)+'permute'+'\\'+'sim_'+str(time)+str(epoch)+'layer2_weight.pickle', 'wb') as file:
            #         pickle.dump(sample_layer1, file)
        weights = model.get_weights() # the weight of the model  
        # layer1 = weights[1]
        layer0 = weights[0]
        # sample_layer1 = layer1
        # with open(path+str(nodes)+'permute'+'\\'+'sim_'+'layer1_weight.pickle', 'wb') as file:
        #     pickle.dump(sample_layer1, file)
        # with open(path+str(nodes)+'permute'+'\\'+'sim_'+'layer0_weight.pickle', 'wb') as file:
        #     pickle.dump(sample_layer0, file)  
        with open(path+str(nodes)+'sim #'+str(time)+'layer0_weight.pickle','wb') as file:
             pickle.dump(layer0,file)              
        print(' simulations done')
        
losses = losses.mean(axis=0)
accs = acc.mean(axis=0)
plt.figure(1)
plt.title('accuracy')
plt.plot(accs,color='red',label="accuracy")
plt.yscale("log")
plt.xlabel('training epochs')
plt.ylabel('training acc')

plt.figure(2)
plt.title('accuracy')
plt.plot(losses,color='blue',label="loss")
plt.yscale("log")
plt.xlabel('training epochs')
plt.ylabel('training loss')
plt.grid()
plt.show() 
model.save_weights('Student_weights.h5')    

# model.save('Student', save_format='tf')
