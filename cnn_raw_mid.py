# -*- coding: utf-8 -*-

from keras.datasets import cifar10
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from cnn_param_mid import DeepConvNet
from trainer import Trainer
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D

#size of parameters
batch_size = 128
input_size = 32*32 # 28*28
input_h= 32
hidden_size= 100
filter_size = 64
filter_pixel = 6
pool_pixel = 2
output_size = 10
epochs = 100

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#1-D
y_test = y_test.reshape(-1)
y_train = y_train.reshape(-1)

#Start Neural Network
model = Sequential()

#convolution 1st layer
model.add(Conv2D(filter_size, kernel_size=(filter_pixel, filter_pixel), padding='same',
                 activation='relu',strides=2,
                 input_shape=x_train.shape[1:])) #0
model.add(Dropout(0.25))

model.add(Conv2D(filter_size*2, kernel_size=(3, 3), padding='same',strides=2,
                 activation='relu')) 
model.add(Dropout(0.25))

#Fully connected 1st layer
model.add(Flatten()) #1

#Fully connected final layer
model.add(Dense(output_size)) #2
model.add(Dropout(0.25))
model.add(Activation('softmax')) #3

model.load_weights("./model/param_mid.hdf5")

W1=model.layers[0].get_weights()
W2=model.layers[2].get_weights()
W3=model.layers[5].get_weights()
##########################
##start frame work########
##########################

#keras format = N H W C
#framework format = N C H W
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

#x_test = x_test.transpose([0,1,3,2])

#keras conv2 H W C FN
#fw conv2 FN C FH FW
W1[0] = W1[0].transpose([3,2,0,1])
W2[0] = W2[0].transpose([3,2,0,1])


network = DeepConvNet(W1,W2,W3)  

print("start inference")
a = network.accuracy(x_test[0:100], y_test[0:100])

print(a)

#trainer = Trainer(network, x_train, y_train, x_test[0:100], y_test[0:100],
#                  epochs=20, mini_batch_size=100,
#                  optimizer='Adam', optimizer_param={'lr':0.001},
#                  evaluate_sample_num_per_epoch=1000)



#trainer.inference()
