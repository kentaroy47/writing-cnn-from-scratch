# -*- coding: utf-8 -*-
"""
Created on Tue May 29 09:47:15 2018

@author: YOSHIOKA
"""

import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *


class DeepConvNet:
    """認識率99%以上の高精度なConvNet
    例題
        conv - relu - conv- relu - pool -
        conv - relu - conv- relu - pool -
        conv - relu - conv- relu - pool -
        affine - relu - dropout - affine - dropout - softmax
        
        my cifar10
        conv - relu - maxpool - conv - relu - conv - relu - pool
        affine - softmax
        
    """
    def __init__(self, W1, W2, W3, W4, W5, W6, W7, W8, input_dim=(3, 32, 32),
                 conv_param_1 = {'filter_num':64, 'filter_size':6, 'pad':2, 'stride':2},
                 conv_param_2 = {'filter_num':128, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_3 = {'filter_num':128, 'filter_size':3, 'pad':1, 'stride':1},
                 output_size=10):
        # 重みの初期化===========
        # 各層のニューロンひとつあたりが、前層のニューロンといくつのつながりがあるか（TODO:自動で計算する）      
        self.params = {}

        self.params['W1'] = W1[0]
        self.params['b1'] = W1[1]
        self.params['W2'] = W2[0]
        self.params['b2'] = W2[1]
        self.params['W3'] = W3[0]
        self.params['b3'] = W3[1]
        self.params['W4'] = W4[0]
        self.params['b4'] = W4[1]
        self.params['W5'] = W5[0]
        self.params['b5'] = W5[1]
        self.params['W6'] = W6[0]
        self.params['b6'] = W6[1]
        self.params['W7'] = W7[0]
        self.params['b7'] = W7[1]
        self.params['W8'] = W8[0]
        self.params['b8'] = W8[1]

        # レイヤの生成===========
        self.layers = []
        self.layers.append(Convolution(self.params['W1'], self.params['b1'], 
                           conv_param_1['stride'], conv_param_1['pad'])) #0
        self.layers.append(Relu()) #1
#        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2)) #2
        
        self.layers.append(Convolution(self.params['W2'], self.params['b2'], 
                           conv_param_2['stride'], conv_param_2['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W3'], self.params['b3'], 
                           conv_param_2['stride'], conv_param_2['pad']))
        self.layers.append(Relu())
        
        self.layers.append(Convolution(self.params['W4'], self.params['b4'], 
                           conv_param_2['stride'], conv_param_2['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W5'], self.params['b5'], 
                           conv_param_1['stride'], conv_param_2['pad']))
        self.layers.append(Relu())
        
        self.layers.append(Convolution(self.params['W6'], self.params['b6'], 
                           conv_param_2['stride'], conv_param_2['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W7'], self.params['b7'], 
                           conv_param_2['stride'], conv_param_2['pad']))
        self.layers.append(Relu())
        
        
#        self.layers.append(Flatten())
        self.layers.append(Affine(self.params['W8'], self.params['b8'])) #8
        
        self.last_layer = SoftmaxWithLoss() #9
 
    def predict(self, x, train_flg=False):
        for layer in self.layers:
            
            if isinstance(layer, Dropout):
                x = layer.forward(x, train_flg)
            if isinstance(layer, Flatten):
                print("layer is flatten")
                x = layer.forward(x)
            else:
                print(x.shape)
                x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x, train_flg=True)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        tmp_layers = self.layers.copy()
        tmp_layers.reverse()
        for layer in tmp_layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        for i, layer_idx in enumerate((0, 3, 5, 8)):
            grads['W' + str(i+1)] = self.layers[layer_idx].dW
            grads['b' + str(i+1)] = self.layers[layer_idx].db

        return grads

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
            self.layers[layer_idx].W = self.params['W' + str(i+1)]
            self.layers[layer_idx].b = self.params['b' + str(i+1)]