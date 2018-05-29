# -*- coding: utf-8 -*-
"""
Created on Tue May 29 11:53:00 2018

@author: YOSHIOKA
"""

import numpy as np
from keras.layers import Input, Flatten
from keras.models import Model
inputs = Input(shape=(4,4,2))

# Define a model consisting only of the Flatten operation
prediction = Flatten()(inputs)
model = Model(inputs=inputs, outputs=prediction)

X = np.arange(0,32)#.reshape(1,3,2,4)
print(X)

X = X.reshape(1, 4, 4, 2)
print(X)

model.predict(X)