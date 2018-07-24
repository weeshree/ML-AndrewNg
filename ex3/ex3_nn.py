# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 01:33:27 2018

@author: weesh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize 
from scipy.io import loadmat

def sigmoid(z):
    return 1/(1+np.exp(-z))

data = loadmat('ex3weights.mat');
Theta1 = np.array(data['Theta1'])
Theta2 = np.array(data['Theta2'])

data = loadmat('ex3data1.mat')
X = np.array(data['X']); y = np.array(data['y'])

A1 = X
A1 = np.c_[np.ones(X.shape[0]), A1]
Z2 = A1 @ Theta1.T
A2 = sigmoid(Z2)
A2 = np.c_[np.ones(X.shape[0]), A2]
Z3 = A2 @ Theta2.T
A3 = sigmoid(Z3)
acc = np.mean(np.array(np.argmax(A3, axis=1).T + 1, ndmin=2) == y.T)
print(acc)