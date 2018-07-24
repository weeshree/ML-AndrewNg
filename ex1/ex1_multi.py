# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 20:18:23 2018

@author: weesh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def featureNormalize(X):
    mu = np.mean(X[:,1:], axis=0);
    sd = np.std(X[:,1:], axis=0);
    X[:,1:] = X[:,1:] - mu;
    X[:,1:] = X[:,1:] / sd;
    return X;

def costFunction(X, theta, y):
    m = len(X)
    J = np.sum(np.power((X @ theta.T - y), 2)) / 2 / m;
    return J;

def gradientDescent(X, theta, y, alpha, iterations):
    m = len(X)
    cost = np.zeros(iterations)
    for i in range(iterations):
        theta = theta - alpha/m * (X @ theta.T - y).T @ X
        cost[i] = costFunction(X, theta, y)
    return theta, cost
    
data = pd.read_csv("ex1data2.txt", header=None, names=['Size', 'Bedroom', 'Price'])
#print(data.head())
data.insert(0, 'Ones', 1)

''' Gradient Descent '''
iterations = 1500
alpha = 0.01
m = data.shape[1]-1;
X = (data.iloc[:,0:m]).values; # 47 x 3
y = (data.iloc[:,m:]).values; # 47 x 1
theta = np.zeros([1,m]); # 1 x 3
X = featureNormalize(X);
#print(np.shape(X), np.shape(y), np.shape(theta))
theta, cost = gradientDescent(X, theta, y, alpha, iterations) 

''' Plot cost v time '''
fig, ax = plt.subplots(figsize=(12,8))
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
ax.plot(np.arange(iterations), cost, 'r')