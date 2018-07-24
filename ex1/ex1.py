# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def computeCost(X, y, theta):
    m = len(X);
    J = np.sum(np.power(X @ theta.T - y, 2)) / 2 / m;
    return J;

def gradientDescent(iterations, X, y, theta, alpha):
    m = len(X);
    cost = np.zeros(iterations)
    for i in range(iterations):
        theta = theta - alpha/m * (X @ theta.T - y).T @ X; 
        cost[i] = computeCost(X, y, theta)
    return theta, cost;
        
''' Load and Plot Data ''' 
data = pd.read_csv("ex1data1.txt", names=['Population', 'Profit']);
data.insert(0, 'Ones', 1);
#print(data.head())
#print(data.describe())
#==============================================================================
# data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
#==============================================================================



''' Gradient Descent '''
alpha = .01;
iterations = 1500;
m = data.shape[1]-1; # 97
X = (data.iloc[:, 0:m]).values; # 97 x 2
y = (data.iloc[:, m:]).values; # 97 x 1

theta = np.zeros([1,m]); # 1 x 2
theta, cost = gradientDescent(iterations, X, y, theta, alpha)
#==============================================================================
# #print(theta)
#==============================================================================


''' Plot linear fit '''
x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = theta[0,0] + (theta[0,1] * x)
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(data.Population, data.Profit, label = 'Training Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
#==============================================================================
# ax.plot(x, f, 'r', label='Prediction')
#==============================================================================

''' Plot cost v time'''
fig, ax = plt.subplots(figsize=(12,8))
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
#==============================================================================
# ax.plot(np.arange(iterations), cost, 'r')
#==============================================================================



