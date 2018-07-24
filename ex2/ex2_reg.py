# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 15:49:25 2018

@author: weesh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

def sigmoid(z):
''' Sigmoid function '''
    return 1/(1+np.exp(-z));

def costFunction(theta, X, y, lam):
''' Computes cost of logistic regression '''
    m = len(X);
#    print(m)
    theta = np.array(theta, ndmin=2);
    J_orig = np.sum(-y*np.log(sigmoid(X @ theta.T)) - (1-y)*np.log(1-sigmoid(X @ theta.T))) / m;
    J = J_orig + (lam/2/m) * np.sum(np.power(theta, 2));
    J -= lam/2/m * np.power(theta[0][0], 2);
    return J;

def getGradients(theta, X, y, lam):
''' Returns gradient vector, d(costFunction)/d(theta) '''
    m = len(X);
    theta = np.array(theta, ndmin=2);
#    print(theta)
#    print(theta.shape, X.shape, y.shape)
    grad_orig = ((sigmoid(X @ theta.T) - y).T @ X)/m;
#    print(grad_orig)
#    print(grad_orig.shape)
    grad = grad_orig + lam/m * theta
#    print("gt",grad_orig[0][1], theta[0][1])
    grad[0][0] = grad_orig[0][0];
    return grad.ravel();
    
def polyAdd(data):
''' Inserts degree 0-7 terms to data '''
    x1 = data.iloc[:, 0].values; x2 = data.iloc[:, 1].values;
    data.insert(3, 'Ones', 1)
    for i in range(7):
        for j in range(7-i):
            data['col'+str(i)+str(j)] = np.power(x1, i) * np.power(x2, j)
    data.drop('Microchip Test 1', axis=1, inplace=True)
    data.drop('Microchip Test 2', axis=1, inplace=True)
    return data

def gradientDescent(X, y, theta, lam, alpha, iterations): 
''' Performs gradient descent to minimize costFunction over theta '''
    cost = np.zeros(iterations)
    for i in range(iterations):
        theta = theta - alpha*getGradients(theta, X, y, lam);
        cost[i] = costFunction(theta, X, y, lam);    
    return theta, cost
  
''' Input '''
data = pd.read_csv("ex2data2.txt", header=None, names=['Microchip Test 1', 'Microchip Test 2', 'Result'])
#print(data.head())

''' Data Plot '''
#==============================================================================
# pos = data[data['Result']==1];
# neg = data[data['Result']==0];
# 
# fig, ax = plt.subplots(figsize=(12, 8))
# ax.scatter(pos['Microchip Test 1'], pos['Microchip Test 2'], s = 400, c = 'm', marker='*');
# ax.scatter(neg['Microchip Test 1'], neg['Microchip Test 2'], s = 100, c = 'c', marker='D');
# ax.set_xlabel("MT1")
# ax.set_ylabel("MT2")
# ax.legend()
#==============================================================================

''' Data Prep '''
data = polyAdd(data) 
print(data.head())
m = data.shape[1]-1; # 28
#print(data.head())
X = (data.iloc[:, 1:]).values # 118 x 28
y = (data.iloc[:, :1]).values # 118 x 1
initial_theta = np.zeros(m); # 28 x ?
initial_theta = np.full(m, .0001);
lam = 1;
c = costFunction(initial_theta, X, y, lam);
g = getGradients(initial_theta, X, y, lam);
#print("Cost and Func w/o Reg (works):",c, g, initial_theta.shape)
#print(g[1])



''' Gradient Descent '''
iterations = 20000; alpha = .01;
fin_theta, cost = gradientDescent(X, y, initial_theta, lam, alpha, iterations);

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(iterations), cost, 'r');

result = opt.fmin_tnc(func=costFunction, x0 = initial_theta, fprime = getGradients, args = (X, y, lam))
fin_theta = result[0];
ot2 = np.array(fin_theta, ndmin=2);
arr = (sigmoid(X@ot2.T) >= 0.5);
#print(y);
acc = np.mean((sigmoid(X @ ot2.T) >= 0.5) == y)
print("Accuracy: ",acc)







