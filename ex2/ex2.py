# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 21:12:05 2018

@author: weesh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

def sigmoid(z):
    return 1/(1+np.exp(-z));

def costFunction(theta, X, y):
    theta = np.array(theta, ndmin=2);
    m = len(X);
    J = np.sum(-y*np.log(sigmoid(X @ theta.T)) - (1-y)*(np.log(1-sigmoid(X @ theta.T))))/m;
    return J;
    
def getGradients(theta, X, y):
    theta = np.array(theta, ndmin=2);
    m = len(X);
    grad = (sigmoid(X @ theta.T) - y).T @ X / m;
    grad = np.array(grad);
    return grad.ravel();
   
data = pd.read_csv("ex2data1.txt", header=None, names = ['Exam 1', 'Exam 2', 'Admitted']);
#print(data.head());
#print(data.describe());

''' Plot Data '''
#==============================================================================
# pos = data[data['Admitted'].isin([1])]
# neg = data[data['Admitted'].isin([0])]
# fig, ax = plt.subplots(figsize=(12,8))
# ax.scatter(pos['Exam 1'], pos['Exam 2'], s=50, c='b', marker='o', label='Admitted')
# ax.scatter(neg['Exam 1'], neg['Exam 2'], s=50, c='r', marker='x', label='Rejected')
# ax.legend()
# ax.set_xlabel('Exam 1')
# ax.set_ylabel('Exam 2')
#==============================================================================

''' Get Data and Check Cost '''
data.insert(0, 'Ones', 1);
m = data.shape[1] - 1; # 2
X = (data.iloc[:, :m]).values; # 100 x 3
y = (data.iloc[:, m:]).values; # 100 x 1
initial_theta = np.zeros(m); # 3 x ?

''' Gradient Descent '''
#gradient = getGradients(initial_theta, X, y)
#cost = costFunction(initial_theta, X, y)
result = opt.fmin_tnc(func = costFunction, x0 = initial_theta, fprime = getGradients, args=(X,y))
opt_theta = result[0];

''' Check Accuracy '''
ot2 = np.array(opt_theta, ndmin=2)
ar = sigmoid(X @ ot2.T) >= 0.5
acc = 1 - np.sum(ar!=y)/y.shape[0]
#==============================================================================
# print(acc)
#==============================================================================

























