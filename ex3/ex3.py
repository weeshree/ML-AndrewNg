# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 21:35:10 2018

@author: weesh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.io import loadmat
from scipy.optimize import minimize

def sigmoid(z):
''' Sigmoid function '''
    return 1/(1+np.exp(-z))
    
def costFunction(theta, X, y, lam):
''' Computes cost of logistic regression '''
    m = len(X);
    theta = np.array(theta, ndmin = 2)    
    J_orig = np.sum(-1*y*np.log(sigmoid(X @ theta.T)) - (1-y)*np.log(1 - sigmoid(X @ theta.T)))/m
    J = J_orig + lam/2/m * np.sum(np.power(theta, 2))
    J -= lam/2/m * np.power(theta[0][0], 2)
    return J

def getGradients(theta, X, y, lam):
''' Returns gradient vector, d(costFunction)/d(theta) '''
    m = len(X);
    theta = np.array(theta, ndmin = 2)
    grad_orig = (sigmoid(X @ theta.T) - y).T @ X / m
    grad = grad_orig + lam / m * theta
    grad[0][0] = grad_orig[0][0]
    return grad.ravel()
    
def gradientDescent(X, y, theta, lam, alpha, iterations):  
''' Performs gradient descent to minimize costFunction over theta (back-up scipy's minimize) '''
    cost = np.zeros(iterations)
    for i in range(iterations):
        theta = theta - alpha*getGradients(theta, X, y, lam);
        cost[i] = costFunction(theta, X, y, lam);    
    return theta, cost
    
    
def oneVsAll(X, y, lam, K):
''' Performs one vs all (logistic regression for each class independently) w/ scipy's minimize '''
#    alpha = 1; iterations = 1500;
    allTheta = np.zeros([K, X.shape[1]])
    for i in range(K):
        yLog = (y==i+1)
#        allTheta[i], cost = gradientDescent(X, yLog, allTheta[i], lam, alpha, iterations)
        fmin = minimize(fun=costFunction, x0=allTheta[i], args=(X, yLog, lam), method='TNC', jac=getGradients)        
        allTheta[i] = fmin.x
#    fig, ax = plt.subplots(figsize=(12, 8))
#    ax.plot(np.arange(iterations), cost, 'r');
    return allTheta
    
    
''' Input '''
data = loadmat('ex3data1.mat')
X = np.array(data['X']); y = np.array(data['y'])
X = np.c_[np.ones([5000, 1]), X]
lam = 1; K = 10;

''' Classify & output '''
allTheta = oneVsAll(X, y, lam, K)
acc = np.mean(np.argmax(allTheta @ X.T, axis=0).T + 1 == y.T)
print(acc)


