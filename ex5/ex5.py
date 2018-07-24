# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 12:57:08 2018

@author: weesh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize
import scipy.optimize as opt

def costFunction(theta, X, y, lam):
''' computes cost of prediction, X @ theta.T, for linear regression '''
    theta = np.array(theta, ndmin=2)
    m = len(X)
    J = np.sum(np.power(X @ theta.T - y, 2)) / 2 / m;
    J += lam/2/m * np.sum(np.power(theta[0][1:], 2))
    return J

def getGradients(theta, X, y, lam):
''' returns gradient vector, d(costFunction)/d(theta) '''
    theta = np.array(theta, ndmin=2)
    m = len(X)
    grad = (X @ theta.T - y).T @ X / m
    grad += lam/m * theta
    grad[0][0] -= lam/m * theta[0][0]
    return grad.ravel()

def learningCurveSize(theta, X, y, Xval, yval, lam):
''' Plots learning curve of error_train/error_validation vs. training set size '''
    m = len(X)
    errTrain = np.zeros(m); errVal = np.zeros(m); 
    for i in range(m):
        res = opt.fmin_tnc(func=costFunction, fprime=getGradients, x0=theta, args=(X[:i+1,:], y[:i+1,:], lam))
        theta = res[0]
        errTrain[i] = costFunction(theta, X[:i+1,:], y[:i+1, :], 0)
        errVal[i] = costFunction(theta, Xval, yval, 0)
#        print(errTrain[i]," ", errVal[i])
    fig, ax = plt.subplots(figsize = (12, 8))
    ax.plot(range(m), errTrain, '-mx', label='Training')
    ax.plot(range(m), errVal, '-co', label='XVal')
    ax.set_xlabel('Size of training set')
    ax.set_ylabel('Error')
    ax.legend();

def learningCurveLambda(thet, X, y, Xval, yval):
''' Plots learning curve of error_train/error_validation vs. lambda (regularization parameter) '''
    errTrain = np.zeros(10); errVal = np.zeros(10); 
    lam = 0; 
    ctr = 0;
    arr = np.zeros(10);
    while lam < 10.1:
        res = opt.fmin_tnc(func=costFunction, fprime=getGradients, x0=thet, args=(X, y, lam))
        theta = res[0]
        errTrain[ctr] = costFunction(theta, X, y, lam)
        errVal[ctr] = costFunction(theta, Xval, yval, lam)
        arr[ctr] = lam;
#        print(errTrain[i]," ", errVal[i])
#        print(lam,errTrain[ctr], errVal[ctr])
        if(lam==0):
            lam+=.001;
        elif(ctr%2 == 0):
            lam *= 10/3;
        else:
            lam *= 3;
        ctr+=1;
    fig, ax = plt.subplots(figsize = (12, 8))

    ax.plot(arr, errTrain, '-mx', label='Training')
    ax.plot(arr, errVal, '-co', label='XVal')
    ax.set_xlabel('Lambda')
    ax.set_ylabel('Error')
    ax.legend();


def addPoly(X, p):
''' Adds polynomial features up until degree (p-1) to 1d vector X'''
    m = len(X)
    ret = np.zeros([m, p])
    for i in range(p):
        ret[:, i] = np.power(X, i).ravel()
    return ret

def norm(X):
''' Normalizes polynomial features of vector X (not column 0) '''
    p = X.shape[1]
    means = np.zeros([p]);
    stds = np.zeros([p])
    for i in range(p):
        means[i] = np.mean(X[:, i])
        stds[i] = np.std(X[:, i])
        X[:, i] -= np.mean(X[:, i])
        X[:, i] /= np.std(X[:, i])

    return X, means, stds;

def normOthers(X, p, means, stds):
''' Given previous normalization conditions [means, stds], normalizes new vector X '''
    for i in range(1,p):
        X[:, i] -= means[i-1]
        X[:, i] /= stds[i-1]
    return X;
    
def polyPlot(means, stds, p, X, y, Xval, yval, lam, X_n, Xval_n):
''' Plots polynomial curves of X and Xval '''
    fig, ax = plt.subplots(figsize = (12, 8))
    ax.plot(X, y, 'mx', label='Training')
    ax.plot(Xval, yval, 'kD', label='X Validation')
    ax.set_xlabel('reservoir H2O lev')
    ax.set_ylabel('dam H2O level')
     
    xr = np.linspace(np.min(X)-10, np.max(X)+10, 100)
    xrP = addPoly(xr, p)
    xrP = normOthers(xrP, p, means, stds)
    
    res = opt.fmin_tnc(func=costFunction, fprime = getGradients, x0=np.zeros(X_n.shape[1]), args=(X_n, y, lam))
    theta = np.array(res[0], ndmin=2);
    f = xrP @ theta.T;

    ax.plot(xr, f, 'c', label='Prediction')
    ax.legend();
    print("Lambda value: ",lam,"X Validation Error", costFunction(theta, Xval_n, yval, lam))

    

''' Input '''
data = loadmat('ex5data1.mat')
X = data['X']
y = data['y']
Xval = data['Xval']
yval = data['yval']

lam = 3
theta = np.zeros(X.shape[1])

''' Plot of linear hypothesis '''
#==============================================================================
# fig, ax = plt.subplots(figsize = (12, 8))
# ax.plot(X, y, 'mx')
# ax.set_xlabel('reservoir water level')
# ax.set_ylabel('dam water level')
# 
# m = len(X)
# X = np.c_[np.ones([X.shape[0],1]), X]
# theta = np.zeros(X.shape[1])
#
# mval = len(Xval)
# Xval = np.c_[np.ones([Xval.shape[0],1]), Xval]
# 
# 
# res = opt.fmin_tnc(func=costFunction, fprime = getGradients, x0=theta, args=(X, y, lam))
# theta = res[0]
# 
# xr = np.linspace(np.min(X), np.max(X), 100)
# f = theta[0] + theta[1]*xr
# ax.plot(xr, f, 'c', label = 'Prediction')
# 
#==============================================================================


''' Add poly features, normalize X_n and Xval_n '''
p = 8

X_n = addPoly(X, p)
X_n[:, 1:], means, stds = norm(X_n[:, 1:])

Xval_n = addPoly(Xval, p)
Xval_n = normOthers(Xval_n, p, means, stds)



''' Compute theta vector using scipy.optimize and plot learning curves / polynomial curves '''
theta = np.zeros(X_n.shape[1])

res = opt.fmin_tnc(func=costFunction, fprime = getGradients, x0=theta, args=(X_n, y, lam))
theta = res[0]

#learningCurveSize(theta, X_n, y, Xval_n, yval, lam)
#learningCurveLambda(theta, X_n, y, Xval_n, yval)

polyPlot(means, stds, p, X, y, Xval, yval, lam, X_n, Xval_n)

