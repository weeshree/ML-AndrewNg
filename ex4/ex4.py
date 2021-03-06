# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 13:14:53 2018

@author: weesh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize

#bpcount = 0;

def wrap(Theta1, Theta2):
''' Linearizes Theta1, Theta2 into single vector '''
    g1 = np.array(Theta1.flatten(), ndmin=2)
    g2 = np.array(Theta2.flatten(), ndmin=2)
    ar = np.c_[g1, g2].flatten()
    return ar;

def unwrap(arr, inputS, hiddenS, K):
''' Reshapes 1-d vector into Theta1 and Theta2 '''
    Theta1 = np.reshape(arr[:hiddenS * (inputS + 1)], (hiddenS, inputS + 1))
    Theta2 = np.reshape(arr[hiddenS * (inputS + 1):], (K, hiddenS + 1))    
    return Theta1, Theta2

def sigmoid(z):
''' Sigmoid function '''
    return 1/(1+np.exp(-z))

def costFunction(params, inputS, hiddenS, K, X, y, lam):
''' First computes hX using forward prop, then computes cost of logistic regression + regularization '''
    Theta1, Theta2 = unwrap(params, inputS, hiddenS, K)
    a1, a2, hX, z1, z2 = forwardProp(Theta1, Theta2, X)
    m = len(X)
    J = np.sum(-y*np.log(hX) - (1-y)*np.log(1-hX))/m
#    ar = Theta1[0:,1:]
#    print(ar.shape)
    J += lam/2/m*np.sum(np.power(Theta1[:,1:], 2)) + lam/2/m*np.sum(np.power(Theta2[:,1:], 2))
    return J

def forwardProp(Theta1, Theta2, X):
''' Given Theta vectors and X, computes activations and Z-values of each layer '''
    A1 = X
    A1 = np.c_[np.ones(A1.shape[0]), A1] # 5000 401
    Z2 = A1 @ Theta1.T # 5000 x 25
    A2 = sigmoid(Z2) 
    A2 = np.c_[np.ones(A2.shape[0]), A2] # 5000 x 26
    Z3 = A2 @ Theta2.T # 5000 x 10
    A3 = sigmoid(Z3) # 5000 x 10
#    print(A3)
    return A1, A2, A3, Z2, Z3

def sigGrad(z):
''' Computes derivative of sigmoid function w/ respect to z '''
    return sigmoid(z)*(1-sigmoid(z))
    
def randInit(pre, post):
''' Randomly initializes theta '''
    eps = np.sqrt(6) / (np.sqrt(pre+post));
    theta = np.random.rand(pre, post) * 2 * eps - eps;
    return theta;

def backProp(params, inputS, hiddenS, K, X, y, lam):   
''' backProp computes gradient vectors, d(costFunction)/d(Theta1/2) '''
#    global bpcount;
#    print()
#    print(bpcount)
#    bpcount = bpcount + 1

    Theta1, Theta2 = unwrap(params, inputS, hiddenS, K)
    m = len(X);
    D1 = np.zeros(Theta1.shape)
    D2 = np.zeros(Theta2.shape)
    J = 0
    
    a1, a2, a3, z2, z3 = forwardProp(Theta1, Theta2, X)

    for i in range(m):

        a1i = a1[i:i+1,:] # 1 x 401
        a2i = a2[i:i+1,:] # 1 x 26
        a3i = a3[i:i+1,:] # 1 x 10
        z2i = z2[i:i+1,:] # 1 x 25
        yi = y[i:i+1, :] # 1 x 10

        d3 = a3i - yi; # 1 x 10
        d2 = (d3 @ Theta2)[:,1:] * sigGrad(z2i) # 1 x 25
        D1 += d2.T @ a1i # 25 x 401 
        D2 += d3.T @ a2i # 10 x 26
        J += -np.sum(yi*np.log(a3i) + (1-yi)*np.log(1-a3i))

    J += lam/2 * (np.sum(np.power(Theta1[:,1:], 2)) + np.sum(np.power(Theta2[:,1:], 2)))  

    D1 += Theta1; D2 += Theta2; 
    D1[:,0:1] -= Theta1[:, 0:1]; D2[:, 0:1] -= Theta2[:, 0:1];

    D1 /= m; D2 /= m; J /= m;

#    print(J)
    return J, wrap(D1, D2).flatten();
    
def gradCheck(X, y, Theta1, Theta2, lam):
''' Checks backProp's computation of gradient vectors in a slow approximation '''
    t1 = Theta1.flatten(); t2 = Theta2.flatten();
    eps = 0.001;
    D1 = np.zeros(t1.shape); D2 = np.zeros(t2.shape);
    for i in range(len(t1)):
        print(i)
        t1[i] += eps;
        JUp = costFunction( wrap(np.reshape(t1, Theta1.shape) , Theta2), X.shape[1], Theta1.shape[0], Theta2.shape[0], X, y, lam)
        t1[i] -= 2*eps;
        JDown = costFunction( wrap(np.reshape(t1, Theta1.shape) , Theta2), X.shape[1], Theta1.shape[0], Theta2.shape[0], X, y, lam)
        D1[i] = (JUp-JDown)/2/eps;
    for i in range(len(t2)):
        print("X",i)
        t2[i] += eps;
        JUp = costFunction( wrap(Theta1, np.reshape(t2, Theta2.shape)) , X.shape[1], Theta1.shape[0], Theta2.shape[0], X, y, lam)
        t2[i] -= 2*eps;
        JDown = costFunction( wrap(Theta1, np.reshape(t2, Theta2.shape)) , X.shape[1], Theta1.shape[0], Theta2.shape[0], X, y, lam)
        D2[i] = (JUp-JDown)/2/eps;
    D1 = np.reshape(D1, Theta1.shape); D2 = np.reshape(D2, Theta2.shape);
    return D1, D2;
    
''' Input '''
data = loadmat('ex4data1.mat')
X = np.array(data['X']) # 5000 x 400
y = np.array(data['y']) 
y = np.reshape(np.eye(10)[y-1], [5000, 10]) # 5000 x 10

data = loadmat('ex4weights.mat')
Theta1 = np.array(data['Theta1'])
Theta2 = np.array(data['Theta2'])



''' Some initialization '''
inputS = 400; hiddenS= 25; K = 10; 

Theta1 = randInit(hiddenS, inputS + 1) # 25 x 401
Theta2 = randInit(K, hiddenS + 1) # 10 x 26

lam = 1.5
#a1, a2, hX, z2, z3 = forwardProp(Theta1, Theta2, X)


''' Run backProp to find gradient vectors and use scipy.minimize to find Theta vectors '''
cos, wT = backProp( wrap(Theta1, Theta2), inputS, hiddenS, K, X, y, lam)
d1, d2 = unwrap(wT, inputS, hiddenS, K);
#dt1, dt2 = gradCheck(X, y, Theta1, Theta2, lam)
#print(d1,dt1)
#print(d2, dt2)

fmin = minimize(fun=backProp, x0=wrap(Theta1, Theta2), args=(inputS, hiddenS, K, X, y, lam),
                method='TNC', jac=True, options={'maxiter': 250, 'disp':True})

Theta1, Theta2 = unwrap(fmin.x, inputS, hiddenS, K);
#print(costFunction(wrap(Theta1, Theta2), inputS, hiddenS, K, X, y, lam))


''' Run forwardProp to find optimal prediction, hX, and compute its accuracy '''
a1, a2, hX, z2, z3 = forwardProp(Theta1, Theta2, X)
#print(hX)

def getMaxAr(ar):
    return np.array(np.argmax(ar, axis=1).T + 1, ndmin=2);

#print(np.array(np.argmax(hX, axis=1).T + 1, ndmin=2))
#print(y.T)
#acc = np.mean(np.array(np.argmax(hX, axis=1).T + 1, ndmin=2) == y.T)
print(getMaxAr(hX))
print(getMaxAr(y))
acc = np.mean(getMaxAr(hX) == getMaxAr(y))
print(acc)








