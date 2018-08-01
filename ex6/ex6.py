# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 20:18:08 2018

@author: weesh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import scipy.optimize as opt
from scipy.io import loadmat

data = loadmat('ex6data1.mat')
df = pd.DataFrame(data['X'], columns=['X1', 'X2'])
df['y'] = data['y']

pos = df[df['y']==1]
neg = df[df['y']==0]

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(pos['X1'], pos['X2'], 'mx')
ax.plot(neg['X1'], neg['X2'], 'co')

from sklearn import svm

X = data['X']
y = data['y']

svc = svm.LinearSVC(C=1, loss='hinge', max_iter=250)
svc.fit(df[['X1', 'X2']])