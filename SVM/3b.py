#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
from scipy.optimize import minimize,Bounds
import warnings
warnings.filterwarnings("ignore")

# In[15]:


train_data = pd.read_csv("bank-note/train.csv",header=None)
test_data = pd.read_csv("bank-note/test.csv",header=None)


# In[16]:


def labels_convert(value):
    if value==0:
        return -1
    else:
        return 1


# In[17]:


train_data.iloc[:,-1] = train_data.iloc[:,-1].map(labels_convert)
test_data.iloc[:,-1] = test_data.iloc[:,-1].map(labels_convert)


# In[18]:


X = train_data.iloc[:,:-1]
Y_train = train_data.iloc[:,-1]
X_train = np.column_stack(([1]*X.shape[0], X))
train_data = np.column_stack(([1]*train_data.shape[0], train_data))
train_data = pd.DataFrame(train_data)
X_test = test_data.iloc[:,:-1]
X_test = np.column_stack(([1]*X_test.shape[0], X_test))
Y_test = test_data.iloc[:,-1]
Y_train = Y_train.replace(0,-1)
Y_test = Y_test.replace(0,-1)


# In[19]:


def gaussian_kernel(x, y, gamma):
    return np.exp(-np.sum(np.square(x - y)) / gamma)

def loss(alpha):
    return (0.5 * np.dot(alpha.T, np.dot(gra, alpha))-alpha.sum())
def predict(test, X, Y, k, alpha, b):
    pred = [0]*len(test)
    for i, value in enumerate(test):
        eval_ = []
        for data, a in zip(X, alpha):
            if a > 0:
                eval_.append(k(value, data))
            else:
                eval_.append(0)
        pred[i] = (alpha * Y).dot(np.array(eval_)) + b
    return pred


# In[ ]:


print("Running 3(b)")
n = X_train.shape[0]
# C = 100/873
C_val = [100/873,500/873,700/873]
gamma = [0.1,0.5,1,5,100]
for C in C_val:
    for g in gamma:
        kernel = lambda x, y: gaussian_kernel(x, y, g)
        n = len(X_train)
        K = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = kernel(X_train[i], X_train[j])

        gra = K * np.outer(Y_train,Y_train)
        constraints = ({'type': 'eq', 'fun': lambda x: np.dot(x, Y_train)})

        print("c = {} and gamma = {}".format(C,g))
        alpha = np.random.rand(n)
        bnds = [(0,C)]*X_train.shape[0]
        result = minimize(loss, alpha, constraints=constraints, method='SLSQP', bounds=bnds)
        alpha = result.x 
        alpha[np.isclose(alpha, 0)] = 0
        alpha[np.isclose(alpha, C)] = C 
        indices_for_support = np.where(0 < alpha)[0]
        margin_indices = np.where((0 < alpha) & (alpha < C))[0]
        print('There are about %d support vectors for the given data' % (len(indices_for_support)))
        sum_ = 0
        for n in margin_indices:
            eval_ = []
            for data, a in zip(X_train, alpha):
                if a > 0:
                    eval_.append(kernel(data, X_train[n]))
                else:
                    eval_.append(0)
            sum_ += Y_train[n] - (alpha * Y_train).dot(np.array(eval_))
        bias = sum_ / len(margin_indices)
        support_vectors = np.where(alpha == C)[0]
        pred = predict(X_train[support_vectors], X_train, Y_train, kernel, alpha, bias)
        c = np.sum((pred * Y_train[support_vectors]) < 0)
        print('Train accuracy: ' + str(1 - c / n))
        print("Train Error", c/n)

        pred = predict(X_test, X_train, Y_train, kernel, alpha, bias)
        c = np.sum((pred * Y_test[list(range(500))]) < 0)
        print('Test accuracy: ' + str(1 - c / n))
        print('Test Error ',c/n)
        print("*"*50)


