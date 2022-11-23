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

# In[2]:


train_data = pd.read_csv("bank-note/train.csv",header=None)
test_data = pd.read_csv("bank-note/test.csv",header=None)


# In[3]:


def labels_convert(value):
    if value==0:
        return -1
    else:
        return 1


# In[4]:


train_data.iloc[:,-1] = train_data.iloc[:,-1].map(labels_convert)
test_data.iloc[:,-1] = test_data.iloc[:,-1].map(labels_convert)


# In[5]:


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


# In[6]:


def gaussian_kernel(x, y, gamma):
    return np.exp(-np.sum(np.square(x - y)) / gamma)

def prediction(kernel,x1,x2,y,count,ga):
    pred = np.sum(count*y*kernel(x1,x2,ga))
    if pred>0:
        return 1
    else:
        return -1


# In[11]:


lr = [10**x for x in range(-5,5)]
c = [0]*len(X_train)
gamma = [0.1, 0.5, 1, 5, 100]
for ga in gamma:
    print("for setting gamma value as {}".format(ga))
    for _ in range(100):
        i = 0
        for x,y in zip(X_train,Y_train):
            pred = prediction(gaussian_kernel,X_train,x,y,np.array(c),ga)
            if pred!=y:
                    c[i] +=1
            i+=1
    error = 0
    for x,y in zip(X_train,Y_test):
        pred_test = prediction(gaussian_kernel,X_train,x,y,np.array(c),ga)
        if pred_test!=y:
            error +=1
    print("The train error is {} and the test accuracy is {}".format(error/len(X_train),(len(X_train)-error)/len(X_train)))
    error = 0
    for x,y in zip(X_test,Y_test):
        pred_test = prediction(gaussian_kernel,X_train,x,y,np.array(c),ga)
        if pred_test!=y:
            error +=1
    print("The test error is {} and the test accuracy is {}".format(error/len(X_test),(len(X_test)-error)/len(X_test)))


# In[ ]:




