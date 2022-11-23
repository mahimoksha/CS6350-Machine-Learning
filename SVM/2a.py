#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt


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


# In[14]:


epochs = 100
C = [100/873,500/873,700/873]
for c in C:
        a = 0.1
        learning_rate = 0.001
        print("c={},a={},learning_rate={}".format(c,a,learning_rate))
        weights = np.zeros(train_data.shape[1]-1)
        for epoch in range(epochs):
            train_data = train_data.sample(frac=1).reset_index(drop=True)
            X = train_data.iloc[:,:-1]
            Y = train_data.iloc[:,-1]
            for i in range(train_data.shape[0]):
                x = np.array(X.iloc[i])
                y = Y.iloc[i]
                if y*np.dot(weights.T,x) <= 1:
                    new_w = weights.copy()
                    new_w[0] = 0
                    weights = weights - learning_rate*new_w+ learning_rate*c*train_data.shape[0]*y*x
                else:
                    weights[1:] = (1-learning_rate)*weights[1:]
            learning_rate = learning_rate/(1+((learning_rate/a)*epoch))
        print("weights : ",weights)
        co = 0
        for i in range(X.shape[0]):
            if np.dot(weights.T,X.iloc[i])>0 and Y[i]==1:
                        co+=1
            elif np.dot(weights.T,X.iloc[i])<0 and Y[i]==-1:
                    co+=1
        print("train accuracy ", (co/train_data.shape[0])*100)
        print("train error ", (train_data.shape[0]-co)/train_data.shape[0])
        co = 0
        for i in range(len(X_test)):
            if np.dot(weights.T,X_test[i])>0 and Y_test[i]==1:
                        co+=1
            elif np.dot(weights.T,X_test[i])<0 and Y_test[i]==-1:
                    co+=1
        print("test accuracy ", (co/len(X_test))*100)
        print("test error ", (len(X_test)-co)/len(X_test))
        print("*"*50)


# In[ ]:




