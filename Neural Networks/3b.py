#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
import random
import warnings
warnings.filterwarnings("ignore") 


# In[2]:


print("running (3b)")
train_data = pd.read_csv("bank-note/train.csv",header=None)
test_data = pd.read_csv("bank-note/test.csv",header=None)


# In[3]:


X = train_data.iloc[:,:-1]
Y_train = train_data.iloc[:,-1]
X_train = np.column_stack(([1]*X.shape[0], X))
train_data = np.column_stack(([1]*train_data.shape[0], train_data))
train_data = pd.DataFrame(train_data)
X_test = test_data.iloc[:,:-1]
X_test = np.column_stack(([1]*X_test.shape[0], X_test))
Y_test = test_data.iloc[:,-1]


# In[4]:


m,n = X_train.shape


# In[6]:


def gradient_dw(x,y,w):
    gradient = ((np.exp(-y*w.T*x)*(-y*x))/(1+np.exp(-y*w.T*x)))+w
    return gradient


# In[7]:


# lr = 0.001
d = 0.1
epochs = 100
for lr in [0.001,0.005,0.1,0.5,1]:
    print("learning rate",lr)
    weights = np.random.normal(0,1,size=n)
    for epoch in range(epochs):
        train_data = train_data.sample(frac=1).reset_index(drop=True)
        X_train = train_data.iloc[:,:-1]
        Y_train = train_data.iloc[:,-1]
        for j in range(len(X_train)):
            dw = gradient_dw(X_train.iloc[j],Y_train[j],weights)
            weights = weights - lr * np.array(dw)
        lr = lr/(1+((lr/d)*epoch))

    count = 0
    for i in range(len(X_train)):
        pred = np.dot(X_train.iloc[i],weights.T)
        if pred<0.5:
            if Y_train[i]==0:
                   count+=1
        else:
            if Y_train[i]==1:
                count+=1
    print("Train Error {} and accuracy is {}".format((len(X_train)-count)/len(X_train),(count/len(X_train)*100)))
    count = 0
    for i in range(len(X_test)):
        pred = np.dot(X_test[i],weights.T)
        if pred<0.5:
            if Y_test[i]==0:
                   count+=1
        else:
            if Y_test[i]==1:
                count+=1
    print("Test Error {} and accuracy is {}".format((len(X_test)-count)/len(X_test),(count/len(X_test)*100)))
    print("*"*100)


# In[ ]:




