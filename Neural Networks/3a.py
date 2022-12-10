#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import numpy as np
import math
import random
import warnings
warnings.filterwarnings("ignore") 


# In[41]:


print("running (3a)")
train_data = pd.read_csv("bank-note/train.csv",header=None)
test_data = pd.read_csv("bank-note/test.csv",header=None)


# In[42]:


X = train_data.iloc[:,:-1]
Y_train = train_data.iloc[:,-1]
X_train = np.column_stack(([1]*X.shape[0], X))
train_data = np.column_stack(([1]*train_data.shape[0], train_data))
train_data = pd.DataFrame(train_data)
X_test = test_data.iloc[:,:-1]
X_test = np.column_stack(([1]*X_test.shape[0], X_test))
Y_test = test_data.iloc[:,-1]


# In[43]:


m,n = X_train.shape


# In[44]:


def gradient_dw(x,y,w,v):
    gradient = ((np.exp(-y*w.T*x)*(-y*x))/(1+np.exp(-y*w.T*x)))+(w/v)
    return gradient


# In[ ]:


lr = 0.001
d = 0.2
epochs = 100
for v in [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]:
    weights = np.random.normal(0,1,size=n)
    for epoch in range(epochs):
        train_data = train_data.sample(frac=1).reset_index(drop=True)
        X_train = train_data.iloc[:,:-1]
        Y_train = train_data.iloc[:,-1]
        for j in range(len(X_train)):
            dw = gradient_dw(X_train.iloc[j],Y_train[j],weights,v)
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
    print("Train Error {} for setting v {} and accuracy is {}".format((len(X_train)-count)/len(X_train),v,(count/len(X_train)*100)))
    count = 0
    for i in range(len(X_test)):
        pred = np.dot(X_test[i],weights.T)
        if pred<0.5:
            if Y_test[i]==0:
                   count+=1
        else:
            if Y_test[i]==1:
                count+=1
    print("Test Error {} for setting v {} and accuracy is {}".format((len(X_test)-count)/len(X_test),v,(count/len(X_test)*100)))


# In[39]:


X_test.shape


# In[ ]:




