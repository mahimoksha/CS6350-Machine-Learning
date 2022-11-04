#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math


# In[2]:


train_data = pd.read_csv("bank-note/train.csv",header=None)


# In[3]:


test_data = pd.read_csv("bank-note/test.csv",header=None)


# In[4]:


X = train_data.iloc[:,:-1]
Y = train_data.iloc[:,-1]
X_train = np.column_stack(([1]*X.shape[0], X))
x_test = test_data.iloc[:,:-1]
x_test = np.column_stack(([1]*x_test.shape[0], x_test))
y_test = test_data.iloc[:,-1]


# In[10]:


m,n = X_train.shape
weights = np.zeros(n)
a = np.zeros(n)
r=0.001
epochs = 10
X = np.array(X_train)
Y = np.array(Y)


# In[11]:


def prediction(x,w):
    if np.dot(w.T,x)>0:        
        return 1
    else:
        return 0


# In[12]:


#lr = [10**x for x in range(-5,1)]
#for r in lr:
for e in range(epochs):
    for x,y in zip(X,Y):
        pred = prediction(x,weights)
        if pred==0 and y==1:
            weights = weights + (r * x)
        if pred == 1 and y==0:
            weights = weights - (r * x)
        a = a + weights
#print(weights)
# pred_test = []
#     print(a)
error = 0
for x,y in zip(x_test,y_test):
    pred_test = prediction(x,a/len(X))
    if pred_test!=y:
        error +=1
print("The learned weight vector is ",a)
print("The Average Learned weight vector is ",a/len(X))
print("for learning rate {} the error is {} and the average prediction error {}".format(r,error,error/len(x_test)))
