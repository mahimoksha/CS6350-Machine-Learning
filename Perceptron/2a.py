#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math


# In[62]:


train_data = pd.read_csv("bank-note/train.csv",header=None)


# In[63]:


test_data = pd.read_csv("bank-note/test.csv",header=None)


# In[64]:


X = train_data.iloc[:,:-1]
Y = train_data.iloc[:,-1]
X_train = np.column_stack(([1]*X.shape[0], X))
x_test = test_data.iloc[:,:-1]
x_test = np.column_stack(([1]*x_test.shape[0], x_test))
y_test = test_data.iloc[:,-1]


# In[93]:


m,n = X_train.shape
weights = np.zeros(n)
r=0.001
epochs = 10
X = np.array(X_train)
Y = np.array(Y)


# In[94]:


def prediction(x,w):
    if np.dot(w.T,x)>0:        
        return 1
    else:
        return 0


# In[101]:


#lr = [10**x for x in range(-5,5)]
#for r in lr:
for e in range(epochs):
    for x,y in zip(X,Y):
        pred = prediction(x,weights)
        if pred==0 and y==1:
            weights = weights + (r * x)
        if pred == 1 and y==0:
            weights = weights - (r * x)
    #     print(weights)
# pred_test = []
error = 0
for x,y in zip(x_test,y_test):
    pred_test = prediction(x,weights)
    if pred_test!=y:
        error +=1
print("Learned weight vector is ",weights)
print("The missclassified points are {} and the average test prediction error {}".format(error,error/len(x_test)))


