#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math


# In[2]:


a = np.array([[1,0.5,-1,0.3],[1,-1,-2,-2],[1,1.5,0.2,-2.5]])
b = [1,-1,1]


# In[3]:


w = np.array([0,0,0,0])
subgrad = np.array([0,0,0,0])
c = 1/3


# In[4]:


lr = 0.01
i = 0
# print(b[i]*np.dot(w.T,a[i]))
if b[i]*np.dot(w.T,a[i])<=1:
    new_w = w.copy()
    new_w[0] = 0
    w = w - lr*new_w+lr*c*len(a)*b[i]*a[i]
    subgrad = new_w-c*len(a)*b[i]*a[i]
else:
    w[1:] = (1-lr)*w[1:]
    subgrad = w[1:]
print("First Step")
print("     weights",w)
print("     Subgradients",subgrad)



# In[5]:


lr = 0.005
i = 1
# print(b[i]*np.dot(w.T,a[i]))
if b[i]*np.dot(w.T,a[i])<=1:
    new_w = w.copy()
    new_w[0] = 0
    w = w - lr*new_w+lr*c*len(a)*b[i]*a[i]
    subgrad = new_w-c*len(a)*b[i]*a[i]
else:
    w[1:] = (1-lr)*w[1:]
    subgrad = w[1:]
print("Second Step")
print("     weights",w)
print("     Subgradients",subgrad)



# In[6]:


lr = 0.0025
i = 2
# print(b[i]*np.dot(w.T,a[i]))
if b[i]*np.dot(w.T,a[i])<=1:
    new_w = w.copy()
    new_w[0] = 0
    w = w - lr*new_w+lr*c*len(a)*b[i]*a[i]
    subgrad = new_w-c*len(a)*b[i]*a[i]
else:
    w[1:] = (1-lr)*w[1:]
    subgrad = w[1:]

print("Third Step")
print("     weights",w)
print("     Subgradients",subgrad)



# In[ ]:




