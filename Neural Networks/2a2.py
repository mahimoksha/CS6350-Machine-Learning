#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
import random


# In[2]:


print("running (2a2)")
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


# In[6]:


class NeuralNetwork:
    def __init__(self,n1,n2):
        self.n1 = n1
        self.n2 = n2
        self.w_1 = np.array([[-1,1],[-2,2],[-3,3]])
        self.w_2 = np.array([[-1,1],[-2,2],[-3,3]])
        self.w_3 = np.array([[-1],[2],[-1.5]])
    
    def sigmoid(self,z):
        return 1/(1 + np.exp(-z))
    
    def derivative_sigmoid(self,output):
        return output*(1-output)
    
    def loss(self,y,pred_y):
        return (1/2)*(pred_y-y)**2

    def forward(self,x,y):
        self.o1 = np.append(1,self.sigmoid(np.dot(x,self.w_1)))
        self.o2 = np.append(1,self.sigmoid(np.dot(self.o1,self.w_2)))
        self.o3 = np.dot(self.o2,self.w_3)
        self.loss = self.loss(y,self.o3)
        return self.o3
    
    def backward(self, output, x, y):
        self.grad_3 = (output-y)*self.o2
        temp = (output-y)*self.w_3[1:].T*self.derivative_sigmoid(self.o2[1:])
        self.grad_2 = (np.repeat(temp,self.n1+1,0).T*self.o1).T
        temp = np.sum((output-y)*self.w_3[1:]*self.derivative_sigmoid(self.o2[1:])*self.w_2[1:].T,axis=0).reshape(1,-1)
        self.grad_1 = (np.repeat(temp, len(x), 0)*self.derivative_sigmoid(self.o1[1:])).T*x
        return self.grad_3,self.grad_2,self.grad_1.T


# In[11]:


obj = NeuralNetwork(2,2)
x = np.array([1,1,1])
y = 1
output = obj.forward(x,y)
weight_3,weight_2,weight_1 = obj.backward(output,x,y)
print("Weight matrix for 3 layer is ")
print(weight_3)
print("*"*50)
print("Weight matrix for 2 layer is ")
print(weight_2)
print("*"*50)
print("Weight matrix for 1 layer is ")
print(weight_1)


# In[ ]:




