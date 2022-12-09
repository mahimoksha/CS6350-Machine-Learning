#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import math
import random
import warnings
warnings.filterwarnings("ignore") 


# In[7]:


print("running (2b)")
train_data = pd.read_csv("bank-note/train.csv",header=None)
test_data = pd.read_csv("bank-note/test.csv",header=None)


# In[8]:


X = train_data.iloc[:,:-1]
Y_train = train_data.iloc[:,-1]
X_train = np.column_stack(([1]*X.shape[0], X))
train_data = np.column_stack(([1]*train_data.shape[0], train_data))
train_data = pd.DataFrame(train_data)
X_test = test_data.iloc[:,:-1]
X_test = np.column_stack(([1]*X_test.shape[0], X_test))
Y_test = test_data.iloc[:,-1]


# In[9]:


class NeuralNetwork:
    def __init__(self,n1,n2):
        self.n1 = n1
        self.n2 = n2
        self.w_1 = np.random.normal(0,1,size=(X_train.shape[1],self.n1))
        self.w_2 = np.random.normal(0,1,size=(self.n1+1,self.n2))
        self.w_3 = np.random.normal(0,1,size=(self.n2+1,1))
    
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
        self.los = self.loss(y,self.o3)
        
        return self.o3,self.los
    
    def backward(self, output, x, y):
        self.grad_3 = (output-y)*self.o2
        temp = (output-y)*self.w_3[1:].T*self.derivative_sigmoid(self.o2[1:])
        self.grad_2 = (np.repeat(temp,self.n1+1,0).T*self.o1).T
        temp = np.sum(np.dot((output-y)*self.w_3[1:]*self.derivative_sigmoid(self.o2[1:]),self.w_2[1:].T),axis=0).reshape(1,-1)
        self.grad_1 = (np.repeat(temp, len(x), 0)*self.derivative_sigmoid(self.o1[1:])).T*x
        return self.grad_3,self.grad_2,self.grad_1.T
    def fit(self,train_data,X_train,Y_train,X_test,Y_test,lr=0.01,epochs=10):
        
        test_pred = []
        train_pred = []
        train_data = train_data.sample(frac=1).reset_index(drop=True)
        X = train_data.iloc[:,:-1]
        Y = train_data.iloc[:,-1]
        for epoch in range(epochs):
            ans_loss = []
            for i in range(len(X)):
                output,lo = self.forward(X.iloc[i].values,Y[i])
                ans_loss.append(lo[0])
                g_3,g_2,g_1 = self.backward(output,X.iloc[i].values,Y[i])
                self.w_1 = self.w_1 - lr * g_1
                self.w_2 = self.w_2 - lr * g_2
                self.w_3 = (self.w_3.reshape(1,-1) - lr * g_3.reshape(1,-1)).reshape(-1,1)
        lr = lr/(1+((lr/0.1)*epoch))
        import pdb;pdb.set_trace
        for x,y in zip(X_train,Y_train):
            pred,_ = self.forward(x,y)
            train_pred.append(pred)
        for x,y in zip(X_test,Y_test):
            pred,_ = self.forward(x,y)
            test_pred.append(pred)
        return train_pred,test_pred


# In[12]:


neurons = [5, 10, 25, 50, 100]
for n in neurons:
    nn = NeuralNetwork(n,n)
    p_tr,p_te = nn.fit(train_data,X_train,Y_train,X_test,Y_test,0.1,50)
    count = 0
    print("Number of hidden neurons (width) {}".format(n))
    for i in range(len(p_tr)):
        if p_tr[i][0]>0.5:
            if Y_train[i]==1:
                count+=1
        else:
            if Y_train[i]==0:
                count+=1
    print("Train error is {} and accuracy is {}".format((len(p_tr)-count)/len(p_tr), (count/len(p_tr))*100))
    count = 0
    for i in range(len(p_te)):
        if p_te[i][0]>0.5:
            if Y_test[i]==1:
                count+=1
        else:
            if Y_test[i]==0:
                count+=1
    print("Test error is {} and accuracy is {}".format((len(p_te)-count)/len(p_te),(count/len(p_te))*100))
    print("*"*50)


# In[ ]:




