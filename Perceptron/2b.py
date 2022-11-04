#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import math


# In[3]:


train_data = pd.read_csv("bank-note/train.csv",header=None)


# In[4]:


test_data = pd.read_csv("bank-note/test.csv",header=None)


# In[5]:


X = train_data.iloc[:,:-1]
Y = train_data.iloc[:,-1]
X_train = np.column_stack(([1]*X.shape[0], X))
x_test = test_data.iloc[:,:-1]
x_test = np.column_stack(([1]*x_test.shape[0], x_test))
y_test = test_data.iloc[:,-1]


# In[6]:


m,n = X_train.shape
weights = np.zeros(n)
r=1
epochs = 10
X = np.array(X_train)
Y = np.array(Y)
cm = 0
m = np.zeros(n)


# In[7]:


def prediction(x,w):
    if np.dot(w.T,x)>0:        
        return 1
    else:
        return 0


def predict(X,all_weights,C):
    preds = []
    for x in X:
        s = 0
        for w,c in zip(all_weights,C):
            s = s + c*np.sign(np.dot(w,x))
        if s>=0:
            preds.append(1)
        else:
            preds.append(0)
        #np.sign(1 if s>= 0 else 0))
    #import pdb;pdb.set_trace()
    return preds


k = 0
all_weights = []
C = []
#weights = [np.zeros_like(X)[0]]
for i in range(len(Y)):
        if Y[i] == 0:
            Y[i] = -1
        k = 0
        weights = [np.zeros_like(X)[0]]
        c = [0]
        for epoch in range(10):
            for i in range(len(X)):
                pred = 1 if np.dot(weights[k], X[i]) > 0 else -1
                if pred == Y[i]:
                    c[k] += 1
                else:
                    weights.append(weights[k] + 0.001*np.dot(Y[i], X[i]))
                    #v.append(np.add(v[k], np.dot(Y[i], X[i])))
                    c.append(1)
                    k += 1
        all_weights = weights
        C = c
        k = k
pred_test = predict(x_test,all_weights,C)
teerror = 0
for true,pred in zip(y_test,pred_test):
    if true!=pred:
        teerror+=1

print("The list of distinct weight vectors are as follows ",all_weights)
print("and its respective counts",C)
print("The number of correctly predicted points are {} and missclassified points are {} and the average test error is {}".format(len(x_test)-teerror,teerror,teerror/len(x_test)))
#lr = [10**x for x in range(-5,5)]
#k = 0
#cm = [0]
#m = 0
#V = []
#V.append(np.zeros(n))
#for r in lr:
    #for e in range(epochs):
        #for x,y in zip(X,Y):
            #pred = prediction(x,weights)
            #if pred==0 and y==1:
                #weights = weights - (r * x)
                #V.append(weights)
                #cm.append(0)
                #m += 1
                #cm[m] = 1
                #k += 1
            #if pred == 1 and y==0:
                #weights = weights - (r * x)
                #V.append(weights)
                #cm.append(0)
                #m +=1
                #cm[m] = 1
                #k += 1
            #else:
                #cm[m] = cm[m]+1
            #print(weights)
    #pred_test = []
    #error = 0
    #pred_test = []
    #for x,y in zip(x_test,y_test):
        #pred_test = prediciton_voted(x,V,cm,y)
        #if pred_test!=y:
            #error+=1
    #print("for learning rate {} the error is {} and the average prediction error {}".format(r,error,error/len(x_test)))





