#!/usr/bin/env python
# coding: utf-8

# In[6]:


import tensorflow as tf
import pandas as pd
import numpy as np
import math
import random
import warnings
warnings.filterwarnings("ignore") 


# In[11]:


print("running (2e)")
train_data = pd.read_csv("bank-note/train.csv",header=None)
test_data = pd.read_csv("bank-note/test.csv",header=None)


# In[12]:


X_train = train_data.iloc[:,:-1]
Y_train = train_data.iloc[:,-1]
X_test = test_data.iloc[:,:-1]
Y_test = test_data.iloc[:,-1]


# In[13]:


initializer_xavier = tf.keras.initializers.GlorotNormal()
initializer_he = tf.keras.initializers.HeNormal()


# In[ ]:


depth = [3,5,9]
width = [5, 10, 25, 50, 100]
for w in width:
    model_1 = tf.keras.Sequential([
        tf.keras.Input(shape=(4,)),
        tf.keras.layers.Dense(units=w, activation='tanh',kernel_initializer = initializer_xavier),
        tf.keras.layers.Dense(units=w, activation='tanh'),
        tf.keras.layers.Dense(units=w, activation='tanh'),
        tf.keras.layers.Dense(units=1, activation='linear')
    ])
    model_2 = tf.keras.Sequential([
        tf.keras.Input(shape=(4,)),
        tf.keras.layers.Dense(units=w, activation='relu',kernel_initializer = initializer_he),
        tf.keras.layers.Dense(units=w, activation='relu'),
        tf.keras.layers.Dense(units=w, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='linear')
    ])
    model_1.compile(optimizer='adam', 
                  loss=tf.losses.MeanSquaredError(),
                  metrics=tf.keras.metrics.BinaryAccuracy(
        name="binary_accuracy", dtype=None, threshold=0.5
    ))
    model_2.compile(optimizer='adam', 
                  loss=tf.losses.MeanSquaredError(),
                  metrics=tf.keras.metrics.BinaryAccuracy(
        name="binary_accuracy", dtype=None, threshold=0.5
    ))
    history = model_1.fit(
        X_train,Y_train, 
        epochs=10, 
        validation_data=(X_test,Y_test),verbose=0
    )
    pred = model_1.predict(X_train,verbose=0)
    count=0
    for i in range(len(pred)):
        if pred[i]>0.5:
            if Y_train[i] == 1:
                count+=1
        else:
            if Y_train[i] == 0:
                count+=1
    print("Model 1")
    print("Train error for neural network implementation with Activation {}, weight initialization {}, width {} and depth {} is {}".format("tanh","Xavier Normal",w,3,(len(Y_train)-count)/len(Y_train)))
    pred = model_1.predict(X_test,verbose=0)
    count=0
    for i in range(len(pred)):
        if pred[i]>0.5:
            if Y_test[i] == 1:
                count+=1
        else:
           if Y_test[i] == 0:
                count+=1
    print("Test error for neural network implementation with Activation {}, weight initialization {}, width {} and depth {} is {}".format("tanh","Xavier Normal", w,3,(len(Y_test)-count)/len(Y_test)))

    history = model_2.fit(
        X_train,Y_train, 
        epochs=100, 
        validation_data=(X_test,Y_test),verbose=0
    )
    pred = model_2.predict(X_train,verbose=0)
    count=0
    for i in range(len(pred)):
        if pred[i]>0.5:
            if Y_train[i] == 1:
                count+=1
        else:
            if Y_train[i] == 0:
                count+=1
    print("Model 2")
    print("Train error for neural network implementation with Activation {}, weight initialization {}, width {} and depth {} is {}".format("ReLu","He Normal",w,3,(len(Y_train)-count)/len(Y_train)))
    pred = model_2.predict(X_test,verbose=0)
    count=0
    for i in range(len(pred)):
        if pred[i]>0.5:
            if Y_test[i] == 1:
                count+=1
        else:
           if Y_test[i] == 0:
                count+=1
    print("Test error for neural network implementation with Activation {}, weight initialization {}, width {} and depth {} is {}".format("ReLu","He Normal", w,3,(len(Y_test)-count)/len(Y_test)))
    print("*"*50)


# In[45]:


for w in width:
    model_1 = tf.keras.Sequential([
        tf.keras.Input(shape=(4,)),
        tf.keras.layers.Dense(units=w, activation='tanh',kernel_initializer = initializer_xavier),
        tf.keras.layers.Dense(units=w, activation='tanh'),
        tf.keras.layers.Dense(units=w, activation='tanh'),
        tf.keras.layers.Dense(units=w, activation='tanh'),
        tf.keras.layers.Dense(units=w, activation='tanh'),
        tf.keras.layers.Dense(units=1, activation='linear')
    ])
    model_2 = tf.keras.Sequential([
        tf.keras.Input(shape=(4,)),
        tf.keras.layers.Dense(units=w, activation='relu',kernel_initializer = initializer_he),
        tf.keras.layers.Dense(units=w, activation='relu'),
        tf.keras.layers.Dense(units=w, activation='relu'),
        tf.keras.layers.Dense(units=w, activation='relu'),
        tf.keras.layers.Dense(units=w, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='linear')
    ])
    model_1.compile(optimizer='adam', 
                  loss=tf.losses.MeanSquaredError(),
                  metrics=tf.keras.metrics.BinaryAccuracy(
        name="binary_accuracy", dtype=None, threshold=0.5
    ))
    model_2.compile(optimizer='adam', 
                  loss=tf.losses.MeanSquaredError(),
                  metrics=tf.keras.metrics.BinaryAccuracy(
        name="binary_accuracy", dtype=None, threshold=0.5
    ))
    history = model_1.fit(
        X_train,Y_train, 
        epochs=100, 
        validation_data=(X_test,Y_test),verbose=0
    )
    pred = model_1.predict(X_train,verbose=0)
    count=0
    for i in range(len(pred)):
        if pred[i]>0.5:
            if Y_train[i] == 1:
                count+=1
        else:
            if Y_train[i] == 0:
                count+=1
    print("Model 1")
    print("Train error for neural network implementation with Activation {}, weight initialization {}, width {} and depth {} is {}".format("tanh","Xavier Normal",w,5,(len(Y_train)-count)/len(Y_train)))
    pred = model_1.predict(X_test,verbose=0)
    count=0
    for i in range(len(pred)):
        if pred[i]>0.5:
            if Y_test[i] == 1:
                count+=1
        else:
           if Y_test[i] == 0:
                count+=1
    print("Test error for neural network implementation with Activation {}, weight initialization {}, width {} and depth {} is {}".format("tanh","Xavier Normal", w,5,(len(Y_test)-count)/len(Y_test)))

    history = model_2.fit(
        X_train,Y_train, 
        epochs=100, 
        validation_data=(X_test,Y_test),verbose=0
    )
    pred = model_2.predict(X_train,verbose=0)
    count=0
    for i in range(len(pred)):
        if pred[i]>0.5:
            if Y_train[i] == 1:
                count+=1
        else:
            if Y_train[i] == 0:
                count+=1
    print("Model 2")
    print("Train error for neural network implementation with Activation {}, weight initialization {}, width {} and depth {} is {}".format("ReLu","He Normal",w,5,(len(Y_train)-count)/len(Y_train)))
    pred = model_2.predict(X_test,verbose=0)
    count=0
    for i in range(len(pred)):
        if pred[i]>0.5:
            if Y_test[i] == 1:
                count+=1
        else:
           if Y_test[i] == 0:
                count+=1
    print("Test error for neural network implementation with Activation {}, weight initialization {}, width {} and depth {} is {}".format("ReLu","He Normal", w,5,(len(Y_test)-count)/len(Y_test)))
    print("*"*50)


# In[ ]:


for w in width:
    model_1 = tf.keras.Sequential([
        tf.keras.Input(shape=(4,)),
        tf.keras.layers.Dense(units=w, activation='tanh',kernel_initializer = initializer_xavier),
        tf.keras.layers.Dense(units=w, activation='tanh'),
        tf.keras.layers.Dense(units=w, activation='tanh'),
        tf.keras.layers.Dense(units=w, activation='tanh'),
        tf.keras.layers.Dense(units=w, activation='tanh'),
        tf.keras.layers.Dense(units=w, activation='tanh'),
        tf.keras.layers.Dense(units=w, activation='tanh'),
        tf.keras.layers.Dense(units=w, activation='tanh'),
        tf.keras.layers.Dense(units=w, activation='tanh'),
        tf.keras.layers.Dense(units=1, activation='linear')
    ])
    model_2 = tf.keras.Sequential([
        tf.keras.Input(shape=(4,)),
        tf.keras.layers.Dense(units=w, activation='relu',kernel_initializer = initializer_he),
        tf.keras.layers.Dense(units=w, activation='relu'),
        tf.keras.layers.Dense(units=w, activation='relu'),
        tf.keras.layers.Dense(units=w, activation='relu'),
        tf.keras.layers.Dense(units=w, activation='relu'),
        tf.keras.layers.Dense(units=w, activation='relu'),
        tf.keras.layers.Dense(units=w, activation='relu'),
        tf.keras.layers.Dense(units=w, activation='relu'),
        tf.keras.layers.Dense(units=w, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='linear')
    ])
    model_1.compile(optimizer='adam', 
                  loss=tf.losses.MeanSquaredError(),
                  metrics=tf.keras.metrics.BinaryAccuracy(
        name="binary_accuracy", dtype=None, threshold=0.5
    ))
    model_2.compile(optimizer='adam', 
                  loss=tf.losses.MeanSquaredError(),
                  metrics=tf.keras.metrics.BinaryAccuracy(
        name="binary_accuracy", dtype=None, threshold=0.5
    ))
    history = model_1.fit(
        X_train,Y_train, 
        epochs=100, 
        validation_data=(X_test,Y_test),verbose=0
    )
    pred = model_1.predict(X_train,verbose=0)
    count=0
    for i in range(len(pred)):
        if pred[i]>0.5:
            if Y_train[i] == 1:
                count+=1
        else:
            if Y_train[i] == 0:
                count+=1
    print("Model 1")
    print("Train error for neural network implementation with Activation {}, weight initialization {}, width {} and depth {} is {}".format("tanh","Xavier Normal",w,9,(len(Y_train)-count)/len(Y_train)))
    pred = model_1.predict(X_test,verbose=0)
    count=0
    for i in range(len(pred)):
        if pred[i]>0.5:
            if Y_test[i] == 1:
                count+=1
        else:
           if Y_test[i] == 0:
                count+=1
    print("Test error for neural network implementation with Activation {}, weight initialization {}, width {} and depth {} is {}".format("tanh","Xavier Normal", w,9,(len(Y_test)-count)/len(Y_test)))

    history = model_2.fit(
        X_train,Y_train, 
        epochs=100, 
        validation_data=(X_test,Y_test),verbose=0
    )
    pred = model_2.predict(X_train,verbose=0)
    count=0
    for i in range(len(pred)):
        if pred[i]>0.5:
            if Y_train[i] == 1:
                count+=1
        else:
            if Y_train[i] == 0:
                count+=1
    print("Model 2")
    print("Train error for neural network implementation with Activation {}, weight initialization {}, width {} and depth {} is {}".format("ReLu","He Normal",w,9,(len(Y_train)-count)/len(Y_train)))
    pred = model_2.predict(X_test,verbose=0)
    count=0
    for i in range(len(pred)):
        if pred[i]>0.5:
            if Y_test[i] == 1:
                count+=1
        else:
           if Y_test[i] == 0:
                count+=1
    print("Test error for neural network implementation with Activation {}, weight initialization {}, width {} and depth {} is {}".format("ReLu","He Normal", w,9,(len(Y_test)-count)/len(Y_test)))
    print("*"*50)

