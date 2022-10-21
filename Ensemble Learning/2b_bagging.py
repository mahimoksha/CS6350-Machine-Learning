#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import pandas as pd
import random
import matplotlib.pyplot as plt


# In[2]:


train_data = pd.read_csv("bank-2/train.csv",header=None)
print("Running part 2(b)...")
train_data_backup = train_data.copy()


# In[3]:


columns = ["age","job","marital","education","default","balance","housing","loan","contact","day","month","duration","campaign","pdays","previous","poutcome","y"]
train_data.columns = columns
train_data_backup.columns = columns


# In[4]:


def num_to_bin_age(x):
    if x>np.median(train_data['age']):
        return "yes"
    else:
        return "no"
def num_to_bin_balance(x):
    if x>np.median(train_data['balance']):
        return "yes"
    else:
        return "no"
def num_to_bin_day(x):
    if x>np.median(train_data['day']):
        return "yes"
    else:
        return "no"
def num_to_bin_duration(x):
    if x>np.median(train_data['duration']):
        return "yes"
    else:
        return "no"
def num_to_bin_campaign(x):
    if x>np.median(train_data['campaign']):
        return "yes"
    else:
        return "no"
def num_to_bin_pdays(x):
    if x>np.median(train_data['pdays']):
        return "yes"
    else:
        return "no"
def num_to_bin_previous(x):
    if x>np.median(train_data['previous']):
        return "yes"
    else:
        return "no"


# In[5]:


train_data['age'] = train_data['age'].map(num_to_bin_age)
train_data['balance'] = train_data['balance'].map(num_to_bin_balance)
train_data['day'] = train_data['day'].map(num_to_bin_day)
train_data['duration'] = train_data['duration'].map(num_to_bin_duration)
train_data['campaign'] = train_data['campaign'].map(num_to_bin_campaign)
train_data['pdays'] = train_data['pdays'].map(num_to_bin_pdays)
train_data['previous'] = train_data['previous'].map(num_to_bin_previous)

X_train = train_data.iloc[:,:-1]
Y_train = train_data.iloc[:,-1]


# In[6]:


def num_to_bin_age(x):
    if x>np.median(train_data_backup['age']):
        return "yes"
    else:
        return "no"
def num_to_bin_balance(x):
    if x>np.median(train_data_backup['balance']):
        return "yes"
    else:
        return "no"
def num_to_bin_day(x):
    if x>np.median(train_data_backup['day']):
        return "yes"
    else:
        return "no"
def num_to_bin_duration(x):
    if x>np.median(train_data_backup['duration']):
        return "yes"
    else:
        return "no"
def num_to_bin_campaign(x):
    if x>np.median(train_data_backup['campaign']):
        return "yes"
    else:
        return "no"
def num_to_bin_pdays(x):
    if x>np.median(train_data_backup['pdays']):
        return "yes"
    else:
        return "no"
def num_to_bin_previous(x):
    if x>np.median(train_data_backup['previous']):
        return "yes"
    else:
        return "no"


# In[7]:


test_data = pd.read_csv("bank-2/test.csv",header=None)
test_data.columns = columns
test_data['age'] = test_data['age'].map(num_to_bin_age)
test_data['balance'] = test_data['balance'].map(num_to_bin_balance)
test_data['day'] = test_data['day'].map(num_to_bin_day)
test_data['duration'] = test_data['duration'].map(num_to_bin_duration)
test_data['campaign'] = test_data['campaign'].map(num_to_bin_campaign)
test_data['pdays'] = test_data['pdays'].map(num_to_bin_pdays)
test_data['previous'] = test_data['previous'].map(num_to_bin_previous)
X_test = test_data.iloc[:,:-1]
Y_test = test_data.iloc[:,-1]


# In[8]:


class ID3:
    def __init__(self):
        self.data = None
        self.features = None
        self.labels = None
        # self.depth = 0
    def total_entropy(self,data,labels):
        label_data = data['y'].value_counts()
        e = 0
        for i in label_data:
            e -= (i/sum(label_data)) * math.log2(i/sum(label_data))
        return e
    def fea_cat_entropy(self,data, feature,labels):
        categories_features = data[feature].value_counts().keys()
        e = 0
        for cat in categories_features:
            label_fea_data = data[data[feature]==cat]['y'].value_counts()
            pd.DataFrame(data)[feature].value_counts()
            s = 0
            for i in label_fea_data:
                s -= (i/sum(label_fea_data)) * math.log2(i/sum(label_fea_data))
            e += (sum(label_fea_data)/len(data)) * (s)
        return e
    def total_gini(self,data,labels):
        label_data = data['y'].value_counts()
        e = 1
        for i in label_data:
            e -= (i/sum(label_data))**2
        return e
    def fea_cat_gini(self,data, feature,labels):
        categories_features = data[feature].value_counts().keys()
        e = 0
        for cat in categories_features:
            label_fea_data = data[data[feature]==cat]['y'].value_counts()
            pd.DataFrame(data)[feature].value_counts()
            s = 1
            for i in label_fea_data:
                s -= (i/sum(label_fea_data))**2
            e += (sum(label_fea_data)/len(data)) * (s)
        return e
    
    def total_me(self,data,labels):
        label_data = data['y'].value_counts()
        e = (min(label_data)/sum(label_data))
        return e
    def fea_cat_me(self,data, feature,labels):
        categories_features = data[feature].value_counts().keys()
        e = 0
#         import pdb;pdb.set_trace()
        for cat in categories_features:
            label_fea_data = data[data[feature]==cat]['y'].value_counts()
            pd.DataFrame(data)[feature].value_counts()
            if len(label_fea_data)!=4:
                   e+=0 
            else:
                e += (min(label_fea_data)/len(data))
        return e
    def IG(self,data,features,labels,split_method):
        if split_method=="entropy":
          return self.total_entropy(data,labels) - self.fea_cat_entropy(data,features,labels)
        elif split_method=="gini":
          return self.total_gini(data,labels) - self.fea_cat_gini(data,features,labels)
        elif split_method=="MajorityError":
          return self.total_me(data,labels) - self.fea_cat_me(data,features,labels)
    def create_root_node(self,data,features,split_method,labels):
        total_fea_ig = dict()
        for i in features:
            total_fea_ig[i] = self.IG(data,i,labels,split_method)
        best_feature = max(total_fea_ig, key=total_fea_ig.get)
#         root_node_dict[best_feature] = 0
        return best_feature
    def find_bestsplits(self,data,data_copy,root_node):
        features_groups = dict()
        temp_x = []
        # import pdb;pdb.set_trace()
#         if depth<max_depth:
        for x in data[root_node].value_counts().keys():
          for i in range(len(data)):
              if data.iloc[i][root_node] == x:
                  if features_groups.get(x,0)!=0:
                      features_groups[x].append((dict(data.iloc[i][:-1]),data.iloc[i]['y']))
                  else:   
                      features_groups[x] = [(dict(data.iloc[i][:-1]),data.iloc[i]['y'])]    
          temp_x.append(x)
        c_ = list(data_copy[root_node].value_counts().keys())
        for i in c_:
          if i not in temp_x:
            label_counts = dict(data['y'].value_counts())
            features_groups[i] =  [({}, max(label_counts,key=label_counts.get))]
        # import pdb;pdb.set_trace()
        return features_groups
    def ID3_Algo(self,data,data_copy,features,labels,split_method,backup_features=[]):
        if len(data['y'].value_counts()) == 1:
            # answer['leafnode'] = data['label'].value_counts().keys()
            return data['y'].value_counts().keys()
        if len(features)==0: 
            label_counts = dict(data['y'].value_counts())
            return max(label_counts,key=label_counts.get)
        root_node = self.create_root_node(data,features,split_method,labels)
#         answer[root_node] = []
        categories = data[root_node].value_counts().keys()
        if(root_node in features):
            features.remove(root_node)
            backup_features.append(root_node)
        feature_groups = self.find_bestsplits(data, data_copy,root_node)
        subtree_dict = {}
        final_tree = tuple()
        # import pdb;pdb.set_trace()
        for i ,j in feature_groups.items():
            data = pd.DataFrame.from_dict({k: dict(v) for k,v in pd.DataFrame(j)[0].items()}, orient='index')
            data['y'] = pd.DataFrame(j)[1]
            subtree_dict[i] = self.ID3_Algo(data, data_copy, features,labels,split_method,backup_features)
            final_tree = (root_node,subtree_dict)
        # subtree_dict = { i : self.ID3_Algo(j, features,labels,backup_features) for i,j in feature_groups.items()} 
        features.append(backup_features[-1])
        backup_features.remove(backup_features[-1])
        return final_tree


# In[9]:


def classify(tree, query):
      labels = ['yes','no']
      if tree in labels:
          return tree
      key = query.get(tree[0])
      if key not in tree[1]:
          key = None
      class_ = classify(tree[1][key], query)
      return class_


# In[10]:


def random_split_data(train_data,samplesize):
    sampled_data = []
    indexs = random.choices(range(train_data.shape[0]),k=samplesize)
    #print(indexs)
    sampled_data = train_data.iloc[indexs].reset_index(drop=True)
    return sampled_data


# In[11]:


def prediction_with_all_classfiers(all_classifiers,sample):
    predictions = []
    for tree in all_classifiers:
        if classify(tree,sample)=='no':
            predictions.append('no')
        elif classify(tree,sample)=='yes':
            predictions.append('yes')
    return max(set(predictions), key=predictions.count)


# In[12]:


def bagging(trees):
    all_classifiers = []
    samplesize = 5000
    for t in range(trees):
        print("*"*10+"for trees = " +str(trees)+"*"*10+"caluclating for t = "+str(t+1))
        sample_data = random_split_data(train_data,samplesize)
        new_weights = []
        error_t = 0
        algo = ID3()
        labels = ['yes','no']
        answer = dict()
#         max_depth = 1000000000
        split_method = "entropy"
        sample_data = pd.DataFrame(sample_data)
        #print(sample_data)
        sample_Y_train = sample_data['y']
        all_classifiers.append(algo.ID3_Algo(sample_data,sample_data,columns[:-1],labels,split_method,[]))
        
    final_pred = []
    for i in range(X_train.shape[0]):
        final_pred.append(prediction_with_all_classfiers(all_classifiers,X_train.iloc[i]))
    c_t = np.count_nonzero(np.array(final_pred)==np.array(Y_train))
    #for i in range(len(final_pred)):
        #if final_pred[i]==Y_train[i]:
            #c_t+=1
    final_pred_test = []
    for i in range(X_test.shape[0]):
        final_pred_test.append(prediction_with_all_classfiers(all_classifiers,X_test.iloc[i]))
    c_test = np.count_nonzero(np.array(final_pred_test)==np.array(Y_test))
    #for i in range(len(final_pred_test)):
        #if final_pred[i]==Y_test[i]:
            #c_test+=1
    return (X_train.shape[0]-c_t)/X_train.shape[0],(X_test.shape[0]-c_test)/X_test.shape[0]


# In[13]:


train_error = []
test_error = []
for i in range(1,501):
    train_e,test_e = bagging(i)
    train_error.append(train_e)
    test_error.append(test_e)
    print(train_error,test_error)

plt.plot(range(len(train_error)),train_error,label = "train_error")
plt.plot(range(len(test_error)),test_error,label = "test_error")
plt.legend()
plt.ylabel("train and test errors")
plt.xlabel("number of trees")
plt.title("Train and Test errors vary along with the trees")
plt.show()

# In[ ]:




