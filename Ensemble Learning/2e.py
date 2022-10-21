#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import pandas as pd
import random
import matplotlib.pyplot as plt
import pickle


# In[2]:


train_data = pd.read_csv("bank-2/train.csv",header=None)
print("Running part 2(e)...")
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
        #total_count = data['y'].value_counts()
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
        #import pdb;pdb.set_trace()
        #total_count = data['y'].value_counts()
        #if total_count['yes']>total_count['no']:
        #   subtree_dict[None] = True
        #else:
        #    subtree_dict[None] = False
        #final_tree = (root_node,subtree_dict)
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
      if key:
        class_ = classify(tree[1][key], query)
      else:
        class_ = "no"
      return class_


# In[10]:


def prediction_with_all_classfiers(all_classifiers,sample):
    predictions = []
    for tree in all_classifiers:
        if classify(tree,sample)=='no':
            predictions.append('no')
        elif classify(tree,sample)=='yes':
            predictions.append('yes')
    return max(set(predictions), key=predictions.count)


# In[13]:


def random_split_data(train_data,samplesize,fea_size):
#     sampled_data = []
#     for i in range(samplesize):
#         index = random.randrange(0,train_data.shape[0])
#         sampled_data.append(train_data.iloc[index])
    sa = random.choices(range(train_data.shape[0]),k=samplesize)
    row_samples = train_data.iloc[sa].reset_index(drop=True)
    samples = random.sample(range(len(train_data.columns)-1),k=fea_size)
    samples.append(16)
    sampled_data = row_samples.iloc[:,samples]
    return sampled_data


# In[18]:


all_trees = dict()
single_tree_predictions = dict()
multi_tree_predictions = dict()
for i in range(0,100):
    single_tree_pred_for_one_tree = []
    multi_tree_pred = []
    all_classifiers = []
    c = 0
    max_pred = []
    for t in range(500):
        print("*"*10+"for iteration t = "+str(i)+" trees = "+str(t)+" feature size = "+str(6)+"*"*10)
        sample_data = random_split_data(train_data,1000,6)
        new_weights = []
        error_t = 0
        algo = ID3()
        labels = ['yes','no']
        answer = dict()
        split_method = "entropy"
        sample_data = pd.DataFrame(sample_data)
        sample_data.reset_index(inplace=True,drop=True)
        sample_Y_train = sample_data['y']
        tree_pred = algo.ID3_Algo(sample_data,sample_data,list(sample_data.columns[:-1]),labels,split_method,[])
        all_classifiers.append(tree_pred)
        if c==0:
            c = 1
            for sample in range(X_test.shape[0]):
                #print(tree_pred)
                sa = dict(X_test.iloc[sample])
                pred = classify(tree_pred,sa)
                single_tree_pred_for_one_tree.append(pred[0])
    single_tree_predictions[i] = single_tree_pred_for_one_tree
    if i!=0:
        #import pdb;pdb.set_trace()
        for sample in range(X_test.shape[0]):
            sa = dict(X_test.iloc[sample])
            pred = prediction_with_all_classfiers(all_classifiers,sa)
            multi_tree_pred.append(pred)
    multi_tree_predictions[i] = multi_tree_pred
    #print("single_tree_predictions",single_tree_predictions)
    #print("multi_tree_predictions",multi_tree_predictions)
with open('singletreeclassifiers_'+str(i)+"_"+"fea_size_"+str(6)+'.pickle', 'wb') as handle:
    pickle.dump(single_tree_predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('multi_tree_predictions_'+str(i)+"_"+"fea_size_"+str(6)+'.pickle', 'wb') as handle:
    pickle.dump(multi_tree_predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)

singletrees = []
with (open("singletreeclassifiers_100_fea_size_6.pickle", "rb")) as openfile:
    while True:
        try:
            singletrees.append(pickle.load(openfile))
        except EOFError:
            break

binary_singletrees = dict()
for i in singletrees[0].keys():
    binary_singletrees[i] = [0 if j=='n' or j=='no' else 1 for j in singletrees[0][i]]
singletrees_df=pd.DataFrame.from_dict(binary_singletrees,orient='index').transpose()


# In[66]:


multitrees = []
with (open("multi_tree_predictions_100_fea_size_6.pickle", "rb")) as openfile:
    while True:
        try:
            multitrees.append(pickle.load(openfile))
        except EOFError:
            break
binary_multitrees = dict()
for i in multitrees[0].keys():
    binary_multitrees[i] = [0 if j=='n' or j=='no' else 1 for j in multitrees[0][i]]
multitrees_df=pd.DataFrame.from_dict(binary_multitrees,orient='index').transpose()
multitrees_df = multitrees_df.drop(0,axis=1)

ground_truth = []
for i in range(len(Y_test)):
    if Y_test[i]=="no":
        ground_truth.append(0)
    else:
        ground_truth.append(1)


# In[186]:


#single tree learner bias and variance
average = np.array(singletrees_df.mean(axis=1))
ground_truth = np.array(ground_truth)
bias = np.mean((average-ground_truth)**2)
# bias = sum((average-ground_truth)**2)
variance = (1/(len(ground_truth)-1))*sum((ground_truth-average)**2)
print("Bias and Variance of single tree learner is :",(bias,variance))
print("Expected error for single tree learner is ", bias+variance)


# In[187]:


average_multi = np.array(multitrees_df.mean(axis=1))
bias = np.mean((average_multi-ground_truth)**2)
# bias = sum((average-ground_truth)**2)
variance = (1/(len(ground_truth)-1))*sum((ground_truth-average_multi)**2)
print("Bias and Variance of whole forest is :",(bias,variance))
print("Expected error for whole forest is ", bias+variance)


# In[56]:





