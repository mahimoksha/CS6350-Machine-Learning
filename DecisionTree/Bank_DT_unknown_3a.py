#!/usr/bin/env python
# coding: utf-8

# In[43]:


import numpy as np
import math
import pandas as pd


# In[44]:


train_data = pd.read_csv("Bank/train.csv",header=None)

# In[46]:

print("Running part 3(a)...")

# In[45]:


train_data_backup = train_data.copy()


# In[48]:


columns = ["age","job","marital","education","default","balance","housing","loan","contact","day","month","duration","campaign","pdays","previous","poutcome","y"]


# In[49]:


train_data.columns = columns
train_data_backup.columns = columns


# In[50]:


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


# In[51]:


train_data['age'] = train_data['age'].map(num_to_bin_age)
train_data['balance'] = train_data['balance'].map(num_to_bin_balance)
train_data['day'] = train_data['day'].map(num_to_bin_day)
train_data['duration'] = train_data['duration'].map(num_to_bin_duration)
train_data['campaign'] = train_data['campaign'].map(num_to_bin_campaign)
train_data['pdays'] = train_data['pdays'].map(num_to_bin_pdays)
train_data['previous'] = train_data['previous'].map(num_to_bin_previous)


# In[52]:


X_train = train_data.iloc[:,:-1]
Y_train = train_data.iloc[:,-1]


# In[53]:


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


# In[54]:


test_data = pd.read_csv("Bank/test.csv",header=None)
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


# In[55]:


class ID3:
    def __init__(self):
        self.data = None
        self.features = None
        self.labels = None
        # self.depth = 0
    def total_entropy(self,data,labels):
        # if not isinstance(data,list):
        label_data = data['y'].value_counts()
        # else:
        #   label_data = pd.DataFrame(data)[1].value_counts()
        e = 0
        for i in label_data:
            e -= (i/sum(label_data)) * math.log2(i/sum(label_data))
        return e
    def fea_cat_entropy(self,data, feature,labels):
        # if not isinstance(data,list):
        categories_features = data[feature].value_counts().keys()
        # else:
        #   categories_features = pd.DataFrame(data)[feature].value_counts().keys()
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
        # if not isinstance(data,list):
        label_data = data['y'].value_counts()
        # else:
        #   label_data = pd.DataFrame(data)[1].value_counts()
        e = 1
        for i in label_data:
            e -= (i/sum(label_data))**2
        return e
    def fea_cat_gini(self,data, feature,labels):
        # if not isinstance(data,list):
        categories_features = data[feature].value_counts().keys()
        # else:
        #   categories_features = pd.DataFrame(data)[feature].value_counts().keys()
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
    def find_bestsplits(self,data,data_copy,max_depth,depth,root_node):
        features_groups = dict()
        temp_x = []
        # import pdb;pdb.set_trace()
        if depth<max_depth:
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
          return features_groups,depth+1
        else:
          c_ = list(data_copy[root_node].value_counts().keys())
          temp_x = list(data[root_node].value_counts().keys())
          for i in c_:
                if i in temp_x:
                  label_counts = dict(data[data[root_node]==i]['y'].value_counts())
                  features_groups[i] =  [({}, max(label_counts,key=label_counts.get))]
                else:
                  label_counts = dict(data['y'].value_counts())
                  features_groups[i] =  [({}, max(label_counts,key=label_counts.get))]
          # import pdb;pdb.set_trace()
          return features_groups,depth
    def ID3_Algo(self,data,data_copy,features,labels,max_depth,depth,split_method,backup_features=[]):
        if len(data['y'].value_counts()) == 1:
            # answer['leafnode'] = data['label'].value_counts().keys()
            return data['y'].value_counts().keys()
        if len(features)==0: 
            label_counts = dict(data['y'].value_counts())
            return max(label_counts,key=label_counts.get)
        root_node = self.create_root_node(data,features,split_method,labels)
#         answer[root_node] = []
        categories = data[root_node].value_counts().keys()
        # for c in categories:
#             answer = dict()
            # answer[root_node] = []
            # answer[root_node].append(c)
            # label_fea_data = data[data[root_node]==c]['label'].value_counts()
        # sv = data[data[root_node]==c]
        # if len(sv)!=0:
        # if max_depth
        if(root_node in features):
            features.remove(root_node)
            backup_features.append(root_node)
        feature_groups,depth = self.find_bestsplits(data, data_copy,max_depth,depth,root_node)
        subtree_dict = {}
        final_tree = tuple()
        # import pdb;pdb.set_trace()
        for i ,j in feature_groups.items():
            data = pd.DataFrame.from_dict({k: dict(v) for k,v in pd.DataFrame(j)[0].items()}, orient='index')
            data['y'] = pd.DataFrame(j)[1]
            subtree_dict[i] = self.ID3_Algo(data, data_copy, features,labels,max_depth,depth,split_method,backup_features)
            # if subtree_dict==4:
            # import pdb;pdb.set_trace()
            # features.append(backup_features[-1])
            # backup_features.remove(backup_features[-1])
            final_tree = (root_node,subtree_dict)
        # subtree_dict = { i : self.ID3_Algo(j, features,labels,backup_features) for i,j in feature_groups.items()} 
        features.append(backup_features[-1])
        backup_features.remove(backup_features[-1])
        return final_tree
            # else:
              #  pass
              #  print(data)
              #  print(root_node,c)
              #  label_counts = dict(data['label'].value_counts())
              #  return max(label_counts,key=label_counts.get)
                # print("printintg sv=0 case:")


# In[56]:


def classify(tree, query):
      if tree in labels:
          return tree
      key = query.get(tree[0])
      if key not in tree[1]:
          key = None
      class_ = classify(tree[1][key], query)
      return class_


# In[57]:


best_split = ["entropy","gini","MajorityError"]
train_error = dict()
test_error = dict()

for md in range(16):
    for j in best_split:
        print("*"*5+" training decision tree with depth "+str(md+1)+" and split method "+j+" "+"*"*5)
        algo = ID3()
        labels = ['yes','no']
        answer = dict()
        max_depth = md+1
        split_method = j
#         print("max-depth",max_depth)
#         print("split-method",split_method)
        s = algo.ID3_Algo(train_data,train_data,columns[:-1],labels,max_depth,1,split_method,[])
        c=0
        for i in range(X_train.shape[0]):
          sample = dict(X_train.iloc[i])
          if classify(s,sample)==Y_train[i]:
            c+=1
        #   else:
            #print(sample,classify(s,sample),Y_train[i])
        print("train missclassified points",(X_train.shape[0]-c))
        print("Training Error",(X_train.shape[0]-c)/X_train.shape[0])
        if train_error.get(max_depth):
            train_error[max_depth].append((j,(X_train.shape[0]-c)/X_train.shape[0]))
        else:
            train_error[max_depth] = [(j,(X_train.shape[0]-c)/X_train.shape[0])]
        c=0
        for i in range(X_test.shape[0]):
          sample = dict(X_test.iloc[i])
          if classify(s,sample)==Y_test[i]:
            c+=1
          # else:
          #   print(sample,classify(s,sample)[0],Y_test[i])
        print("test missclassified points: ",(X_test.shape[0]-c))
        print("Testing Error: ",(X_test.shape[0]-c)/X_test.shape[0])
        if test_error.get(max_depth):
            test_error[max_depth] .append((j,(X_test.shape[0]-c)/X_test.shape[0]))
        else:
            test_error[max_depth] = [(j,(X_test.shape[0]-c)/X_test.shape[0])]
        print("*"*50)


# In[58]:


# train_error


# In[59]:


# test_error


# In[61]:


# train_ans = []
# k = 0
# for i in train_error.keys():
#     train_ans.append(["Depth = "+ str(i)])
#     for j in range(len(train_error[i])):
#         train_ans[k].append(train_error[i][j][1])
#     k+=1
# test_ans = []
# k = 0
# for i in test_error.keys():
#     test_ans.append(["Depth = "+str(i)])
#     for j in range(len(test_error[i])):
#         test_ans[k].append(test_error[i][j][1])
#     k+=1


# In[65]:


# from prettytable import PrettyTable
# x = PrettyTable()
# x.field_names = ["depth \ split_method","Entropy", "Gini Index", "Majority Error"]
# x.add_rows(train_ans)
# print(x)
# x = PrettyTable()
# x.field_names = ["depth \ split_method","Entropy", "Gini Index", "Majority Error"]
# x.add_rows(test_ans)
# print(x)

