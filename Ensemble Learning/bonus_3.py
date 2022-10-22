#!/usr/bin/env python
# coding: utf-8

# In[114]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math


# In[115]:


data = pd.read_excel("credit_card/default of credit card clients.xls")
data = data.drop("Unnamed: 0",axis=1)
data.columns = data.iloc[0]
Y = data['default payment next month']
data = data.drop("default payment next month",axis=1)
data = data.drop(0).reset_index(drop=True)
normalized_data=(data-data.mean())/data.std()


# In[116]:


train_indexs = random.sample(range(normalized_data.shape[0]),k=24000)
test_indexs = list(set(range(normalized_data.shape[0])) - set(train_indexs))
# test_indexs = list(test)
train_data = normalized_data.iloc[train_indexs].reset_index(drop=True)
Y_train = Y[train_indexs].reset_index(drop=True)
test_data = normalized_data.iloc[test_indexs].reset_index(drop=True)
Y_test = Y[test_indexs].reset_index(drop=True)
train_data['y'] = Y_train
test_data['y'] = Y_test


# In[117]:


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
# else:
#           c_ = list(data_copy[root_node].value_counts().keys())
#           temp_x = list(data[root_node].value_counts().keys())
#           for i in c_:
#                 if i in temp_x:
#                   label_counts = dict(data[data[root_node]==i]['y'].value_counts())
#                   features_groups[i] =  [({}, max(label_counts,key=label_counts.get))]
#                 else:
#                   label_counts = dict(data['y'].value_counts())
#                   features_groups[i] =  [({}, max(label_counts,key=label_counts.get))]
#           # import pdb;pdb.set_trace()
#           return features_groups,depth
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


# In[122]:


def classify(tree, query):
      labels = ['yes','no']
      if tree in labels:
          return tree
      key = query.get(tree[0])
      if key not in tree[1]:
          key = None
      class_ = classify(tree[1][key], query)
      return class_


# In[123]:


def prediction_with_all_classfiers(all_classifiers,sample):
    predictions = []
    for tree in all_classifiers:
        if classify(tree,sample)=='no':
            predictions.append('no')
        elif classify(tree,sample)=='yes':
            predictions.append('yes')
    return max(set(predictions), key=predictions.count)


# In[134]:


def random_split_data(train_data,samplesize,fea_size):
#     sampled_data = []
#     for i in range(samplesize):
#         index = random.randrange(0,train_data.shape[0])
#         sampled_data.append(train_data.iloc[index])
    sa = random.choices(range(train_data.shape[0]),k=samplesize)
    row_samples = train_data.iloc[sa].reset_index(drop=True)
    samples = random.sample(range(len(train_data.columns)-1),k=fea_size)
    samples.append(23)
    sampled_data = row_samples.iloc[:,samples]
    return sampled_data


# In[135]:


def random_forest(trees,fea_size):
    all_classifiers = []
    for t in range(trees):
            print("*"*10+"trees = "+str(t+1)+" for feature sizes = "+str(fea_size)+"*"*10)
            sample_data = random_split_data(train_data,train_data.shape[0],fea_size)
            new_weights = []
            error_t = 0
            algo = ID3()
            labels = ['yes','no']
            answer = dict()
    #         max_depth = 1000000000
            split_method = "entropy"
            sample_data = pd.DataFrame(sample_data)
            sample_Y_train = sample_data['y']
            all_classifiers.append(algo.ID3_Algo(sample_data,sample_data,list(sample_data.columns[:-1]),labels,split_method,[]))
#     X_train = train_data.iloc[:,:-1]
#     Y_train = train_data.iloc[:,-1]
    final_pred = []
    for i in range(X_train.shape[0]):
        final_pred.append(prediction_with_all_classfiers(all_classifiers,X_train.iloc[i]))
    c_t = 0
    for i in range(len(final_pred)):
        if final_pred[i]==Y_train[i]:
            c_t+=1
    final_pred_test = []
    for i in range(X_test.shape[0]):
        final_pred_test.append(prediction_with_all_classfiers(all_classifiers,X_test.iloc[i]))
    c_test = 0
    for i in range(len(final_pred_test)):
        if final_pred[i]==Y_test[i]:
            c_test+=1
    return (X_train.shape[0]-c_t)/X_train.shape[0],(X_test.shape[0]-c_test)/X_test.shape[0]


# In[136]:


print("Random Forest")
train_error = []
train_error_fea = dict()
test_error_fea = dict()
test_error = []
for fea_size in [6]:
    for i in range(1,501):
        train_e,test_e = random_forest(i,fea_size)
        train_error.append(train_e)
        test_error.append(test_e)
        print(train_error,test_error)
    train_error_fea[fea_size] = train_error
    test_error_fea[fea_size] = test_error
print("train errors: ",train_error_fea)
print("test errors: ",test_error_fea)


# In[ ]:


file = open("train_errors.txt", "w+")
content = str(train_error)
file.write(content)
file.close()
file = open("test_errors.txt", "w+")
content = str(test_error)
file.write(content)
file.close()


# In[118]:


def random_split_data(train_data,samplesize):
    sampled_data = []
    indexs = random.choices(range(train_data.shape[0]),k=samplesize)
    #print(indexs)
    sampled_data = train_data.iloc[indexs].reset_index(drop=True)
    #for i in range(samplesize):
        #index = random.randrange(0,train_data.shape[0])
        #sampled_data.append(train_data.iloc[index])
    return sampled_data


# In[119]:



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
        all_classifiers.append(algo.ID3_Algo(sample_data,sample_data,list(sample_data.columns)[:-1],labels,split_method,[]))
        
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


# In[120]:


print("Bagging")
train_error = []
test_error = []
for i in range(1,501):
    train_e,test_e = bagging(i)
    train_error.append(train_e)
    test_error.append(test_e)
    print(train_error,test_error)


# In[ ]:


class ID3:
    def __init__(self):
        self.data = None
        self.features = None
        self.labels = None
    def total_entropy(self,data,labels,weights):
        label_data = data['y'].value_counts()
        e = 0
        for i in label_data.keys():
            e -= sum(data[data['y']==i]['weights'])/sum(data['weights']) * math.log2(sum(data[data['y']==i]['weights'])/sum(data['weights']))
        return e
    def fea_cat_entropy(self,data, feature,labels,weights):
        categories_features = data[feature].value_counts().keys()
        e = 0
        for cat in categories_features:
            label_fea_data = data[data[feature]==cat]['y'].value_counts()
            pd.DataFrame(data)[feature].value_counts()
            temp_age = data[data[feature]==cat]
            s = 0
            for i in label_fea_data.keys():
                s -= (sum(temp_age[temp_age['y']==i]['weights'])/sum(temp_age['weights'])) * math.log2((sum(temp_age[temp_age['y']==i]['weights'])/sum(temp_age['weights'])))
            e += (sum(label_fea_data)/len(data)) * (s)
        return e
    def total_gini(self,data,labels,weights):
        label_data = data['y'].value_counts()
        e = 1
        for i in label_data:
            e -= (i/sum(label_data))**2
        return e
    def fea_cat_gini(self,data, feature,labels,weights):
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
    
    def total_me(self,data,labels,weights):
        label_data = data['y'].value_counts()
        e = (min(label_data)/sum(label_data))
        return e
    def fea_cat_me(self,data, feature,labels,weights):
        categories_features = data[feature].value_counts().keys()
        e = 0
        for cat in categories_features:
            label_fea_data = data[data[feature]==cat]['y'].value_counts()
            pd.DataFrame(data)[feature].value_counts()
            if len(label_fea_data)!=4:
                   e+=0 
            else:
                e += (min(label_fea_data)/len(data))
        return e
    def IG(self,data,features,labels,split_method,weights):
        if split_method=="entropy":
          return self.total_entropy(data,labels,weights) - self.fea_cat_entropy(data,features,labels,weights)
        elif split_method=="gini":
          return self.total_gini(data,labels,weights) - self.fea_cat_gini(data,features,labels,weights)
        elif split_method=="MajorityError":
          return self.total_me(data,labels,weights) - self.fea_cat_me(data,features,labels,weights)
    def create_root_node(self,data,features,split_method,labels,weights):
        total_fea_ig = dict()
        for i in features:
            total_fea_ig[i] = self.IG(data,i,labels,split_method,weights)
        best_feature = max(total_fea_ig, key=total_fea_ig.get)
        print(total_fea_ig,best_feature)
#         root_node_dict[best_feature] = 0
        return best_feature
    def find_bestsplits(self,data,data_copy,max_depth,depth,root_node):
        features_groups = dict()
        temp_x = []
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
          return features_groups,depth
    def ID3_Algo(self,data,data_copy,features,labels,max_depth,depth,split_method,weights,backup_features=[]):
        if len(data['y'].value_counts()) == 1:
            # answer['leafnode'] = data['label'].value_counts().keys()
            return data['y'].value_counts().keys()
        if len(features)==0: 
            label_counts = dict(data['y'].value_counts())
            return max(label_counts,key=label_counts.get)
        data['weights'] = weights
        root_node = self.create_root_node(data,features,split_method,labels,weights)
        categories = data[root_node].value_counts().keys()
        if(root_node in features):
            features.remove(root_node)
            backup_features.append(root_node)
        feature_groups,depth = self.find_bestsplits(data, data_copy,max_depth,depth,root_node)
        subtree_dict = {}
        final_tree = tuple()
        for i ,j in feature_groups.items():
            data = pd.DataFrame.from_dict({k: dict(v) for k,v in pd.DataFrame(j)[0].items()}, orient='index')
            data['y'] = pd.DataFrame(j)[1]
            subtree_dict[i] = self.ID3_Algo(data, data_copy, features,labels,max_depth,depth,split_method,weights,backup_features)
            final_tree = (root_node,subtree_dict)
        features.append(backup_features[-1])
        backup_features.remove(backup_features[-1])
        return final_tree


# In[ ]:


def classify(tree, query):
      if tree in labels:
          return tree
      key = query.get(tree[0])
      if key not in tree[1]:
          key = None
      class_ = classify(tree[1][key], query)
      return class_


# In[ ]:


all_weights = []
vote_alpha = []
D = [1/train_data.shape[0]] * train_data.shape[0]
all_weights.append(D)
all_weak_classifiers = []
train_errors = []
test_errors = []
all_errors = []
for t in range(500):
    print("*"*10+"Iteration {}".format(t+1)+"*"*10)
    new_weights = []
    error_t = 0
    algo = ID3()
    labels = ['yes','no']
    answer = dict()
    max_depth = 1
    split_method = "entropy"
    all_weak_classifiers.append(algo.ID3_Algo(train_data,train_data,columns[:-1],labels,max_depth,1,split_method,all_weights[t],[]))
    print(all_weak_classifiers)
    c=0
    for i in range(X_train.shape[0]):
      sample = dict(X_train.iloc[i])
      if classify(all_weak_classifiers[t],sample)!=Y_train[i]:
            error_t += train_data['weights'][i]*1
      else:
            c+=1
    error_t = error_t/sum(train_data['weights'])
    print(error_t)
    vote_alpha.append((1/2)*(math.log(1-error_t)/error_t))
    for i in range(X_train.shape[0]):
        sample = dict(X_train.iloc[i])
        if classify(all_weak_classifiers[t],sample)!=Y_train[i]:
            new_weights.append(all_weights[t][i] * math.exp(vote_alpha[t]))
        else:
            new_weights.append(all_weights[t][i] * math.exp(-vote_alpha[t]))
    all_weights.append(new_weights)
#     c_t=0
#     for i in range(X_train.shape[0]):
#       sample = dict(X_train.iloc[i])
#       if classify(all_weak_classifiers[t],sample)==Y_train[i]:
#         c_t+=1
#       else:
#         if classify(all_weak_classifiers[t],sample)=="y" or classify(all_weak_classifiers[t],sample)=="n":
#             print(classify(all_weak_classifiers[t],sample))
    train_errors.append((X_train.shape[0]-c)/X_train.shape[0])
    c_test=0
    for i in range(X_test.shape[0]):
      sample = dict(X_test.iloc[i])
      if classify(all_weak_classifiers[t],sample)==Y_test[i]:
        c_test+=1
      else:
        if classify(all_weak_classifiers[t],sample)=="y" or classify(all_weak_classifiers[t],sample)=="n":
            print(classify(all_weak_classifiers[t],sample))
    test_errors.append((X_test.shape[0]-c_test)/X_test.shape[0])
    all_weights[t+1] = (np.array(all_weights[t+1]) / sum(all_weights[t+1]))
    print(train_errors,test_errors)

