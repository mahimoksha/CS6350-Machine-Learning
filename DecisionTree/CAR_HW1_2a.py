 


import numpy as np
import math
import pandas as pd
import sys


max_depth = int(sys.argv[1])
split_method = sys.argv[2]
# In[55]:


train_data = pd.read_csv("Car/train.csv",header=None)

# In[46]:

print("Running part 2(a)...")

# In[56]:


with open('Car/data-desc.txt') as f:
    lines = f.readlines()
columns = []
for i in range(len(lines)):
    if '|' in lines[i] and lines[i].split(" ")[1]=="columns\n":
        columns.append(lines[i+1].split(","))
        break
columns = columns[0] 
columns[-1] = columns[-1][:-1]
train_data.columns = columns


# In[57]:


X_train = train_data.iloc[:,:-1]
Y_train = train_data.iloc[:,-1]


# In[58]:


test_data = pd.read_csv("Car/test.csv",header=None)
test_data.columns = columns


# In[59]:


X_test = test_data.iloc[:,:-1]
Y_test = test_data.iloc[:,-1]


# In[60]:


class ID3:
    def __init__(self):
        self.data = None
        self.features = None
        self.labels = None
    def total_entropy(self,data,labels):
        label_data = data['label'].value_counts()
        e = 0
        for i in label_data:
            e -= (i/sum(label_data)) * math.log2(i/sum(label_data))
        return e
    def fea_cat_entropy(self,data, feature,labels):
        categories_features = data[feature].value_counts().keys()
        e = 0
        for cat in categories_features:
            label_fea_data = data[data[feature]==cat]['label'].value_counts()
            pd.DataFrame(data)[feature].value_counts()
            s = 0
            for i in label_fea_data:
                s -= (i/sum(label_fea_data)) * math.log2(i/sum(label_fea_data))
            e += (sum(label_fea_data)/len(data)) * (s)
        return e
    def total_gini(self,data,labels):
        label_data = data['label'].value_counts()
        e = 1
        for i in label_data:
            e -= (i/sum(label_data))**2
        return e
    def fea_cat_gini(self,data, feature,labels):
        categories_features = data[feature].value_counts().keys()
        e = 0
        for cat in categories_features:
            label_fea_data = data[data[feature]==cat]['label'].value_counts()
            pd.DataFrame(data)[feature].value_counts()
            s = 1
            for i in label_fea_data:
                s -= (i/sum(label_fea_data))**2
            e += (sum(label_fea_data)/len(data)) * (s)
        return e
    
    def total_me(self,data,labels):
        label_data = data['label'].value_counts()
        if len(label_data)==1:
                   ee=0
        else:
            ee = (min(label_data)/sum(label_data))
        return ee
    def fea_cat_me(self,data, feature,labels):
        categories_features = data[feature].value_counts().keys()
        e = 0
#         import pdb;pdb.set_trace()
        for cat in categories_features:
            label_fea_data = data[data[feature]==cat]['label'].value_counts()
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
#           import pdb; pdb.set_trace()
          return self.total_me(data,labels) - self.fea_cat_me(data,features,labels)
    def create_root_node(self,data,features,split_method,labels):
        total_fea_ig = dict()
        for i in features:
            total_fea_ig[i] = self.IG(data,i,labels,split_method)
        best_feature = max(total_fea_ig, key=total_fea_ig.get)
        return best_feature
    def find_bestsplits(self,data,data_copy,max_depth,depth,root_node):
        features_groups = dict()
        temp_x = []
        if depth<max_depth:
          for x in data[root_node].value_counts().keys():
              for i in range(len(data)):
                  if data.iloc[i][root_node] == x:
                      if features_groups.get(x,0)==0:
                          features_groups[x] = [(dict(data.iloc[i][:-1]),data.iloc[i]['label'])]
                      else:
                          features_groups[x].append((dict(data.iloc[i][:-1]),data.iloc[i]['label']))
              temp_x.append(x)
          c_ = list(data_copy[root_node].value_counts().keys())
          for i in c_:
              if i not in temp_x:
                label_counts = dict(data['label'].value_counts())
                features_groups[i] =  [({}, max(label_counts,key=label_counts.get))]
          return features_groups,depth+1
        else:
          c_ = list(data_copy[root_node].value_counts().keys())
          temp_x = list(data[root_node].value_counts().keys())
          for i in c_:
                if i in temp_x:
                  label_counts = dict(data[data[root_node]==i]['label'].value_counts())
                  features_groups[i] =  [({}, max(label_counts,key=label_counts.get))]
                else:
                  label_counts = dict(data['label'].value_counts())
                  features_groups[i] =  [({}, max(label_counts,key=label_counts.get))]
          return features_groups,depth
    def ID3_Algo(self,data,data_copy,features,labels,max_depth,depth,split_method,backup_features=[]):
        if len(data['label'].value_counts()) == 1:
            return data['label'].value_counts().keys()
        if len(features)==0: 
            label_counts = dict(data['label'].value_counts())
            return max(label_counts,key=label_counts.get)
        root_node = self.create_root_node(data,features,split_method,labels)
        categories = data[root_node].value_counts().keys()
        if(root_node in features):
            features.remove(root_node)
            backup_features.append(root_node)
        feature_groups,depth = self.find_bestsplits(data, data_copy,max_depth,depth,root_node)
        subtree_dict = {}
        final_tree = tuple()
        for i ,j in feature_groups.items():
            data = pd.DataFrame.from_dict({k: dict(v) for k,v in pd.DataFrame(j)[0].items()}, orient='index')
            data['label'] = pd.DataFrame(j)[1]
            subtree_dict[i] = self.ID3_Algo(data, data_copy, features,labels,max_depth,depth,split_method,backup_features)
            final_tree = (root_node,subtree_dict)
        features.append(backup_features[-1])
        backup_features.remove(backup_features[-1])
        return final_tree


# In[61]:


def classify(tree, query):
      if tree in labels:
          return tree
      key = query.get(tree[0])
      if key not in tree[1]:
          key = None
      class_ = classify(tree[1][key], query)
      return class_


# In[65]:


# best_split = ["entropy","gini","MajorityError"]
train_error = dict()
test_error = dict()

# for md in range(6):
#     for j in best_split:
print("*"*5+" training decision tree with depth "+str(max_depth)+" and split method "+split_method+" "+"*"*5)
algo = ID3()
labels = list(train_data['label'].value_counts().keys())
answer = dict()
max_depth = max_depth
#         print("max-depth",max_depth)
#         print("split-method",split_method)
s = algo.ID3_Algo(train_data,train_data,columns[:-1],labels,max_depth,1,split_method,[])
c=0
for i in range(X_train.shape[0]):
    sample = dict(X_train.iloc[i])
    if classify(s,sample)[0]==Y_train[i]:
        c+=1
    # else:
    #   print(sample,classify(s,sample)[0],Y_train[0])
print("test missclassified points: ",(X_train.shape[0]-c)) 
print("Training Error: ",(X_train.shape[0]-c)/X_train.shape[0])
if train_error.get(max_depth):
    train_error[max_depth].append((split_method,(X_train.shape[0]-c)/X_train.shape[0]))
else:
    train_error[max_depth] = [(split_method,(X_train.shape[0]-c)/X_train.shape[0])]
c=0
for i in range(X_test.shape[0]):
    sample = dict(X_test.iloc[i])
    if classify(s,sample)[0]==Y_test[i]:
        c+=1
    # else:
    #   print(sample,classify(s,sample)[0],Y_test[i])
print("test missclassified points: ",(X_test.shape[0]-c))
print("Testing Error: ",(X_test.shape[0]-c)/X_test.shape[0])
if test_error.get(max_depth):
    test_error[max_depth].append((split_method,(X_test.shape[0]-c)/X_test.shape[0]))
else:
    test_error[max_depth] = [(split_method,(X_test.shape[0]-c)/X_test.shape[0])]
print("*"*50)


# In[75]:


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
#         test_ans[k].append(round(test_error[i][j][1],4))
#     k+=1


# In[76]:


# from prettytable import PrettyTable
# x = PrettyTable()
# x.field_names = ["depth \ split_method","Entropy", "Gini Index", "Majority Error"]
# x.add_rows(train_ans)
# print(x)
# x = PrettyTable()
# x.field_names = ["depth \ split_method","Entropy", "Gini Index", "Majority Error"]
# x.add_rows(test_ans)
# print(x)

