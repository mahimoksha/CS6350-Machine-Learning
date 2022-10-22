#!/usr/bin/env python
# coding: utf-8

# In[116]:


import numpy as np
import math
import pandas as pd


# In[117]:


train_data = pd.read_csv("bank-2/train.csv",header=None)
print("Running part 2(a)...")
train_data_backup = train_data.copy()


# In[118]:


columns = ["age","job","marital","education","default","balance","housing","loan","contact","day","month","duration","campaign","pdays","previous","poutcome","y"]
train_data.columns = columns
train_data_backup.columns = columns


# In[119]:


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


# In[120]:


train_data['age'] = train_data['age'].map(num_to_bin_age)
train_data['balance'] = train_data['balance'].map(num_to_bin_balance)
train_data['day'] = train_data['day'].map(num_to_bin_day)
train_data['duration'] = train_data['duration'].map(num_to_bin_duration)
train_data['campaign'] = train_data['campaign'].map(num_to_bin_campaign)
train_data['pdays'] = train_data['pdays'].map(num_to_bin_pdays)
train_data['previous'] = train_data['previous'].map(num_to_bin_previous)

X_train = train_data.iloc[:,:-1]
Y_train = train_data.iloc[:,-1]


# In[121]:


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


# In[122]:


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


# In[105]:


# temp_age = train_data[train_data['age']=='yes']
# temp_age[temp_age['y']=="no"]['weights']


# In[106]:


# p_pos = sum(train_data[train_data['y']=="yes"]['weights'])
# p_neg = sum(train_data[train_data['y']=="no"]['weights'])
# print(p_pos,p_neg)
# train_age_yes = train_data[train_data['age']=="yes"]
# p_pos_age_yes = sum(train_age_yes[train_age_yes['y']=="yes"]['weights'])/sum(train_age_yes['weights'])
# p_neg_age_yes = sum(train_age_yes[train_age_yes['y']=="no"]['weights'])/sum(train_age_yes['weights'])
# print(p_pos_age_yes,p_neg_age_yes)
# train_age_no = train_data[train_data['age']=="no"] 
# p_pos_age_no = sum(train_age_no[train_age_no['y']=="yes"]['weights'])/sum(train_age_no['weights'])
# p_neg_age_no = sum(train_age_no[train_age_no['y']=="no"]['weights'])/sum(train_age_no['weights'])
# print(p_pos_age_no,p_neg_age_no)
# total_entropy = -p_pos*math.log2(p_pos)-p_neg*math.log2(p_neg)
# print(total_entropy) #correct
# entropy1 = -p_pos_age_yes*math.log2(p_pos_age_yes)-p_neg_age_yes*math.log2(p_neg_age_yes)
# entropy2 = -p_pos_age_no*math.log2(p_pos_age_no)-p_neg_age_no*math.log2(p_neg_age_no)
# print(entropy1,entropy2)
# total_entropy-((len(train_age_yes)/train_data.shape[0])* entropy1 + (len(train_age_no)/train_data.shape[0])* entropy2)


# In[113]:


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


# In[127]:


def classify(tree, query):
      if tree in labels:
          return tree
      key = query.get(tree[0])
      if key not in tree[1]:
          key = None
      class_ = classify(tree[1][key], query)
      return class_


# In[128]:


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
    c=0
    for i in range(X_train.shape[0]):
      sample = dict(X_train.iloc[i])
      if classify(all_weak_classifiers[t],sample)!=Y_train[i]:
            error_t += train_data['weights'][i]*1
      else:
            c+=1
    error_t = error_t/sum(train_data['weights'])
    print("erorr: ",error_t)
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
    test_errors.append((X_test.shape[0]-c_test)/X_test.shape[0])
    all_weights[t+1] = (np.array(all_weights[t+1]) / sum(all_weights[t+1]))
    print(train_errors,test_errors)
plt.plot(range(len(train_errors)),train_errors,label = "train_error")
plt.plot(range(len(test_errors)),test_errors,label="test_error")
plt.legend()
plt.ylabel("train and test errors")
plt.xlabel("number of trees")
plt.title("Train and Test errors vary along with the trees")
plt.show()

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




