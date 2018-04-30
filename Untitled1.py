
# coding: utf-8

# In[15]:


import sys, os, time, pickle
from timeit import default_timer as timer
from humanfriendly import format_timespan


# In[16]:


import pandas as pd
import numpy as np


# In[17]:


from dotenv import load_dotenv
load_dotenv('admin.env')


# In[18]:


from db_connect_mag import Session, Paper, PaperAuthorAffiliation


# In[19]:


# test_papers_df = pd.read_pickle('data/collect_haystack_20180409/test_papers.pickle')
# target_papers_df = pd.read_pickle('data/collect_haystack_20180409/target_papers.pickle')
# train_papers_df = pd.read_pickle('data/collect_haystack_20180409/train_papers.pickle')


# In[20]:


test_papers_df = pd.read_pickle('data/collect_haystack_2490420619/test_papers_checkpoint.pickle')
target_papers_df = pd.read_pickle('data/collect_haystack_2490420619/target_papers.pickle')
train_papers_df = pd.read_pickle('data/collect_haystack_2490420619/train_papers.pickle')


# In[21]:


# with open('data/collect_haystack_20180409/counter.pickle', 'rb') as f:
#     c = pickle.load(f)


# In[22]:


with open('data/collect_haystack_2490420619/counter_checkpoint.pickle', 'rb') as f:
    c = pickle.load(f)


# In[23]:


def get_target_in_test(test, target, id_colname='Paper_ID'):
    return set.intersection(set(test[id_colname]), set(target[id_colname]))
len(get_target_in_test(test_papers_df, target_papers_df))


# In[24]:


len(target_papers_df)


# In[25]:


start = timer()
target_ids = set(target_papers_df.Paper_ID)
test_papers_df['target'] = test_papers_df.Paper_ID.apply(lambda x: x in target_ids)
print(format_timespan(timer()-start))


# In[26]:


def tree_distance(n1, n2, sep=":"):
    # https://en.wikipedia.org/wiki/Lowest_common_ancestor
    # the distance from v to w can be computed as 
    # the distance from the root to v, plus the distance from 
    # the root to w, minus twice the distance from 
    # the root to their lowest common ancestor
    v, w = [n.split(sep) for n in [n1, n2]]
    distance_root_to_v = len(v)
    distance_root_to_w = len(w)
    
    distance_root_to_lca = 0
    for i in range(min(distance_root_to_v, distance_root_to_w)):
        if v[i] == w[i]:
            distance_root_to_lca += 1
        else:
            break
    return distance_root_to_v + distance_root_to_w - (2*distance_root_to_lca)


# In[27]:


def avg_distance(cl, cl_group):
    distances = []
    for x in cl_group:
        distances.append(tree_distance(cl, x))
    return sum(distances) / len(distances)


# In[28]:


start = timer()
test_papers_df['avg_distance_to_train'] = test_papers_df.cl.apply(avg_distance, cl_group=train_papers_df.cl.tolist())
print(format_timespan(timer()-start))


# In[29]:


test_papers_df.sort_values(['avg_distance_to_train', 'EF'], ascending=[True, False]).head(50)


# In[30]:


test_papers_df.groupby('target')['EF', 'avg_distance_to_train'].describe().T


# In[29]:


session = Session()


# In[32]:


pid = train_papers_df.Paper_ID.iloc[0]
p = session.query(Paper).get(str(pid))


# In[ ]:


p.

