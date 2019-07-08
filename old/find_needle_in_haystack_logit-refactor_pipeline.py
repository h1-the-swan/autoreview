
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


from db_connect_mag import Session, Paper, PaperAuthorAffiliation, db


# In[19]:


# test_papers_df = pd.read_pickle('data/collect_haystack_20180409/test_papers.pickle')
# target_papers_df = pd.read_pickle('data/collect_haystack_20180409/target_papers.pickle')
# train_papers_df = pd.read_pickle('data/collect_haystack_20180409/train_papers.pickle')


# In[20]:


# this is the data for the fortunato review on Community Detection in Graphs
start = timer()
test_papers_df = pd.read_pickle('data/collect_haystack_2127048411_seed-1/test_papers.pickle')
target_papers_df = pd.read_pickle('data/collect_haystack_2127048411_seed-1/target_papers.pickle')
train_papers_df = pd.read_pickle('data/collect_haystack_2127048411_seed-1/seed_papers.pickle')
print("data loaded. {}".format(format_timespan(timer()-start)))


# In[21]:


# with open('data/collect_haystack_20180409_2/counter.pickle', 'rb') as f:
#     c = pickle.load(f)


# In[22]:


def get_target_in_test(test, target, id_colname='Paper_ID'):
    return set.intersection(set(test[id_colname]), set(target[id_colname]))
print("target_in_test:")
print(len(get_target_in_test(test_papers_df, target_papers_df)))
print()


# In[23]:


print("len(target_papers_df)")
print(len(target_papers_df))


# In[24]:


print("len(test_papers_df)")
print(len(test_papers_df))



# remove the train (seed) papers from the test set (haystack)
n_before = len(test_papers_df)
test_papers_df = test_papers_df.drop(train_papers_df.index, errors='ignore')
n_after = len(test_papers_df)
print("removed {} seed papers from the haystack. size of haystack: {}".format(n_before-n_after, n_after))


# In[26]:


start = timer()
target_ids = set(target_papers_df.Paper_ID)
test_papers_df['target'] = test_papers_df.Paper_ID.apply(lambda x: x in target_ids)


# In[27]:


# def tree_distance(n1, n2, sep=":"):
#     # https://en.wikipedia.org/wiki/Lowest_common_ancestor
#     # the distance from v to w can be computed as 
#     # the distance from the root to v, plus the distance from 
#     # the root to w, minus twice the distance from 
#     # the root to their lowest common ancestor
#     v, w = [n.split(sep) for n in [n1, n2]]
#     distance_root_to_v = len(v)
#     distance_root_to_w = len(w)
    
#     distance_root_to_lca = 0
#     for i in range(min(distance_root_to_v, distance_root_to_w)):
#         if v[i] == w[i]:
#             distance_root_to_lca += 1
#         else:
#             break
#     return distance_root_to_v + distance_root_to_w - (2*distance_root_to_lca)


# In[28]:


def tree_distance(n1, n2, sep=":"):
    # since depth is sort of arbitrary, let's try this
    v, w = [n.split(sep) for n in [n1, n2]]
    distance_root_to_v = len(v)
    distance_root_to_w = len(w)
    avg_depth = (distance_root_to_v + distance_root_to_w) * .5
    
    distance_root_to_lca = 0
    for i in range(min(distance_root_to_v, distance_root_to_w)):
        if v[i] == w[i]:
            distance_root_to_lca += 1
        else:
            break
    return (avg_depth - distance_root_to_lca) / avg_depth


# In[29]:


def avg_distance(cl, cl_group):
    distances = []
    for x in cl_group:
        distances.append(tree_distance(cl, x))
    return sum(distances) / len(distances)


# In[30]:


start = timer()
test_papers_df['avg_distance_to_train'] = test_papers_df.cl.apply(avg_distance, cl_group=train_papers_df.cl.tolist())
print(format_timespan(timer()-start))


# In[31]:


test_papers_df.sort_values(['avg_distance_to_train', 'EF'], ascending=[True, False]).head(50)


# In[32]:


test_papers_df.groupby('target')['EF', 'avg_distance_to_train'].describe().T





from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


# In[42]:


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report


# In[37]:


# http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html
class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to scikit-learn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


# In[100]:


class ClusterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, colname='cl'):
        self.colname = colname
        
    def fit(self, x, y=None):
        return self
    
    def transform(self, df):
        avg_dist = df[self.colname].apply(avg_distance, cl_group=train_papers_df.cl.tolist())
        return avg_dist.as_matrix().reshape(-1, 1)


# In[101]:


class DataFrameColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, colname):
        self.colname = colname
        
    def fit(self, x, y=None):
        return self
    
    def transform(self, df):
        return df[self.colname].as_matrix().reshape(-1, 1)


# In[190]:





# In[191]:

def run_pipeline(pipeline, outfname=None):

    # X = test_papers_df[['EF', 'avg_distance_to_train']]
    X = test_papers_df[test_papers_df.title.notnull()]
    # Fortunato paper was published in 2010
    X = X[X.year<=2010]

    # y = test_papers_df['target']
    y = X['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=999)


    # In[192]:


    start = timer()
    print("fitting pipeline...")
    pipeline.fit(X_train, y_train)
    print(format_timespan(timer()-start))
    print()


    # In[193]:


    start = timer()
    # y_pred_proba = model.predict_proba(X_test)[:, 1]
    print("predicting probabilities...")
    y_pred_proba = pipeline.predict_proba(X)[:, 1]
    print(format_timespan(timer()-start))
    print()




    pred_ranks = pd.Series(y_pred_proba, index=X.index, name='pred_ranks')
    # test_papers_df.join(pred_ranks).sort_values('pred_ranks', ascending=False).head()


    # In[196]:


    # len(test_papers_df)


    # In[197]:


    # len(X)


    # In[198]:


    # top_predictions = test_papers_df.join(pred_ranks).sort_values('pred_ranks', ascending=False).head(len(target_papers_df))
    X_predictions = X.join(pred_ranks).sort_values('pred_ranks', ascending=False)
    top_predictions = X_predictions.head(len(target_papers_df))


    # In[199]:


    print("counts of correctly identified papers in the top {}:".format(len(target_papers_df)))
    print(top_predictions.groupby('target')['Paper_ID'].count())
    print()

    if outfname:
        print("saving to {}".format(outfname))
        X_predictions.to_pickle(outfname)
    print()


print("=======Pipeline: avg_distance_to_train and EF -- SVC")

pipeline = Pipeline([
    ('union', FeatureUnion(
        transformer_list = [
            ('avg_distance_to_train', Pipeline([
#                 ('selector', ItemSelector(key='avg_distance_to_train')),
#                 ('vect', DictVectorizer(X.avg_distance_to_train.to_dict))
                ('cl_feat', ClusterTransformer()),
            ])),
            ('ef', Pipeline([
#                 ('selector', ItemSelector(key='avg_distance_to_train')),
#                 ('vect', DictVectorizer(X.avg_distance_to_train.to_dict))
                ('ef_feat', DataFrameColumnTransformer('EF')),
            ])),
        ],
    )),
    
    ('svc', SVC(kernel='linear', probability=True))
])

run_pipeline(pipeline, "data/collect_haystack_2127048411_seed-1/pipeline/pipeline01-dist_and_ef-SVC.pickle")
print()




print("=======Pipeline: avg_distance_to_train and EF and TfIdf -- SVC")

pipeline = Pipeline([
    ('union', FeatureUnion(
        transformer_list = [
            ('avg_distance_to_train', Pipeline([
#                 ('selector', ItemSelector(key='avg_distance_to_train')),
#                 ('vect', DictVectorizer(X.avg_distance_to_train.to_dict))
                ('cl_feat', ClusterTransformer()),
            ])),
            ('ef', Pipeline([
#                 ('selector', ItemSelector(key='avg_distance_to_train')),
#                 ('vect', DictVectorizer(X.avg_distance_to_train.to_dict))
                ('ef_feat', DataFrameColumnTransformer('EF')),
            ])),
            
            # NOTE: this is just to test.
            # we probably want features that relate the titles to the seed papers. not just straight features in test set.
            ('title_bow', Pipeline([
                ('selector', ItemSelector(key='title')),
                ('tfidf', TfidfVectorizer(min_df=10)),
            ]))
        ],
    )),
    
    ('svc', SVC(kernel='linear', probability=True))
])

run_pipeline(pipeline, "data/collect_haystack_2127048411_seed-1/pipeline/pipeline02-dist_and_ef_and_tfidf-SVC.pickle")
print()


# In[159]:


print("=======Pipeline: avg_distance_to_train and EF -- SVC")

pipeline = Pipeline([
    ('union', FeatureUnion(
        transformer_list = [
            ('avg_distance_to_train', Pipeline([
#                 ('selector', ItemSelector(key='avg_distance_to_train')),
#                 ('vect', DictVectorizer(X.avg_distance_to_train.to_dict))
                ('cl_feat', ClusterTransformer()),
            ])),
            ('ef', Pipeline([
#                 ('selector', ItemSelector(key='avg_distance_to_train')),
#                 ('vect', DictVectorizer(X.avg_distance_to_train.to_dict))
                ('ef_feat', DataFrameColumnTransformer('EF')),
            ])),
            
            # NOTE: this is just to test.
            # we probably want features that relate the titles to the seed papers. not just straight features in test set.
#             ('title_bow', Pipeline([
#                 ('selector', ItemSelector(key='title')),
#                 ('tfidf', TfidfVectorizer(min_df=10)),
#             ]))
        ],
    )),
    
    ('logreg', LogisticRegression())
])

run_pipeline(pipeline, "data/collect_haystack_2127048411_seed-1/pipeline/pipeline03-dist_and_ef-logreg.pickle")
print()


# # In[160]:
#
#
# # X = test_papers_df[['EF', 'avg_distance_to_train']]
# X = test_papers_df[test_papers_df.title.notnull()]
# # Fortunato paper was published in 2010
# X = X[X.year<=2010]
#
# # y = test_papers_df['target']
# y = X['target']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=999)
#
#
# # In[161]:
#
#
# start = timer()
# pipeline.fit(X_train, y_train)
# print(format_timespan(timer()-start))
#
#
# # In[162]:
#
#
# start = timer()
# # y_pred_proba = model.predict_proba(X_test)[:, 1]
# y_pred_proba = pipeline.predict_proba(X)[:, 1]
# print(format_timespan(timer()-start))
# y_pred_proba
#
#
# # In[163]:
#
#
# y_pred_proba.shape
#
#
# # In[164]:
#
#
# pred_ranks = pd.Series(y_pred_proba, index=X.index, name='pred_ranks')
# test_papers_df.join(pred_ranks).sort_values('pred_ranks', ascending=False).head()
#
#
# # In[165]:
#
#
# len(test_papers_df)
#
#
# # In[166]:
#
#
# len(X)
#
#
# # In[167]:
#
#
# # top_predictions = test_papers_df.join(pred_ranks).sort_values('pred_ranks', ascending=False).head(len(target_papers_df))
# top_predictions = X.join(pred_ranks).sort_values('pred_ranks', ascending=False).head(len(target_papers_df))
#
#
# # In[168]:
#
#
# top_predictions.groupby('target')['Paper_ID'].count()
#
#
# # In[169]:
#
#
# top_predictions.pred_ranks.min()
#
#
# # In[170]:
#
#
# start = timer()
# y_test_pred = pipeline.predict(X_test)
# print(format_timespan(timer()-start))
#
#
# # In[171]:
#
#
# print(classification_report(y_test, y_test_pred))
#
#
# # In[32]:
#
#
# # what if we only use pagerank?
# X = test_papers_df[['EF']]
# y = test_papers_df['target']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=999)
#
# start = timer()
# _model = LogisticRegression()
# _model.fit(X_train, y_train)
# print(format_timespan(timer()-start))
#
# # y_pred_proba = model.predict_proba(X_test)[:, 1]
# _y_pred_proba = _model.predict_proba(X)[:, 1]
# #y_pred_proba
#
# print(y_pred_proba.shape)
#
# _pred_ranks = pd.Series(_y_pred_proba, index=X.index, name='pred_ranks')
# #test_papers_df.join(_pred_ranks).sort_values('pred_ranks', ascending=False).head()
#
#
#
# _top_predictions = test_papers_df.join(_pred_ranks).sort_values('pred_ranks', ascending=False).head(len(target_papers_df))
#
# _top_predictions.groupby('target')['Paper_ID'].count()
#
#
# # In[33]:
#
#
# # what if we only use avg distance?
# X = test_papers_df[['avg_distance_to_train']]
# y = test_papers_df['target']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=999)
#
# start = timer()
# _model = LogisticRegression()
# _model.fit(X_train, y_train)
# print(format_timespan(timer()-start))
#
# # y_pred_proba = model.predict_proba(X_test)[:, 1]
# _y_pred_proba = _model.predict_proba(X)[:, 1]
# #y_pred_proba
#
# print(y_pred_proba.shape)
#
# _pred_ranks = pd.Series(_y_pred_proba, index=X.index, name='pred_ranks')
# #test_papers_df.join(_pred_ranks).sort_values('pred_ranks', ascending=False).head()
#
#
#
# _top_predictions = test_papers_df.join(_pred_ranks).sort_values('pred_ranks', ascending=False).head(len(target_papers_df))
#
# _top_predictions.groupby('target')['Paper_ID'].count()
#
#
# # In[34]:
#
#
# start = timer()
# toplevels = test_papers_df.cl.apply(lambda x: x.split(":")[0])
# print(format_timespan(timer()-start))
#
#
# # In[55]:
#
#
# toplevels.name = 'toplevel'
#
#
# # In[37]:
#
#
# toplevels_set = set(toplevels)
#
#
# # In[46]:
#
#
# start = timer()
# tbl = db.tables['clusters_meta_tree']
# sq = tbl.select(tbl.c.toplevel_in_tree.in_(toplevels_set))
# # r = db.engine.execute(sq).fetchall()
# cl_meta = db.read_sql(sq)
# print(format_timespan(timer()-start))
#
#
# # In[50]:
#
#
# cl_meta = cl_meta.set_index('id')
#
#
# # In[82]:
#
#
# train_papers_df['toplevel'] = train_papers_df.cl.apply(lambda x: x.split(":")[0]).astype(int)
#
#
# # In[83]:
#
#
# meta_map = cl_meta.set_index('toplevel_in_tree').meta_cl
#
#
# # In[84]:
#
#
# train_papers_df['cl_meta'] = train_papers_df.toplevel.map(meta_map)
#
#
# # In[87]:
#
#
# test_papers_df['toplevel'] = toplevels.astype(int)
# test_papers_df['cl_meta'] = test_papers_df.toplevel.map(meta_map)
#
#
# # In[89]:
#
#
# start = timer()
# test_papers_df['meta_avg_distance_to_train'] = test_papers_df.cl_meta.apply(avg_distance, cl_group=train_papers_df.cl_meta.tolist())
# print(format_timespan(timer()-start))
#
#
# # In[94]:
#
#
# # logistic regression including meta cl
# X = test_papers_df[['EF', 'avg_distance_to_train', 'meta_avg_distance_to_train']]
# y = test_papers_df['target']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=999)
#
# start = timer()
# model_meta = LogisticRegression()
# model_meta.fit(X_train, y_train)
# print(format_timespan(timer()-start))
#
# # y_pred_proba = model.predict_proba(X_test)[:, 1]
# y_pred_proba_meta = model_meta.predict_proba(X)[:, 1]
# #y_pred_proba
#
# print(y_pred_proba_meta.shape)
#
# pred_ranks_meta = pd.Series(y_pred_proba_meta, index=X.index, name='pred_ranks')
# #test_papers_df.join(_pred_ranks).sort_values('pred_ranks', ascending=False).head()
#
#
#
# top_predictions_meta = test_papers_df.join(pred_ranks_meta).sort_values('pred_ranks', ascending=False).head(len(target_papers_df))
#
# top_predictions_meta.groupby('target')['Paper_ID'].count()
#
#
# # In[105]:
#
#
# from sklearn.metrics import roc_auc_score
# print(roc_auc_score(y, y_pred_proba))
# print(roc_auc_score(y, y_pred_proba_meta))
# print(roc_auc_score(y, _y_pred_proba))
#
