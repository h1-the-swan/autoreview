# -*- coding: utf-8 -*-
import sys, os, time
from glob import glob
from datetime import datetime
from timeit import default_timer as timer
try:
    from humanfriendly import format_timespan
except ImportError:
    def format_timespan(seconds):
        return "{:.2f} seconds".format(seconds)

import logging
logger = logging.getLogger(__name__)

# import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.externals import joblib
# import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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


def avg_distance(cl, cl_group):
    distances = []
    for x in cl_group:
        distances.append(tree_distance(cl, x))
    return sum(distances) / len(distances)

class ClusterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, seed_papers=None, colname='cl'):
        self.seed_papers = seed_papers
        self.colname = colname
        
    def fit(self, x, y=None):
        return self
    
    def transform(self, df):
        avg_dist = df[self.colname].apply(avg_distance, cl_group=self.seed_papers.cl.tolist())
        return avg_dist.to_numpy().reshape(-1, 1)


class DataFrameColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, colname):
        self.colname = colname
        
    def fit(self, x, y=None):
        return self
    
    def transform(self, df):
        return df[self.colname].to_numpy().reshape(-1, 1)

class AverageTfidfCosSimTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, seed_papers=None, colname='title'):
        self.seed_papers = seed_papers
        self.colname = colname

    def fit(self, x, y=None):
        return self

    def transform(self, df):
        # fit a term-frequency counter over all titles in seed papers and test papers
        vect = CountVectorizer()
        docs_test = df[self.colname]
        docs_seed = self.seed_papers[self.colname]
        docs_global = docs_test.append(docs_seed).tolist()
        vect.fit(docs_global)
        tf_global = vect.transform(docs_global)
        tf_test = vect.transform(docs_test.tolist())
        tf_seed = vect.transform(docs_seed.tolist())

        tfidf_transform = TfidfTransformer()
        tfidf_transform.fit(tf_global)
        tfidf_test = tfidf_transform.transform(tf_test)
        tfidf_seed = tfidf_transform.transform(tf_seed)
        csims = cosine_similarity(tfidf_test, tfidf_seed.mean(axis=0))
        return csims

class AbsoluteDistanceToSeedTransformer(BaseEstimator, TransformerMixin):
    """
    For a continuous variable, calculate the absolute deviation from the seed papers' mean.
    """
    def __init__(self, colname, seed_papers=None):
        self.seed_papers = seed_papers
        self.colname = colname
        
    def fit(self, x, y=None):
        return self
    
    def transform(self, df):
        seed_mean = self.seed_papers[self.colname].mean()
        abs_dist = df[self.colname].apply(lambda x: abs(x - seed_mean))
        return abs_dist.to_numpy().reshape(-1, 1)

class EmbeddingSimilarityTransformer(BaseEstimator, TransformerMixin):
    """
    cosine similarity between the embedding vector and the average vector of the seed papers
    """
    def __init__(self, seed_papers=None, embeddings=None, id_colname='ID'):
        """
        :seed_papers: dataframe of seed papers
        :embeddings: dictionary mapping paper ID to embedding vector. should include all of the seed and test papers
        """
        self.seed_papers = seed_papers
        self.embeddings = embeddings
        self.id_colname = id_colname

    def fit(self, x, y=None):
        return self

    def transform(self, df):
        seed_embeddings = [self.embeddings[id_] for id_ in self.seed_papers[self.id_colname].values if id_ in self.embeddings]
        seed_embeddings = np.array(seed_embeddings)
        test_embeddings = [self.embeddings[id_] for id_ in df[self.id_colname].values if id_ in self.embeddings]
        test_embeddings = np.array(test_embeddings)
        avg_seed_embeddings = seed_embeddings.mean(axis=0).reshape(1, -1)
        csims = cosine_similarity(test_embeddings, avg_seed_embeddings)
        return csims

