# -*- coding: utf-8 -*-
from __future__ import division
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
logging.basicConfig(format='%(asctime)s %(name)s.%(lineno)d %(levelname)s : %(message)s',
        datefmt="%H:%M:%S",
        level=logging.INFO)
# logger = logging.getLogger(__name__)
logger = logging.getLogger('__main__').getChild(__name__)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_random_state(random_state=None):
    if random_state is None:
        random_state = np.random.RandomState()
    elif not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(seed=random_state)
    return random_state

def prepare_directory(outdir):
    """Make sure the output directory does not exist, then create it.
    """
    outdir = os.path.abspath(outdir)
    if os.path.exists(outdir):
        raise RuntimeError("output path {} already exists!".format(outdir))
    os.mkdir(outdir)

def load_spark_dataframe(path_to_data, spark, fmt=None):
    if (fmt and fmt.lower()=='tsv') or ('parquet' not in path_to_data.lower() and ('csv' in path_to_data.lower() or 'tsv' in path_to_data.lower())):
        # ASSUME TAB SEPARATED
        return spark.read.csv(path_to_data, sep='\t', header=True)
    else:
        # Assume parquet format
        return spark.read.parquet(path_to_data)

def get_year(x):
    if x:
        return x.year
    else:
        return None

def save_pandas_dataframe_to_pickle(df, outfname):
    if 'year' not in df.columns:
        df['year'] = df['pub_date'].apply(get_year)
    columns_rename = {
        'ID': 'Paper_ID',
        'flow': 'EF'
    }
    df.rename(columns=columns_rename, inplace=True)
    df.to_pickle(outfname)

# def prepare_directory(mag_id, description=None):
#     dirname = "review_{}".format(mag_id)
#     if description:
#         dirname = dirname + "_{}".format(description)
#     dirname = os.path.join('data', dirname)
#     if not os.path.exists(dirname):
#         logger.debug("creating directory: {}".format(dirname))
#         os.mkdir(dirname)
#     else:
#         logger.debug("directory {} already exists. using this directory.".format(dirname))
#     return dirname

def load_data_from_pickles(data_dir, files=['test_papers', 'seed_papers', 'target_papers'], ext='.pickle'):
    dfs = []
    for fn in files:
        df = pd.read_pickle(os.path.join(data_dir, fn + ext))
        df = df.drop_duplicates()
        dfs.append(df)
    return dfs

def remove_seed_papers_from_test_set(test_papers, seed_papers):
    n_before = len(test_papers)
    test_papers = test_papers.drop(seed_papers.index, errors='ignore')
    n_after = len(test_papers)
    logger.debug("removed {} seed papers from the haystack. size of haystack: {}".format(n_before-n_after, n_after))
    return test_papers

def remove_missing_titles(df, colname='title'):
    n_before = len(df)
    df = df.dropna(subset=[colname])
    n_after = len(df)
    logger.debug("removed {} papers with missing titles. size of haystack: {}".format(n_before-n_after, n_after))
    return df

def year_lowpass_filter(df, year=None):
    # only keep papers published on or before a given year
    n_before = len(df)
    if year is not None:
        df = df[df.year<=year]
    n_after = len(df)
    logger.debug("removed {} papers published after year {}. size of haystack: {}".format(n_before-n_after, year, n_after))
    return df

def prepare_data_for_model(data_dir, year=None):
    test_papers, seed_papers, target_papers = load_data_from_pickles(data_dir)
    # test_subset = test_papers.sample(n=args.subset_size, random_state=args.seed)
    test_papers = remove_seed_papers_from_test_set(test_papers, seed_papers)
    target_ids = set(target_papers.Paper_ID)
    test_papers['target'] = test_papers.Paper_ID.apply(lambda x: x in target_ids)
    test_papers = remove_missing_titles(test_papers)
    if year is None:
        year = get_year_from_datadir(data_dir)
    test_papers = year_lowpass_filter(test_papers, year=year)
    return test_papers, seed_papers, target_papers

def prepare_data_for_pretrained_model(data_dir, year, random_seed=999):
    test_papers, seed_papers, target_papers = prepare_data_for_model(data_dir, year)
    X = test_papers.reset_index()
    y = X['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)
    return X_train, X_test, y_train, y_test

def get_paper_info(datadir):
    g = glob(os.path.join(datadir, '..', '*paperinfo.json'))
    if len(g) != 1:
        raise RuntimeError("failed to find paper info")
    return pd.read_json(os.path.join(g[0]), typ='series')

def get_year_from_datadir(datadir):
    paper_info = get_paper_info(datadir)
    mag_date = datetime.utcfromtimestamp(paper_info['mag_date']/1000)
    year = mag_date.year
    return year

# these are broken. the best model isn't necessarily the most recent
# def get_best_model_path(datadir):
#     g = glob(os.path.join(datadir, 'best_model*'))
#     g.sort()
#     best_model_dirname = g[-1]
#     return os.path.join(best_model_dirname, 'best_model.pickle')
#
# def get_best_model_from_datadir(datadir):
#     return joblib.load(get_best_model_path(datadir))

def predict_ranks_from_data(pipeline, df):
    start = timer()
    y_score = pipeline.predict_proba(df)[:, 1]
    logger.debug("predicted probabilities in {}".format(format_timespan(timer()-start)))
    pred_ranks = pd.Series(y_score, index=df.index, name="pred_ranks")
    return df.join(pred_ranks).sort_values('pred_ranks', ascending=False)

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
        return avg_dist.as_matrix().reshape(-1, 1)


class DataFrameColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, colname):
        self.colname = colname
        
    def fit(self, x, y=None):
        return self
    
    def transform(self, df):
        return df[self.colname].as_matrix().reshape(-1, 1)

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
