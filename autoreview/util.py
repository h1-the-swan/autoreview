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
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.externals import joblib
# import joblib
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
    path_to_data = str(path_to_data)
    if (fmt and fmt.lower()=='tsv') or ('parquet' not in path_to_data.lower() and ('csv' in path_to_data.lower() or 'tsv' in path_to_data.lower())):
        # ASSUME TAB SEPARATED
        return spark.read.csv(path_to_data, sep='\t', header=True)
    else:
        # Assume parquet format
        return spark.read.parquet(path_to_data)

def load_pandas_dataframe(path_to_data, fmt=None):
    path_to_data = str(path_to_data)
    if (fmt and fmt.lower()=='tsv') or ('parquet' not in path_to_data.lower() and ('csv' in path_to_data.lower() or 'tsv' in path_to_data.lower())):
        # ASSUME TAB SEPARATED
        return pd.read_csv(path_to_data, sep='\t', header=True)
    else:
        # Assume parquet format
        return pd.read_parquet(path_to_data)

def get_year(x):
    if x:
        return x.year
    else:
        return None

def save_pandas_dataframe_to_pickle(df, outfname):
    if 'year' not in df.columns:
        df['year'] = df['pub_date'].apply(get_year)
    columns_rename = {
        # 'ID': 'Paper_ID',
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

def infer_year_column(df):
    cols = df.columns
    choices = ['year', 'paper_year', 'pub_year']
    for c in choices:
        for col in cols:
            if col.lower() == c:
                return col
    for col in cols:
        if 'year' in col.lower():
            return col
    return None

def year_lowpass_filter(df, year=None, year_colname=None):
    # only keep papers published on or before a given year
    n_before = len(df)
    if year is not None:
        if not year_colname:
            year_colname = infer_year_column(df) or 'year'
        df = df[df[year_colname]<=year]
    n_after = len(df)
    logger.debug("removed {} papers published after year {}. size of haystack: {}".format(n_before-n_after, year, n_after))
    return df

def prepare_data_for_model(data_dir, year=None, id_colname='Paper_ID'):
    test_papers, seed_papers, target_papers = load_data_from_pickles(data_dir)
    # test_subset = test_papers.sample(n=args.subset_size, random_state=args.seed)
    test_papers = remove_seed_papers_from_test_set(test_papers, seed_papers)
    target_ids = set(target_papers[id_colname])
    test_papers['target'] = test_papers[id_colname].apply(lambda x: x in target_ids)
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

def prec_at_n(preds, n=10):
    """Precision at n

    :preds: True/False predictions (in rank order)
    :returns: precision at n (float)

    """
    a = np.asarray(preds)
    return a[:n].mean()

def recall_at_n(preds, num_relevant, n=10):
    """Recall at n

    :preds: True/False predictions (in rank order)
    :num_relevant: total number of relevant documents
    :returns: recall at n (float)

    """
    a = np.asarray(preds)
    return a[:n].sum() / num_relevant

def f1_score(prec, recall):
    return (2 * prec * recall) / (prec + recall)

def prec_recall_f1_at_n(preds, num_relevant, n=10):
    prec = prec_at_n(preds, n)
    recall = recall_at_n(preds, num_relevant, n)
    return prec, recall, f1_score(prec, recall)

def average_precision(preds, num_relevant):
    """Average precision

    :preds: True/False predictions (in rank order)
    :num_relevant: total number of relevant documents
    :returns: average precision (float)

    """
    a = np.asarray(preds)
    prec_scores = [prec_at_n(a, n) for n in range(1, a.size) if a[n]]
    if not prec_scores:
        return 0
    return np.sum(prec_scores) / num_relevant
