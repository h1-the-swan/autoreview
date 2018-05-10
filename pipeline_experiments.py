import sys, os, time
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

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics import classification_report

from utils.autoreview_utils import ItemSelector, DataFrameColumnTransformer, ClusterTransformer


class PipelineExperiment(object):

    """configure and run a pipeline for a review paper classifier"""

    def __init__(self, clf, seed_papers=None, random_state=999):
        """

        :clf: a classifier instance: the classifier to be used for this experiment instance (e.g., LogisticRegression())

        """

        self.clf = clf
        self.seed_papers = seed_papers
        self.load_random_state(random_state)
        self.pipeline_init()

    def load_random_state(self, random_state):
        if isinstance(random_state, np.random.RandomState):
            pass
        elif isinstance(random_state, int):
            random_state = np.random.RandomState(random_state)
        else:
            raise RuntimeError('argument random_state must be type np.random.RandomState or integer')
        self.random_state = random_state
        return self

    def pipeline_init(self):
        """initialize the pipeline
        :returns: self

        """
        pipeline = Pipeline([
            ('union', FeatureUnion(
                transformer_list = [
                    ('avg_distance_to_train', Pipeline([
                        ('cl_feat', ClusterTransformer(seed_papers=self.seed_papers)),
                    ])),
                    ('ef', Pipeline([
                        ('ef_feat', DataFrameColumnTransformer('EF')),
                    ])),
                ],
            )),
            
            ('clf', self.clf)
        ])
        self.pipeline = pipeline
        return self

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X):
        self.y_pred_proba = self.pipeline.predict_proba(X)[:, 1]
        pred_ranks = pd.Series(self.y_pred_proba, index=X.index, name='pred_ranks')
        self.predictions = X.join(pred_ranks).sort_values('pred_ranks', ascending=False)
        return self

    def top_predictions(self, n=200, id_colname='Paper_ID'):
        _top_predictions = self.predictions.head(n)
        return _top_predictions.groupby('target')[id_colname].count()

    def run(self, X, y, random_state=None, num_target=None):
        if random_state is not None:
            self.load_random_state(random_state)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state)

        start = timer()
        logger.debug("Fitting pipeline...")
        self.fit(X_train, y_train)
        logger.debug("Done fitting. Took {}".format(format_timespan(timer()-start)))

        start = timer()
        logger.debug("Predicting probabilities...")
        self.predict_proba(X)
        logger.debug("Done predicting. Took {}".format(format_timespan(timer()-start)))

        if num_target is None:
            num_target = (X.target==True).sum()
        logger.info("TOP PREDICTIONS: True is count of target papers in the top predicted")
        logger.info(self.top_predictions(n=num_target))

def load_data_from_pickles(data_dir, files=['test_papers', 'seed_papers', 'target_papers'], ext='.pickle'):
    dfs = []
    for fn in files:
        df = pd.read_pickle(os.path.join(data_dir, fn + ext))
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

def quickcheck(args):
    # quick check to make sure train-test split is always the same
    data_dir = "data/collect_haystack_2127048411_seed-3/"
    logger.debug("data_dir: {}".format(data_dir))
    # test_papers_df = pd.read_pickle(os.path.join(data_dir, 'test_papers.pickle'))
    # seed_papers = pd.read_pickle(os.path.join(data_dir, 'seed_papers.pickle'))
    # target_papers = pd.read_pickle(os.path.join(data_dir, 'target_papers.pickle'))
    test_papers, seed_papers, target_papers = load_data_from_pickles(data_dir)
    # test_subset = test_papers.sample(n=args.subset_size, random_state=args.seed)
    test_papers = remove_seed_papers_from_test_set(test_papers, seed_papers)
    target_ids = set(target_papers.Paper_ID)
    test_papers['target'] = test_papers.Paper_ID.apply(lambda x: x in target_ids)
    test_papers = remove_missing_titles(test_papers)
    # Fortunato review paper was published in 2010
    test_papers = year_lowpass_filter(test_papers, year=2010)

    X = test_papers.reset_index()
    y = X['target']
    splits = []
    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X, y, test_size=0.2, random_state=999)
    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X, y, test_size=0.2, random_state=999)
    logger.info(X_train_1.equals(X_train_2))
        
def test(args):
    logger.debug('testing mode')
    data_dir = "data/collect_haystack_2127048411_seed-3/"
    # test_papers_df = pd.read_pickle(os.path.join(data_dir, 'test_papers.pickle'))
    # seed_papers = pd.read_pickle(os.path.join(data_dir, 'seed_papers.pickle'))
    # target_papers = pd.read_pickle(os.path.join(data_dir, 'target_papers.pickle'))
    test_papers, seed_papers, target_papers = load_data_from_pickles(data_dir)
    test_subset = test_papers.sample(n=args.subset_size, random_state=args.seed)
    test_subset = remove_seed_papers_from_test_set(test_subset, seed_papers)
    target_ids = set(target_papers.Paper_ID)
    test_subset['target'] = test_subset.Paper_ID.apply(lambda x: x in target_ids)
    test_subset = remove_missing_titles(test_subset)
    # Fortunato review paper was published in 2010
    test_subset = year_lowpass_filter(test_subset, year=2010)

    X = test_subset.reset_index()
    y = X['target']
    clfs = [
        LogisticRegression(),
        SVC(probability=True),
        SVC(kernel='linear', probability=True),
        GaussianNB(),
        RandomForestClassifier(),
        AdaBoostClassifier(),
    ]
    for clf in clfs:
        experiment = PipelineExperiment(clf, seed_papers, random_state=args.seed)
        logger.info("========Pipeline:")
        logger.info(experiment.pipeline._final_estimator)
        experiment.run(X, y, num_target=len(target_papers))
        logger.info("\n\n")

def main(args):
    data_dir = "data/collect_haystack_2127048411_seed-3/"
    logger.debug("data_dir: {}".format(data_dir))
    # test_papers_df = pd.read_pickle(os.path.join(data_dir, 'test_papers.pickle'))
    # seed_papers = pd.read_pickle(os.path.join(data_dir, 'seed_papers.pickle'))
    # target_papers = pd.read_pickle(os.path.join(data_dir, 'target_papers.pickle'))
    test_papers, seed_papers, target_papers = load_data_from_pickles(data_dir)
    # test_subset = test_papers.sample(n=args.subset_size, random_state=args.seed)
    test_papers = remove_seed_papers_from_test_set(test_papers, seed_papers)
    target_ids = set(target_papers.Paper_ID)
    test_papers['target'] = test_papers.Paper_ID.apply(lambda x: x in target_ids)
    test_papers = remove_missing_titles(test_papers)
    # Fortunato review paper was published in 2010
    test_papers = year_lowpass_filter(test_papers, year=2010)

    X = test_papers.reset_index()
    y = X['target']
    clfs = [
        LogisticRegression(),
        SVC(probability=True),
        SVC(kernel='linear', probability=True),
        GaussianNB(),
        RandomForestClassifier(),
        AdaBoostClassifier(),
    ]
    for clf in clfs:
        experiment = PipelineExperiment(clf, seed_papers, random_state=args.seed)
        logger.info("========Pipeline:")
        logger.info(experiment.pipeline._final_estimator)
        experiment.run(X, y, num_target=len(target_papers))
        logger.info("\n\n")

if __name__ == "__main__":
    total_start = timer()
    logger = logging.getLogger(__name__)
    logger.info(" ".join(sys.argv))
    logger.info( '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()) )
    import argparse
    # parser.add_argument("--debug", action='store_true', help="output debugging info")
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--seed", type=int, default=999, help="random seed")
    parent_parser.add_argument("--debug", action='store_true', help="output debugging info")
    parent_parser.set_defaults(func=main)
    parser = argparse.ArgumentParser(description="run classifier experiments", parents=[parent_parser])

    subparsers = parser.add_subparsers()
    parser_test = subparsers.add_parser("test", parents=[parent_parser], help="test on subset")
    parser_test.add_argument('--subset-size', default=100000, help="size of random sample to test on (default: 100000)")
    parser_test.set_defaults(func=test)

    parser_quickcheck = subparsers.add_parser("quickcheck", parents=[parent_parser], help="quick check")
    parser_quickcheck.set_defaults(func=quickcheck)

    global args
    args = parser.parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug('debug mode is on')
    else:
        logger.setLevel(logging.INFO)
    if hasattr(args, 'func'):
        args.func(args)
    # main(args)
    total_end = timer()
    logger.info('all finished. total time: {}'.format(format_timespan(total_end-total_start)))
