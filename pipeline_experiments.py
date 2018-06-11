from __future__ import division
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

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics import classification_report

from utils.autoreview_utils import ItemSelector, DataFrameColumnTransformer, ClusterTransformer, AverageTfidfCosSimTransformer
# from features import avgDist_EF, avgDist_EF_avgTitleTfidfCosSim

from dotenv import load_dotenv
logger.debug("loading dotenv...")
success_load_dotenv = load_dotenv('admin.env')
if success_load_dotenv:
    logger.debug("dotenv loaded successfully.")
else:
    logger.warn("failed to load dotenv")
from mysql_connect import get_db_connection
db = get_db_connection('jp_autoreview')

from sqlalchemy.orm import sessionmaker
from models.jp_autoreview import PipelineTest

Session = sessionmaker(bind=db.engine)

class PipelineExperiment(object):

    """configure and run a pipeline for a review paper classifier"""

    def __init__(self, clf, transformer_list, seed_papers=None, random_state=999):
        """

        :clf: a classifier instance: the classifier to be used for this experiment instance (e.g., LogisticRegression())

        """

        self.clf = clf
        self.transformer_list = transformer_list
        self.seed_papers = seed_papers

        # these will be filled in at runtime
        self.time_fit = None  # in seconds
        self.time_predict = None  # in seconds
        self.num_candidates = None
        self.num_target_papers = None
        self.num_correctly_predicted = None

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
                transformer_list=self.transformer_list
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
        self.num_correctly_predicted = len(_top_predictions[_top_predictions.target==True])
        return _top_predictions.groupby('target')[id_colname].count()

    def run(self, X, y, random_state=None, num_target=None):
        if random_state is not None:
            self.load_random_state(random_state)

        self.num_candidates = len(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state)

        start = timer()
        logger.debug("Fitting pipeline...")
        self.fit(X_train, y_train)
        self.time_fit = timer()-start
        logger.debug("Done fitting. Took {}".format(format_timespan(self.time_fit)))

        start = timer()
        logger.debug("Predicting probabilities...")
        self.predict_proba(X)
        self.time_predict = timer()-start
        logger.debug("Done predicting. Took {}".format(format_timespan(self.time_predict)))

        self.num_target_in_candidates = int((X.target==True).sum())
        if num_target is None:
            # figure out the number of target papers
            num_target = self.num_target_in_candidates
        self.num_target_papers = int(num_target)
        logger.info("TOP PREDICTIONS: True is count of target papers in the top predicted")
        logger.info(self.top_predictions(n=num_target))
        self.score_correctly_predicted = self.num_correctly_predicted / self.num_target_papers

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


def log_to_db(experiment, data_dir, review_id, seed):
    session = Session()
    d = PipelineTest()
    d.review_paper_id = review_id
    d.random_seed = seed
    d.datadir = data_dir
    d.num_seed_papers = len(experiment.seed_papers)

    clf = experiment.pipeline._final_estimator
    d.clf = str(clf)
    d.clf_type = clf.__class__.__name__
    d.features = str([item[0] for item in experiment.transformer_list])

    d.time_fit = round(experiment.time_fit)
    d.time_predict = round(experiment.time_predict)

    d.num_candidates = experiment.num_candidates
    d.num_correctly_predicted = experiment.num_correctly_predicted
    d.num_target_in_candidates = experiment.num_target_in_candidates
    d.num_target_papers = experiment.num_target_papers
    d.score_correctly_predicted = experiment.score_correctly_predicted

    session.add(d)
    session.commit()

    session.close()

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
    # data_dir = "data/collect_haystack_2127048411_seed-3/"
    data_dir = args.datadir
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
        # SVC(probability=True),
        # SVC(kernel='linear', probability=True),
        # GaussianNB(),
        RandomForestClassifier(n_estimators=50),
        RandomForestClassifier(n_estimators=100),
        RandomForestClassifier(n_estimators=500),
        RandomForestClassifier(n_estimators=100, criterion="entropy"),
        RandomForestClassifier(n_estimators=500, criterion="entropy"),
        AdaBoostClassifier(n_estimators=500),
    ]
    for clf in clfs:
        experiment = PipelineExperiment(clf, seed_papers, random_state=args.seed)
        logger.info("========Pipeline:")
        logger.info(experiment.pipeline._final_estimator)
        experiment.run(X, y, num_target=len(target_papers))
        logger.info("\n\n")

def main(args):
    # data_dir = "data/collect_haystack_2127048411_seed-3/"
    data_dir = args.datadir
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
    test_papers = year_lowpass_filter(test_papers, year=args.year)
    ### TODO: MOVE ABOVE TO UTILS
    # logger.debug("There are {} target papers. {} of these appear in the haystack.".format(len(target_ids), test_papers['target'].sum()))
    logger.debug("There are {} target papers. {} of these appear in the haystack.".format(target_papers.Paper_ID.nunique(), test_papers['target'].sum()))

    logger.debug("\nSEED PAPERS: seed_papers.head()")
    logger.debug(seed_papers.head())
    logger.debug("\nTARGET PAPERS: target_papers.head()")
    logger.debug(target_papers.head())

    X = test_papers.reset_index()
    y = X['target']
    clfs = [
        LogisticRegression(random_state=args.seed),
        LogisticRegression(random_state=args.seed, class_weight='balanced'),
        LogisticRegression(random_state=args.seed, penalty='l1'),
        # SVC(probability=True, random_state=args.seed),  # this one doesn't perform well
        # SVC(probability=True, random_state=args.seed, class_weight='balanced'),  # this one takes a long time (8hours?)
        SVC(kernel='linear', probability=True, random_state=args.seed),
        SGDClassifier(loss='modified_huber', random_state=args.seed),
        GaussianNB(),
        RandomForestClassifier(n_estimators=50, random_state=args.seed),
        RandomForestClassifier(n_estimators=100, random_state=args.seed),
        RandomForestClassifier(n_estimators=500, random_state=args.seed),
        RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=args.seed),
        RandomForestClassifier(n_estimators=500, criterion="entropy", random_state=args.seed),
        AdaBoostClassifier(n_estimators=500, random_state=args.seed),
    ]
    transformer_list = [
        ('avg_distance_to_train', Pipeline([
            ('cl_feat', ClusterTransformer(seed_papers=seed_papers)),
        ])),
        ('ef', Pipeline([
            ('ef_feat', DataFrameColumnTransformer('EF')),
        ])),
    ]

    # if command line option is set, include a feature for similarity of titles
    if args.titles_cossim:
        transformer_list.append(
            ('avg_title_tfidf_cosine_similarity', Pipeline([
                ('title_feat', AverageTfidfCosSimTransformer(seed_papers=seed_papers, colname='title')),
            ]))
        )

    if args.save_best:
        from sklearn.externals import joblib
        best_model_dir = os.path.join(data_dir, "best_model_{:%Y%m%d%H%M%S%f}".format(datetime.now()))
        os.mkdir(best_model_dir)
        best_model_fname = os.path.join(best_model_dir, "best_model.pickle")

    best_score = 0
    for clf in clfs:
        experiment = PipelineExperiment(clf, transformer_list, seed_papers, random_state=args.seed)
        logger.info("\n========Pipeline:")
        # logger.info(experiment.pipeline.steps)
        # feature_union = experiment.pipeline.named_steps.get('union')
        # feature_names = [item[0] for item in feature_union.transformer_list]
        feature_names = [item[0] for item in transformer_list]
        logger.info("feature names: {}".format(feature_names))
        logger.info(experiment.pipeline._final_estimator)
        experiment.run(X, y, num_target=len(target_papers))
        if experiment.score_correctly_predicted > best_score:
            best_score = experiment.score_correctly_predicted
            if args.save_best:
                start = timer()
                logger.debug("This is the best model so far. Saving to {}...".format(best_model_fname))
                joblib.dump(experiment.pipeline, best_model_fname)
                logger.debug("Saved model in {}".format(format_timespan(timer()-start)))
        log_to_db(experiment, data_dir=data_dir, review_id=args.review_id, seed=args.dataset_seed)
        logger.info("\n")

if __name__ == "__main__":
    total_start = timer()
    logger = logging.getLogger(__name__)
    logger.info(" ".join(sys.argv))
    logger.info( '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now()) )
    import argparse
    # parser.add_argument("--debug", action='store_true', help="output debugging info")
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("datadir", help="data directory")
    parent_parser.add_argument("--seed", type=int, default=999, help="random seed")
    parent_parser.add_argument("--year", type=int, default=2010, help="publication year of the review paper")
    parent_parser.add_argument("--titles-cossim", action='store_true', help="use feature: average cosine similarity for titles")
    parent_parser.add_argument("-s", "--save-best", action='store_true', help="save the best performing model to disk")
    parent_parser.add_argument("--review-id", help="the paper id of the review article for these experimments")
    parent_parser.add_argument("--dataset-seed", type=int, help="this is the random seed used when separating the data into seed papers and target papers")
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
