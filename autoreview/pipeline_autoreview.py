# -*- coding: utf-8 -*-
#
import logging
logger = logging.getLogger(__name__)

import sys, os, time
from datetime import datetime
from timeit import default_timer as timer
try:
    from humanfriendly import format_timespan
except ImportError:
    def format_timespan(seconds):
        return "{:.2f} seconds".format(seconds)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion

from .util import prec_recall_f1_at_n, average_precision

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

    def top_predictions(self, n=200, id_colname='ID'):
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
        self.score_correctly_predicted = self.num_correctly_predicted / self.num_target_papers  # R-Precision score

        preds = self.predictions.target
        logger.debug("Precision, Recall, F1 scores at n:")
        self.prec_at_n = {}
        self.recall_at_n = {}
        self.f1_at_n = {}
        for n in [10, 50, 100, 500, 1000, 10000, 50000, 100000, 500000, 1e6, 5e6, 1e7, 5e7, 1e8]:
            n = int(n)
            n = min(n, len(preds))
            prec, recall, f1 = prec_recall_f1_at_n(preds, self.num_target_papers, n)
            self.prec_at_n[n] = prec
            self.recall_at_n[n] = recall
            self.f1_at_n[n] = f1
            logger.debug("n=={}: prec=={}, recall=={}, f1=={}".format(n, prec, recall, f1))
            if n >= len(preds):
                break
        self.average_precision = average_precision(preds, self.num_target_papers)
        logger.debug("average_precision=={}".format(self.average_precision))


