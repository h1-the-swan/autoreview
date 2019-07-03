# -*- coding: utf-8 -*-
#
import logging
logging.basicConfig(format='%(asctime)s %(name)s.%(lineno)d %(levelname)s : %(message)s',
        datefmt="%H:%M:%S",
        level=logging.INFO)
# logger = logging.getLogger(__name__)
logger = logging.getLogger('__main__').getChild(__name__)

import sys, os, time
from datetime import datetime
from timeit import default_timer as timer
try:
    from humanfriendly import format_timespan
except ImportError:
    def format_timespan(seconds):
        return "{:.2f} seconds".format(seconds)

from .config import Config
from .util import load_random_state, prepare_directory, load_spark_dataframe, save_pandas_dataframe_to_pickle, remove_seed_papers_from_test_set, remove_missing_titles, year_lowpass_filter, predict_ranks_from_data

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics import classification_report

from .util import ItemSelector, DataFrameColumnTransformer, ClusterTransformer, AverageTfidfCosSimTransformer

from .pipeline_autoreview import PipelineExperiment

class Autoreview(object):

    """Toplevel Autoreview object"""

    def __init__(self, id_list, citations, papers, outdir, sample_size, random_seed=None, id_colname='UID', citing_colname=None, cited_colname='cited_UID', config=None):
        """
        :id_list: list of strings: IDs for the seed set
        :citations: path to citations data
        :papers: path to papers data
        :outdir: output directory. will raise RuntimeError if the directory already exists
        :sample_size: integer: size of the seed set to split off from the initial. the rest will be used as target papers
        :random_seed: integer

        """
        self.id_list = id_list
        self.citations = citations
        self.papers = papers
        self.outdir = outdir
        self.sample_size = sample_size
        self.random_state = load_random_state(random_seed)
        self.id_colname = id_colname
        if citing_colname is not None:
            self.citing_colname = citing_colname
        else:
            self.citing_colname = id_colname
        self.cited_colname = cited_colname

        if config is not None:
            assert isinstance(config, Config)
            self._config = config
        else:
            self._config = Config()
        self.spark = self._config.spark

        self.best_model_pipeline_experiment = None

    def follow_citations(self, sdf, sdf_citations):
        """follow in- and out-citations

        :sdf: a spark dataframe with one column `ID` that contains the ids to follow in- and out-citations
        :sdf_citations: spark dataframe with citation data. columns are `ID` and `cited_ID`
        :returns: spark dataframe with one column `ID` that contains deduplicated IDs for in- and out-citations

        """
        sdf_outcitations = sdf_citations.join(sdf, on='ID', how='inner')
        _sdf_renamed = sdf.withColumnRenamed('ID', 'cited_ID')
        sdf_incitations = sdf_citations.join(_sdf_renamed, on='cited_ID')

        sdf_combined = self.combine_ids([sdf_incitations, sdf_outcitations])
        return sdf_combined

    def combine_ids(self, sdfs):
        """Given a list of spark dataframes with columns ['UID', 'cited_UID']
        return a dataframe with one column 'UID' containing all of the UIDs in both columns of the input dataframes
        """
        sdf_combined = self.spark.createDataFrame([], schema='ID string')
        for sdf in sdfs:
            # add 'UID' column
            sdf_combined = sdf_combined.union(sdf.select(['ID']))
            
            # add 'cited_UID' column (need to rename)
            sdf_combined = sdf_combined.union(sdf.select(['cited_ID']).withColumnRenamed('cited_ID', 'ID'))
        return sdf_combined.drop_duplicates()

    def get_papers_2_degrees_out(self):
        """For a list of paper IDs (in `self.id_list`),
        get all papers citing or cited by those, then repeat for 
        all these new papers.

        For the three sets of papers---seed, target, and test (candidate) papers--- 
        save a pickled pandas dataframe.
        :returns: pandas dataframes for seed, target, and test papers

        """
        df_id_list = pd.DataFrame(self.id_list, columns=['ID'])
        seed_papers, target_papers = train_test_split(df_id_list, train_size=self.sample_size, random_state=self.random_state)

        sdf_papers = load_spark_dataframe(self.papers, self.spark) 
        sdf_papers = sdf_papers.withColumnRenamed(self.id_colname, 'ID')
        sdf_papers = sdf_papers.dropna(subset=['cl'])

        sdf_seed = self.spark.createDataFrame(seed_papers[['ID']])
        sdf_target = self.spark.createDataFrame(target_papers[['ID']])

        outfname = os.path.join(self.outdir, 'seed_papers.pickle')
        logger.debug('saving seed papers to {}'.format(outfname))
        start = timer()
        df_seed = sdf_seed.join(sdf_papers, on='ID', how='inner').toPandas()
        save_pandas_dataframe_to_pickle(df_seed, outfname)
        logger.debug("done saving seed papers. took {}".format(format_timespan(timer()-start)))

        outfname = os.path.join(self.outdir, 'target_papers.pickle')
        logger.debug('saving target papers to {}'.format(outfname))
        start = timer()
        df_target = sdf_target.join(sdf_papers, on='ID', how='inner').toPandas()
        save_pandas_dataframe_to_pickle(df_target, outfname)
        logger.debug("done saving target papers. took {}".format(format_timespan(timer()-start)))

        sdf_citations = load_spark_dataframe(self.citations, self.spark)
        sdf_citations = sdf_citations.withColumnRenamed(self.citing_colname, 'ID')
        sdf_citations = sdf_citations.withColumnRenamed(self.cited_colname, 'cited_ID')

        # collect IDs for in- and out-citations
        sdf_combined = self.follow_citations(sdf_seed, sdf_citations)
        # do it all again to get second degree
        sdf_combined = self.follow_citations(sdf_combined, sdf_citations)
        outfname = os.path.join(self.outdir, 'test_papers.pickle')
        logger.debug("saving test papers to {}".format(outfname))
        start = timer()
        df_combined = sdf_combined.join(sdf_papers, on='ID', how='inner').toPandas()
        save_pandas_dataframe_to_pickle(df_combined, outfname)
        logger.debug("done saving test papers. took {}".format(format_timespan(timer()-start)))
        return df_seed, df_target, df_combined  # seed, target, test (candidate) papers


    def run(self):
        """Run and save output

        """
        try:
            prepare_directory(self.outdir)
            seed_papers, target_papers, candidate_papers = self.get_papers_2_degrees_out()

            test_papers = remove_seed_papers_from_test_set(candidate_papers, seed_papers)
            target_ids = set(target_papers.Paper_ID)
            test_papers['target'] = test_papers.Paper_ID.apply(lambda x: x in target_ids)
            test_papers = remove_missing_titles(test_papers)
            # TODO IMPLEMENT YEAR LOWPASS FILTER
            # test_papers = year_lowpass_filter(test_papers, year=args.year)
            logger.debug("There are {} target papers. {} of these appear in the haystack.".format(target_papers.Paper_ID.nunique(), test_papers['target'].sum()))

            logger.debug("\nSEED PAPERS: seed_papers.head()")
            logger.debug(seed_papers.head())
            logger.debug("\nTARGET PAPERS: target_papers.head()")
            logger.debug(target_papers.head())

            X = test_papers.reset_index()
            y = X['target']
            clfs = [
                LogisticRegression(random_state=self.random_state),
                LogisticRegression(random_state=self.random_state, class_weight='balanced'),
                LogisticRegression(random_state=self.random_state, penalty='l1'),
                # SVC(probability=True, random_state=args.seed),  # this one doesn't perform well
                # SVC(probability=True, random_state=args.seed, class_weight='balanced'),  # this one takes a long time (8hours?)
                SVC(kernel='linear', probability=True, random_state=self.random_state),
                SGDClassifier(loss='modified_huber', random_state=self.random_state),
                GaussianNB(),
                RandomForestClassifier(n_estimators=50, random_state=self.random_state),
                RandomForestClassifier(n_estimators=100, random_state=self.random_state),
                RandomForestClassifier(n_estimators=500, random_state=self.random_state),
                RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=self.random_state),
                RandomForestClassifier(n_estimators=500, criterion="entropy", random_state=self.random_state),
                AdaBoostClassifier(n_estimators=500, random_state=self.random_state),
            ]
            transformer_list = [
                ('avg_distance_to_train', Pipeline([
                    ('cl_feat', ClusterTransformer(seed_papers=seed_papers)),
                ])),
                ('ef', Pipeline([
                    ('ef_feat', DataFrameColumnTransformer('EF')),
                ])),
            ]

            # # if command line option is set, include a feature for similarity of titles
            # if args.titles_cossim:
            transformer_list.append(
                ('avg_title_tfidf_cosine_similarity', Pipeline([
                    ('title_feat', AverageTfidfCosSimTransformer(seed_papers=seed_papers, colname='title')),
                ]))
            )

            # if args.save_best:
            #     from sklearn.externals import joblib
            #     best_model_dir = os.path.join(data_dir, "best_model_{:%Y%m%d%H%M%S%f}".format(datetime.now()))
            #     os.mkdir(best_model_dir)
            #     best_model_fname = os.path.join(best_model_dir, "best_model.pickle")
            # else:
            #     best_model_fname = None

            from sklearn.externals import joblib
            best_model_dir = os.path.join(self.outdir, "best_model_{:%Y%m%d%H%M%S%f}".format(datetime.now()))
            os.mkdir(best_model_dir)
            best_model_fname = os.path.join(best_model_dir, "best_model.pickle")

            best_score = 0
            for clf in clfs:
                experiment = PipelineExperiment(clf, transformer_list, seed_papers, random_state=self.random_state)
                logger.info("\n========Pipeline:")
                # logger.info(experiment.pipeline.steps)
                # feature_union = experiment.pipeline.named_steps.get('union')
                # feature_names = [item[0] for item in feature_union.transformer_list]
                feature_names = [item[0] for item in transformer_list]
                logger.info("feature names: {}".format(feature_names))
                logger.info(experiment.pipeline._final_estimator)
                experiment.run(X, y, num_target=len(target_papers))

                # turn off db logging for now
                # session = Session()
                # try:
                #     db_rec_id = log_to_db(session, experiment, data_dir=data_dir, review_id=args.review_id, seed=args.dataset_seed)
                #     session.commit()
                # except Exception as e:
                #     logger.debug("Exception encountered when adding record to db! {}".format(e))
                #     session.rollback()
                #     db_rec_id = None
                # finally:
                #     session.close()
                if experiment.score_correctly_predicted > best_score:
                    best_score = experiment.score_correctly_predicted
                    # if args.save_best:
                    start = timer()
                    logger.debug("This is the best model so far. Saving to {}...".format(best_model_fname))
                    joblib.dump(experiment.pipeline, best_model_fname)
                    logger.debug("Saved model in {}".format(format_timespan(timer()-start)))
                    # best_rec_id = db_rec_id
                    self.best_model_pipeline_experiment = experiment
                logger.info("\n")

            logger.info("Done with experiments. Using best model: {}".format(self.best_model_pipeline_experiment.pipeline._final_estimator))
            logger.info("Scoring all test papers...")
            df_predictions = predict_ranks_from_data(self.best_model_pipeline_experiment.pipeline, test_papers)
            df_predictions = df_predictions[df_predictions.target==False].drop(columns='target')
            outfname = os.path.join(self.outdir, 'predictions.tsv')
            df_predictions.head(100).to_csv(outfname, sep='\t')
            
        finally:
            self._config.teardown()
